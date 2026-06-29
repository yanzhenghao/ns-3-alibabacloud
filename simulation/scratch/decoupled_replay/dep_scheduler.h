/*
 * Copyright (c) 2024, Alibaba Group;
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * ---
 * DECOUPLED REPLAY: Bucket-constrained flow scheduler with dependency graph.
 * [CORE NEW LOGIC - not extracted from SimAI]
 *
 * === Scheduling model ===
 *
 * Two per-flow gates + one per-bucket compute gap control flow injection:
 *
 *   1. Flow dependency (hard gate):
 *      A flow is eligible only after ALL flows listed in its parent_flow_id[]
 *      field have completed.  parent_flow_id[] holds FLOW ids — the genuine
 *      predecessor DAG authored by MockNcclGroup (NOT prev[], which holds RANK
 *      ids for recv-side sender identity).
 *
 *   2. Bucket constraint (hard gate):
 *      A flow is only eligible when its emission-order bucket is unlocked.
 *      Buckets advance in emission order (the order flows appear in the file),
 *      which faithfully reproduces the SimAI event-chain order — including the
 *      backward pass (decreasing layer_num) and the three loopstates
 *      (FWD/WG/IG) that share a layer_num. A single monotonic layer counter
 *      cannot represent either; the bucket frontier can.
 *
 *   3. Bucket compute gap (soft gate):
 *      When a bucket is fully complete, the next bucket is not unlocked
 *      immediately.  Instead the GPU compute time for the next bucket
 *      (compute_before_ns from the flow file's bucket metadata) is scheduled
 *      via Simulator::Schedule(...).  Only after this compute delay does the
 *      next bucket unlock and eligible flows are injected.
 *      compute_before_ns is sourced per-(layer, loopstate) from AIOB GPU
 *      profiling (fwd + wg + ig), written by SimAI's finalizeFlowFile.
 *
 * === Key invariants ===
 *
 *   - Single QP per flow (_QPS_PER_CONNECTION_ == 1)
 *   - Bucket 0 unlocks at Start() (after its own compute_before_ns, if any)
 *   - Higher buckets unlock in emission order after their compute gap
 *   - parent_flow_id[] entries are FLOW ids (validated against _states)
 *   - prev[] is recv-side rank metadata, NEVER a scheduling dependency
 *   - Empty/already-complete buckets cascade-forward without stalling
 *     (no flow-completion event fires for an empty bucket, so the frontier
 *      must self-advance — see advanceBucketFrontier()).
 */

#ifndef __DECOUPLED_DEP_SCHEDULER_H__
#define __DECOUPLED_DEP_SCHEDULER_H__

#include "common_types.h"
#include "flow_reader.h"
#include "flow_sender.h"

#include <unordered_map>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <iostream>
#include <functional>
#include <cassert>

// ============================================================================
// DepScheduler
// ============================================================================

class DepScheduler {
public:
    struct FlowState {
        FlowFileRecord record;
        bool completed = false;
        bool scheduled = false;
        uint64_t scheduled_time_ns = 0;   // when SendFlow was called
        uint64_t completed_time_ns = 0;   // when qp_finish completed the flow

        // Dependency graph (built from parent_flow_id)
        int pending_deps = 0;              // count of uncompleted parent_flow_id[] flows
        std::vector<uint32_t> dependents;  // flows that list this flow in their parent_flow_id[]

        // For expeRecvHash: buffer allocated for receive expectation
        char* recv_buffer = nullptr;
    };

    DepScheduler() = default;

    // ------------------------------------------------------------------
    // Init: build dependency graph and bucket tracking from loaded flows
    // ------------------------------------------------------------------
    bool Init(const std::vector<FlowFileRecord>& flows, const BucketMetaMap& bucket_meta) {
        _total_flows = flows.size();
        _completed_flows = 0;
        _states.clear();
        _in_flight.clear();
        _bucket_flow_count.clear();
        _bucket_completed.clear();
        _bucket_compute_before.clear();
        _bucket_layer.clear();
        _bucket_loopstate.clear();
        _requests.clear();
        _request_by_id.clear();
        _current_unlocked_bucket.clear();
        _max_bucket.clear();
        _bucket_unlock_scheduled.clear();
        _jobs.clear();
        _started = false;

        // Store bucket compute timing + phase labels, keyed by (job_id, bucket_id).
        for (const auto& kv : bucket_meta) {
            uint32_t job = kv.second.job_id;
            JobBucket jb{job, (int)kv.second.bucket_id};
            _bucket_compute_before[jb] = kv.second.compute_before_ns;
            _bucket_layer[jb] = kv.second.layer_num;
            _bucket_loopstate[jb] = kv.second.loopstate;
            _jobs.insert(job);
            if ((int)kv.second.bucket_id > _max_bucket[job]) _max_bucket[job] = (int)kv.second.bucket_id;
        }

        // Build flow_id -> index
        std::unordered_map<uint32_t, size_t> id_to_idx;
        for (size_t i = 0; i < flows.size(); i++) {
            id_to_idx[flows[i].flow_id] = i;
        }

        // Pass 1: create FlowState for each flow.
        // Dependencies are gated on parent_flow_id (FLOW ids — the genuine DAG),
        // NOT prev[] (which holds RANK ids for recv-side sender identity).
        for (const auto& rec : flows) {
            FlowState st;
            st.record = rec;
            st.pending_deps = (int)rec.parent_flow_id.size();

            _bucket_flow_count[JobBucket{rec.job_id, (int)rec.bucket_id}]++;
            _jobs.insert(rec.job_id);
            _states[rec.flow_id] = st;
        }

        // Pass 2: build dependents (reverse edges) from parent_flow_id and validate
        // that every parent is a real flow-id (eliminates the rank-as-flow-id corruption).
        for (auto& kv : _states) {
            uint32_t fid = kv.first;
            FlowState& st = kv.second;
            for (uint32_t pid : st.record.parent_flow_id) {
                if (_states.find(pid) == _states.end()) {
                    std::cerr << "[DepScheduler] ERROR: flow " << fid
                              << " depends on non-existent parent flow " << pid
                              << " (corrupt dependency graph — parent_flow_id must be a real flow-id)"
                              << std::endl;
                    return false;
                }
                _states[pid].dependents.push_back(fid);
            }
        }

        // Init invariant: bucket flow counts sum to total flows
        uint64_t bucket_sum = 0;
        for (const auto& kv : _bucket_flow_count) bucket_sum += kv.second;
        if (bucket_sum != _total_flows) {
            std::cerr << "[DepScheduler] ERROR: bucket flow count sum (" << bucket_sum
                      << ") != total flows (" << _total_flows << ")" << std::endl;
            return false;
        }

        // Allocate FlowRequest objects for each flow (heap-allocated, stable pointers
        // for the simulation lifetime -- SendFlow stores the pointer in sender_src_port_map).
        _requests.resize(_total_flows);
        _request_by_id.clear();
        for (size_t i = 0; i < flows.size(); i++) {
            _requests[i] = new FlowRequest();
            const auto& rec = flows[i];
            _requests[i]->flowTag.current_flow_id = rec.flow_id;
            _requests[i]->flowTag.nvls_on =
                (rec.conn_type == "NVLS" || rec.conn_type == "NVLS_TREE");
            _requests[i]->srcRank = rec.src;
            _requests[i]->dstRank = rec.dst;
            _requests[i]->reqCount = rec.flow_size;
            _request_by_id[rec.flow_id] = _requests[i];
        }

        // Setup expeRecvHash for each flow (receive expectation)
        for (const auto& rec : flows) {
            int tag = (int)rec.flow_id;
            auto key = std::make_pair(tag, std::make_pair((int)rec.src, (int)rec.dst));
            if (expeRecvHash.find(key) == expeRecvHash.end()) {
                RecvTask t;
                t.src = (int)rec.src;
                t.dest = (int)rec.dst;
                t.count = rec.flow_size;
                t.type = 1;
                t.msg_handler = [](void*){ /* no-op: completion tracked via g_on_flow_completed */ };
                char* buf = new char[168]();
                t.fun_arg = buf;
                expeRecvHash[key] = t;

                if (id_to_idx.count(rec.flow_id)) {
                    _states[rec.flow_id].recv_buffer = buf;
                }
            }
        }

        int total_buckets = 0;
        for (const auto& kv : _max_bucket) total_buckets += kv.second + 1;
        std::cout << "[DepScheduler] Init: " << _total_flows << " flows, "
                  << total_buckets << " buckets across " << _jobs.size()
                  << " job(s)" << std::endl;

        return true;
    }

    // ------------------------------------------------------------------
    // Start: kick off the simulation — unlock bucket 0 (after its compute gap)
    // ------------------------------------------------------------------
    void Start() {
        if (_started) return;
        _started = true;
        // Unlock bucket 0 of EVERY job independently. The frontier starts at -1
        // (nothing unlocked) so bucket-0 flows stay gated until their compute gap
        // elapses; DoUnlockBucket is what actually unlocks bucket 0 and schedules
        // its flows. Jobs with no bucket-0 gap unlock immediately. This is what
        // lets multiple jobs' first collectives fire concurrently while each still
        // respects its own compute pacing.
        for (uint32_t job : _jobs) {
            _current_unlocked_bucket[job] = -1;
            _bucket_unlock_scheduled[job] = false;
            uint64_t delay = 0;
            auto it = _bucket_compute_before.find(JobBucket{job, 0});
            if (it != _bucket_compute_before.end()) delay = it->second;
            if (delay > 0) {
                _bucket_unlock_scheduled[job] = true;
                std::cout << "[DepScheduler] Job " << job << " bucket 0 compute gap: "
                          << delay << " ns" << std::endl;
                Simulator::Schedule(NanoSeconds(delay),
                                    &DepScheduler::DoUnlockBucket, this, packJB(job, 0));
            } else {
                // No gap → unlock bucket 0 now (DoUnlockBucket cascades + schedules).
                DoUnlockBucket(packJB(job, 0));
            }
        }
    }

    // ------------------------------------------------------------------
    // ScheduleReadyFlows: find eligible flows and schedule them
    // ------------------------------------------------------------------
    void ScheduleReadyFlows() {
        int scheduled_this_round = 0;
        for (auto& kv : _states) {
            uint32_t fid = kv.first;
            FlowState& st = kv.second;

            if (st.scheduled || st.completed) continue;

            // Gate 1: all parent_flow_id[] dependencies must be satisfied
            if (st.pending_deps > 0) continue;

            // Gate 2: this flow's bucket must be unlocked by ITS OWN job's frontier
            // (per-job — jobs advance independently and overlap on the wire).
            uint32_t job = st.record.job_id;
            auto fit = _current_unlocked_bucket.find(job);
            if (fit == _current_unlocked_bucket.end() ||
                (int)st.record.bucket_id > fit->second) {
                continue;
            }

            // Flow is eligible — schedule immediately (no artificial inter-flow delay).
            // Timing is determined by NS3 network simulation + bucket compute gaps.
            st.scheduled = true;
            st.scheduled_time_ns = Simulator::Now().GetNanoSeconds();
            _in_flight.insert(fid);

            Simulator::Schedule(NanoSeconds(0),
                              &DepScheduler::DoSendFlow, this, fid);

            scheduled_this_round++;
        }

        if (scheduled_this_round > 0) {
            NS_LOG_DEBUG("[DepScheduler] Scheduled " << scheduled_this_round
                         << " flows this round, " << _in_flight.size()
                         << " total in-flight");
        }
    }

    // ------------------------------------------------------------------
    // OnFlowCompleted: called from qp_finish via g_on_flow_completed
    // ------------------------------------------------------------------
    void OnFlowCompleted(uint32_t flow_id) {
        auto it = _states.find(flow_id);
        if (it == _states.end()) {
            std::cerr << "[DepScheduler] WARNING: OnFlowCompleted for unknown flow "
                      << flow_id << std::endl;
            return;
        }

        FlowState& st = it->second;
        if (st.completed) {
            return;  // already completed (can happen with multiple QPs)
        }

        st.completed = true;
        st.completed_time_ns = Simulator::Now().GetNanoSeconds();
        _completed_flows++;
        _in_flight.erase(flow_id);
        uint32_t job = st.record.job_id;
        _bucket_completed[JobBucket{job, (int)st.record.bucket_id}]++;

        NS_LOG_DEBUG("[DepScheduler] Flow " << flow_id << " completed (job "
                     << job << " bucket " << st.record.bucket_id << "), " << _completed_flows
                     << "/" << _total_flows << " total, in-flight: " << _in_flight.size());

        // Decrement pending_deps for all dependent flows
        for (uint32_t dep_fid : st.dependents) {
            auto dep_it = _states.find(dep_fid);
            if (dep_it != _states.end()) {
                dep_it->second.pending_deps--;
                NS_LOG_DEBUG("[DepScheduler] Flow " << dep_fid
                             << " pending_deps now " << dep_it->second.pending_deps);
            }
        }

        // Advance ONLY the completing flow's job frontier (others are independent).
        advanceBucketFrontier(job);
        if (!_bucket_unlock_scheduled[job]) {
            ScheduleReadyFlows();
        }
    }

    // ------------------------------------------------------------------
    // DoSendFlow: public callback from Simulator::Schedule
    // ------------------------------------------------------------------
    void DoSendFlow(uint32_t flow_id) {
        auto it = _states.find(flow_id);
        if (it == _states.end()) return;

        FlowState& st = it->second;
        auto rit = _request_by_id.find(flow_id);
        if (rit == _request_by_id.end()) {
            std::cerr << "[DepScheduler] ERROR: no request found for flow "
                      << flow_id << std::endl;
            return;
        }
        FlowRequest* req = rit->second;

        NS_LOG_DEBUG("[DepScheduler] Sending flow " << flow_id
                     << " src=" << st.record.src << " dst=" << st.record.dst
                     << " size=" << st.record.flow_size
                     << " at tick=" << Simulator::Now().GetNanoSeconds());

        SendFlow((int)st.record.src, (int)st.record.dst,
                 st.record.flow_size,
                 [](void*){}, nullptr,
                 (int)flow_id, req);
    }

    // ------------------------------------------------------------------
    // AllCompleted
    // ------------------------------------------------------------------
    bool AllCompleted() const {
        return _completed_flows == _total_flows;
    }

    // ------------------------------------------------------------------
    // LastCompletionTime
    // ------------------------------------------------------------------
    uint64_t LastCompletionTime() const {
        uint64_t max_t = 0;
        for (const auto& kv : _states) {
            if (kv.second.completed_time_ns > max_t) {
                max_t = kv.second.completed_time_ns;
            }
        }
        return max_t;
    }

    // ------------------------------------------------------------------
    // VerifyCompletion
    // ------------------------------------------------------------------
    bool VerifyCompletion() {
        if (!AllCompleted()) {
            std::cerr << "[DepScheduler] WARNING: Not all flows completed! "
                      << _completed_flows << "/" << _total_flows << std::endl;

            for (const auto& kv : _states) {
                if (!kv.second.completed) {
                    std::cerr << "  Incomplete: flow " << kv.first
                              << " bucket=" << kv.second.record.bucket_id
                              << " pending_deps=" << kv.second.pending_deps
                              << " scheduled=" << kv.second.scheduled
                              << std::endl;
                }
            }
            return false;
        }

        std::cout << "[DepScheduler] All " << _total_flows << " flows completed." << std::endl;
        std::cout << "[DepScheduler] Last completion time: "
                  << LastCompletionTime() << " ns" << std::endl;

        return true;
    }

    // ------------------------------------------------------------------
    // DumpBucketStats: per-bucket timing statistics
    // ------------------------------------------------------------------
    void DumpBucketStats() {
        std::cout << "\n========== Per-Bucket Timing Statistics ==========\n";
        std::map<JobBucket, std::vector<uint64_t>> bucket_fcts;

        for (const auto& kv : _states) {
            const FlowState& st = kv.second;
            if (st.completed && st.scheduled_time_ns > 0) {
                uint64_t fct = st.completed_time_ns - st.scheduled_time_ns;
                bucket_fcts[JobBucket{st.record.job_id, (int)st.record.bucket_id}].push_back(fct);
            }
        }

        for (uint32_t job : _jobs) {
            int maxb = _max_bucket.count(job) ? _max_bucket[job] : -1;
            for (int b = 0; b <= maxb; b++) {
                JobBucket jb{job, b};
                auto it = bucket_fcts.find(jb);
                int layer = _bucket_layer.count(jb) ? _bucket_layer[jb] : -1;
                int loopstate = _bucket_loopstate.count(jb) ? _bucket_loopstate[jb] : -1;
                if (it == bucket_fcts.end() || it->second.empty()) {
                    std::cout << "Job " << job << " bucket " << b << " (layer=" << layer
                              << ", loopstate=" << loopstate << "): no completed flows\n";
                    continue;
                }

                const auto& fcts = it->second;
                uint64_t sum = 0, min_fct = UINT64_MAX, max_fct = 0;
                for (uint64_t f : fcts) {
                    sum += f;
                    if (f < min_fct) min_fct = f;
                    if (f > max_fct) max_fct = f;
                }
                uint64_t avg = sum / fcts.size();

                int total_in_bucket = _bucket_flow_count[jb];
                int completed_in_bucket = _bucket_completed[jb];

                std::cout << "Job " << job << " bucket " << b << " (layer=" << layer
                          << ", loopstate=" << loopstate << "): "
                          << completed_in_bucket << "/" << total_in_bucket << " flows, "
                          << "avg FCT=" << avg << " ns, "
                          << "min=" << min_fct << " ns, "
                          << "max=" << max_fct << " ns\n";
            }
        }
        std::cout << "==================================================\n";
    }

    // ------------------------------------------------------------------
    // VerifyDAG: DFS-based cycle detection on the dependency graph
    // ------------------------------------------------------------------
    bool VerifyDAG() {
        // Iterative (explicit-stack) DFS with white/gray/black coloring. Ring
        // collectives produce long linear parent_flow_id chains, so a recursive
        // DFS would recurse O(chain length) deep and stack-overflow on large
        // collectives — fatal now that VerifyDAG runs unconditionally.
        std::unordered_map<uint32_t, int> color;  // 0=white, 1=gray, 2=black
        for (const auto& kv : _states) color[kv.first] = 0;

        for (const auto& kv0 : _states) {
            if (color[kv0.first] != 0) continue;
            // frame = (flow id, index of next dependent to visit)
            std::vector<std::pair<uint32_t, size_t>> stack;
            stack.push_back({kv0.first, 0});
            color[kv0.first] = 1;
            while (!stack.empty()) {
                uint32_t fid = stack.back().first;
                size_t& next = stack.back().second;
                const FlowState& st = _states[fid];
                if (next < st.dependents.size()) {
                    uint32_t dep = st.dependents[next++];
                    if (color[dep] == 1) {
                        std::cerr << "[DepScheduler] ERROR: Cycle detected involving flow "
                                  << dep << std::endl;
                        std::cerr << "[DepScheduler] DAG verification FAILED" << std::endl;
                        return false;
                    }
                    if (color[dep] == 0) {
                        color[dep] = 1;
                        stack.push_back({dep, 0});
                    }
                } else {
                    color[fid] = 2;  // done
                    stack.pop_back();
                }
            }
        }

        std::cout << "[DepScheduler] DAG verification PASSED (no cycles)" << std::endl;
        return true;
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    ~DepScheduler() {
        for (auto& kv : _states) {
            delete[] kv.second.recv_buffer;
        }
        for (auto* req : _requests) {
            delete req;
        }
    }

private:
    std::unordered_map<uint32_t, FlowState> _states;
    uint32_t _total_flows = 0;
    uint32_t _completed_flows = 0;
    std::set<uint32_t> _in_flight;

    // Per-job bucket frontier + compute gap (emission-order, independent per job).
    // Multiple training jobs can be packed into one flow file (FLOWFMT v4 job_id);
    // each job advances its OWN frontier so their collectives overlap in NS3 time
    // and contend on the wire, instead of running strictly one-after-another.
    using JobBucket = std::pair<uint32_t, int>;  // (job_id, bucket_id)
    // Simulator::Schedule binds a single scalar arg, so pack (job_id, bucket) into
    // one int for the DoUnlockBucket callback. job_id in the high 16 bits, bucket
    // in the low 16 (both comfortably fit: jobs and per-job buckets are small).
    static int packJB(uint32_t job, int bucket) { return (int)((job << 16) | (bucket & 0xFFFF)); }
    static uint32_t jbJob(int packed) { return (uint32_t)((packed >> 16) & 0xFFFF); }
    static int jbBucket(int packed) { return packed & 0xFFFF; }
    std::map<uint32_t, int> _current_unlocked_bucket;  // job_id -> unlocked frontier
    std::map<uint32_t, int> _max_bucket;               // job_id -> max bucket
    std::map<uint32_t, bool> _bucket_unlock_scheduled; // job_id -> compute-gap timer pending
    std::set<uint32_t> _jobs;                           // all job_ids present
    bool _started = false;
    std::map<JobBucket, int> _bucket_flow_count;       // (job,bucket) -> total flows
    std::map<JobBucket, int> _bucket_completed;        // (job,bucket) -> completed flows
    std::map<JobBucket, uint64_t> _bucket_compute_before;  // (job,bucket) -> compute_before_ns
    std::map<JobBucket, int> _bucket_layer;            // (job,bucket) -> layer_num (stats)
    std::map<JobBucket, int> _bucket_loopstate;        // (job,bucket) -> loopstate (stats)

    // ------------------------------------------------------------------
    // advanceBucketFrontier: skip fully-complete / empty buckets, scheduling
    // the next non-empty bucket's compute gap when needed.
    //
    // This is the single place that decides whether to unlock the next bucket.
    // It self-advances over empty/already-complete buckets (Step 5: no
    // flow-completion event fires for an empty bucket, so the frontier must
    // cascade-forward here, or the simulation would stall).
    // ------------------------------------------------------------------
    void advanceBucketFrontier(uint32_t job) {
        int& frontier = _current_unlocked_bucket[job];
        int maxb = _max_bucket.count(job) ? _max_bucket[job] : -1;
        while (frontier <= maxb && !_bucket_unlock_scheduled[job]) {
            int b = frontier;
            int total = _bucket_flow_count[JobBucket{job, b}];
            int completed = _bucket_completed[JobBucket{job, b}];
            if (total == 0 || completed >= total) {
                // Bucket (job,b) is done (or empty) — advance to b+1.
                int next = b + 1;
                if (next > maxb) {
                    return;  // this job's buckets exhausted
                }
                uint64_t delay = 0;
                auto it = _bucket_compute_before.find(JobBucket{job, next});
                if (it != _bucket_compute_before.end() && it->second > 0) {
                    delay = it->second;
                }
                if (delay > 0) {
                    _bucket_unlock_scheduled[job] = true;
                    NS_LOG_DEBUG("[DepScheduler] Job " << job << " bucket " << b
                                 << " complete, scheduling bucket " << next
                                 << " unlock after " << delay << " ns");
                    Simulator::Schedule(NanoSeconds(delay),
                                        &DepScheduler::DoUnlockBucket, this, packJB(job, next));
                    return;  // wait for the scheduled callback
                } else {
                    NS_LOG_DEBUG("[DepScheduler] Job " << job << " bucket " << b
                                 << " complete, unlocking bucket " << next << " immediately");
                    frontier = next;
                    // continue loop: next may also be empty/complete
                }
            } else {
                return;  // bucket (job,b) still has in-flight flows
            }
        }
    }

    // ------------------------------------------------------------------
    // DoUnlockBucket: scheduled callback after a bucket's compute delay.
    // The packed arg carries (job_id, bucket) — see packJB.
    // ------------------------------------------------------------------
    void DoUnlockBucket(int packed) {
        uint32_t job = jbJob(packed);
        int bucket_to_unlock = jbBucket(packed);
        int& frontier = _current_unlocked_bucket[job];
        // Valid: unlocking this job's next bucket (advance), or its bucket 0 from Start().
        if (bucket_to_unlock != frontier + 1 && bucket_to_unlock != frontier) {
            std::cerr << "[DepScheduler] WARNING: job " << job << " DoUnlockBucket("
                      << bucket_to_unlock << ") but expected " << (frontier + 1)
                      << " (stale or reordered callback?)" << std::endl;
        }
        frontier = bucket_to_unlock;
        _bucket_unlock_scheduled[job] = false;
        std::cout << "[DepScheduler] Job " << job << " bucket " << bucket_to_unlock
                  << " unlocked (compute delay complete)" << std::endl;
        // The newly-unlocked bucket may itself be empty/complete — cascade forward.
        advanceBucketFrontier(job);
        ScheduleReadyFlows();
    }

    // Heap-allocated FlowRequest objects (lifetime = simulation duration)
    std::vector<FlowRequest*> _requests;
    std::unordered_map<uint32_t, FlowRequest*> _request_by_id;
};

#endif // __DECOUPLED_DEP_SCHEDULER_H__
