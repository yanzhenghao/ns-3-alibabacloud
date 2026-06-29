/*
 * Copyright (c) 2024, Alibaba Group;
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * ---
 * DECOUPLED REPLAY: Complete flow file parser.
 * Parses layer metadata header + 16-field flow body.
 * Based on: loadFlowsFromFile() in MockNcclGroup.cc:2245-2265
 *
 * Flow file format (FLOWFMT v4):
 *   [line 1]    FLOWFMT <version> <total_flows> <bucket_count> [send_lat_us]
 *   [line 2..B] bucket: <id>  layer: <N>  loopstate: <S>  total_flows: <F>  compute_before_ns: <C>  job_id: <J>
 *   [body]      flow_id src dest flow_size channel_id chunk_id chunk_count conn_type
 *               start_time pg maxPacketCount port dport
 *               np prev[0..np-1]                 (prev = RANK ids, recv-side identity)
 *               npar parent_flow_id[0..npar-1]   (FLOW ids = the dependency DAG)
 *               nchi child_flow_id[0..nchi-1]    (FLOW ids = successor list)
 *               layer_num group_type op loopstate job_id
 */

#ifndef __DECOUPLED_FLOW_READER_H__
#define __DECOUPLED_FLOW_READER_H__

#include "common_types.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <iostream>

// Flow-file format version. MUST stay in sync with the writer in
// MockNcclGroup.cc::finalizeFlowFile ("FLOWFMT 3 ..."). Bump on any body-format change.
// v3: bucket metadata keyed by (layer, loopstate) in emission order.
static constexpr uint32_t FLOW_FILE_FORMAT_VERSION = 4;

// ============================================================================
// FlowFileRecord: parsed flow body record (16 fields)
// ============================================================================

struct FlowFileRecord {
    uint32_t flow_id = 0;
    uint32_t src = 0;
    uint32_t dst = 0;
    uint64_t flow_size = 0;
    int channel_id = 0;
    int chunk_id = 0;
    int chunk_count = 0;
    std::string conn_type;
    double start_time = 0.0;
    uint32_t pg = 0;
    uint32_t maxPacketCount = 0;
    uint32_t port = 0;
    uint32_t dport = 0;
    std::vector<uint32_t> prev;            // RANK ids (recv-side sender identity, NOT a scheduling dep)
    std::vector<uint32_t> parent_flow_id;  // FLOW ids: the genuine predecessor dependency DAG
    std::vector<uint32_t> child_flow_id;   // FLOW ids: successor list
    uint32_t layer_num = 0;
    uint32_t group_type = 0;
    uint32_t op = 0;
    uint32_t loopstate = 0;
    uint32_t bucket_id = 0;   // emission-order bucket (reconstructed by LoadFlows, job-local)
    uint32_t job_id = 0;      // FLOWFMT v4: source job (0 for single-task files)

    bool valid() const {
        return flow_size > 0 || (src != dst);
    }
};

// ============================================================================
// BucketMeta: per-(layer, loopstate) bucket in emission order
// ============================================================================

struct BucketMeta {
    uint32_t bucket_id = 0;
    uint32_t layer_num = 0;
    uint32_t loopstate = 0;
    uint32_t total_flows = 0;
    uint64_t compute_before_ns = 0;  // GPU compute time before this bucket's communication
    uint32_t job_id = 0;             // FLOWFMT v4: source job (0 for single-task files)
};

// Keyed by (job_id, bucket_id) packed into a uint64 (job in the high 32 bits),
// so job-local bucket ids from different jobs don't collide when several jobs are
// packed into one file.
using BucketMetaMap = std::map<uint64_t, BucketMeta>;
static inline uint64_t bucketMetaKey(uint32_t job_id, uint32_t bucket_id) {
    return ((uint64_t)job_id << 32) | bucket_id;
}


// ============================================================================
// LoadFlows: Parse complete flow file
// Extracted from: MockNcclGroup.cc:2245-2265 (loadFlowsFromFile)
// ============================================================================

// Extracted from: MockNcclGroup.cc:2245-2265 (parse loop pattern)
// send_lat_us_out (optional): receives the per-flow send latency (us) recorded in
// the flow-file header, so the replay can match the value the coupled export used.
inline std::vector<FlowFileRecord> LoadFlows(const std::string& flow_file_path,
                                              BucketMetaMap& bucket_meta_out,
                                              int* send_lat_us_out = nullptr) {
    std::vector<FlowFileRecord> flows;
    bucket_meta_out.clear();

    std::ifstream ff(flow_file_path);
    if (!ff.is_open()) {
        std::cerr << "[LoadFlows] ERROR: Cannot open flow file: "
                  << flow_file_path << std::endl;
        return flows;
    }

    // ── Parse header: FLOWFMT <version> <total_flows> <bucket_count> ──
    std::string header_line;
    if (!std::getline(ff, header_line) || header_line.empty()) {
        std::cerr << "[LoadFlows] ERROR: Empty or invalid flow file: "
                  << flow_file_path << std::endl;
        return flows;
    }

    std::istringstream hs(header_line);
    std::string fmt_tag;
    uint32_t version = 0;
    uint32_t total = 0;
    uint32_t bucket_count = 0;
    hs >> fmt_tag;
    if (fmt_tag != "FLOWFMT") {
        std::cerr << "[LoadFlows] ERROR: missing FLOWFMT version token in header (got '"
                  << fmt_tag << "'). This file is an old/incompatible format — regenerate it."
                  << std::endl;
        return flows;
    }
    hs >> version >> total >> bucket_count;
    if (version != FLOW_FILE_FORMAT_VERSION) {
        std::cerr << "[LoadFlows] ERROR: flow file format version " << version
                  << " != expected " << FLOW_FILE_FORMAT_VERSION
                  << ". Regenerate the flow file." << std::endl;
        return flows;
    }

    // Optional 5th header token: the send_lat (us) the coupled export run used.
    // Absent in older v3 files → leave the caller's default in place.
    if (send_lat_us_out) {
        int sl = -1;
        if (hs >> sl && sl > 0) *send_lat_us_out = sl;
    }

    if (total == 0) {
        std::cerr << "[LoadFlows] WARNING: Flow file header says 0 flows." << std::endl;
        return flows;
    }

    flows.reserve(total);

    // ── Parse bucket metadata ──
    // Format: "bucket: <id>  layer: <N>  loopstate: <S>  total_flows: <F>  compute_before_ns: <C>"
    for (uint32_t b = 0; b < bucket_count; b++) {
        std::string meta_line;
        if (!std::getline(ff, meta_line)) {
            std::cerr << "[LoadFlows] ERROR: Expected bucket metadata line "
                      << b << " but got EOF" << std::endl;
            break;
        }
        BucketMeta bm;
        std::string kw_bucket, kw_layer, kw_loop, kw_flows, kw_compute;
        std::istringstream ms(meta_line);
        ms >> kw_bucket >> bm.bucket_id;            // "bucket:" id
        ms >> kw_layer  >> bm.layer_num;            // "layer:" N
        ms >> kw_loop   >> bm.loopstate;            // "loopstate:" S
        ms >> kw_flows  >> bm.total_flows;          // "total_flows:" F
        ms >> kw_compute >> bm.compute_before_ns;   // "compute_before_ns:" C
        // Optional "job_id: <J>" (FLOWFMT v4). Absent → job 0.
        { std::string kw_job; uint32_t j;
          if (ms >> kw_job >> j && kw_job == "job_id:") bm.job_id = j; }
        if (!ms.fail() &&
            kw_bucket == "bucket:" && kw_layer == "layer:" &&
            kw_loop == "loopstate:" && kw_flows == "total_flows:" &&
            kw_compute == "compute_before_ns:") {
            bucket_meta_out[bucketMetaKey(bm.job_id, bm.bucket_id)] = bm;
        } else {
            std::cerr << "[LoadFlows] WARNING: Bad bucket metadata line, skipping: "
                      << meta_line << std::endl;
        }
    }

    // ── Parse flow body ──
    uint32_t body_line_num = 1 + bucket_count;  // header + metadata lines
    std::string line;
    while (std::getline(ff, line)) {
        body_line_num++;
        if (line.empty()) continue;

        std::istringstream is(line);
        FlowFileRecord r;

        // Fields 1-8
        if (!(is >> r.flow_id >> r.src >> r.dst >> r.flow_size
                  >> r.channel_id >> r.chunk_id >> r.chunk_count
                  >> r.conn_type)) {
            std::cerr << "[LoadFlows] ERROR: Line " << body_line_num
                      << " truncated (cannot read fields 1-8)" << std::endl;
            continue;
        }

        // Fields 9-14: start_time, pg, maxPacketCount, port, dport, np
        double st; uint32_t pg, mpc, port, dport; uint32_t np;
        if (!(is >> st >> pg >> mpc >> port >> dport >> np)) {
            std::cerr << "[LoadFlows] ERROR: Line " << body_line_num
                      << " truncated (cannot read fields 9-14)" << std::endl;
            continue;
        }
        r.start_time = st;
        r.pg = pg;
        r.maxPacketCount = mpc;
        r.port = port;
        r.dport = dport;

        // Field prev[] (variable length)
        r.prev.reserve(np);
        for (uint32_t j = 0; j < np; j++) {
            uint32_t pid;
            if (!(is >> pid)) {
                std::cerr << "[LoadFlows] ERROR: Line " << body_line_num
                          << " truncated (prev[" << j << "/" << np << "])" << std::endl;
                break;
            }
            r.prev.push_back(pid);
        }

        // parent_flow_id[] (FLOW ids — the genuine dependency DAG the scheduler gates on)
        uint32_t npar = 0;
        if (!(is >> npar)) {
            std::cerr << "[LoadFlows] ERROR: Line " << body_line_num
                      << " truncated (cannot read npar)" << std::endl;
            continue;
        }
        r.parent_flow_id.reserve(npar);
        for (uint32_t j = 0; j < npar; j++) {
            uint32_t pid;
            if (!(is >> pid)) {
                std::cerr << "[LoadFlows] ERROR: Line " << body_line_num
                          << " truncated (parent_flow_id[" << j << "/" << npar << "])" << std::endl;
                break;
            }
            r.parent_flow_id.push_back(pid);
        }

        // child_flow_id[] (FLOW ids — successor list)
        uint32_t nchi = 0;
        if (!(is >> nchi)) {
            std::cerr << "[LoadFlows] ERROR: Line " << body_line_num
                      << " truncated (cannot read nchi)" << std::endl;
            continue;
        }
        r.child_flow_id.reserve(nchi);
        for (uint32_t j = 0; j < nchi; j++) {
            uint32_t cid;
            if (!(is >> cid)) {
                std::cerr << "[LoadFlows] ERROR: Line " << body_line_num
                          << " truncated (child_flow_id[" << j << "/" << nchi << "])" << std::endl;
                break;
            }
            r.child_flow_id.push_back(cid);
        }

        // Fields: layer_num, group_type, op, loopstate (16 fields total)
        if (!(is >> r.layer_num >> r.group_type >> r.op >> r.loopstate)) {
            std::cerr << "[LoadFlows] ERROR: Line " << body_line_num
                      << " truncated (cannot read layer_num/group_type/op/loopstate)"
                      << std::endl;
            continue;
        }
        // Optional trailing job_id (FLOWFMT v4); absent → job 0.
        { uint32_t j; if (is >> j) r.job_id = j; }

        flows.push_back(r);
    }

    ff.close();

    // ── Reconstruct emission-order bucket_id for each flow ──
    // A bucket is a maximal run of consecutive flows with the same (layer, loopstate),
    // matching the writer's grouping in finalizeFlowFile. This lets the scheduler
    // advance an unlock frontier over the true emission order (backward pass included).
    {
        uint32_t cur_bucket = 0;
        for (size_t i = 0; i < flows.size(); i++) {
            if (i > 0) {
                // Reset to 0 at a job boundary so each job has its own 0..max_j
                // bucket numbering (the scheduler runs an independent frontier per
                // job). Within a job, advance on each (layer, loopstate) change.
                if (flows[i].job_id != flows[i-1].job_id) {
                    cur_bucket = 0;
                } else if (flows[i].layer_num != flows[i-1].layer_num ||
                           flows[i].loopstate != flows[i-1].loopstate) {
                    cur_bucket++;
                }
            }
            flows[i].bucket_id = cur_bucket;
        }
    }

    std::cout << "[LoadFlows] Parsed " << flows.size() << " flows, "
              << bucket_meta_out.size() << " bucket metadata entries from "
              << flow_file_path << " (header said " << total << ")" << std::endl;

    // The header carries the authoritative count; a mismatch means the file was
    // truncated or a body line was malformed. Surface it loudly rather than
    // replay a silently-short flow set.
    if (flows.size() != total) {
        std::cerr << "[LoadFlows] ERROR: parsed " << flows.size() << " flows but header declared "
                  << total << " — flow file is truncated or corrupt. Regenerate it." << std::endl;
        return std::vector<FlowFileRecord>{};  // fail closed: empty → main() aborts
    }

    return flows;
}

#endif // __DECOUPLED_FLOW_READER_H__
