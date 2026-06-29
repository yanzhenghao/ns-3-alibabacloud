/*
 * Copyright (c) 2024, Alibaba Group;
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * ---
 * DECOUPLED REPLAY: Multi-task manifest parser + multi-job assembler.
 *
 * A manifest describes a concurrent multi-task SCENARIO: how many tasks, each
 * task's name, and each task's absolute arrival time (start_ns). The actual
 * traffic lives in N separate SINGLE-TASK flow files (each one untouched, exactly
 * as the coupled sim / flowgen emitted it). This keeps WORKLOAD (the flow files)
 * cleanly separate from SCENARIO (arrival/placement) so one flow file can be
 * reused across many scenarios without re-merging.
 *
 * Manifest format (plain text):
 *
 *   TASKS 3
 *   task: A  flow: jobA_flows.txt  start_ns: 0
 *   task: B  flow: jobB_flows.txt  start_ns: 20000
 *   task: C  flow: jobC_flows.txt  start_ns: 50000
 *
 *   - First non-blank/non-comment line: "TASKS <n>".
 *   - Then exactly <n> "task:" lines. Each carries name / flow / start_ns in any
 *     order (parsed by key, like the bucket metadata lines in flow_reader.h).
 *   - "flow:" paths are resolved relative to the MANIFEST file's directory
 *     (absolute paths are used verbatim).
 *   - Lines starting with '#' and blank lines are ignored.
 *
 * Assembly (LoadJobsFromManifest):
 *   Each single-task flow file numbers flow_id from 0 and is internally job 0.
 *   Running N of them concurrently in one DepScheduler requires:
 *     - flow_id / parent_flow_id[] / child_flow_id[] offset by a per-job flow_base
 *       (cumulative flow count) so the global flow-id space stays collision-free
 *       and the dependency DAG stays self-consistent.
 *     - job_id stamped = task index i (overriding the file's 0) so the scheduler's
 *       per-job bucket frontier advances each task independently.
 *   src/dst/prev are NOT touched: GPU/NPU placement (npu_matrix) is handled
 *   externally — the flow files arriving here are already in the shared fabric's
 *   global node space, on disjoint GPUs.
 */

#ifndef __DECOUPLED_MANIFEST_READER_H__
#define __DECOUPLED_MANIFEST_READER_H__

#include "flow_reader.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <iostream>

// ============================================================================
// ManifestEntry: one task in the scenario
// ============================================================================
struct ManifestEntry {
    std::string name;
    std::string flow_path;   // resolved relative to the manifest's directory
    uint64_t start_ns = 0;   // absolute arrival time of this task's first collective
};

// Directory portion of a path (everything up to and including the last '/'),
// or "" if the path has no directory component.
inline std::string ManifestDir(const std::string& manifest_path) {
    size_t slash = manifest_path.find_last_of('/');
    if (slash == std::string::npos) return "";
    return manifest_path.substr(0, slash + 1);  // keep trailing '/'
}

// ============================================================================
// ParseManifest: read the scenario manifest. Returns empty on any error
// (fail-loud: the caller aborts rather than replay a partial scenario).
// ============================================================================
inline std::vector<ManifestEntry> ParseManifest(const std::string& manifest_path) {
    std::vector<ManifestEntry> entries;

    std::ifstream mf(manifest_path);
    if (!mf.is_open()) {
        std::cerr << "[ParseManifest] ERROR: cannot open manifest: "
                  << manifest_path << std::endl;
        return entries;
    }

    const std::string base_dir = ManifestDir(manifest_path);

    int declared = -1;
    std::string raw;
    while (std::getline(mf, raw)) {
        // Strip a trailing CR (CRLF files) and skip blank / comment lines.
        if (!raw.empty() && raw.back() == '\r') raw.pop_back();
        std::string trimmed = raw;
        size_t first = trimmed.find_first_not_of(" \t");
        if (first == std::string::npos) continue;        // blank
        if (trimmed[first] == '#') continue;             // comment

        std::istringstream ls(raw);
        std::string tok;
        ls >> tok;

        if (tok == "TASKS") {
            if (!(ls >> declared) || declared < 0) {
                std::cerr << "[ParseManifest] ERROR: bad 'TASKS <n>' line: "
                          << raw << std::endl;
                return std::vector<ManifestEntry>{};
            }
            continue;
        }

        if (tok == "task:") {
            // Parse key/value pairs in any order: name / flow / start_ns.
            ManifestEntry e;
            bool have_flow = false;
            std::string key;
            // We already consumed "task:"; the next token is the name value.
            // Treat "task:" specially as the leading key whose value is `name`.
            ls >> e.name;
            while (ls >> key) {
                if (key == "flow:") {
                    std::string p; ls >> p;
                    e.flow_path = (!p.empty() && p[0] == '/') ? p : base_dir + p;
                    have_flow = true;
                } else if (key == "start_ns:") {
                    ls >> e.start_ns;
                } else if (key == "name:") {
                    ls >> e.name;  // allow explicit "name:" override
                } else {
                    std::cerr << "[ParseManifest] WARNING: unknown key '" << key
                              << "' in: " << raw << std::endl;
                    std::string ignore; ls >> ignore;  // skip its value
                }
            }
            if (!have_flow || e.flow_path.empty()) {
                std::cerr << "[ParseManifest] ERROR: task line missing 'flow:': "
                          << raw << std::endl;
                return std::vector<ManifestEntry>{};
            }
            if (e.name.empty()) e.name = "task" + std::to_string(entries.size());
            entries.push_back(e);
            continue;
        }

        std::cerr << "[ParseManifest] WARNING: ignoring unrecognized line: "
                  << raw << std::endl;
    }

    if (declared < 0) {
        std::cerr << "[ParseManifest] ERROR: manifest has no 'TASKS <n>' header: "
                  << manifest_path << std::endl;
        return std::vector<ManifestEntry>{};
    }
    if ((int)entries.size() != declared) {
        std::cerr << "[ParseManifest] ERROR: manifest declared TASKS " << declared
                  << " but found " << entries.size() << " task lines." << std::endl;
        return std::vector<ManifestEntry>{};
    }
    if (entries.empty()) {
        std::cerr << "[ParseManifest] ERROR: manifest declares 0 tasks." << std::endl;
        return std::vector<ManifestEntry>{};
    }

    std::cout << "[ParseManifest] " << entries.size() << " task(s) from "
              << manifest_path << std::endl;
    return entries;
}

// ============================================================================
// LoadJobsFromManifest: load every task's single-task flow file, offset flow-ids
// + stamp job_id per task, and assemble one combined flow set for DepScheduler.
//
// Returns true on success. On any failure returns false with flows_out cleared.
//   flows_out        : all jobs' flows, flow-ids offset to disjoint ranges, job_id set
//   bucket_meta_out  : all jobs' bucket metadata, re-keyed by (job_id, bucket_id)
//   start_ns_out     : job_id -> absolute arrival time (from the manifest)
//   send_lat_us_out  : per-flow send latency (us); jobs must agree, else error
// ============================================================================
inline bool LoadJobsFromManifest(const std::vector<ManifestEntry>& entries,
                                 std::vector<FlowFileRecord>& flows_out,
                                 BucketMetaMap& bucket_meta_out,
                                 std::map<uint32_t, uint64_t>& start_ns_out,
                                 int* send_lat_us_out = nullptr) {
    flows_out.clear();
    bucket_meta_out.clear();
    start_ns_out.clear();

    uint32_t flow_base = 0;       // cumulative flow count = this job's flow-id offset
    int agreed_send_lat = -1;     // first job's recorded send_lat; later jobs must match

    for (size_t i = 0; i < entries.size(); i++) {
        const ManifestEntry& e = entries[i];
        const uint32_t job_id = (uint32_t)i;

        // 1) Parse this task's single-task flow file with the canonical reader
        //    (per-file size==total check + job-local bucket_id reconstruction).
        BucketMetaMap bm;
        int file_send_lat = -1;
        std::vector<FlowFileRecord> flows = LoadFlows(e.flow_path, bm, &file_send_lat);
        if (flows.empty()) {
            std::cerr << "[LoadJobsFromManifest] ERROR: task " << job_id
                      << " (" << e.name << ") loaded 0 flows from "
                      << e.flow_path << std::endl;
            flows_out.clear();
            return false;
        }

        // 2) send_lat consistency across jobs (mirrors merge_flows.py semantics).
        if (file_send_lat > 0) {
            if (agreed_send_lat < 0) agreed_send_lat = file_send_lat;
            else if (agreed_send_lat != file_send_lat) {
                std::cerr << "[LoadJobsFromManifest] ERROR: task " << job_id
                          << " send_lat=" << file_send_lat << "us disagrees with "
                          << agreed_send_lat << "us from earlier task(s). "
                          << "Regenerate flow files with a consistent AS_SEND_LAT."
                          << std::endl;
                flows_out.clear();
                return false;
            }
        }

        // 3) Offset FLOW-ids into a disjoint range + stamp job_id. Placement
        //    (src/dst/prev) is left untouched — handled externally (npu_matrix).
        for (FlowFileRecord& r : flows) {
            r.flow_id += flow_base;
            for (uint32_t& p : r.parent_flow_id) p += flow_base;
            for (uint32_t& c : r.child_flow_id)  c += flow_base;
            r.job_id = job_id;
            flows_out.push_back(r);
        }

        // 4) Re-key this job's bucket metadata as (job_id, bucket_id).
        for (const auto& kv : bm) {
            BucketMeta meta = kv.second;
            meta.job_id = job_id;
            bucket_meta_out[bucketMetaKey(job_id, meta.bucket_id)] = meta;
        }

        // 5) Record arrival; advance the flow-id base for the next job.
        start_ns_out[job_id] = e.start_ns;
        std::cout << "[LoadJobsFromManifest] task " << job_id << " (" << e.name
                  << "): " << flows.size() << " flows, flow_base=" << flow_base
                  << ", start_ns=" << e.start_ns << std::endl;
        flow_base += (uint32_t)flows.size();
    }

    if (send_lat_us_out && agreed_send_lat > 0) *send_lat_us_out = agreed_send_lat;

    std::cout << "[LoadJobsFromManifest] assembled " << flows_out.size()
              << " flows across " << entries.size() << " task(s)." << std::endl;
    return true;
}

#endif // __DECOUPLED_MANIFEST_READER_H__
