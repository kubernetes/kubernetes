// Copyright 2024 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package model is the stable data schema libcadvisor exposes to its consumer
// (the Kubernetes kubelet). It is the widest part of the libcadvisor <-> kubelet
// boundary — imported by ~16 kubelet packages across ~60 files — and is therefore
// treated as a FROZEN v1 contract: exported types and their fields must not be
// renamed, reordered, or removed without a coordinated kubelet change.
//
// Provenance: a near-verbatim copy of upstream cAdvisor's info/v1 with a few
// info/v2 types folded in (see derived.go / projection.go). Keep the stat types in
// sync with upstream info/v1; a CI drift-check against upstream is recommended,
// since there is no automatic merge path for a copied package.
//
// Maintainer notes:
//
//   - SchemaVersion (below) is the self-describing contract version. Bump it only on
//     an intentional, coordinated breaking change.
//
//   - Several stat types are retained for frozen-contract / upstream info/v1 fidelity
//     even though NO collector enabled by the kubelet reads them in this fork: the
//     perf stats (PerfStat / PerfUncoreStat / PerfValue), resctrl stats (ResctrlStats
//     / MemoryBandwidthStats), NUMA (MemoryNumaStats), hugepages (HugetlbStats), and
//     the ~100-counter TcpAdvancedStat. Their /metrics/cadvisor registration blocks
//     were removed in C3 (the kubelet enables none of those MetricKinds), so they are
//     NOT load-bearing — keep them only to preserve the copied schema's shape for
//     upstream drift-checks, and do not assume they are wired to anything live.
//
//   - The custom-metric types (MetricSpec / MetricVal / MetricType / DataType) feed
//     the kubelet's public Summary API field userDefinedMetrics
//     (pkg/kubelet/stats/helper.go -> statsapi.UserDefinedMetric). Removing them is a
//     Kubernetes API change (KEP-2371 retires UserDefinedMetrics via the deprecation
//     process), out of scope for a library prune.
//
//   - Add-a-field hazards: MachineInfo.Clone() (machine.go) enumerates every field by
//     hand — a new field is silently dropped unless Clone() is updated. The Eq /
//     StatsEq methods use reflect.DeepEqual, so a new field silently participates in
//     equality. Update Clone() and review equality whenever the schema grows.
package model

// SchemaVersion identifies the libcadvisor data-model contract version exposed to
// consumers. It is informational/self-describing; bump it only on an intentional,
// coordinated breaking change to the exported types in this package.
const SchemaVersion = "v1"
