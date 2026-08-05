// Copyright 2026 Google Inc. All Rights Reserved.
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

package model

// Types in this file are the additions over info/v1 that the model needs to be
// the single canonical, kubelet-facing type set (design §6). The rest of model/
// is seeded verbatim from info/v1 so the eventual consumer repoint is mechanical.

// CpuInstStats is instantaneous (rate) CPU usage. On Linux it is derived by
// expose/library from two cumulative ContainerStats samples; on Windows winstats
// supplies it directly from perf counters — hence an optional field, not a pure
// read-time projection. (Shape adapted from info/v2.)
type CpuInstStats struct {
	Usage CpuInstUsage `json:"usage"`
}

type CpuInstUsage struct {
	// Total CPU usage. Units: nanocores per second.
	Total uint64 `json:"total"`
	// Per CPU/core usage. Units: nanocores per second.
	PerCpu []uint64 `json:"per_cpu_usage,omitempty"`
	// Time spent in user space. Units: nanocores per second.
	User uint64 `json:"user"`
	// Time spent in kernel space. Units: nanocores per second.
	System uint64 `json:"system"`
}

// Container is the per-container envelope the library returns to embedders
// (mirrors info/v2.ContainerInfo). It carries the identity the kubelet's
// ContainerLabelsFunc keys pod lookup on (Name/Aliases/Namespace), plus the spec
// and a time-ordered window of samples from the cache.
type Container struct {
	Name      string            `json:"name"`
	Aliases   []string          `json:"aliases,omitempty"`
	Namespace string            `json:"namespace,omitempty"`
	Spec      ContainerSpec     `json:"spec,omitempty"`
	Stats     []*ContainerStats `json:"stats,omitempty"`
}

// FilesystemSummary is the per-container filesystem rollup expose/library projects
// from the per-device ContainerStats.Filesystem, reproducing today's single-fs /
// skip-multi-device v2 shape by default (design §6.1 caveat). Library-owned.
type FilesystemSummary struct {
	TotalUsageBytes *uint64 `json:"totalUsageBytes,omitempty"`
	BaseUsageBytes  *uint64 `json:"baseUsageBytes,omitempty"`
	InodeUsage      *uint64 `json:"inodeUsage,omitempty"`
}
