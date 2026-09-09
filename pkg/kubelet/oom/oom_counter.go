/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package oom

import "sync"

// oomEventCounts holds the cumulative number of OOM kills observed per
// container, keyed by the cgroup name. The OOM watcher
// writes it and the kubelet metrics server reads it to generate the
// container_oom_events_total metric.
//
// ponytail: process global because the OOM watcher is a singleton. If more than
// one watcher ever exists, pass an explicit counter to NewWatcher and to the
// metrics server instead.
//
// ponytail: entries are never deleted, so the map grows by one for each distinct
// cgroup that ever OOMs. OOM kills are rare and cgroup names are unique per
// container, so this is a slow leak, not stale metrics (the collector only emits
// for live containers). Prune against the live container set if it matters.
var oomEventCounts = struct {
	sync.Mutex
	byContainer map[string]uint64
}{byContainer: make(map[string]uint64)}

// recordOOMKill increments the OOM kill counter for the given cgroup name.
func recordOOMKill(containerName string) {
	oomEventCounts.Lock()
	defer oomEventCounts.Unlock()
	oomEventCounts.byContainer[containerName]++
}

// OOMEventsForContainer returns the number of OOM kills observed for the given
// cgroup name. It returns 0 for an unknown container and on
// platforms where the OOM watcher does not run.
func OOMEventsForContainer(containerName string) uint64 {
	oomEventCounts.Lock()
	defer oomEventCounts.Unlock()
	return oomEventCounts.byContainer[containerName]
}
