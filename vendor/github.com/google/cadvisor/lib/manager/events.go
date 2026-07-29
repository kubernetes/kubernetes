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

package manager

import (
	info "github.com/google/cadvisor/lib/model"

	"k8s.io/klog/v2"
)

// EventSink receives the container lifecycle (creation/deletion) and OOM events
// the manager emits. The full cAdvisor binary injects its events.EventManager
// here (it satisfies this interface) and serves them over the /events REST API;
// the kubelet leaves it nil, so the manager emits nothing and the library
// carries no event-storage machinery. This is the same injection pattern as the
// perf/resctrl factories — see plugins.go.
type EventSink interface {
	AddEvent(*info.Event) error
}

// SetEventSink wires an event sink. Safe to call once before Start; passing nil
// (the default) disables event emission.
func (m *manager) SetEventSink(sink EventSink) {
	m.eventSink = sink
}

// addEvent delivers an event to the sink, best-effort. It is a no-op when no
// sink is wired (the kubelet case).
func (m *manager) addEvent(e *info.Event) {
	if m.eventSink == nil {
		return
	}
	if err := m.eventSink.AddEvent(e); err != nil {
		klog.Errorf("failed to add %s event for %q: %v", e.EventType, e.ContainerName, err)
	}
}

// EventStorageAgeLimit and EventStorageEventLimit expose the values of the
// event_storage_* flags this package registers (for kubelet flag-compatibility)
// so the root binary can build its events storage policy without re-registering
// — and double-registering — the same flags.
func EventStorageAgeLimit() string   { return *eventStorageAgeLimit }
func EventStorageEventLimit() string { return *eventStorageEventLimit }
