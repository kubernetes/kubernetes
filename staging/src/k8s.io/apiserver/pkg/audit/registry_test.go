/*
Copyright 2018 The Kubernetes Authors.

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

package audit

import (
	"strconv"
	"testing"

	"k8s.io/apimachinery/pkg/types"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
)

func TestRegistry(t *testing.T) {
	backends := []Backend{
		new(fakeBackend),
		new(fakeBackend),
		new(fakeBackend),
	}

	b := WithRegistry(Union(backends...))

	var events []*auditinternal.Event
	for i := 0; i < 15; i++ {
		events = append(events, &auditinternal.Event{
			AuditID: types.UID(strconv.Itoa(i)),
		})
	}
	b.ProcessEvents(events[0:5]...)

	for i, b := range backends {
		checkEvents(events[0:5], b, i, t)
	}

	dynamicBackend := new(fakeBackend)
	b.Register(dynamicBackend)
	b.ProcessEvents(events[5:10]...)
	for i, b := range backends {
		checkEvents(events[0:10], b, i, t)
	}
	checkEvents(events[5:10], dynamicBackend, 3, t)

	b.UnRegister(dynamicBackend)
	b.ProcessEvents(events[10:15]...)
	for i, b := range backends {
		checkEvents(events[0:15], b, i, t)
	}
	checkEvents(events[5:10], dynamicBackend, 3, t)
}

func checkEvents(expected []*auditinternal.Event, b Backend, i int, t *testing.T) {
	// so we can inspect the underlying events.
	backend := b.(*fakeBackend)

	if got := len(backend.events); got != len(expected) {
		t.Errorf("backend %d wanted %d events, got %d", i, len(expected), got)
		return
	}
	for j, event := range backend.events {
		wantID := expected[j].AuditID
		if event.AuditID != wantID {
			t.Errorf("backend %d event %d wanted event id %s, got %s", i, j, wantID, event.AuditID)
		}
	}
}
