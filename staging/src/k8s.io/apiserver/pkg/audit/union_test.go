/*
Copyright 2017 The Kubernetes Authors.

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

type fakeBackend struct {
	events []*auditinternal.Event
}

func (f *fakeBackend) ProcessEvents(events ...*auditinternal.Event) bool {
	f.events = append(f.events, events...)
	return true
}

func (f *fakeBackend) Run(stopCh <-chan struct{}) error {
	return nil
}

func (f *fakeBackend) Shutdown() {
	// Nothing to do here.
}

func (f *fakeBackend) String() string {
	return ""
}

func TestUnion(t *testing.T) {
	backends := []Backend{
		new(fakeBackend),
		new(fakeBackend),
		new(fakeBackend),
	}

	b := Union(backends...)

	n := 5

	var events []*auditinternal.Event
	for i := 0; i < n; i++ {
		events = append(events, &auditinternal.Event{
			AuditID: types.UID(strconv.Itoa(i)),
		})
	}
	b.ProcessEvents(events...)

	for i, b := range backends {
		// so we can inspect the underlying events.
		backend := b.(*fakeBackend)

		if got := len(backend.events); got != n {
			t.Errorf("backend %d wanted %d events, got %d", i, n, got)
			continue
		}
		for j, event := range backend.events {
			wantID := types.UID(strconv.Itoa(j))
			if event.AuditID != wantID {
				t.Errorf("backend %d event %d wanted id %s, got %s", i, j, wantID, event.AuditID)
			}
		}
	}
}

type cannotMultipleRunBackend struct {
	started chan struct{}
}

func (b *cannotMultipleRunBackend) ProcessEvents(events ...*auditinternal.Event) bool {
	return true
}

func (b *cannotMultipleRunBackend) Run(stopCh <-chan struct{}) error {
	close(b.started)
	return nil
}

func (b *cannotMultipleRunBackend) Shutdown() {}

func (b *cannotMultipleRunBackend) String() string {
	return "cannotMultipleRunBackend"
}

func TestUnionRun(t *testing.T) {
	backends := []Backend{
		&cannotMultipleRunBackend{started: make(chan struct{})},
		&cannotMultipleRunBackend{started: make(chan struct{})},
		&cannotMultipleRunBackend{started: make(chan struct{})},
	}

	b := Union(backends...)

	if err := b.Run(make(chan struct{})); err != nil {
		t.Errorf("union backend run: %v", err)
	}
}
