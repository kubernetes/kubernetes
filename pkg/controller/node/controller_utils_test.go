/*
Copyright 2016 The Kubernetes Authors.

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

package node

import (
	"fmt"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
)

// fakeRecorder is used as a fake during testing.
type fakeRecorder struct {
	source api.EventSource
	events chan *api.Event
	clock  util.Clock
}

func (f *fakeRecorder) Event(obj runtime.Object, eventtype, reason, message string) {
	f.generateEvent(obj, unversioned.Now(), eventtype, reason, message)
}

func (f *fakeRecorder) Eventf(obj runtime.Object, eventtype, reason, messageFmt string, args ...interface{}) {
	f.Event(obj, eventtype, reason, fmt.Sprintf(messageFmt, args...))
}

func (f *fakeRecorder) PastEventf(obj runtime.Object, timestamp unversioned.Time, eventtype, reason, messageFmt string, args ...interface{}) {

}

func (f *fakeRecorder) generateEvent(obj runtime.Object, timestamp unversioned.Time, eventtype, reason, message string) {
	ref, err := api.GetReference(obj)
	if err != nil {
		return
	}
	event := f.makeEvent(ref, eventtype, reason, message)
	event.Source = f.source
	if f.events != nil {
		f.events <- event
	}
}

func (f *fakeRecorder) makeEvent(ref *api.ObjectReference, eventtype, reason, message string) *api.Event {
	t := unversioned.Time{Time: f.clock.Now()}
	namespace := ref.Namespace
	if namespace == "" {
		namespace = api.NamespaceDefault
	}
	return &api.Event{
		ObjectMeta: api.ObjectMeta{
			Name:      fmt.Sprintf("%v.%x", ref.Name, t.UnixNano()),
			Namespace: namespace,
		},
		InvolvedObject: *ref,
		Reason:         reason,
		Message:        message,
		FirstTimestamp: t,
		LastTimestamp:  t,
		Count:          1,
		Type:           eventtype,
	}
}

func NewFakeRecorder(bufferSize int) *fakeRecorder {
	return &fakeRecorder{
		source: api.EventSource{Component: "nodeControllerTest"},
		events: make(chan *api.Event, bufferSize),
		clock:  util.NewFakeClock(time.Now()),
	}
}

func TestRecordNodeEvent(t *testing.T) {
	table := map[string]struct {
		name      string
		uid       string
		eventType string
		reason    string
		event     string
		expectUID string
	}{
		"TestNode01": {
			name:      "TestNode01",
			uid:       "11111",
			eventType: api.EventTypeNormal,
			reason:    "Started",
			event:     "start",
			expectUID: "11111",
		},
		"TestNode02": {
			name:      "TestNode02",
			uid:       "22222",
			eventType: api.EventTypeNormal,
			reason:    "Started",
			event:     "start",
			expectUID: "22222",
		},
		"TestNode03": {
			name:      "TestNode03",
			uid:       "33333",
			eventType: api.EventTypeNormal,
			reason:    "Started",
			event:     "start",
			expectUID: "33333",
		},
	}
	recorder := NewFakeRecorder(len(table))
	for _, item := range table {
		recordNodeEvent(recorder, item.name, item.uid, item.eventType, item.reason, item.event)
	}
	for i := 0; i < len(table); i++ {
		event := <-recorder.events
		involvedObject := event.InvolvedObject
		actualUID := string(involvedObject.UID)
		item, _ := table[involvedObject.Name]
		expectUID := item.uid
		if actualUID != expectUID {
			t.Fatalf("expect %s but got %s", expectUID, actualUID)
		}
	}
}
