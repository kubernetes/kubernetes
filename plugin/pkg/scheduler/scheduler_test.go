/*
Copyright 2014 Google Inc. All rights reserved.

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

package scheduler

import (
	"errors"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/scheduler"
)

type fakeBinder struct {
	b func(binding *api.Binding) error
}

func (fb fakeBinder) Bind(binding *api.Binding) error { return fb.b(binding) }

func podWithID(id string) *api.Pod {
	return &api.Pod{ObjectMeta: api.ObjectMeta{Name: id, SelfLink: testapi.SelfLink("pods", id)}}
}

type mockScheduler struct {
	machine string
	err     error
}

func (es mockScheduler) Schedule(pod api.Pod, ml scheduler.MinionLister) (string, error) {
	return es.machine, es.err
}

func TestScheduler(t *testing.T) {
	defer record.StartLogging(t.Logf).Stop()
	errS := errors.New("scheduler")
	errB := errors.New("binder")

	table := []struct {
		injectBindError error
		sendPod         *api.Pod
		algo            scheduler.Scheduler
		expectErrorPod  *api.Pod
		expectError     error
		expectBind      *api.Binding
		eventReason     string
	}{
		{
			sendPod:     podWithID("foo"),
			algo:        mockScheduler{"machine1", nil},
			expectBind:  &api.Binding{PodID: "foo", Host: "machine1"},
			eventReason: "scheduled",
		}, {
			sendPod:        podWithID("foo"),
			algo:           mockScheduler{"machine1", errS},
			expectError:    errS,
			expectErrorPod: podWithID("foo"),
			eventReason:    "failedScheduling",
		}, {
			sendPod:         podWithID("foo"),
			algo:            mockScheduler{"machine1", nil},
			expectBind:      &api.Binding{PodID: "foo", Host: "machine1"},
			injectBindError: errB,
			expectError:     errB,
			expectErrorPod:  podWithID("foo"),
			eventReason:     "failedScheduling",
		},
	}

	for i, item := range table {
		var gotError error
		var gotPod *api.Pod
		var gotBinding *api.Binding
		c := &Config{
			MinionLister: scheduler.FakeMinionLister(
				api.MinionList{Items: []api.Minion{{ObjectMeta: api.ObjectMeta{Name: "machine1"}}}},
			),
			Algorithm: item.algo,
			Binder: fakeBinder{func(b *api.Binding) error {
				gotBinding = b
				return item.injectBindError
			}},
			Error: func(p *api.Pod, err error) {
				gotPod = p
				gotError = err
			},
			NextPod: func() *api.Pod {
				return item.sendPod
			},
		}
		s := New(c)
		called := make(chan struct{})
		events := record.GetEvents(func(e *api.Event) {
			if e, a := item.eventReason, e.Reason; e != a {
				t.Errorf("%v: expected %v, got %v", i, e, a)
			}
			close(called)
		})
		s.scheduleOne()
		if e, a := item.expectErrorPod, gotPod; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error pod: wanted %v, got %v", i, e, a)
		}
		if e, a := item.expectError, gotError; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error: wanted %v, got %v", i, e, a)
		}
		if e, a := item.expectBind, gotBinding; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: error: wanted %v, got %v", i, e, a)
		}
		<-called
		events.Stop()
	}
}
