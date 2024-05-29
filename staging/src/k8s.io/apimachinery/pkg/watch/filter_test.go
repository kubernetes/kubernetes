/*
Copyright 2014 The Kubernetes Authors.

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

package watch_test

import (
	"reflect"
	"testing"

	. "k8s.io/apimachinery/pkg/watch"
)

func TestFilter(t *testing.T) {
	table := []Event{
		{Type: Added, Object: testType("foo")},
		{Type: Added, Object: testType("bar")},
		{Type: Added, Object: testType("baz")},
		{Type: Added, Object: testType("qux")},
		{Type: Added, Object: testType("zoo")},
	}

	source := NewFake()
	filtered := Filter(source, func(e Event) (Event, bool) {
		return e, e.Object.(testType)[0] != 'b'
	})

	go func() {
		defer source.Close()
		for _, item := range table {
			source.Action(item.Type, item.Object)
		}
	}()

	resultCh := filtered.ResultChan()
	var got []string
	for {
		event, ok := <-resultCh
		if !ok {
			break
		}
		got = append(got, string(event.Object.(testType)))
	}

	if e, a := []string{"foo", "qux", "zoo"}, got; !reflect.DeepEqual(e, a) {
		t.Errorf("got %v, wanted %v", e, a)
	}
}

func TestFilterStop(t *testing.T) {
	source := NewFake()
	filtered := Filter(source, func(e Event) (Event, bool) {
		return e, e.Object.(testType)[0] != 'b'
	})

	go func() {
		source.Add(testType("foo"))
		// Wait for watcher to stop, when close the result channel
		<-source.StopChan()
		source.Close()
	}()

	resultCh := filtered.ResultChan()
	var got []string
	for {
		event, ok := <-resultCh
		if !ok {
			break
		}
		got = append(got, string(event.Object.(testType)))
		if e, a := []string{"foo"}, got; !reflect.DeepEqual(e, a) {
			t.Errorf("got %v, wanted %v", e, a)
		} else {
			// Got what we were looking for. Stop watching.
			filtered.Stop()
		}
	}
}

func TestRecorder(t *testing.T) {
	events := []Event{
		{Type: Added, Object: testType("foo")},
		{Type: Added, Object: testType("bar")},
		{Type: Added, Object: testType("baz")},
		{Type: Added, Object: testType("qux")},
		{Type: Added, Object: testType("zoo")},
	}

	source := NewFake()
	go func() {
		defer source.Close()
		for _, item := range events {
			source.Action(item.Type, item.Object)
		}
	}()

	recorder := NewRecorder(source)
	resultCh := recorder.Interface.ResultChan()
	for {
		_, ok := <-resultCh
		if !ok {
			break
		}
	}
	recordedEvents := recorder.Events()
	if !reflect.DeepEqual(recordedEvents, events) {
		t.Errorf("got %v, expected %v", recordedEvents, events)
	}
}
