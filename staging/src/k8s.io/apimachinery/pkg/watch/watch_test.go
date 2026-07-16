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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	. "k8s.io/apimachinery/pkg/watch"
)

type testType string

func (obj testType) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }
func (obj testType) DeepCopyObject() runtime.Object   { return obj }

func TestFake(t *testing.T) {
	f := NewFake()

	table := []struct {
		t EventType
		s testType
	}{
		{Added, testType("foo")},
		{Modified, testType("qux")},
		{Modified, testType("bar")},
		{Deleted, testType("bar")},
		{Error, testType("error: blah")},
	}

	// Prove that f implements Interface by phrasing this as a function.
	consumer := func(w Interface) {
		for _, expect := range table {
			got, ok := <-w.ResultChan()
			if !ok {
				t.Fatalf("closed early")
			}
			if e, a := expect.t, got.Type; e != a {
				t.Fatalf("Expected %v, got %v", e, a)
			}
			if a, ok := got.Object.(testType); !ok || a != expect.s {
				t.Fatalf("Expected %v, got %v", expect.s, a)
			}
		}
		_, stillOpen := <-w.ResultChan()
		if stillOpen {
			t.Fatal("Never stopped")
		}
	}

	sender := func() {
		f.Add(testType("foo"))
		f.Action(Modified, testType("qux"))
		f.Modify(testType("bar"))
		f.Delete(testType("bar"))
		f.Error(testType("error: blah"))
		f.Stop()
	}

	go sender()
	consumer(f)
}

func TestRaceFreeFake(t *testing.T) {
	f := NewRaceFreeFake()

	table := []struct {
		t EventType
		s testType
	}{
		{Added, testType("foo")},
		{Modified, testType("qux")},
		{Modified, testType("bar")},
		{Deleted, testType("bar")},
		{Error, testType("error: blah")},
	}

	// Prove that f implements Interface by phrasing this as a function.
	consumer := func(w Interface) {
		for _, expect := range table {
			got, ok := <-w.ResultChan()
			if !ok {
				t.Fatalf("closed early")
			}
			if e, a := expect.t, got.Type; e != a {
				t.Fatalf("Expected %v, got %v", e, a)
			}
			if a, ok := got.Object.(testType); !ok || a != expect.s {
				t.Fatalf("Expected %v, got %v", expect.s, a)
			}
		}
		_, stillOpen := <-w.ResultChan()
		if stillOpen {
			t.Fatal("Never stopped")
		}
	}

	sender := func() {
		f.Add(testType("foo"))
		f.Action(Modified, testType("qux"))
		f.Modify(testType("bar"))
		f.Delete(testType("bar"))
		f.Error(testType("error: blah"))
		f.Stop()
	}

	go sender()
	consumer(f)
}

func TestFakeWatcherLifecycle(t *testing.T) {
	f := NewFakeWithChanSize(2, false)

	if f.IsStopped() {
		t.Errorf("unexpected stopped watcher")
	}

	// The buffered channel accepts events without a consumer.
	f.Add(testType("foo"))
	f.Add(testType("bar"))

	f.Stop()
	if !f.IsStopped() {
		t.Errorf("expected watcher to be stopped")
	}
	// Stop is idempotent.
	f.Stop()

	f.Reset()
	if f.IsStopped() {
		t.Errorf("unexpected stopped watcher after reset")
	}

	// Events flow again on the fresh channel.
	go f.Add(testType("baz"))
	got, ok := <-f.ResultChan()
	if !ok {
		t.Fatalf("closed early")
	}
	if e, a := (Event{Added, testType("baz")}), got; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestRaceFreeFakeWatcherLifecycle(t *testing.T) {
	f := NewRaceFreeFake()

	if f.IsStopped() {
		t.Errorf("unexpected stopped watcher")
	}

	f.Stop()
	if !f.IsStopped() {
		t.Errorf("expected watcher to be stopped")
	}
	// Stop is idempotent.
	f.Stop()

	// Sending on a stopped watcher is a silent no-op.
	f.Add(testType("foo"))
	f.Modify(testType("foo"))
	f.Delete(testType("foo"))
	f.Error(testType("foo"))
	f.Action(Added, testType("foo"))
	if _, ok := <-f.ResultChan(); ok {
		t.Errorf("unexpected event on stopped watcher")
	}

	f.Reset()
	if f.IsStopped() {
		t.Errorf("unexpected stopped watcher after reset")
	}

	// Events flow again on the fresh channel.
	f.Add(testType("baz"))
	got, ok := <-f.ResultChan()
	if !ok {
		t.Fatalf("closed early")
	}
	if e, a := (Event{Added, testType("baz")}), got; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestRaceFreeFakeWatcherPanicsWhenFull(t *testing.T) {
	expectPanic := func(name string, fn func()) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("%s: expected panic on full channel", name)
			}
		}()
		fn()
	}

	f := NewRaceFreeFake()
	// Fill the buffered channel without a consumer.
	for i := int32(0); i < DefaultChanSize; i++ {
		f.Add(testType("foo"))
	}

	expectPanic("Add", func() { f.Add(testType("foo")) })
	expectPanic("Modify", func() { f.Modify(testType("foo")) })
	expectPanic("Delete", func() { f.Delete(testType("foo")) })
	expectPanic("Error", func() { f.Error(testType("foo")) })
	expectPanic("Action", func() { f.Action(Added, testType("foo")) })
}

func TestEmpty(t *testing.T) {
	w := NewEmptyWatch()
	_, ok := <-w.ResultChan()
	if ok {
		t.Errorf("unexpected result channel result")
	}
	w.Stop()
	_, ok = <-w.ResultChan()
	if ok {
		t.Errorf("unexpected result channel result")
	}
}

func TestProxyWatcher(t *testing.T) {
	events := []Event{
		{Added, testType("foo")},
		{Modified, testType("qux")},
		{Modified, testType("bar")},
		{Deleted, testType("bar")},
		{Error, testType("error: blah")},
	}

	ch := make(chan Event, len(events))
	w := NewProxyWatcher(ch)

	for _, e := range events {
		ch <- e
	}

	for _, e := range events {
		g := <-w.ResultChan()
		if !reflect.DeepEqual(e, g) {
			t.Errorf("Expected %#v, got %#v", e, g)
			continue
		}
	}

	if w.Stopping() {
		t.Errorf("unexpected stopping watcher")
	}

	w.Stop()

	if !w.Stopping() {
		t.Errorf("expected watcher to be stopping")
	}

	select {
	// Closed channel always reads immediately
	case <-w.StopChan():
	default:
		t.Error("Channel isn't closed")
	}

	// Test double close
	w.Stop()
}

func TestMockWatcher(t *testing.T) {
	stopped := false
	ch := make(chan Event)
	w := MockWatcher{
		StopFunc:       func() { stopped = true },
		ResultChanFunc: func() <-chan Event { return ch },
	}

	if got := w.ResultChan(); got != (<-chan Event)(ch) {
		t.Errorf("expected ResultChan to return the provided channel")
	}

	w.Stop()
	if !stopped {
		t.Errorf("expected Stop to call StopFunc")
	}
}

func TestEventDeepCopy(t *testing.T) {
	e := Event{Type: Added, Object: testType("foo")}
	c := e.DeepCopy()
	if !reflect.DeepEqual(e, *c) {
		t.Errorf("expected %#v, got %#v", e, *c)
	}

	empty := Event{}
	if got := empty.DeepCopy(); !reflect.DeepEqual(empty, *got) {
		t.Errorf("expected %#v, got %#v", empty, *got)
	}
}
