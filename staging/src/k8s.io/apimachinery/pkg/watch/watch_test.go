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
	"context"
	"reflect"
	"testing"
	"time"

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
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		resultCh := w.ResultChan()
		for _, expect := range table {
			got, ok := <-resultCh
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
		for {
			select {
			case <-ctx.Done():
				t.Fatalf("Result channel never closed: %v", ctx.Err())
			case got, ok := <-resultCh:
				if !ok {
					// closed after last event, as expected
					return
				}
				t.Errorf("Unexpected event: %v", got)
			}
		}
	}

	sender := func() {
		defer f.Close()
		f.Add(testType("foo"))
		f.Action(Modified, testType("qux"))
		f.Modify(testType("bar"))
		f.Delete(testType("bar"))
		f.Error(testType("error: blah"))
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
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		resultCh := w.ResultChan()
		for _, expect := range table {
			got, ok := <-resultCh
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
		for {
			select {
			case <-ctx.Done():
				t.Fatalf("Result channel never closed: %v", ctx.Err())
			case got, ok := <-resultCh:
				if !ok {
					// closed after last event, as expected
					return
				}
				t.Errorf("Unexpected event: %v", got)
			}
		}
	}

	sender := func() {
		f.Add(testType("foo"))
		f.Action(Modified, testType("qux"))
		f.Modify(testType("bar"))
		f.Delete(testType("bar"))
		f.Error(testType("error: blah"))
		// TODO: Upgrade RaceFreeFake to use Close, not just Stop
		f.Stop()
	}

	go sender()
	consumer(f)
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

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	eventCh := make(chan Event, len(events))
	w := NewProxyWatcher(eventCh)

	go func() {
		for _, e := range events {
			eventCh <- e
		}
		// Wait for consumer to stop watching, then close the result channel
		<-w.StopChan()
		close(eventCh)
	}()

	resultCh := w.ResultChan()
	for _, expect := range events {
		got, ok := <-resultCh
		if !ok {
			t.Fatalf("closed early")
		}
		if !reflect.DeepEqual(expect, got) {
			t.Errorf("Expected %#v, got %#v", expect, got)
		}
	}
	// Tell the producer that the consumer is done watching
	w.Stop()

loop:
	for {
		select {
		case <-ctx.Done():
			t.Fatalf("Result channel never closed: %v", ctx.Err())
		case got, ok := <-resultCh:
			if !ok {
				// closed after last event, as expected
				break loop
			}
			t.Errorf("Unexpected event: %v", got)
		}
	}

	// Test double stop doesn't panic
	w.Stop()
}
