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
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/util/wait"
	. "k8s.io/kubernetes/pkg/watch"
)

type myType struct {
	ID    string
	Value string
}

func (obj *myType) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }

func TestBroadcaster(t *testing.T) {
	table := []Event{
		{Type: Added, Object: &myType{"foo", "hello world 1"}},
		{Type: Added, Object: &myType{"bar", "hello world 2"}},
		{Type: Modified, Object: &myType{"foo", "goodbye world 3"}},
		{Type: Deleted, Object: &myType{"bar", "hello world 4"}},
	}

	// The broadcaster we're testing
	m := NewBroadcaster(0, WaitIfChannelFull)

	// Add a bunch of watchers
	const testWatchers = 2
	wg := sync.WaitGroup{}
	wg.Add(testWatchers)
	for i := 0; i < testWatchers; i++ {
		// Verify that each watcher gets the events in the correct order
		go func(watcher int, w Interface) {
			tableLine := 0
			for {
				event, ok := <-w.ResultChan()
				if !ok {
					break
				}
				if e, a := table[tableLine], event; !reflect.DeepEqual(e, a) {
					t.Errorf("Watcher %v, line %v: Expected (%v, %#v), got (%v, %#v)",
						watcher, tableLine, e.Type, e.Object, a.Type, a.Object)
				} else {
					t.Logf("Got (%v, %#v)", event.Type, event.Object)
				}
				tableLine++
			}
			wg.Done()
		}(i, m.Watch())
	}

	for i, item := range table {
		t.Logf("Sending %v", i)
		m.Action(item.Type, item.Object)
	}

	m.Shutdown()

	wg.Wait()
}

func TestBroadcasterWatcherClose(t *testing.T) {
	m := NewBroadcaster(0, WaitIfChannelFull)
	w := m.Watch()
	w2 := m.Watch()
	w.Stop()
	m.Shutdown()
	if _, open := <-w.ResultChan(); open {
		t.Errorf("Stop didn't work?")
	}
	if _, open := <-w2.ResultChan(); open {
		t.Errorf("Shutdown didn't work?")
	}
	// Extra stops don't hurt things
	w.Stop()
	w2.Stop()
}

func TestBroadcasterWatcherStopDeadlock(t *testing.T) {
	done := make(chan bool)
	m := NewBroadcaster(0, WaitIfChannelFull)
	go func(w0, w1 Interface) {
		// We know Broadcaster is in the distribute loop once one watcher receives
		// an event. Stop the other watcher while distribute is trying to
		// send to it.
		select {
		case <-w0.ResultChan():
			w1.Stop()
		case <-w1.ResultChan():
			w0.Stop()
		}
		close(done)
	}(m.Watch(), m.Watch())
	m.Action(Added, &myType{})
	select {
	case <-time.After(wait.ForeverTestTimeout):
		t.Error("timeout: deadlocked")
	case <-done:
	}
	m.Shutdown()
}

func TestBroadcasterDropIfChannelFull(t *testing.T) {
	m := NewBroadcaster(1, DropIfChannelFull)

	event1 := Event{Type: Added, Object: &myType{"foo", "hello world 1"}}
	event2 := Event{Type: Added, Object: &myType{"bar", "hello world 2"}}

	// Add a couple watchers
	watches := make([]Interface, 2)
	for i := range watches {
		watches[i] = m.Watch()
	}

	// Send a couple events before closing the broadcast channel.
	t.Log("Sending event 1")
	m.Action(event1.Type, event1.Object)
	t.Log("Sending event 2")
	m.Action(event2.Type, event2.Object)
	m.Shutdown()

	// Pull events from the queue.
	wg := sync.WaitGroup{}
	wg.Add(len(watches))
	for i := range watches {
		// Verify that each watcher only gets the first event because its watch
		// queue of length one was full from the first one.
		go func(watcher int, w Interface) {
			defer wg.Done()
			e1, ok := <-w.ResultChan()
			if !ok {
				t.Errorf("Watcher %v failed to retrieve first event.", watcher)
			}
			if e, a := event1, e1; !reflect.DeepEqual(e, a) {
				t.Errorf("Watcher %v: Expected (%v, %#v), got (%v, %#v)",
					watcher, e.Type, e.Object, a.Type, a.Object)
			}
			t.Logf("Got (%v, %#v)", e1.Type, e1.Object)
			e2, ok := <-w.ResultChan()
			if ok {
				t.Errorf("Watcher %v received second event (%v, %#v) even though it shouldn't have.",
					watcher, e2.Type, e2.Object)
			}
		}(i, watches[i])
	}
	wg.Wait()
}
