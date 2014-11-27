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

package watch

import (
	"reflect"
	"sync"
	"testing"
	"time"
)

type myType struct {
	ID    string
	Value string
}

func (*myType) IsAnAPIObject() {}

func TestMux(t *testing.T) {
	table := []Event{
		{Added, &myType{"foo", "hello world 1"}},
		{Added, &myType{"bar", "hello world 2"}},
		{Modified, &myType{"foo", "goodbye world 3"}},
		{Deleted, &myType{"bar", "hello world 4"}},
	}

	// The mux we're testing
	m := NewMux(0)

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

func TestMuxWatcherClose(t *testing.T) {
	m := NewMux(0)
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

func TestMuxWatcherStopDeadlock(t *testing.T) {
	done := make(chan bool)
	m := NewMux(0)
	go func(w0, w1 Interface) {
		// We know Mux is in the distribute loop once one watcher receives
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
	case <-time.After(5 * time.Second):
		t.Error("timeout: deadlocked")
	case <-done:
	}
	m.Shutdown()
}
