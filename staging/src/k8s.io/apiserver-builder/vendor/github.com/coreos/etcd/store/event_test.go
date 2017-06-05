// Copyright 2015 The etcd Authors
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

package store

import "testing"

// TestEventQueue tests a queue with capacity = 100
// Add 200 events into that queue, and test if the
// previous 100 events have been swapped out.
func TestEventQueue(t *testing.T) {

	eh := newEventHistory(100)

	// Add
	for i := 0; i < 200; i++ {
		e := newEvent(Create, "/foo", uint64(i), uint64(i))
		eh.addEvent(e)
	}

	// Test
	j := 100
	i := eh.Queue.Front
	n := eh.Queue.Size
	for ; n > 0; n-- {
		e := eh.Queue.Events[i]
		if e.Index() != uint64(j) {
			t.Fatalf("queue error!")
		}
		j++
		i = (i + 1) % eh.Queue.Capacity
	}
}

func TestScanHistory(t *testing.T) {
	eh := newEventHistory(100)

	// Add
	eh.addEvent(newEvent(Create, "/foo", 1, 1))
	eh.addEvent(newEvent(Create, "/foo/bar", 2, 2))
	eh.addEvent(newEvent(Create, "/foo/foo", 3, 3))
	eh.addEvent(newEvent(Create, "/foo/bar/bar", 4, 4))
	eh.addEvent(newEvent(Create, "/foo/foo/foo", 5, 5))

	// Delete a dir
	de := newEvent(Delete, "/foo", 6, 6)
	de.PrevNode = newDir(nil, "/foo", 1, nil, Permanent).Repr(false, false, nil)
	eh.addEvent(de)

	e, err := eh.scan("/foo", false, 1)
	if err != nil || e.Index() != 1 {
		t.Fatalf("scan error [/foo] [1] %d (%v)", e.Index(), err)
	}

	e, err = eh.scan("/foo/bar", false, 1)

	if err != nil || e.Index() != 2 {
		t.Fatalf("scan error [/foo/bar] [2] %d (%v)", e.Index(), err)
	}

	e, err = eh.scan("/foo/bar", true, 3)

	if err != nil || e.Index() != 4 {
		t.Fatalf("scan error [/foo/bar/bar] [4] %d (%v)", e.Index(), err)
	}

	e, err = eh.scan("/foo/foo/foo", false, 6)
	if err != nil || e.Index() != 6 {
		t.Fatalf("scan error [/foo/foo/foo] [6] %d (%v)", e.Index(), err)
	}

	e, _ = eh.scan("/foo/bar", true, 7)
	if e != nil {
		t.Fatalf("bad index shoud reuturn nil")
	}
}

// TestFullEventQueue tests a queue with capacity = 10
// Add 1000 events into that queue, and test if scanning
// works still for previous events.
func TestFullEventQueue(t *testing.T) {

	eh := newEventHistory(10)

	// Add
	for i := 0; i < 1000; i++ {
		ce := newEvent(Create, "/foo", uint64(i), uint64(i))
		eh.addEvent(ce)
		e, err := eh.scan("/foo", true, uint64(i-1))
		if i > 0 {
			if e == nil || err != nil {
				t.Fatalf("scan error [/foo] [%v] %v", i-1, i)
			}
		}
	}
}

func TestCloneEvent(t *testing.T) {
	e1 := &Event{
		Action:    Create,
		EtcdIndex: 1,
		Node:      nil,
		PrevNode:  nil,
	}
	e2 := e1.Clone()
	if e2.Action != Create {
		t.Fatalf("Action=%q, want %q", e2.Action, Create)
	}
	if e2.EtcdIndex != e1.EtcdIndex {
		t.Fatalf("EtcdIndex=%d, want %d", e2.EtcdIndex, e1.EtcdIndex)
	}
	// Changing the cloned node should not affect the original
	e2.Action = Delete
	e2.EtcdIndex = uint64(5)
	if e1.Action != Create {
		t.Fatalf("Action=%q, want %q", e1.Action, Create)
	}
	if e1.EtcdIndex != uint64(1) {
		t.Fatalf("EtcdIndex=%d, want %d", e1.EtcdIndex, uint64(1))
	}
	if e2.Action != Delete {
		t.Fatalf("Action=%q, want %q", e2.Action, Delete)
	}
	if e2.EtcdIndex != uint64(5) {
		t.Fatalf("EtcdIndex=%d, want %d", e2.EtcdIndex, uint64(5))
	}
}
