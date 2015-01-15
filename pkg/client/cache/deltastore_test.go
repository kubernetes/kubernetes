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

package cache

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func TestDeltaStore_basic(t *testing.T) {
	q := NewDeltaStore()

	const amount = 500
	go func() {
		for i := 0; i < amount; i++ {
			q.Add(string([]rune{'a', rune(i)}), i+1)
		}
	}()
	go func() {
		for u := uint(0); u < amount; u++ {
			q.Add(string([]rune{'b', rune(u)}), u+1)
		}
	}()

	lastInt := int(0)
	lastUint := uint(0)
	for i := 0; i < amount*2; i++ {
		_, obj := q.Pop()

		switch obj.(type) {
		case int:
			if obj.(int) <= lastInt {
				t.Errorf("got %v (int) out of order, last was %v", obj, lastInt)
			}
			lastInt = obj.(int)
		case uint:
			if obj.(uint) <= lastUint {
				t.Errorf("got %v (uint) out of order, last was %v", obj, lastUint)
			} else {
				lastUint = obj.(uint)
			}
		default:
			t.Fatalf("unexpected type %#v", obj)
		}
	}
}

func TestDeltaStore_initialEventIsDelete(t *testing.T) {
	q := NewDeltaStore()

	q.Replace(map[string]interface{}{
		"foo": 2,
	})

	q.Delete("foo")

	event, thing := q.Pop()

	if thing != 2 {
		t.Fatalf("expected %v, got %v", 2, thing)
	}

	if event != watch.Deleted {
		t.Fatalf("expected %s, got %s", watch.Added, event)
	}
}

func TestDeltaStore_compressAddDelete(t *testing.T) {
	q := NewDeltaStore()

	q.Add("foo", 10)
	q.Delete("foo")
	q.Add("zab", 30)

	event, thing := q.Pop()

	if thing != 30 {
		t.Fatalf("expected %v, got %v", 30, thing)
	}

	if event != watch.Added {
		t.Fatalf("expected %s, got %s", watch.Added, event)
	}
}

func TestDeltaStore_compressAddUpdate(t *testing.T) {
	q := NewDeltaStore()

	q.Add("foo", 10)
	q.Update("foo", 11)

	event, thing := q.Pop()

	if thing != 11 {
		t.Fatalf("expected %v, got %v", 11, thing)
	}

	if event != watch.Added {
		t.Fatalf("expected %s, got %s", watch.Added, event)
	}
}

func TestDeltaStore_compressTwoUpdates(t *testing.T) {
	q := NewDeltaStore()

	q.Replace(map[string]interface{}{
		"foo": 2,
	})

	q.Update("foo", 3)
	q.Update("foo", 4)

	event, thing := q.Pop()

	if thing != 4 {
		t.Fatalf("expected %v, got %v", 4, thing)
	}

	if event != watch.Modified {
		t.Fatalf("expected %s, got %s", watch.Modified, event)
	}
}

func TestDeltaStore_compressUpdateDelete(t *testing.T) {
	q := NewDeltaStore()

	q.Replace(map[string]interface{}{
		"foo": 2,
	})

	q.Update("foo", 3)
	q.Delete("foo")

	event, thing := q.Pop()

	if thing != 3 {
		t.Fatalf("expected %v, got %v", 3, thing)
	}

	if event != watch.Deleted {
		t.Fatalf("expected %s, got %s", watch.Deleted, event)
	}
}

func TestDeltaStore_modifyEventsFromReplace(t *testing.T) {
	q := NewDeltaStore()

	q.Replace(map[string]interface{}{
		"foo": 2,
	})

	q.Update("foo", 2)

	event, thing := q.Pop()

	if thing != 2 {
		t.Fatalf("expected %v, got %v", 3, thing)
	}

	if event != watch.Modified {
		t.Fatalf("expected %s, got %s", watch.Modified, event)
	}
}

func TestDeltaStore_replaceHandlesMissedDeletes(t *testing.T) {
	q := NewDeltaStore()

	q.Replace(map[string]interface{}{
		"foo": 2,
	})

	q.Replace(map[string]interface{}{})

	event, thing := q.Pop()

	if thing != 2 {
		t.Fatalf("expected %v, got %v", 3, thing)
	}

	if event != watch.Deleted {
		t.Fatalf("expected %s, got %s", watch.Deleted, event)
	}	
}
