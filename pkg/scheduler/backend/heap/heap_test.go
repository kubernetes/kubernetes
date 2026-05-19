/*
Copyright 2018 The Kubernetes Authors.

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

// This file was copied from client-go/tools/cache/heap.go and modified
// for our non thread-safe heap

package heap

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func testHeapObjectKeyFunc(obj testHeapObject) string {
	return obj.name
}

type testHeapObject struct {
	name string
	val  interface{}
	size int
}

func (obj testHeapObject) Size() int {
	return obj.size
}

type testMetricRecorder int

func (tmr *testMetricRecorder) Add(val int) {
	if tmr != nil {
		*tmr += testMetricRecorder(val)
	}
}

func (tmr *testMetricRecorder) Inc() {
	if tmr != nil {
		*tmr++
	}
}

func (tmr *testMetricRecorder) Dec() {
	if tmr != nil {
		*tmr--
	}
}

func (tmr *testMetricRecorder) Clear() {
	if tmr != nil {
		*tmr = 0
	}
}

func mkHeapObj(name string, val interface{}) testHeapObject {
	return testHeapObject{name: name, val: val, size: 1}
}

func mkHeapObjWithSize(name string, val interface{}, size int) testHeapObject {
	return testHeapObject{name: name, val: val, size: size}
}

func compareInts(val1 testHeapObject, val2 testHeapObject) bool {
	first := val1.val.(int)
	second := val2.val.(int)
	return first < second
}

func expectPopOrder(t *testing.T, h *Heap[testHeapObject], want []int) {
	t.Helper()
	var got []int
	for h.Len() > 0 {
		item, err := h.Pop()
		if err != nil {
			t.Fatalf("unexpected pop error: %v", err)
		}
		got = append(got, item.val.(int))
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("unexpected pop order (-want, +got):\n%s", diff)
	}
}

// TestHeap_Peek tests that Peek returns the minimum element after each insertion.
func TestHeap_Peek(t *testing.T) {
	h := New(testHeapObjectKeyFunc, compareInts)
	const amount = 500
	var zero testHeapObject

	if item, ok := h.Peek(); ok || item != zero {
		t.Errorf("expected nil object but got %v", item)
	}

	for i := amount; i > 0; i-- {
		h.AddOrUpdate(mkHeapObj(string([]rune{'a', rune(i)}), i))
		head, ok := h.Peek()
		if e, a := i, head.val; !ok || a != e {
			t.Errorf("expected %d, got %d", e, a)
		}
	}
}

// TestHeap_Delete tests Heap.Delete and ensures that heap invariant is
// preserved after deleting items.
func TestHeap_Delete(t *testing.T) {
	h := New(testHeapObjectKeyFunc, compareInts)
	h.AddOrUpdate(mkHeapObj("foo", 10))
	h.AddOrUpdate(mkHeapObj("bar", 1))
	h.AddOrUpdate(mkHeapObj("bal", 31))
	h.AddOrUpdate(mkHeapObj("baz", 11))

	// Delete head. Delete should work with "key" and doesn't care about the value.
	if obj := h.Delete(mkHeapObj("bar", 200)); obj.name == "" {
		t.Fatalf("Failed to delete head.")
	}
	item, err := h.Pop()
	if e, a := 10, item.val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	h.AddOrUpdate(mkHeapObj("zab", 30))
	h.AddOrUpdate(mkHeapObj("faz", 30))
	len := h.data.Len()
	// Delete non-existing item.
	if obj := h.Delete(mkHeapObj("non-existent", 10)); obj.name != "" || len != h.data.Len() {
		t.Fatalf("Didn't expect any item removal")
	}
	// Delete tail.
	if obj := h.Delete(mkHeapObj("bal", 31)); obj.name == "" {
		t.Fatalf("Failed to delete tail.")
	}
	// Delete one of the items with value 30.
	if obj := h.Delete(mkHeapObj("zab", 30)); obj.name == "" {
		t.Fatalf("Failed to delete item.")
	}
	item, err = h.Pop()
	if e, a := 11, item.val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	item, err = h.Pop()
	if e, a := 30, item.val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	if h.data.Len() != 0 {
		t.Fatalf("expected an empty heap.")
	}
}

// TestHeap_AddOrUpdate_Update tests update capabilities of Heap.Update
// and ensures that heap invariant is preserved after adding items.
func TestHeap_AddOrUpdate_Update(t *testing.T) {
	h := New(testHeapObjectKeyFunc, compareInts)
	h.AddOrUpdate(mkHeapObj("foo", 10))
	h.AddOrUpdate(mkHeapObj("bar", 1))
	h.AddOrUpdate(mkHeapObj("bal", 31))
	h.AddOrUpdate(mkHeapObj("baz", 11))

	// Update an item to a value that should push it to the head.
	h.AddOrUpdate(mkHeapObj("baz", 0))
	if h.data.queue[0].key != "baz" || h.data.keyIndex["baz"] != 0 {
		t.Fatalf("expected baz to be at the head")
	}
	item, err := h.Pop()
	if e, a := 0, item.val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	// Update bar to push it farther back in the queue.
	h.AddOrUpdate(mkHeapObj("bar", 100))
	if h.data.queue[0].key != "foo" || h.data.keyIndex["foo"] != 0 {
		t.Fatalf("expected foo to be at the head")
	}
}

// TestHeap_Get tests Heap.Get.
func TestHeap_Get(t *testing.T) {
	h := New(testHeapObjectKeyFunc, compareInts)
	h.AddOrUpdate(mkHeapObj("foo", 10))
	h.AddOrUpdate(mkHeapObj("bar", 1))
	h.AddOrUpdate(mkHeapObj("bal", 31))
	h.AddOrUpdate(mkHeapObj("baz", 11))

	// Get works with the key.
	item, exists := h.Get(mkHeapObj("baz", 0))
	if !exists || item.val != 11 {
		t.Fatalf("unexpected error in getting element")
	}
	// Get non-existing object.
	_, exists = h.Get(mkHeapObj("non-existing", 0))
	if exists {
		t.Fatalf("didn't expect to get any object")
	}
}

// TestHeap_GetByKey tests Heap.GetByKey and is very similar to TestHeap_Get.
func TestHeap_GetByKey(t *testing.T) {
	h := New(testHeapObjectKeyFunc, compareInts)
	h.AddOrUpdate(mkHeapObj("foo", 10))
	h.AddOrUpdate(mkHeapObj("bar", 1))
	h.AddOrUpdate(mkHeapObj("bal", 31))
	h.AddOrUpdate(mkHeapObj("baz", 11))

	item, exists := h.GetByKey("baz")
	if !exists || item.val != 11 {
		t.Fatalf("unexpected error in getting element")
	}
	// Get non-existing object.
	_, exists = h.GetByKey("non-existing")
	if exists {
		t.Fatalf("didn't expect to get any object")
	}
}

// TestHeap_List tests Heap.List function.
func TestHeap_List(t *testing.T) {
	h := New(testHeapObjectKeyFunc, compareInts)
	list := h.List()
	if len(list) != 0 {
		t.Errorf("expected an empty list")
	}

	items := map[string]int{
		"foo": 10,
		"bar": 1,
		"bal": 30,
		"baz": 11,
		"faz": 30,
	}
	for k, v := range items {
		h.AddOrUpdate(mkHeapObj(k, v))
	}
	list = h.List()
	if len(list) != len(items) {
		t.Errorf("expected %d items, got %d", len(items), len(list))
	}
	for _, heapObj := range list {
		v, ok := items[heapObj.name]
		if !ok || v != heapObj.val {
			t.Errorf("unexpected item in the list: %v", heapObj)
		}
	}
}

func TestHeapHas(t *testing.T) {
	t.Run("empty heap", func(t *testing.T) {
		h := New(testHeapObjectKeyFunc, compareInts)
		if got := h.Has(mkHeapObj("foo", 0)); got != false {
			t.Errorf("Has(%q) = %v, want %v", "foo", got, false)
		}
	})

	t.Run("non-existing item in populated heap", func(t *testing.T) {
		h := New(testHeapObjectKeyFunc, compareInts)
		h.AddOrUpdate(mkHeapObj("bar", 1))
		if got := h.Has(mkHeapObj("foo", 0)); got != false {
			t.Errorf("Has(%q) = %v, want %v", "foo", got, false)
		}
	})

	t.Run("existing item (single element)", func(t *testing.T) {
		h := New(testHeapObjectKeyFunc, compareInts)
		h.AddOrUpdate(mkHeapObj("foo", 10))
		if got := h.Has(mkHeapObj("foo", 0)); got != true {
			t.Errorf("Has(%q) = %v, want %v", "foo", got, true)
		}
	})

	t.Run("existing item (multiple elements)", func(t *testing.T) {
		h := New(testHeapObjectKeyFunc, compareInts)
		h.AddOrUpdate(mkHeapObj("bar", 1))
		h.AddOrUpdate(mkHeapObj("foo", 10))
		h.AddOrUpdate(mkHeapObj("baz", 11))
		if got := h.Has(mkHeapObj("foo", 0)); got != true {
			t.Errorf("Has(%q) = %v, want %v", "foo", got, true)
		}
	})

	t.Run("item missing after deletion", func(t *testing.T) {
		h := New(testHeapObjectKeyFunc, compareInts)
		h.AddOrUpdate(mkHeapObj("foo", 10))
		h.Delete(mkHeapObj("foo", 0))
		if got := h.Has(mkHeapObj("foo", 0)); got != false {
			t.Errorf("Has(%q) = %v, want %v", "foo", got, false)
		}
	})
}

func TestHeapZeroAndOneElement(t *testing.T) {
	t.Run("pop single element", func(t *testing.T) {
		h := New(testHeapObjectKeyFunc, compareInts)
		h.AddOrUpdate(mkHeapObj("foo", 42))
		item, err := h.Pop()
		if e, a := 42, item.val; err != nil || a != e {
			t.Fatalf("expected %d, got %d", e, a)
		}
		if e, a := 0, h.Len(); a != e {
			t.Fatalf("expected %d, got %d", e, a)
		}
	})

	t.Run("pop empty heap errors", func(t *testing.T) {
		h := New(testHeapObjectKeyFunc, compareInts)
		_, err := h.Pop()
		if err == nil {
			t.Fatalf("expected error popping empty heap")
		}
	})

	t.Run("delete single element", func(t *testing.T) {
		h := New(testHeapObjectKeyFunc, compareInts)
		h.AddOrUpdate(mkHeapObj("bar", 7))
		h.Delete(mkHeapObj("bar", 0))
		if e, a := 0, h.Len(); a != e {
			t.Fatalf("expected %d, got %d", e, a)
		}
	})

	t.Run("peek single element", func(t *testing.T) {
		h := New(testHeapObjectKeyFunc, compareInts)
		h.AddOrUpdate(mkHeapObj("baz", 99))
		item, ok := h.Peek()
		if e, a := 99, item.val; !ok || a != e {
			t.Fatalf("expected %d, got %d", e, a)
		}
	})
}

func TestHeapPopOrder(t *testing.T) {
	tests := []struct {
		name     string
		inserts  []testHeapObject
		expected []int
	}{
		{
			name: "ascending insertion",
			inserts: []testHeapObject{
				mkHeapObj("foo", 1),
				mkHeapObj("bar", 2),
				mkHeapObj("baz", 3),
			},
			expected: []int{1, 2, 3},
		},
		{
			name: "descending insertion",
			inserts: []testHeapObject{
				mkHeapObj("foo", 30),
				mkHeapObj("bar", 20),
				mkHeapObj("baz", 10),
			},
			expected: []int{10, 20, 30},
		},
		{
			name: "random insertion",
			inserts: []testHeapObject{
				mkHeapObj("foo", 15),
				mkHeapObj("bar", 3),
				mkHeapObj("baz", 42),
				mkHeapObj("zab", 1),
				mkHeapObj("faz", 27),
			},
			expected: []int{1, 3, 15, 27, 42},
		},
		{
			name: "duplicate values",
			inserts: []testHeapObject{
				mkHeapObj("foo", 5),
				mkHeapObj("bar", 5),
				mkHeapObj("baz", 5),
			},
			expected: []int{5, 5, 5},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := New(testHeapObjectKeyFunc, compareInts)
			for _, obj := range tt.inserts {
				h.AddOrUpdate(obj)
			}

			expectPopOrder(t, h, tt.expected)
		})
	}
}

func TestHeapDeleteMiddle(t *testing.T) {
	tests := []struct {
		name         string
		inserts      []testHeapObject
		deletes      []string
		expectedPops []int
	}{
		{
			name: "delete middle items",
			inserts: []testHeapObject{
				mkHeapObj("foo", 10),
				mkHeapObj("bar", 1),
				mkHeapObj("bal", 20),
				mkHeapObj("baz", 15),
				mkHeapObj("zab", 25),
				mkHeapObj("faz", 5),
			},
			deletes:      []string{"foo", "baz"},
			expectedPops: []int{1, 5, 20, 25},
		},
		{
			name: "delete head",
			inserts: []testHeapObject{
				mkHeapObj("foo", 1),
				mkHeapObj("bar", 10),
				mkHeapObj("baz", 20),
			},
			deletes:      []string{"foo"},
			expectedPops: []int{10, 20},
		},
		{
			name: "delete all but one",
			inserts: []testHeapObject{
				mkHeapObj("foo", 10),
				mkHeapObj("bar", 1),
				mkHeapObj("baz", 20),
			},
			deletes:      []string{"foo", "baz"},
			expectedPops: []int{1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := New(testHeapObjectKeyFunc, compareInts)
			for _, obj := range tt.inserts {
				h.AddOrUpdate(obj)
			}
			for _, key := range tt.deletes {
				h.Delete(mkHeapObj(key, 0))
			}

			expectPopOrder(t, h, tt.expectedPops)
		})
	}
}

func TestHeapUpdatePriority(t *testing.T) {
	tests := []struct {
		name         string
		inserts      []testHeapObject
		updates      []testHeapObject
		expectedPops []int
	}{
		{
			name: "update to same value",
			inserts: []testHeapObject{
				mkHeapObj("foo", 10),
				mkHeapObj("bar", 1),
				mkHeapObj("baz", 11),
			},
			updates:      []testHeapObject{mkHeapObj("foo", 10)},
			expectedPops: []int{1, 10, 11},
		},
		{
			name: "update to lower value (move to head)",
			inserts: []testHeapObject{
				mkHeapObj("foo", 10),
				mkHeapObj("bar", 1),
				mkHeapObj("baz", 11),
			},
			updates:      []testHeapObject{mkHeapObj("baz", 0)},
			expectedPops: []int{0, 1, 10},
		},
		{
			name: "update to higher value (move to tail)",
			inserts: []testHeapObject{
				mkHeapObj("foo", 10),
				mkHeapObj("bar", 1),
				mkHeapObj("baz", 11),
			},
			updates:      []testHeapObject{mkHeapObj("bar", 100)},
			expectedPops: []int{10, 11, 100},
		},
		{
			name: "full order inversion",
			inserts: []testHeapObject{
				mkHeapObj("foo", 1),
				mkHeapObj("bar", 2),
				mkHeapObj("baz", 3),
				mkHeapObj("zab", 4),
				mkHeapObj("faz", 5),
			},
			updates: []testHeapObject{
				mkHeapObj("faz", 0),
				mkHeapObj("foo", 100),
			},
			expectedPops: []int{0, 2, 3, 4, 100},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := New(testHeapObjectKeyFunc, compareInts)
			for _, obj := range tt.inserts {
				h.AddOrUpdate(obj)
			}
			for _, obj := range tt.updates {
				h.AddOrUpdate(obj)
			}

			expectPopOrder(t, h, tt.expectedPops)
		})
	}
}

func TestHeapLen(t *testing.T) {
	tests := []struct {
		name        string
		action      func(h *Heap[testHeapObject])
		expectedLen int
	}{
		{
			name:        "empty heap",
			action:      func(h *Heap[testHeapObject]) {},
			expectedLen: 0,
		},
		{
			name: "after one add",
			action: func(h *Heap[testHeapObject]) {
				h.AddOrUpdate(mkHeapObj("foo", 10))
			},
			expectedLen: 1,
		},
		{
			name: "after three adds",
			action: func(h *Heap[testHeapObject]) {
				h.AddOrUpdate(mkHeapObj("foo", 10))
				h.AddOrUpdate(mkHeapObj("bar", 1))
				h.AddOrUpdate(mkHeapObj("baz", 11))
			},
			expectedLen: 3,
		},
		{
			name: "update does not change len",
			action: func(h *Heap[testHeapObject]) {
				h.AddOrUpdate(mkHeapObj("foo", 10))
				h.AddOrUpdate(mkHeapObj("bar", 1))
				h.AddOrUpdate(mkHeapObj("foo", 20))
			},
			expectedLen: 2,
		},
		{
			name: "after add and delete",
			action: func(h *Heap[testHeapObject]) {
				h.AddOrUpdate(mkHeapObj("foo", 10))
				h.AddOrUpdate(mkHeapObj("bar", 1))
				h.Delete(mkHeapObj("bar", 0))
			},
			expectedLen: 1,
		},
		{
			name: "after add and pop",
			action: func(h *Heap[testHeapObject]) {
				h.AddOrUpdate(mkHeapObj("foo", 10))
				h.AddOrUpdate(mkHeapObj("bar", 1))
				_, _ = h.Pop()
			},
			expectedLen: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := New(testHeapObjectKeyFunc, compareInts)
			tt.action(h)
			if e, a := tt.expectedLen, h.Len(); a != e {
				t.Fatalf("expected %d, got %d", e, a)
			}
		})
	}
}

func TestHeapPopAllAndReuse(t *testing.T) {
	tests := []struct {
		name         string
		firstRound   []testHeapObject
		secondRound  []testHeapObject
		expectedPops []int
	}{
		{
			name: "reuse with different keys",
			firstRound: []testHeapObject{
				mkHeapObj("foo", 3),
				mkHeapObj("bar", 1),
				mkHeapObj("baz", 2),
			},
			secondRound: []testHeapObject{
				mkHeapObj("x", 20),
				mkHeapObj("y", 10),
				mkHeapObj("z", 15),
			},
			expectedPops: []int{10, 15, 20},
		},
		{
			name: "reuse with same keys",
			firstRound: []testHeapObject{
				mkHeapObj("foo", 100),
				mkHeapObj("bar", 200),
			},
			secondRound: []testHeapObject{
				mkHeapObj("foo", 5),
				mkHeapObj("bar", 3),
			},
			expectedPops: []int{3, 5},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := New(testHeapObjectKeyFunc, compareInts)
			for _, obj := range tt.firstRound {
				h.AddOrUpdate(obj)
			}
			for h.Len() > 0 {
				_, _ = h.Pop()
			}
			for _, obj := range tt.secondRound {
				h.AddOrUpdate(obj)
			}
			expectPopOrder(t, h, tt.expectedPops)
		})
	}
}

func TestHeapDeleteAll(t *testing.T) {
	tests := []struct {
		name        string
		inserts     []testHeapObject
		deleteOrder []string
	}{
		{
			name: "delete in random order",
			inserts: []testHeapObject{
				mkHeapObj("foo", 10),
				mkHeapObj("bar", 1),
				mkHeapObj("bal", 31),
				mkHeapObj("baz", 11),
				mkHeapObj("zab", 30),
			},
			deleteOrder: []string{"baz", "foo", "zab", "bar", "bal"},
		},
		{
			name: "delete head first",
			inserts: []testHeapObject{
				mkHeapObj("foo", 1),
				mkHeapObj("bar", 10),
				mkHeapObj("baz", 20),
			},
			deleteOrder: []string{"foo", "bar", "baz"},
		},
		{
			name: "delete tail first",
			inserts: []testHeapObject{
				mkHeapObj("foo", 1),
				mkHeapObj("bar", 10),
				mkHeapObj("baz", 20),
			},
			deleteOrder: []string{"baz", "bar", "foo"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := New(testHeapObjectKeyFunc, compareInts)
			for _, obj := range tt.inserts {
				h.AddOrUpdate(obj)
			}
			for _, key := range tt.deleteOrder {
				h.Delete(mkHeapObj(key, 0))
			}
			if e, a := 0, h.Len(); a != e {
				t.Fatalf("expected %d, got %d", e, a)
			}
			// Heap should still be usable.
			h.AddOrUpdate(mkHeapObj("new", 5))
			item, err := h.Pop()
			if e, a := 5, item.val; err != nil || a != e {
				t.Fatalf("expected %d, got %d", e, a)
			}
		})
	}
}

func TestHeapLargeScale(t *testing.T) {
	h := New(testHeapObjectKeyFunc, compareInts)
	const amount = 1000

	// Insert items in descending order.
	for i := amount; i > 0; i-- {
		h.AddOrUpdate(mkHeapObj(string([]rune{'a', rune(i)}), i))
	}

	// Update a subset to new priorities.
	for i := 1; i <= amount; i += 10 {
		h.AddOrUpdate(mkHeapObj(string([]rune{'a', rune(i)}), amount+i))
	}

	// Delete a subset.
	for i := 5; i <= amount; i += 10 {
		h.Delete(mkHeapObj(string([]rune{'a', rune(i)}), 0))
	}

	// Pop all remaining and verify sorted order.
	prevNum := 0
	for h.Len() > 0 {
		item, err := h.Pop()
		if err != nil {
			t.Fatalf("unexpected pop error: %v", err)
		}
		num := item.val.(int)
		if prevNum > num {
			t.Fatalf("got %d out of order, last was %d", num, prevNum)
		}
		prevNum = num
	}
}

func TestHeapWithRecorder(t *testing.T) {
	metricRecorder := new(testMetricRecorder)
	h := NewWithRecorder(testHeapObjectKeyFunc, compareInts, metricRecorder)
	h.AddOrUpdate(mkHeapObjWithSize("foo", 10, 1))
	h.AddOrUpdate(mkHeapObjWithSize("bar", 1, 2))
	h.AddOrUpdate(mkHeapObjWithSize("baz", 100, 3))
	h.AddOrUpdate(mkHeapObjWithSize("qux", 11, 4))

	if *metricRecorder != 10 {
		t.Errorf("expected count to be 10 (1+2+3+4) but got %d", *metricRecorder)
	}
	if obj := h.Delete(mkHeapObjWithSize("bar", 1, 2)); obj.name == "" {
		t.Fatalf("Failed to delete item")
	}
	if *metricRecorder != 8 {
		t.Errorf("expected count to be 8 (1+3+4) but got %d", *metricRecorder)
	}
	if _, err := h.Pop(); err != nil {
		t.Fatal(err)
	}
	if *metricRecorder != 7 {
		t.Errorf("expected count to be 7 (3+4) but got %d", *metricRecorder)
	}

	h.metricRecorder.Clear()
	if *metricRecorder != 0 {
		t.Errorf("expected count to be 0 but got %d", *metricRecorder)
	}
}
