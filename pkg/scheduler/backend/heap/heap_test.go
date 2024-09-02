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
)

func testHeapObjectKeyFunc(obj testHeapObject) string {
	return obj.name
}

type testHeapObject struct {
	name string
	val  interface{}
}

type testMetricRecorder int

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
	return testHeapObject{name: name, val: val}
}

func compareInts(val1 testHeapObject, val2 testHeapObject) bool {
	first := val1.val.(int)
	second := val2.val.(int)
	return first < second
}

// TestHeapBasic tests Heap invariant
func TestHeapBasic(t *testing.T) {
	h := New(testHeapObjectKeyFunc, compareInts)
	const amount = 500
	var i int
	var zero testHeapObject

	// empty queue
	if item, ok := h.Peek(); ok || item != zero {
		t.Errorf("expected nil object but got %v", item)
	}

	for i = amount; i > 0; i-- {
		h.AddOrUpdate(mkHeapObj(string([]rune{'a', rune(i)}), i))
		// Retrieve head without removing it
		head, ok := h.Peek()
		if e, a := i, head.val; !ok || a != e {
			t.Errorf("expected %d, got %d", e, a)
		}
	}

	// Make sure that the numbers are popped in ascending order.
	prevNum := 0
	for i := 0; i < amount; i++ {
		item, err := h.Pop()
		num := item.val.(int)
		// All the items must be sorted.
		if err != nil || prevNum > num {
			t.Errorf("got %v out of order, last was %v", item, prevNum)
		}
		prevNum = num
	}

	_, err := h.Pop()
	if err == nil {
		t.Errorf("expected Pop() to error on empty heap")
	}
}

// TestHeap_AddOrUpdate_Add tests add capabilities of Heap.AddOrUpdate
// and ensures that heap invariant is preserved after adding items.
func TestHeap_AddOrUpdate_Add(t *testing.T) {
	h := New(testHeapObjectKeyFunc, compareInts)
	h.AddOrUpdate(mkHeapObj("foo", 10))
	h.AddOrUpdate(mkHeapObj("bar", 1))
	h.AddOrUpdate(mkHeapObj("baz", 11))
	h.AddOrUpdate(mkHeapObj("zab", 30))
	h.AddOrUpdate(mkHeapObj("foo", 13)) // This updates "foo".

	item, err := h.Pop()
	if e, a := 1, item.val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	item, err = h.Pop()
	if e, a := 11, item.val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	if err := h.Delete(mkHeapObj("baz", 11)); err == nil { // Nothing is deleted.
		t.Fatalf("nothing should be deleted from the heap")
	}
	h.AddOrUpdate(mkHeapObj("foo", 14)) // foo is updated.
	item, err = h.Pop()
	if e, a := 14, item.val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	item, err = h.Pop()
	if e, a := 30, item.val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
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
	if err := h.Delete(mkHeapObj("bar", 200)); err != nil {
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
	if err = h.Delete(mkHeapObj("non-existent", 10)); err == nil || len != h.data.Len() {
		t.Fatalf("Didn't expect any item removal")
	}
	// Delete tail.
	if err = h.Delete(mkHeapObj("bal", 31)); err != nil {
		t.Fatalf("Failed to delete tail.")
	}
	// Delete one of the items with value 30.
	if err = h.Delete(mkHeapObj("zab", 30)); err != nil {
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
	if h.data.queue[0] != "baz" || h.data.items["baz"].index != 0 {
		t.Fatalf("expected baz to be at the head")
	}
	item, err := h.Pop()
	if e, a := 0, item.val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	// Update bar to push it farther back in the queue.
	h.AddOrUpdate(mkHeapObj("bar", 100))
	if h.data.queue[0] != "foo" || h.data.items["foo"].index != 0 {
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

func TestHeapWithRecorder(t *testing.T) {
	metricRecorder := new(testMetricRecorder)
	h := NewWithRecorder(testHeapObjectKeyFunc, compareInts, metricRecorder)
	h.AddOrUpdate(mkHeapObj("foo", 10))
	h.AddOrUpdate(mkHeapObj("bar", 1))
	h.AddOrUpdate(mkHeapObj("baz", 100))
	h.AddOrUpdate(mkHeapObj("qux", 11))

	if *metricRecorder != 4 {
		t.Errorf("expected count to be 4 but got %d", *metricRecorder)
	}
	if err := h.Delete(mkHeapObj("bar", 1)); err != nil {
		t.Fatal(err)
	}
	if *metricRecorder != 3 {
		t.Errorf("expected count to be 3 but got %d", *metricRecorder)
	}
	if _, err := h.Pop(); err != nil {
		t.Fatal(err)
	}
	if *metricRecorder != 2 {
		t.Errorf("expected count to be 2 but got %d", *metricRecorder)
	}

	h.metricRecorder.Clear()
	if *metricRecorder != 0 {
		t.Errorf("expected count to be 0 but got %d", *metricRecorder)
	}
}
