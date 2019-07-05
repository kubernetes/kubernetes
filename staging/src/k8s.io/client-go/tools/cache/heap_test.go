/*
Copyright 2017 The Kubernetes Authors.

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
	"sync"
	"testing"
	"time"
)

func testHeapObjectKeyFunc(obj interface{}) (string, error) {
	return obj.(testHeapObject).name, nil
}

type testHeapObject struct {
	name string
	val  interface{}
}

func mkHeapObj(name string, val interface{}) testHeapObject {
	return testHeapObject{name: name, val: val}
}

func compareInts(val1 interface{}, val2 interface{}) bool {
	first := val1.(testHeapObject).val.(int)
	second := val2.(testHeapObject).val.(int)
	return first < second
}

// TestHeapBasic tests Heap invariant and synchronization.
func TestHeapBasic(t *testing.T) {
	h := NewHeap(testHeapObjectKeyFunc, compareInts)
	var wg sync.WaitGroup
	wg.Add(2)
	const amount = 500
	var i, u int
	// Insert items in the heap in opposite orders in two go routines.
	go func() {
		for i = amount; i > 0; i-- {
			h.Add(mkHeapObj(string([]rune{'a', rune(i)}), i))
		}
		wg.Done()
	}()
	go func() {
		for u = 0; u < amount; u++ {
			h.Add(mkHeapObj(string([]rune{'b', rune(u)}), u+1))
		}
		wg.Done()
	}()
	// Wait for the two go routines to finish.
	wg.Wait()
	// Make sure that the numbers are popped in ascending order.
	prevNum := 0
	for i := 0; i < amount*2; i++ {
		obj, err := h.Pop()
		num := obj.(testHeapObject).val.(int)
		// All the items must be sorted.
		if err != nil || prevNum > num {
			t.Errorf("got %v out of order, last was %v", obj, prevNum)
		}
		prevNum = num
	}
}

// Tests Heap.Add and ensures that heap invariant is preserved after adding items.
func TestHeap_Add(t *testing.T) {
	h := NewHeap(testHeapObjectKeyFunc, compareInts)
	h.Add(mkHeapObj("foo", 10))
	h.Add(mkHeapObj("bar", 1))
	h.Add(mkHeapObj("baz", 11))
	h.Add(mkHeapObj("zab", 30))
	h.Add(mkHeapObj("foo", 13)) // This updates "foo".

	item, err := h.Pop()
	if e, a := 1, item.(testHeapObject).val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	item, err = h.Pop()
	if e, a := 11, item.(testHeapObject).val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	h.Delete(mkHeapObj("baz", 11)) // Nothing is deleted.
	h.Add(mkHeapObj("foo", 14))    // foo is updated.
	item, err = h.Pop()
	if e, a := 14, item.(testHeapObject).val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	item, err = h.Pop()
	if e, a := 30, item.(testHeapObject).val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
}

// TestHeap_BulkAdd tests Heap.BulkAdd functionality and ensures that all the
// items given to BulkAdd are added to the queue before Pop reads them.
func TestHeap_BulkAdd(t *testing.T) {
	h := NewHeap(testHeapObjectKeyFunc, compareInts)
	const amount = 500
	// Insert items in the heap in opposite orders in a go routine.
	go func() {
		l := []interface{}{}
		for i := amount; i > 0; i-- {
			l = append(l, mkHeapObj(string([]rune{'a', rune(i)}), i))
		}
		h.BulkAdd(l)
	}()
	prevNum := -1
	for i := 0; i < amount; i++ {
		obj, err := h.Pop()
		num := obj.(testHeapObject).val.(int)
		// All the items must be sorted.
		if err != nil || prevNum >= num {
			t.Errorf("got %v out of order, last was %v", obj, prevNum)
		}
		prevNum = num
	}
}

// TestHeapEmptyPop tests that pop returns properly after heap is closed.
func TestHeapEmptyPop(t *testing.T) {
	h := NewHeap(testHeapObjectKeyFunc, compareInts)
	go func() {
		time.Sleep(1 * time.Second)
		h.Close()
	}()
	_, err := h.Pop()
	if err == nil || err.Error() != closedMsg {
		t.Errorf("pop should have returned heap closed error: %v", err)
	}
}

// TestHeap_AddIfNotPresent tests Heap.AddIfNotPresent and ensures that heap
// invariant is preserved after adding items.
func TestHeap_AddIfNotPresent(t *testing.T) {
	h := NewHeap(testHeapObjectKeyFunc, compareInts)
	h.AddIfNotPresent(mkHeapObj("foo", 10))
	h.AddIfNotPresent(mkHeapObj("bar", 1))
	h.AddIfNotPresent(mkHeapObj("baz", 11))
	h.AddIfNotPresent(mkHeapObj("zab", 30))
	h.AddIfNotPresent(mkHeapObj("foo", 13)) // This is not added.

	if len := len(h.data.items); len != 4 {
		t.Errorf("unexpected number of items: %d", len)
	}
	if val := h.data.items["foo"].obj.(testHeapObject).val; val != 10 {
		t.Errorf("unexpected value: %d", val)
	}
	item, err := h.Pop()
	if e, a := 1, item.(testHeapObject).val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	item, err = h.Pop()
	if e, a := 10, item.(testHeapObject).val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	// bar is already popped. Let's add another one.
	h.AddIfNotPresent(mkHeapObj("bar", 14))
	item, err = h.Pop()
	if e, a := 11, item.(testHeapObject).val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	item, err = h.Pop()
	if e, a := 14, item.(testHeapObject).val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
}

// TestHeap_Delete tests Heap.Delete and ensures that heap invariant is
// preserved after deleting items.
func TestHeap_Delete(t *testing.T) {
	h := NewHeap(testHeapObjectKeyFunc, compareInts)
	h.Add(mkHeapObj("foo", 10))
	h.Add(mkHeapObj("bar", 1))
	h.Add(mkHeapObj("bal", 31))
	h.Add(mkHeapObj("baz", 11))

	// Delete head. Delete should work with "key" and doesn't care about the value.
	if err := h.Delete(mkHeapObj("bar", 200)); err != nil {
		t.Fatalf("Failed to delete head.")
	}
	item, err := h.Pop()
	if e, a := 10, item.(testHeapObject).val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	h.Add(mkHeapObj("zab", 30))
	h.Add(mkHeapObj("faz", 30))
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
	if e, a := 11, item.(testHeapObject).val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	item, err = h.Pop()
	if e, a := 30, item.(testHeapObject).val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	if h.data.Len() != 0 {
		t.Fatalf("expected an empty heap.")
	}
}

// TestHeap_Update tests Heap.Update and ensures that heap invariant is
// preserved after adding items.
func TestHeap_Update(t *testing.T) {
	h := NewHeap(testHeapObjectKeyFunc, compareInts)
	h.Add(mkHeapObj("foo", 10))
	h.Add(mkHeapObj("bar", 1))
	h.Add(mkHeapObj("bal", 31))
	h.Add(mkHeapObj("baz", 11))

	// Update an item to a value that should push it to the head.
	h.Update(mkHeapObj("baz", 0))
	if h.data.queue[0] != "baz" || h.data.items["baz"].index != 0 {
		t.Fatalf("expected baz to be at the head")
	}
	item, err := h.Pop()
	if e, a := 0, item.(testHeapObject).val; err != nil || a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	// Update bar to push it farther back in the queue.
	h.Update(mkHeapObj("bar", 100))
	if h.data.queue[0] != "foo" || h.data.items["foo"].index != 0 {
		t.Fatalf("expected foo to be at the head")
	}
}

// TestHeap_Get tests Heap.Get.
func TestHeap_Get(t *testing.T) {
	h := NewHeap(testHeapObjectKeyFunc, compareInts)
	h.Add(mkHeapObj("foo", 10))
	h.Add(mkHeapObj("bar", 1))
	h.Add(mkHeapObj("bal", 31))
	h.Add(mkHeapObj("baz", 11))

	// Get works with the key.
	obj, exists, err := h.Get(mkHeapObj("baz", 0))
	if err != nil || exists == false || obj.(testHeapObject).val != 11 {
		t.Fatalf("unexpected error in getting element")
	}
	// Get non-existing object.
	_, exists, err = h.Get(mkHeapObj("non-existing", 0))
	if err != nil || exists == true {
		t.Fatalf("didn't expect to get any object")
	}
}

// TestHeap_GetByKey tests Heap.GetByKey and is very similar to TestHeap_Get.
func TestHeap_GetByKey(t *testing.T) {
	h := NewHeap(testHeapObjectKeyFunc, compareInts)
	h.Add(mkHeapObj("foo", 10))
	h.Add(mkHeapObj("bar", 1))
	h.Add(mkHeapObj("bal", 31))
	h.Add(mkHeapObj("baz", 11))

	obj, exists, err := h.GetByKey("baz")
	if err != nil || exists == false || obj.(testHeapObject).val != 11 {
		t.Fatalf("unexpected error in getting element")
	}
	// Get non-existing object.
	_, exists, err = h.GetByKey("non-existing")
	if err != nil || exists == true {
		t.Fatalf("didn't expect to get any object")
	}
}

// TestHeap_Close tests Heap.Close and Heap.IsClosed functions.
func TestHeap_Close(t *testing.T) {
	h := NewHeap(testHeapObjectKeyFunc, compareInts)
	h.Add(mkHeapObj("foo", 10))
	h.Add(mkHeapObj("bar", 1))

	if h.IsClosed() {
		t.Fatalf("didn't expect heap to be closed")
	}
	h.Close()
	if !h.IsClosed() {
		t.Fatalf("expect heap to be closed")
	}
}

// TestHeap_List tests Heap.List function.
func TestHeap_List(t *testing.T) {
	h := NewHeap(testHeapObjectKeyFunc, compareInts)
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
		h.Add(mkHeapObj(k, v))
	}
	list = h.List()
	if len(list) != len(items) {
		t.Errorf("expected %d items, got %d", len(items), len(list))
	}
	for _, obj := range list {
		heapObj := obj.(testHeapObject)
		v, ok := items[heapObj.name]
		if !ok || v != heapObj.val {
			t.Errorf("unexpected item in the list: %v", heapObj)
		}
	}
}

// TestHeap_ListKeys tests Heap.ListKeys function. Scenario is the same as
// TestHeap_list.
func TestHeap_ListKeys(t *testing.T) {
	h := NewHeap(testHeapObjectKeyFunc, compareInts)
	list := h.ListKeys()
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
		h.Add(mkHeapObj(k, v))
	}
	list = h.ListKeys()
	if len(list) != len(items) {
		t.Errorf("expected %d items, got %d", len(items), len(list))
	}
	for _, key := range list {
		_, ok := items[key]
		if !ok {
			t.Errorf("unexpected item in the list: %v", key)
		}
	}
}

// TestHeapAddAfterClose tests that heap returns an error if anything is added
// after it is closed.
func TestHeapAddAfterClose(t *testing.T) {
	h := NewHeap(testHeapObjectKeyFunc, compareInts)
	h.Close()
	if err := h.Add(mkHeapObj("test", 1)); err == nil || err.Error() != closedMsg {
		t.Errorf("expected heap closed error")
	}
	if err := h.AddIfNotPresent(mkHeapObj("test", 1)); err == nil || err.Error() != closedMsg {
		t.Errorf("expected heap closed error")
	}
	if err := h.BulkAdd([]interface{}{mkHeapObj("test", 1)}); err == nil || err.Error() != closedMsg {
		t.Errorf("expected heap closed error")
	}
}
