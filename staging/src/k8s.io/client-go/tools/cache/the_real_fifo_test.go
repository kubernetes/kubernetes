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

package cache

import (
	"fmt"
	"reflect"
	"runtime"
	"testing"
	"time"
)

func (f *RealFIFO) getItems() []Delta {
	f.lock.Lock()
	defer f.lock.Unlock()

	ret := make([]Delta, len(f.items))
	copy(ret, f.items)
	return ret
}

const closedFIFOName = "FIFO WAS CLOSED"

func popN(queue Queue, count int) []interface{} {
	result := []interface{}{}
	for i := 0; i < count; i++ {
		queue.Pop(func(obj interface{}, isInInitialList bool) error {
			result = append(result, obj)
			return nil
		})
	}
	return result
}

// helper function to reduce stuttering
func testRealFIFOPop(f *RealFIFO) testFifoObject {
	val := Pop(f)
	if val == nil {
		return testFifoObject{name: closedFIFOName}
	}
	return val.(Deltas).Newest().Object.(testFifoObject)
}

func emptyKnownObjects() KeyListerGetter {
	return literalListerGetter(
		func() []testFifoObject {
			return []testFifoObject{}
		},
	)
}

func TestRealFIFO_basic(t *testing.T) {
	f := NewRealFIFO(testFifoObjectKeyFunc, emptyKnownObjects(), nil)
	const amount = 500
	go func() {
		for i := 0; i < amount; i++ {
			f.Add(mkFifoObj(string([]rune{'a', rune(i)}), i+1))
		}
	}()
	go func() {
		for u := uint64(0); u < amount; u++ {
			f.Add(mkFifoObj(string([]rune{'b', rune(u)}), u+1))
		}
	}()

	lastInt := int(0)
	lastUint := uint64(0)
	for i := 0; i < amount*2; i++ {
		switch obj := testRealFIFOPop(f).val.(type) {
		case int:
			if obj <= lastInt {
				t.Errorf("got %v (int) out of order, last was %v", obj, lastInt)
			}
			lastInt = obj
		case uint64:
			if obj <= lastUint {
				t.Errorf("got %v (uint) out of order, last was %v", obj, lastUint)
			} else {
				lastUint = obj
			}
		default:
			t.Fatalf("unexpected type %#v", obj)
		}
	}
}

// TestRealFIFO_replaceWithDeleteDeltaIn tests that a `Sync` delta for an
// object `O` with ID `X` is added when .Replace is called and `O` is among the
// replacement objects even if the RealFIFO already stores in terminal position
// a delta of type `Delete` for ID `X`. Not adding the `Sync` delta causes
// SharedIndexInformers to miss `O`'s create notification, see https://github.com/kubernetes/kubernetes/issues/83810
// for more details.
func TestRealFIFO_replaceWithDeleteDeltaIn(t *testing.T) {
	oldObj := mkFifoObj("foo", 1)
	newObj := mkFifoObj("foo", 2)

	f := NewRealFIFO(
		testFifoObjectKeyFunc,
		literalListerGetter(func() []testFifoObject {
			return []testFifoObject{oldObj}
		}),
		nil,
	)

	f.Delete(oldObj)
	f.Replace([]interface{}{newObj}, "")

	actualDeltas := f.getItems()
	expectedDeltas := []Delta{
		{Type: Deleted, Object: oldObj},
		{Type: Replaced, Object: newObj},
	}
	if !reflect.DeepEqual(expectedDeltas, actualDeltas) {
		t.Errorf("expected %#v, got %#v", expectedDeltas, actualDeltas)
	}
}

func TestRealFIFOW_ReplaceMakesDeletionsForObjectsOnlyInQueue(t *testing.T) {
	obj := mkFifoObj("foo", 2)
	objV2 := mkFifoObj("foo", 3)
	table := []struct {
		name           string
		operations     func(f *RealFIFO)
		expectedDeltas Deltas
	}{
		{
			name: "Added object should be deleted on Replace",
			operations: func(f *RealFIFO) {
				f.Add(obj)
				f.Replace([]interface{}{}, "0")
			},
			expectedDeltas: Deltas{
				{Added, obj},
				{Deleted, DeletedFinalStateUnknown{Key: "foo", Obj: obj}},
			},
		},
		//{
		//	// ATTENTION: difference with delta_fifo_test, there is no option for emitDeltaTypeReplaced
		//	name: "Replaced object should have only a single Delete",
		//	operations: func(f *RealFIFO) {
		//		f.emitDeltaTypeReplaced = true
		//		f.Add(obj)
		//		f.Replace([]interface{}{obj}, "0")
		//		f.Replace([]interface{}{}, "0")
		//	},
		//	expectedDeltas: Deltas{
		//		{Added, obj},
		//		{Replaced, obj},
		//		{Deleted, DeletedFinalStateUnknown{Key: "foo", Obj: obj}},
		//	},
		//},
		{
			name: "Deleted object should have only a single Delete",
			operations: func(f *RealFIFO) {
				f.Add(obj)
				f.Delete(obj)
				f.Replace([]interface{}{}, "0")
			},
			expectedDeltas: Deltas{
				{Added, obj},
				{Deleted, obj},
			},
		},
		{
			name: "Synced objects should have a single delete",
			operations: func(f *RealFIFO) {
				f.Add(obj)
				f.Replace([]interface{}{obj}, "0")
				f.Replace([]interface{}{obj}, "0")
				f.Replace([]interface{}{}, "0")
			},
			expectedDeltas: Deltas{
				{Added, obj},
				{Replaced, obj},
				{Replaced, obj},
				{Deleted, DeletedFinalStateUnknown{Key: "foo", Obj: obj}},
			},
		},
		{
			name: "Added objects should have a single delete on multiple Replaces",
			operations: func(f *RealFIFO) {
				f.Add(obj)
				f.Replace([]interface{}{}, "0")
				f.Replace([]interface{}{}, "1")
			},
			expectedDeltas: Deltas{
				{Added, obj},
				{Deleted, DeletedFinalStateUnknown{Key: "foo", Obj: obj}},
			},
		},
		{
			name: "Added and deleted and added object should be deleted",
			operations: func(f *RealFIFO) {
				f.Add(obj)
				f.Delete(obj)
				f.Add(objV2)
				f.Replace([]interface{}{}, "0")
			},
			expectedDeltas: Deltas{
				{Added, obj},
				{Deleted, obj},
				{Added, objV2},
				{Deleted, DeletedFinalStateUnknown{Key: "foo", Obj: objV2}},
			},
		},
	}
	for _, tt := range table {
		tt := tt

		t.Run(tt.name, func(t *testing.T) {
			// Test with a RealFIFO with a backing KnownObjects
			fWithKnownObjects := NewRealFIFO(
				testFifoObjectKeyFunc,
				literalListerGetter(func() []testFifoObject {
					return []testFifoObject{}
				}),
				nil,
			)
			tt.operations(fWithKnownObjects)
			actualDeltasWithKnownObjects := popN(fWithKnownObjects, len(fWithKnownObjects.getItems()))
			actualAsDeltas := collapseDeltas(actualDeltasWithKnownObjects)
			if !reflect.DeepEqual(tt.expectedDeltas, actualAsDeltas) {
				t.Errorf("expected %#v, got %#v", tt.expectedDeltas, actualAsDeltas)
			}
			if len(fWithKnownObjects.items) != 0 {
				t.Errorf("expected no extra deltas (empty map), got %#v", fWithKnownObjects.items)
			}

			// ATTENTION: difference with delta_fifo_test, there is no option without knownObjects
		})
	}
}

func collapseDeltas(ins []interface{}) Deltas {
	ret := Deltas{}
	for _, curr := range ins {
		for _, delta := range curr.(Deltas) {
			ret = append(ret, delta)
		}
	}
	return ret
}

// ATTENTION: difference with delta_fifo_test, there is no requeue option anymore
// func TestDeltaFIFO_requeueOnPop(t *testing.T) {

func TestRealFIFO_addUpdate(t *testing.T) {
	f := NewRealFIFO(
		testFifoObjectKeyFunc,
		emptyKnownObjects(),
		nil,
	)
	f.Add(mkFifoObj("foo", 10))
	f.Update(mkFifoObj("foo", 12))
	f.Delete(mkFifoObj("foo", 15))

	// ATTENTION: difference with delta_fifo_test, all items on the list.  DeltaFIFO.List only showed newest, but Pop processed all.
	expected1 := []Delta{
		{
			Type:   Added,
			Object: mkFifoObj("foo", 10),
		},
		{
			Type:   Updated,
			Object: mkFifoObj("foo", 12),
		},
		{
			Type:   Deleted,
			Object: mkFifoObj("foo", 15),
		},
	}
	if e, a := expected1, f.getItems(); !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %+v, got %+v", e, a)
	}

	got := make(chan testFifoObject, 4)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			obj := testRealFIFOPop(f)
			if obj.name == closedFIFOName {
				break
			}
			t.Logf("got a thing %#v", obj)
			t.Logf("D len: %v", len(f.items))
			got <- obj
		}
	}()

	first := <-got
	if e, a := 10, first.val; e != a {
		t.Errorf("Didn't get updated value (%v), got %v", e, a)
	}
	second := <-got
	if e, a := 12, second.val; e != a {
		t.Errorf("Didn't get updated value (%v), got %v", e, a)
	}
	third := <-got
	if e, a := 15, third.val; e != a {
		t.Errorf("Didn't get updated value (%v), got %v", e, a)
	}
	select {
	case unexpected := <-got:
		t.Errorf("Got second value %v", unexpected.val)
	case <-time.After(50 * time.Millisecond):
	}

	if e, a := 0, len(f.getItems()); e != a {
		t.Errorf("Didn't get updated value (%v), got %v", e, a)
	}
	f.Close()
	<-done
}

func TestRealFIFO_transformer(t *testing.T) {
	mk := func(name string, rv int) testFifoObject {
		return mkFifoObj(name, &rvAndXfrm{rv, 0})
	}
	xfrm := TransformFunc(func(obj interface{}) (interface{}, error) {
		switch v := obj.(type) {
		case testFifoObject:
			v.val.(*rvAndXfrm).xfrm++
		case DeletedFinalStateUnknown:
			if x := v.Obj.(testFifoObject).val.(*rvAndXfrm).xfrm; x != 1 {
				return nil, fmt.Errorf("object has been transformed wrong number of times: %#v", obj)
			}
		default:
			return nil, fmt.Errorf("unexpected object: %#v", obj)
		}
		return obj, nil
	})

	must := func(err error) {
		if err != nil {
			t.Fatal(err)
		}
	}
	mustTransform := func(obj interface{}) interface{} {
		ret, err := xfrm(obj)
		must(err)
		return ret
	}

	f := NewRealFIFO(
		testFifoObjectKeyFunc,
		emptyKnownObjects(),
		xfrm,
	)
	must(f.Add(mk("foo", 10)))
	must(f.Add(mk("bar", 11)))
	must(f.Update(mk("foo", 12)))
	must(f.Delete(mk("foo", 15)))
	must(f.Replace([]interface{}{}, ""))
	must(f.Add(mk("bar", 16)))
	must(f.Replace([]interface{}{}, ""))

	// ATTENTION: difference with delta_fifo_test, without compression, we keep all the items, including bar being deleted multiple times.
	//            DeltaFIFO starts by checking keys, we start by checking types and keys
	expected1 := []Delta{
		{Type: Added, Object: mustTransform(mk("foo", 10))},
		{Type: Added, Object: mustTransform(mk("bar", 11))},
		{Type: Updated, Object: mustTransform(mk("foo", 12))},
		{Type: Deleted, Object: mustTransform(mk("foo", 15))},
		{Type: Deleted, Object: DeletedFinalStateUnknown{Key: "bar", Obj: mustTransform(mk("bar", 11))}},
		{Type: Added, Object: mustTransform(mk("bar", 16))},
		{Type: Deleted, Object: DeletedFinalStateUnknown{Key: "bar", Obj: mustTransform(mk("bar", 16))}},
	}
	actual1 := f.getItems()
	if len(expected1) != len(actual1) {
		t.Fatalf("Expected %+v, got %+v", expected1, actual1)
	}
	for i := 0; i < len(actual1); i++ {
		e := expected1[i]
		a := actual1[i]
		if e.Type != a.Type {
			t.Errorf("%d Expected %+v, got %+v", i, e, a)
		}
		eKey, err := f.keyOf(e)
		if err != nil {
			t.Fatal(err)
		}
		aKey, err := f.keyOf(a)
		if err != nil {
			t.Fatal(err)
		}
		if eKey != aKey {
			t.Errorf("%d Expected %+v, got %+v", i, eKey, aKey)
		}
	}

	for i := 0; i < len(expected1); i++ {
		obj, err := f.Pop(func(o interface{}, isInInitialList bool) error { return nil })
		if err != nil {
			t.Fatalf("got nothing on try %v?", i)
		}
		a := obj.(Deltas)[0]
		e := expected1[i]
		if !reflect.DeepEqual(e, a) {
			t.Errorf("%d Expected %+v, got %+v", i, e, a)
		}
	}
}

func TestRealFIFO_enqueueingNoLister(t *testing.T) {
	f := NewRealFIFO(
		testFifoObjectKeyFunc,
		emptyKnownObjects(),
		nil,
	)
	f.Add(mkFifoObj("foo", 10))
	f.Update(mkFifoObj("bar", 15))
	f.Add(mkFifoObj("qux", 17))
	f.Delete(mkFifoObj("qux", 18))

	// RealFIFO queues everything
	f.Delete(mkFifoObj("baz", 20))

	// ATTENTION: difference with delta_fifo_test, without compression every item is queued
	expectList := []int{10, 15, 17, 18, 20}
	for _, expect := range expectList {
		if e, a := expect, testRealFIFOPop(f).val; e != a {
			t.Errorf("Didn't get updated value (%v), got %v", e, a)
		}
	}
	if e, a := 0, len(f.getItems()); e != a {
		t.Errorf("queue unexpectedly not empty: %v != %v\n%#v", e, a, f.getItems())
	}
}

func TestRealFIFO_enqueueingWithLister(t *testing.T) {
	f := NewRealFIFO(
		testFifoObjectKeyFunc,
		literalListerGetter(func() []testFifoObject {
			return []testFifoObject{mkFifoObj("foo", 5), mkFifoObj("bar", 6), mkFifoObj("baz", 7)}
		}),
		nil,
	)
	f.Add(mkFifoObj("foo", 10))
	f.Update(mkFifoObj("bar", 15))

	// This delete does enqueue the deletion, because "baz" is in the key lister.
	f.Delete(mkFifoObj("baz", 20))

	expectList := []int{10, 15, 20}
	for _, expect := range expectList {
		if e, a := expect, testRealFIFOPop(f).val; e != a {
			t.Errorf("Didn't get updated value (%v), got %v", e, a)
		}
	}
	if e, a := 0, len(f.items); e != a {
		t.Errorf("queue unexpectedly not empty: %v != %v", e, a)
	}
}

func TestRealFIFO_addReplace(t *testing.T) {
	f := NewRealFIFO(
		testFifoObjectKeyFunc,
		emptyKnownObjects(),
		nil,
	)
	f.Add(mkFifoObj("foo", 10))
	f.Replace([]interface{}{mkFifoObj("foo", 15)}, "0")
	got := make(chan testFifoObject, 3)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			obj := testRealFIFOPop(f)
			if obj.name == closedFIFOName {
				break
			}
			got <- obj
		}
	}()

	// ATTENTION: difference with delta_fifo_test, we get every event instead of the .Newest making us skip some for the test, but not at runtime.
	curr := <-got
	if e, a := 10, curr.val; e != a {
		t.Errorf("Didn't get updated value (%v), got %v", e, a)
	}
	curr = <-got
	if e, a := 15, curr.val; e != a {
		t.Errorf("Didn't get updated value (%v), got %v", e, a)
	}

	select {
	case unexpected := <-got:
		t.Errorf("Got second value %v", unexpected.val)
	case <-time.After(50 * time.Millisecond):
	}

	if items := f.getItems(); len(items) > 0 {
		t.Errorf("item did not get removed")
	}
	f.Close()
	<-done
}

func TestRealFIFO_ResyncNonExisting(t *testing.T) {
	f := NewRealFIFO(
		testFifoObjectKeyFunc,
		literalListerGetter(func() []testFifoObject {
			return []testFifoObject{mkFifoObj("foo", 5)}
		}),
		nil,
	)
	f.Delete(mkFifoObj("foo", 10))
	f.Resync()

	deltas := f.getItems()
	if len(deltas) != 1 {
		t.Fatalf("unexpected deltas length: %v", deltas)
	}
	if deltas[0].Type != Deleted {
		t.Errorf("unexpected delta: %v", deltas[0])
	}
}

func TestRealFIFO_Resync(t *testing.T) {
	f := NewRealFIFO(
		testFifoObjectKeyFunc,
		literalListerGetter(func() []testFifoObject {
			return []testFifoObject{mkFifoObj("foo", 5)}
		}),
		nil,
	)
	f.Resync()

	deltas := f.getItems()
	if len(deltas) != 1 {
		t.Fatalf("unexpected deltas length: %v", deltas)
	}
	if deltas[0].Type != Sync {
		t.Errorf("unexpected delta: %v", deltas[0])
	}
}

func TestRealFIFO_DeleteExistingNonPropagated(t *testing.T) {
	f := NewRealFIFO(
		testFifoObjectKeyFunc,
		emptyKnownObjects(),
		nil,
	)
	f.Add(mkFifoObj("foo", 5))
	f.Delete(mkFifoObj("foo", 6))

	deltas := f.getItems()
	if len(deltas) != 2 {
		t.Fatalf("unexpected deltas length: %v", deltas)
	}
	if deltas[len(deltas)-1].Type != Deleted {
		t.Errorf("unexpected delta: %v", deltas[len(deltas)-1])
	}
}

func TestRealFIFO_ReplaceMakesDeletions(t *testing.T) {
	// We test with only one pre-existing object because there is no
	// promise about how their deletes are ordered.

	// Try it with a pre-existing Delete
	f := NewRealFIFO(
		testFifoObjectKeyFunc,
		literalListerGetter(func() []testFifoObject {
			return []testFifoObject{mkFifoObj("foo", 5), mkFifoObj("bar", 6), mkFifoObj("baz", 7)}
		}),
		nil,
	)
	f.Delete(mkFifoObj("baz", 10))
	f.Replace([]interface{}{mkFifoObj("foo", 5)}, "0")

	expectedList := []Deltas{
		{{Deleted, mkFifoObj("baz", 10)}},
		// ATTENTION: difference with delta_fifo_test, logically the deletes of known items should happen BEFORE newItems are added, so this delete happens early now
		// Since "bar" didn't have a delete event and wasn't in the Replace list
		// it should get a tombstone key with the right Obj.
		{{Deleted, DeletedFinalStateUnknown{Key: "bar", Obj: mkFifoObj("bar", 6)}}},
		{{Replaced, mkFifoObj("foo", 5)}},
	}

	for _, expected := range expectedList {
		cur := Pop(f).(Deltas)
		if e, a := expected, cur; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %#v, got %#v", e, a)
		}
	}

	// Now try starting with an Add instead of a Delete
	f = NewRealFIFO(
		testFifoObjectKeyFunc,
		literalListerGetter(func() []testFifoObject {
			return []testFifoObject{mkFifoObj("foo", 5), mkFifoObj("bar", 6), mkFifoObj("baz", 7)}
		}),
		nil,
	)
	f.Add(mkFifoObj("baz", 10))
	f.Replace([]interface{}{mkFifoObj("foo", 5)}, "0")

	// ATTENTION: difference with delta_fifo_test, every event is its own Deltas with one item
	expectedList = []Deltas{
		{{Added, mkFifoObj("baz", 10)}},
		{{Deleted, DeletedFinalStateUnknown{Key: "baz", Obj: mkFifoObj("baz", 10)}}},
		// ATTENTION: difference with delta_fifo_test, logically the deletes of known items should happen BEFORE newItems are added, so this delete happens early now
		// Since "bar" didn't have a delete event and wasn't in the Replace list
		// it should get a tombstone key with the right Obj.
		{{Deleted, DeletedFinalStateUnknown{Key: "bar", Obj: mkFifoObj("bar", 6)}}},
		{{Replaced, mkFifoObj("foo", 5)}},
	}

	for _, expected := range expectedList {
		cur := Pop(f).(Deltas)
		if e, a := expected, cur; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %#v, got %#v", e, a)
		}
	}

	// Now try deleting and recreating the object in the queue, then delete it by a Replace call
	f = NewRealFIFO(
		testFifoObjectKeyFunc,
		literalListerGetter(func() []testFifoObject {
			return []testFifoObject{mkFifoObj("foo", 5), mkFifoObj("bar", 6), mkFifoObj("baz", 7)}
		}),
		nil,
	)
	f.Delete(mkFifoObj("bar", 6))
	f.Add(mkFifoObj("bar", 100))
	f.Replace([]interface{}{mkFifoObj("foo", 5)}, "0")

	// ATTENTION: difference with delta_fifo_test, every event is its own Deltas with one item
	expectedList = []Deltas{
		{{Deleted, mkFifoObj("bar", 6)}},
		{{Added, mkFifoObj("bar", 100)}},
		// Since "bar" has a newer object in the queue than in the state,
		// it should get a tombstone key with the latest object from the queue
		{{Deleted, DeletedFinalStateUnknown{Key: "bar", Obj: mkFifoObj("bar", 100)}}},
		// ATTENTION: difference with delta_fifo_test, logically the deletes of known items should happen BEFORE newItems are added, so this delete happens early now
		{{Deleted, DeletedFinalStateUnknown{Key: "baz", Obj: mkFifoObj("baz", 7)}}},
		{{Replaced, mkFifoObj("foo", 5)}},
	}

	for _, expected := range expectedList {
		cur := Pop(f).(Deltas)
		if e, a := expected, cur; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %#v, got %#v", e, a)
		}
	}

	// Now try syncing it first to ensure the delete use the latest version
	f = NewRealFIFO(
		testFifoObjectKeyFunc,
		literalListerGetter(func() []testFifoObject {
			return []testFifoObject{mkFifoObj("foo", 5), mkFifoObj("bar", 6), mkFifoObj("baz", 7)}
		}),
		nil,
	)
	f.Replace([]interface{}{mkFifoObj("bar", 100), mkFifoObj("foo", 5)}, "0")
	f.Replace([]interface{}{mkFifoObj("foo", 5)}, "0")

	// ATTENTION: difference with delta_fifo_test, every event is its own Deltas with one item
	// ATTENTION: difference with delta_fifo_test, deltaFifo associated by key, but realFIFO orders across all keys, so this ordering changed
	expectedList = []Deltas{
		// ATTENTION: difference with delta_fifo_test, logically the deletes of known items should happen BEFORE newItems are added, so this delete happens early now
		// Since "baz" didn't have a delete event and wasn't in the Replace list
		{{Deleted, DeletedFinalStateUnknown{Key: "baz", Obj: mkFifoObj("baz", 7)}}},
		{{Replaced, mkFifoObj("bar", 100)}},
		{{Replaced, mkFifoObj("foo", 5)}},
		// Since "bar" didn't have a delete event and wasn't in the Replace list
		// it should get a tombstone key with the right Obj.
		{{Deleted, DeletedFinalStateUnknown{Key: "bar", Obj: mkFifoObj("bar", 100)}}},
		{{Replaced, mkFifoObj("foo", 5)}},
	}

	for i, expected := range expectedList {
		cur := Pop(f).(Deltas)
		if e, a := expected, cur; !reflect.DeepEqual(e, a) {
			t.Errorf("%d Expected %#v, got %#v", i, e, a)
		}
	}

	// Now try starting without an explicit KeyListerGetter
	f = NewRealFIFO(
		testFifoObjectKeyFunc,
		emptyKnownObjects(),
		nil,
	)
	f.Add(mkFifoObj("baz", 10))
	f.Replace([]interface{}{mkFifoObj("foo", 5)}, "0")

	expectedList = []Deltas{
		{{Added, mkFifoObj("baz", 10)}},
		{{Deleted, DeletedFinalStateUnknown{Key: "baz", Obj: mkFifoObj("baz", 10)}}},
		{{Replaced, mkFifoObj("foo", 5)}},
	}

	for _, expected := range expectedList {
		cur := Pop(f).(Deltas)
		if e, a := expected, cur; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %#v, got %#v", e, a)
		}
	}
}

// TestRealFIFO_ReplaceMakesDeletionsReplaced is the same as the above test, but
// ensures that a Replaced DeltaType is emitted.
func TestRealFIFO_ReplaceMakesDeletionsReplaced(t *testing.T) {
	f := NewRealFIFO(
		testFifoObjectKeyFunc,
		literalListerGetter(func() []testFifoObject {
			return []testFifoObject{mkFifoObj("foo", 5), mkFifoObj("bar", 6), mkFifoObj("baz", 7)}
		}),
		nil,
	)

	f.Delete(mkFifoObj("baz", 10))
	f.Replace([]interface{}{mkFifoObj("foo", 6)}, "0")

	expectedList := []Deltas{
		{{Deleted, mkFifoObj("baz", 10)}},
		// ATTENTION: difference with delta_fifo_test, logically the deletes of known items should happen BEFORE newItems are added, so this delete happens early now
		// Since "bar" didn't have a delete event and wasn't in the Replace list
		// it should get a tombstone key with the right Obj.
		{{Deleted, DeletedFinalStateUnknown{Key: "bar", Obj: mkFifoObj("bar", 6)}}},
		{{Replaced, mkFifoObj("foo", 6)}},
	}

	for _, expected := range expectedList {
		cur := Pop(f).(Deltas)
		if e, a := expected, cur; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %#v, got %#v", e, a)
		}
	}
}

// ATTENTION: difference with delta_fifo_test, the previous value was hardcoded as use "Replace" so I've eliminated the option to set it differently
//func TestRealFIFO_ReplaceDeltaType(t *testing.T) {

func TestRealFIFO_UpdateResyncRace(t *testing.T) {
	f := NewRealFIFO(
		testFifoObjectKeyFunc,
		literalListerGetter(func() []testFifoObject {
			return []testFifoObject{mkFifoObj("foo", 5)}
		}),
		nil,
	)
	f.Update(mkFifoObj("foo", 6))
	f.Resync()

	expectedList := []Deltas{
		{{Updated, mkFifoObj("foo", 6)}},
	}

	for _, expected := range expectedList {
		cur := Pop(f).(Deltas)
		if e, a := expected, cur; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %#v, got %#v", e, a)
		}
	}
}

func TestRealFIFO_HasSyncedCorrectOnDeletion(t *testing.T) {
	f := NewRealFIFO(
		testFifoObjectKeyFunc,
		literalListerGetter(func() []testFifoObject {
			return []testFifoObject{mkFifoObj("foo", 5), mkFifoObj("bar", 6), mkFifoObj("baz", 7)}
		}),
		nil,
	)
	f.Replace([]interface{}{mkFifoObj("foo", 5)}, "0")

	expectedList := []Deltas{
		// ATTENTION: difference with delta_fifo_test, logically the deletes of known items should happen BEFORE newItems are added, so this delete happens early now
		// Since "bar" didn't have a delete event and wasn't in the Replace list
		// it should get a tombstone key with the right Obj.
		{{Deleted, DeletedFinalStateUnknown{Key: "bar", Obj: mkFifoObj("bar", 6)}}},
		{{Deleted, DeletedFinalStateUnknown{Key: "baz", Obj: mkFifoObj("baz", 7)}}},
		{{Replaced, mkFifoObj("foo", 5)}},
	}

	for _, expected := range expectedList {
		if f.HasSynced() {
			t.Errorf("Expected HasSynced to be false")
		}
		cur, initial := pop2[Deltas](f)
		if e, a := expected, cur; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %#v, got %#v", e, a)
		}
		if initial != true {
			t.Error("Expected initial list item")
		}
	}
	if !f.HasSynced() {
		t.Errorf("Expected HasSynced to be true")
	}
}

func TestRealFIFO_detectLineJumpers(t *testing.T) {
	f := NewRealFIFO(
		testFifoObjectKeyFunc,
		emptyKnownObjects(),
		nil,
	)

	f.Add(mkFifoObj("foo", 10))
	f.Add(mkFifoObj("bar", 1))
	f.Add(mkFifoObj("foo", 11))
	f.Add(mkFifoObj("foo", 13))
	f.Add(mkFifoObj("zab", 30))

	// ATTENTION: difference with delta_fifo_test, every event is delivered in order

	if e, a := 10, testRealFIFOPop(f).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}

	f.Add(mkFifoObj("foo", 14)) // ensure foo doesn't jump back in line

	if e, a := 1, testRealFIFOPop(f).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	if e, a := 11, testRealFIFOPop(f).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	if e, a := 13, testRealFIFOPop(f).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	if e, a := 30, testRealFIFOPop(f).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
	if e, a := 14, testRealFIFOPop(f).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
}

func TestRealFIFO_KeyOf(t *testing.T) {
	f := RealFIFO{keyFunc: testFifoObjectKeyFunc}

	table := []struct {
		obj interface{}
		key string
	}{
		{obj: testFifoObject{name: "A"}, key: "A"},
		{obj: DeletedFinalStateUnknown{Key: "B", Obj: nil}, key: "B"},
		{obj: Deltas{{Object: testFifoObject{name: "C"}}}, key: "C"},
		{obj: Deltas{{Object: DeletedFinalStateUnknown{Key: "D", Obj: nil}}}, key: "D"},
	}

	for _, item := range table {
		got, err := f.keyOf(item.obj)
		if err != nil {
			t.Errorf("Unexpected error for %q: %v", item.obj, err)
			continue
		}
		if e, a := item.key, got; e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}
}

func TestRealFIFO_HasSynced(t *testing.T) {
	tests := []struct {
		actions        []func(f *RealFIFO)
		expectedSynced bool
	}{
		{
			actions:        []func(f *RealFIFO){},
			expectedSynced: false,
		},
		{
			actions: []func(f *RealFIFO){
				func(f *RealFIFO) { f.Add(mkFifoObj("a", 1)) },
			},
			expectedSynced: true,
		},
		{
			actions: []func(f *RealFIFO){
				func(f *RealFIFO) { f.Replace([]interface{}{}, "0") },
			},
			expectedSynced: true,
		},
		{
			actions: []func(f *RealFIFO){
				func(f *RealFIFO) { f.Replace([]interface{}{mkFifoObj("a", 1), mkFifoObj("b", 2)}, "0") },
			},
			expectedSynced: false,
		},
		{
			actions: []func(f *RealFIFO){
				func(f *RealFIFO) { f.Replace([]interface{}{mkFifoObj("a", 1), mkFifoObj("b", 2)}, "0") },
				func(f *RealFIFO) { Pop(f) },
			},
			expectedSynced: false,
		},
		{
			actions: []func(f *RealFIFO){
				func(f *RealFIFO) { f.Replace([]interface{}{mkFifoObj("a", 1), mkFifoObj("b", 2)}, "0") },
				func(f *RealFIFO) { Pop(f) },
				func(f *RealFIFO) { Pop(f) },
			},
			expectedSynced: true,
		},
		{
			// This test case won't happen in practice since a Reflector, the only producer for delta_fifo today, always passes a complete snapshot consistent in time;
			// there cannot be duplicate keys in the list or apiserver is broken.
			actions: []func(f *RealFIFO){
				func(f *RealFIFO) { f.Replace([]interface{}{mkFifoObj("a", 1), mkFifoObj("a", 2)}, "0") },
				func(f *RealFIFO) { Pop(f) },
				// ATTENTION: difference with delta_fifo_test, every event is delivered, so a is listed twice and must be popped twice to remove both
				func(f *RealFIFO) { Pop(f) },
			},
			expectedSynced: true,
		},
	}

	for i, test := range tests {
		f := NewRealFIFO(
			testFifoObjectKeyFunc,
			emptyKnownObjects(),
			nil,
		)

		for _, action := range test.actions {
			action(f)
		}
		if e, a := test.expectedSynced, f.HasSynced(); a != e {
			t.Errorf("test case %v failed, expected: %v , got %v", i, e, a)
		}
	}
}

// TestRealFIFO_PopShouldUnblockWhenClosed checks that any blocking Pop on an empty queue
// should unblock and return after Close is called.
func TestRealFIFO_PopShouldUnblockWhenClosed(t *testing.T) {
	f := NewRealFIFO(
		testFifoObjectKeyFunc,
		literalListerGetter(func() []testFifoObject {
			return []testFifoObject{mkFifoObj("foo", 5)}
		}),
		nil,
	)

	c := make(chan struct{})
	const jobs = 10
	for i := 0; i < jobs; i++ {
		go func() {
			f.Pop(func(obj interface{}, isInInitialList bool) error {
				return nil
			})
			c <- struct{}{}
		}()
	}

	runtime.Gosched()
	f.Close()

	for i := 0; i < jobs; i++ {
		select {
		case <-c:
		case <-time.After(500 * time.Millisecond):
			t.Fatalf("timed out waiting for Pop to return after Close")
		}
	}
}
