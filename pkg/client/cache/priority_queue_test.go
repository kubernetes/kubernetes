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
	"testing"
	"time"
)

func testPriorityQueueObjectKeyFunc(obj interface{}) (string, error) {
	return obj.(testPriorityQueueObject).name, nil
}

type testPriorityQueueObject struct {
	name string
	val  interface{}
}

func mkPriorityQueueObj(name string, val interface{}) testPriorityQueueObject {
	return testPriorityQueueObject{name: name, val: val}
}

//TODO need to assign priority to an object before adding it.
//this is testing FIFO currently...
func TestPriorityQueue_basic(t *testing.T) {
	f := NewPriorityQueue(testPriorityQueueObjectKeyFunc)
	const amount = 500
	go func() {
		for i := 0; i < amount; i++ {
			f.Add(mkPriorityQueueObj(string([]rune{'a', rune(i)}), i+1))
		}
	}()
	go func() {
		for u := uint64(0); u < amount; u++ {
			f.Add(mkPriorityQueueObj(string([]rune{'b', rune(u)}), u+1))
		}
	}()

	lastInt := int(0)
	lastUint := uint64(0)
	for i := 0; i < amount*2; i++ {
		switch obj := Pop(f).(testPriorityQueueObject).val.(type) {
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

func TestPriorityQueue_requeueOnPop(t *testing.T) {
	f := NewPriorityQueue(testPriorityQueueObjectKeyFunc)

	f.Add(mkPriorityQueueObj("foo", 10))
	_, err := f.Pop(func(obj interface{}) error {
		if obj.(testPriorityQueueObject).name != "foo" {
			t.Fatalf("unexpected object: %#v", obj)
		}
		return ErrRequeue{Err: nil}
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok, err := f.GetByKey("foo"); !ok || err != nil {
		t.Fatalf("object should have been requeued: %t %v", ok, err)
	}

	_, err = f.Pop(func(obj interface{}) error {
		if obj.(testPriorityQueueObject).name != "foo" {
			t.Fatalf("unexpected object: %#v", obj)
		}
		return ErrRequeue{Err: fmt.Errorf("test error")}
	})
	if err == nil || err.Error() != "test error" {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok, err := f.GetByKey("foo"); !ok || err != nil {
		t.Fatalf("object should have been requeued: %t %v", ok, err)
	}

	_, err = f.Pop(func(obj interface{}) error {
		if obj.(testPriorityQueueObject).name != "foo" {
			t.Fatalf("unexpected object: %#v", obj)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok, err := f.GetByKey("foo"); ok || err != nil {
		t.Fatalf("object should have been removed: %t %v", ok, err)
	}
}

func TestPriorityQueue_addUpdate(t *testing.T) {
	f := NewPriorityQueue(testPriorityQueueObjectKeyFunc)
	f.Add(mkPriorityQueueObj("foo", 10))
	f.Update(mkPriorityQueueObj("foo", 15))

	if e, a := []interface{}{mkPriorityQueueObj("foo", 15)}, f.List(); !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %+v, got %+v", e, a)
	}
	if e, a := []string{"foo"}, f.ListKeys(); !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %+v, got %+v", e, a)
	}

	got := make(chan testPriorityQueueObject, 2)
	go func() {
		for {
			got <- Pop(f).(testPriorityQueueObject)
		}
	}()

	first := <-got
	if e, a := 15, first.val; e != a {
		t.Errorf("Didn't get updated value (%v), got %v", e, a)
	}
	select {
	case unexpected := <-got:
		t.Errorf("Got second value %v", unexpected.val)
	case <-time.After(50 * time.Millisecond):
	}
	_, exists, _ := f.Get(mkPriorityQueueObj("foo", ""))
	if exists {
		t.Errorf("item did not get removed")
	}
}

func TestPriorityQueue_addReplace(t *testing.T) {
	f := NewPriorityQueue(testPriorityQueueObjectKeyFunc)
	f.Add(mkPriorityQueueObj("foo", 10))
	f.Replace([]interface{}{mkPriorityQueueObj("foo", 15)}, "15")
	got := make(chan testPriorityQueueObject, 2)
	go func() {
		for {
			got <- Pop(f).(testPriorityQueueObject)
		}
	}()

	first := <-got
	if e, a := 15, first.val; e != a {
		t.Errorf("Didn't get updated value (%v), got %v", e, a)
	}
	select {
	case unexpected := <-got:
		t.Errorf("Got second value %v", unexpected.val)
	case <-time.After(50 * time.Millisecond):
	}
	_, exists, _ := f.Get(mkPriorityQueueObj("foo", ""))
	if exists {
		t.Errorf("item did not get removed")
	}
}

func TestPriorityQueue_detectLineJumpers(t *testing.T) {
	f := NewPriorityQueue(testPriorityQueueObjectKeyFunc)

	f.Add(mkPriorityQueueObj("foo", 10))
	f.Add(mkPriorityQueueObj("bar", 1))
	f.Add(mkPriorityQueueObj("foo", 11))
	f.Add(mkPriorityQueueObj("foo", 13))
	f.Add(mkPriorityQueueObj("zab", 30))

	if e, a := 13, Pop(f).(testPriorityQueueObject).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}

	f.Add(mkPriorityQueueObj("foo", 14)) // ensure foo doesn't jump back in line

	if e, a := 1, Pop(f).(testPriorityQueueObject).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}

	if e, a := 30, Pop(f).(testPriorityQueueObject).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}

	if e, a := 14, Pop(f).(testPriorityQueueObject).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
}

func TestPriorityQueue_addIfNotPresent(t *testing.T) {
	f := NewPriorityQueue(testPriorityQueueObjectKeyFunc)

	f.Add(mkPriorityQueueObj("a", 1))
	f.Add(mkPriorityQueueObj("b", 2))
	f.AddIfNotPresent(mkPriorityQueueObj("b", 3))
	f.AddIfNotPresent(mkPriorityQueueObj("c", 4))

	if e, a := 3, len(f.items); a != e {
		t.Fatalf("expected queue length %d, got %d", e, a)
	}

	expectedValues := []int{1, 2, 4}
	for _, expected := range expectedValues {
		if actual := Pop(f).(testPriorityQueueObject).val; actual != expected {
			t.Fatalf("expected value %d, got %d", expected, actual)
		}
	}
}

func TestPriorityQueue_HasSynced(t *testing.T) {
	tests := []struct {
		actions        []func(f *PriorityQueue)
		expectedSynced bool
	}{
		{
			actions:        []func(f *PriorityQueue){},
			expectedSynced: false,
		},
		{
			actions: []func(f *PriorityQueue){
				func(f *PriorityQueue) { f.Add(mkPriorityQueueObj("a", 1)) },
			},
			expectedSynced: true,
		},
		{
			actions: []func(f *PriorityQueue){
				func(f *PriorityQueue) { f.Replace([]interface{}{}, "0") },
			},
			expectedSynced: true,
		},
		{
			actions: []func(f *PriorityQueue){
				func(f *PriorityQueue) { f.Replace([]interface{}{mkPriorityQueueObj("a", 1), mkPriorityQueueObj("b", 2)}, "0") },
			},
			expectedSynced: false,
		},
		{
			actions: []func(f *PriorityQueue){
				func(f *PriorityQueue) { f.Replace([]interface{}{mkPriorityQueueObj("a", 1), mkPriorityQueueObj("b", 2)}, "0") },
				func(f *PriorityQueue) { Pop(f) },
			},
			expectedSynced: false,
		},
		{
			actions: []func(f *PriorityQueue){
				func(f *PriorityQueue) { f.Replace([]interface{}{mkPriorityQueueObj("a", 1), mkPriorityQueueObj("b", 2)}, "0") },
				func(f *PriorityQueue) { Pop(f) },
				func(f *PriorityQueue) { Pop(f) },
			},
			expectedSynced: true,
		},
	}

	for i, test := range tests {
		f := NewPriorityQueue(testPriorityQueueObjectKeyFunc)

		for _, action := range test.actions {
			action(f)
		}
		if e, a := test.expectedSynced, f.HasSynced(); a != e {
			t.Errorf("test case %v failed, expected: %v , got %v", i, e, a)
		}
	}
}

//func TestPriorityQueue_MetaPriorityFunc(t *testing.T) {
//    tests := []struct{
//        input               interface{}
//        expectedPriority    int
//        expectedErr         error
//    }{
//        {
//            //happy case
//            input:              //make an object,
//            expectedPriority:   -1,
//            expectedErr:        nil,
//        },
//    }
//    for i, test := range tests {
//        pq := NewPriorityQueue(testPriorityQueueObjectKeyFunc)
//
//        for _, input := range test.input {
//            priority, err := pq.MetaPriorityFunc(input)
//            if priority != test.expectedPriority {
//                t.Errorf("test case %v failed, expected: %v , got %v", i, expectedPriority, priority)
//            }
//            if err != test.expectedPriority {
//                t.Errorf("test case %v failed, expected: %v , got %v", i, err, expectedErr)
//            }
//        }
//    }
//    }
//}
