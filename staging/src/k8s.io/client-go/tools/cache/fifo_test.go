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

func testFifoObjectKeyFunc(obj interface{}) (string, error) {
	return obj.(testFifoObject).name, nil
}

type testFifoObject struct {
	name string
	val  interface{}
}

func mkFifoObj(name string, val interface{}) testFifoObject {
	return testFifoObject{name: name, val: val}
}

func TestFIFO_basic(t *testing.T) {
	f := NewFIFO(testFifoObjectKeyFunc)
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
		switch obj := Pop(f).(testFifoObject).val.(type) {
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

func TestFIFO_requeueOnPop(t *testing.T) {
	f := NewFIFO(testFifoObjectKeyFunc)

	f.Add(mkFifoObj("foo", 10))
	_, err := f.Pop(func(obj interface{}) error {
		if obj.(testFifoObject).name != "foo" {
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
		if obj.(testFifoObject).name != "foo" {
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
		if obj.(testFifoObject).name != "foo" {
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

func TestFIFO_addUpdate(t *testing.T) {
	f := NewFIFO(testFifoObjectKeyFunc)
	f.Add(mkFifoObj("foo", 10))
	f.Update(mkFifoObj("foo", 15))

	if e, a := []interface{}{mkFifoObj("foo", 15)}, f.List(); !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %+v, got %+v", e, a)
	}
	if e, a := []string{"foo"}, f.ListKeys(); !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %+v, got %+v", e, a)
	}

	got := make(chan testFifoObject, 2)
	go func() {
		for {
			got <- Pop(f).(testFifoObject)
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
	_, exists, _ := f.Get(mkFifoObj("foo", ""))
	if exists {
		t.Errorf("item did not get removed")
	}
}

func TestFIFO_addReplace(t *testing.T) {
	f := NewFIFO(testFifoObjectKeyFunc)
	f.Add(mkFifoObj("foo", 10))
	f.Replace([]interface{}{mkFifoObj("foo", 15)}, "15")
	got := make(chan testFifoObject, 2)
	go func() {
		for {
			got <- Pop(f).(testFifoObject)
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
	_, exists, _ := f.Get(mkFifoObj("foo", ""))
	if exists {
		t.Errorf("item did not get removed")
	}
}

func TestFIFO_detectLineJumpers(t *testing.T) {
	f := NewFIFO(testFifoObjectKeyFunc)

	f.Add(mkFifoObj("foo", 10))
	f.Add(mkFifoObj("bar", 1))
	f.Add(mkFifoObj("foo", 11))
	f.Add(mkFifoObj("foo", 13))
	f.Add(mkFifoObj("zab", 30))

	if e, a := 13, Pop(f).(testFifoObject).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}

	f.Add(mkFifoObj("foo", 14)) // ensure foo doesn't jump back in line

	if e, a := 1, Pop(f).(testFifoObject).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}

	if e, a := 30, Pop(f).(testFifoObject).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}

	if e, a := 14, Pop(f).(testFifoObject).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
}

func TestFIFO_addIfNotPresent(t *testing.T) {
	f := NewFIFO(testFifoObjectKeyFunc)

	f.Add(mkFifoObj("a", 1))
	f.Add(mkFifoObj("b", 2))
	f.AddIfNotPresent(mkFifoObj("b", 3))
	f.AddIfNotPresent(mkFifoObj("c", 4))

	if e, a := 3, len(f.items); a != e {
		t.Fatalf("expected queue length %d, got %d", e, a)
	}

	expectedValues := []int{1, 2, 4}
	for _, expected := range expectedValues {
		if actual := Pop(f).(testFifoObject).val; actual != expected {
			t.Fatalf("expected value %d, got %d", expected, actual)
		}
	}
}

func TestFIFO_HasSynced(t *testing.T) {
	tests := []struct {
		actions        []func(f *FIFO)
		expectedSynced bool
	}{
		{
			actions:        []func(f *FIFO){},
			expectedSynced: false,
		},
		{
			actions: []func(f *FIFO){
				func(f *FIFO) { f.Add(mkFifoObj("a", 1)) },
			},
			expectedSynced: true,
		},
		{
			actions: []func(f *FIFO){
				func(f *FIFO) { f.Replace([]interface{}{}, "0") },
			},
			expectedSynced: true,
		},
		{
			actions: []func(f *FIFO){
				func(f *FIFO) { f.Replace([]interface{}{mkFifoObj("a", 1), mkFifoObj("b", 2)}, "0") },
			},
			expectedSynced: false,
		},
		{
			actions: []func(f *FIFO){
				func(f *FIFO) { f.Replace([]interface{}{mkFifoObj("a", 1), mkFifoObj("b", 2)}, "0") },
				func(f *FIFO) { Pop(f) },
			},
			expectedSynced: false,
		},
		{
			actions: []func(f *FIFO){
				func(f *FIFO) { f.Replace([]interface{}{mkFifoObj("a", 1), mkFifoObj("b", 2)}, "0") },
				func(f *FIFO) { Pop(f) },
				func(f *FIFO) { Pop(f) },
			},
			expectedSynced: true,
		},
	}

	for i, test := range tests {
		f := NewFIFO(testFifoObjectKeyFunc)

		for _, action := range test.actions {
			action(f)
		}
		if e, a := test.expectedSynced, f.HasSynced(); a != e {
			t.Errorf("test case %v failed, expected: %v , got %v", i, e, a)
		}
	}
}

// TestFIFO_PopShouldUnblockWhenClosed checks that any blocking Pop on an empty queue
// should unblock and return after Close is called.
func TestFIFO_PopShouldUnblockWhenClosed(t *testing.T) {
	f := NewFIFO(testFifoObjectKeyFunc)

	c := make(chan struct{})
	const jobs = 10
	for i := 0; i < jobs; i++ {
		go func() {
			f.Pop(func(obj interface{}) error {
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
