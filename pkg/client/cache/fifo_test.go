/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"reflect"
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
		switch obj := f.Pop().(testFifoObject).val.(type) {
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
			got <- f.Pop().(testFifoObject)
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
			got <- f.Pop().(testFifoObject)
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

	if e, a := 13, f.Pop().(testFifoObject).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}

	f.Add(mkFifoObj("foo", 14)) // ensure foo doesn't jump back in line

	if e, a := 1, f.Pop().(testFifoObject).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}

	if e, a := 30, f.Pop().(testFifoObject).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}

	if e, a := 14, f.Pop().(testFifoObject).val; a != e {
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
		if actual := f.Pop().(testFifoObject).val; actual != expected {
			t.Fatalf("expected value %d, got %d", expected, actual)
		}
	}
}
