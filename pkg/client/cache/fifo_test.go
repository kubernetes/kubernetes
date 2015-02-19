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
	"time"
)

func testFifoObjectKeyFunc(obj interface{}) (string, error) {
	return obj.(testFifoObject).name, nil
}

type testFifoObject struct {
	name string
	val  interface{}
}

func TestFIFO_basic(t *testing.T) {
	mkObj := func(name string, val interface{}) testFifoObject {
		return testFifoObject{name: name, val: val}
	}

	f := NewFIFO(testFifoObjectKeyFunc)
	const amount = 500
	go func() {
		for i := 0; i < amount; i++ {
			f.Add(mkObj(string([]rune{'a', rune(i)}), i+1))
		}
	}()
	go func() {
		for u := uint64(0); u < amount; u++ {
			f.Add(mkObj(string([]rune{'b', rune(u)}), u+1))
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
	mkObj := func(name string, val interface{}) testFifoObject {
		return testFifoObject{name: name, val: val}
	}

	f := NewFIFO(testFifoObjectKeyFunc)
	f.Add(mkObj("foo", 10))
	f.Update(mkObj("foo", 15))
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
	_, exists, _ := f.Get(mkObj("foo", ""))
	if exists {
		t.Errorf("item did not get removed")
	}
}

func TestFIFO_addReplace(t *testing.T) {
	mkObj := func(name string, val interface{}) testFifoObject {
		return testFifoObject{name: name, val: val}
	}

	f := NewFIFO(testFifoObjectKeyFunc)
	f.Add(mkObj("foo", 10))
	f.Replace([]interface{}{mkObj("foo", 15)})
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
	_, exists, _ := f.Get(mkObj("foo", ""))
	if exists {
		t.Errorf("item did not get removed")
	}
}

func TestFIFO_detectLineJumpers(t *testing.T) {
	mkObj := func(name string, val interface{}) testFifoObject {
		return testFifoObject{name: name, val: val}
	}

	f := NewFIFO(testFifoObjectKeyFunc)

	f.Add(mkObj("foo", 10))
	f.Add(mkObj("bar", 1))
	f.Add(mkObj("foo", 11))
	f.Add(mkObj("foo", 13))
	f.Add(mkObj("zab", 30))

	if e, a := 13, f.Pop().(testFifoObject).val; a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}

	f.Add(mkObj("foo", 14)) // ensure foo doesn't jump back in line

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
	mkObj := func(name string, val interface{}) testFifoObject {
		return testFifoObject{name: name, val: val}
	}

	f := NewFIFO(testFifoObjectKeyFunc)

	f.Add(mkObj("a", 1))
	f.Add(mkObj("b", 2))
	f.AddIfNotPresent(mkObj("b", 3))
	f.AddIfNotPresent(mkObj("c", 4))

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
