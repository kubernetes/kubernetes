/*
Copyright 2015 The Kubernetes Authors.

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

package queue

import (
	"fmt"
	"testing"
	"time"
)

type _int int
type _uint uint

func (i _int) Copy() Copyable {
	return i
}

func (i _int) GetUID() string {
	return fmt.Sprintf("INT%d", int(i))
}

func (i _uint) Copy() Copyable {
	return i
}

func (i _uint) GetUID() string {
	return fmt.Sprintf("UINT%d", uint64(i))
}

type testObj struct {
	id    string
	value int
}

func (i *testObj) Copy() Copyable {
	if i == nil {
		return nil
	} else {
		return &testObj{i.id, i.value}
	}
}

func (i *testObj) GetUID() string {
	return i.id
}

func TestFIFO_basic(t *testing.T) {
	f := NewHistorical(nil)
	const amount = 500
	go func() {
		for i := 0; i < amount; i++ {
			f.Add(_int(i + 1))
		}
	}()
	go func() {
		for u := uint(0); u < amount; u++ {
			f.Add(_uint(u + 1))
		}
	}()

	lastInt := _int(0)
	lastUint := _uint(0)
	for i := 0; i < amount*2; i++ {
		switch obj := f.Pop(WithoutCancel()).(type) {
		case _int:
			if obj <= lastInt {
				t.Errorf("got %v (int) out of order, last was %v", obj, lastInt)
			}
			lastInt = obj
		case _uint:
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
	f := NewHistorical(nil)
	f.Add(&testObj{"foo", 10})
	f.Update(&testObj{"foo", 15})
	got := make(chan *testObj, 2)
	go func() {
		for {
			got <- f.Pop(WithoutCancel()).(*testObj)
		}
	}()

	first := <-got
	if e, a := 15, first.value; e != a {
		t.Errorf("Didn't get updated value (%v), got %v", e, a)
	}
	select {
	case unexpected := <-got:
		t.Errorf("Got second value %v", unexpected)
	case <-time.After(50 * time.Millisecond):
	}
	_, exists, _ := f.GetByKey("foo")
	if exists {
		t.Errorf("item did not get removed")
	}
}

func TestFIFO_addDeleteAdd(t *testing.T) {
	f := NewHistorical(nil)
	testobj := &testObj{"foo", 10}
	f.Add(testobj)
	f.Delete(testobj)
	f.Add(testobj)

	_, exists, _ := f.GetByKey("foo")
	if !exists {
		t.Errorf("item did not get readded")
	}
}

func TestFIFO_addPopAdd(t *testing.T) {
	f := NewHistorical(nil)
	testobj := &testObj{"foo", 10}
	f.Add(testobj)
	f.Pop(nil)
	f.Add(testobj)

	_, exists, _ := f.GetByKey("foo")
	if !exists {
		t.Errorf("item did not get readded")
	}
}

func TestFIFO_addReplace(t *testing.T) {
	f := NewHistorical(nil)
	f.Add(&testObj{"foo", 10})
	f.Replace([]interface{}{&testObj{"foo", 15}}, "0")
	got := make(chan *testObj, 2)
	go func() {
		for {
			got <- f.Pop(WithoutCancel()).(*testObj)
		}
	}()

	first := <-got
	if e, a := 15, first.value; e != a {
		t.Errorf("Didn't get updated value (%v), got %v", e, a)
	}
	select {
	case unexpected := <-got:
		t.Errorf("Got second value %v", unexpected)
	case <-time.After(50 * time.Millisecond):
	}
	_, exists, _ := f.GetByKey("foo")
	if exists {
		t.Errorf("item did not get removed")
	}
}

func TestFIFO_detectLineJumpers(t *testing.T) {
	f := NewHistorical(nil)

	f.Add(&testObj{"foo", 10})
	f.Add(&testObj{"bar", 1})
	f.Add(&testObj{"foo", 11})
	f.Add(&testObj{"foo", 13})
	f.Add(&testObj{"zab", 30})

	err := error(nil)
	done := make(chan struct{})
	go func() {
		defer close(done)
		if e, a := 13, f.Pop(WithoutCancel()).(*testObj).value; a != e {
			err = fmt.Errorf("expected %d, got %d", e, a)
			return
		}

		f.Add(&testObj{"foo", 14}) // ensure foo doesn't jump back in line

		if e, a := 1, f.Pop(WithoutCancel()).(*testObj).value; a != e {
			err = fmt.Errorf("expected %d, got %d", e, a)
			return
		}

		if e, a := 30, f.Pop(WithoutCancel()).(*testObj).value; a != e {
			err = fmt.Errorf("expected %d, got %d", e, a)
			return
		}

		if e, a := 14, f.Pop(WithoutCancel()).(*testObj).value; a != e {
			err = fmt.Errorf("expected %d, got %d", e, a)
			return
		}
	}()
	select {
	case <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("Deadlocked unit test")
	}
}
