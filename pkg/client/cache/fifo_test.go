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

func TestFIFO_basic(t *testing.T) {
	f := NewFIFO()
	const amount = 500
	go func() {
		for i := 0; i < amount; i++ {
			f.Add(string([]rune{'a', rune(i)}), i+1)
		}
	}()
	go func() {
		for u := uint(0); u < amount; u++ {
			f.Add(string([]rune{'b', rune(u)}), u+1)
		}
	}()

	lastInt := int(0)
	lastUint := uint(0)
	for i := 0; i < amount*2; i++ {
		switch obj := f.Pop().(type) {
		case int:
			if obj <= lastInt {
				t.Errorf("got %v (int) out of order, last was %v", obj, lastInt)
			}
			lastInt = obj
		case uint:
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
	f := NewFIFO()
	f.Add("foo", 10)
	f.Update("foo", 15)
	got := make(chan int, 2)
	go func() {
		for {
			got <- f.Pop().(int)
		}
	}()

	first := <-got
	if e, a := 15, first; e != a {
		t.Errorf("Didn't get updated value (%v), got %v", e, a)
	}
	select {
	case unexpected := <-got:
		t.Errorf("Got second value %v", unexpected)
	case <-time.After(50 * time.Millisecond):
	}
	_, exists := f.Get("foo")
	if exists {
		t.Errorf("item did not get removed")
	}
}

func TestFIFO_addReplace(t *testing.T) {
	f := NewFIFO()
	f.Add("foo", 10)
	f.Replace(map[string]interface{}{"foo": 15})
	got := make(chan int, 2)
	go func() {
		for {
			got <- f.Pop().(int)
		}
	}()

	first := <-got
	if e, a := 15, first; e != a {
		t.Errorf("Didn't get updated value (%v), got %v", e, a)
	}
	select {
	case unexpected := <-got:
		t.Errorf("Got second value %v", unexpected)
	case <-time.After(50 * time.Millisecond):
	}
	_, exists := f.Get("foo")
	if exists {
		t.Errorf("item did not get removed")
	}
}

func TestFIFO_detectLineJumpers(t *testing.T) {
	f := NewFIFO()

	f.Add("foo", 10)
	f.Add("bar", 1)
	f.Add("foo", 11)
	f.Add("foo", 13)
	f.Add("zab", 30)

	if e, a := 13, f.Pop().(int); a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}

	f.Add("foo", 14) // ensure foo doesn't jump back in line

	if e, a := 1, f.Pop().(int); a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}

	if e, a := 30, f.Pop().(int); a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}

	if e, a := 14, f.Pop().(int); a != e {
		t.Fatalf("expected %d, got %d", e, a)
	}
}
