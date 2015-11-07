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

package util

import (
	"math/rand"
	"sync"
	"testing"
	"time"
)

func ExpectValue(t *testing.T, atomicValue *AtomicValue, expectedValue interface{}) {
	actualValue := atomicValue.Load()
	if actualValue != expectedValue {
		t.Errorf("Expected to find %v, found %v", expectedValue, actualValue)
	}
	ch := make(chan interface{})
	go func() {
		ch <- atomicValue.Load()
	}()
	select {
	case actualValue = <-ch:
		if actualValue != expectedValue {
			t.Errorf("Expected to find %v, found %v", expectedValue, actualValue)
			return
		}
	case <-time.After(ForeverTestTimeout):
		t.Error("Value could not be read")
		return
	}
}

func TestAtomicValue(t *testing.T) {
	atomicValue := &AtomicValue{}
	ExpectValue(t, atomicValue, nil)
	atomicValue.Store(10)
	ExpectValue(t, atomicValue, 10)
}

func TestHighWaterMark(t *testing.T) {
	var h HighWaterMark

	for i := int64(10); i < 20; i++ {
		if !h.Check(i) {
			t.Errorf("unexpected false for %v", i)
		}
		if h.Check(i - 1) {
			t.Errorf("unexpected true for %v", i-1)
		}
	}

	m := int64(0)
	wg := sync.WaitGroup{}
	for i := 0; i < 300; i++ {
		wg.Add(1)
		v := rand.Int63()
		go func(v int64) {
			defer wg.Done()
			h.Check(v)
		}(v)
		if v > m {
			m = v
		}
	}
	wg.Wait()
	if m != int64(h) {
		t.Errorf("unexpected value, wanted %v, got %v", m, int64(h))
	}
}
