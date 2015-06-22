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
	"testing"
	"time"
)

func ExpectValue(t *testing.T, atomicValue *AtomicValue, expectedValue interface{}) {
	actualValue := atomicValue.Load()
	if actualValue != expectedValue {
		t.Error("Expected to find %v, found %v", expectedValue, actualValue)
	}
	ch := make(chan interface{})
	go func() {
		ch <- atomicValue.Load()
	}()
	select {
	case actualValue = <-ch:
		if actualValue != expectedValue {
			t.Error("Expected to find %v, found %v", expectedValue, actualValue)
			return
		}
	case <-time.After(time.Second * 5):
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
