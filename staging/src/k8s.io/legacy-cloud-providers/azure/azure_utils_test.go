/*
Copyright 2019 The Kubernetes Authors.

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

package azure

import (
	"fmt"
	"testing"
	"time"
)

func TestSimpleLockEntry(t *testing.T) {
	testLockMap := newLockMap()

	callbackChan1 := make(chan interface{})
	go testLockMap.lockAndCallback(t, "entry1", callbackChan1)
	ensureCallbackHappens(t, callbackChan1)
}

func TestSimpleLockUnlockEntry(t *testing.T) {
	testLockMap := newLockMap()

	callbackChan1 := make(chan interface{})
	go testLockMap.lockAndCallback(t, "entry1", callbackChan1)
	ensureCallbackHappens(t, callbackChan1)
	testLockMap.UnlockEntry("entry1")
}

func TestConcurrentLockEntry(t *testing.T) {
	testLockMap := newLockMap()

	callbackChan1 := make(chan interface{})
	callbackChan2 := make(chan interface{})

	go testLockMap.lockAndCallback(t, "entry1", callbackChan1)
	ensureCallbackHappens(t, callbackChan1)

	go testLockMap.lockAndCallback(t, "entry1", callbackChan2)
	ensureNoCallback(t, callbackChan2)

	testLockMap.UnlockEntry("entry1")
	ensureCallbackHappens(t, callbackChan2)
	testLockMap.UnlockEntry("entry1")
}

func (lm *lockMap) lockAndCallback(t *testing.T, entry string, callbackChan chan<- interface{}) {
	lm.LockEntry(entry)
	callbackChan <- true
}

var callbackTimeout = 2 * time.Second

func ensureCallbackHappens(t *testing.T, callbackChan <-chan interface{}) bool {
	select {
	case <-callbackChan:
		return true
	case <-time.After(callbackTimeout):
		t.Fatalf("timed out waiting for callback")
		return false
	}
}

func ensureNoCallback(t *testing.T, callbackChan <-chan interface{}) bool {
	select {
	case <-callbackChan:
		t.Fatalf("unexpected callback")
		return false
	case <-time.After(callbackTimeout):
		return true
	}
}

// running same unit tests as https://github.com/kubernetes/apimachinery/blob/master/pkg/util/errors/errors_test.go#L371
func TestAggregateGoroutinesWithDelay(t *testing.T) {
	testCases := []struct {
		errs     []error
		expected map[string]bool
	}{
		{
			[]error{},
			nil,
		},
		{
			[]error{nil},
			nil,
		},
		{
			[]error{nil, nil},
			nil,
		},
		{
			[]error{fmt.Errorf("1")},
			map[string]bool{"1": true},
		},
		{
			[]error{fmt.Errorf("1"), nil},
			map[string]bool{"1": true},
		},
		{
			[]error{fmt.Errorf("1"), fmt.Errorf("267")},
			map[string]bool{"1": true, "267": true},
		},
		{
			[]error{fmt.Errorf("1"), nil, fmt.Errorf("1234")},
			map[string]bool{"1": true, "1234": true},
		},
		{
			[]error{nil, fmt.Errorf("1"), nil, fmt.Errorf("1234"), fmt.Errorf("22")},
			map[string]bool{"1": true, "1234": true, "22": true},
		},
	}
	for i, testCase := range testCases {
		funcs := make([]func() error, len(testCase.errs))
		for i := range testCase.errs {
			err := testCase.errs[i]
			funcs[i] = func() error { return err }
		}
		agg := aggregateGoroutinesWithDelay(100*time.Millisecond, funcs...)
		if agg == nil {
			if len(testCase.expected) > 0 {
				t.Errorf("%d: expected %v, got nil", i, testCase.expected)
			}
			continue
		}
		if len(agg.Errors()) != len(testCase.expected) {
			t.Errorf("%d: expected %d errors in aggregate, got %v", i, len(testCase.expected), agg)
			continue
		}
		for _, err := range agg.Errors() {
			if !testCase.expected[err.Error()] {
				t.Errorf("%d: expected %v, got aggregate containing %v", i, testCase.expected, err)
			}
		}
	}
}
