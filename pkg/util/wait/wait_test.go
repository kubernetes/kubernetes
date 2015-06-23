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

package wait

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestPoller(t *testing.T) {
	w := poller(time.Millisecond, 2*time.Millisecond)
	ch := w()
	count := 0
DRAIN:
	for {
		select {
		case _, open := <-ch:
			if !open {
				break DRAIN
			}
			count++
		case <-time.After(time.Second):
			t.Errorf("unexpected timeout after poll")
		}
	}
	if count > 3 {
		t.Errorf("expected up to three values, got %d", count)
	}
}

func fakeTicker(count int) WaitFunc {
	return func() <-chan struct{} {
		ch := make(chan struct{})
		go func() {
			for i := 0; i < count; i++ {
				ch <- struct{}{}
			}
			close(ch)
		}()
		return ch
	}
}

func TestPoll(t *testing.T) {
	invocations := 0
	f := ConditionFunc(func() (bool, error) {
		invocations++
		return true, nil
	})
	if err := Poll(time.Microsecond, time.Microsecond, f); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if invocations == 0 {
		t.Errorf("Expected at least one invocation, got zero")
	}
	expectedError := errors.New("Expected error")
	f = ConditionFunc(func() (bool, error) {
		return false, expectedError
	})
	if err := Poll(time.Microsecond, time.Microsecond, f); err == nil || err != expectedError {
		t.Fatalf("Expected error %v, got none %v", expectedError, err)
	}
}

func TestPollForever(t *testing.T) {
	ch := make(chan struct{})
	done := make(chan struct{}, 1)
	complete := make(chan struct{})
	go func() {
		f := ConditionFunc(func() (bool, error) {
			ch <- struct{}{}
			select {
			case <-done:
				return true, nil
			default:
			}
			return false, nil
		})
		if err := Poll(time.Microsecond, 0, f); err != nil {
			t.Fatalf("unexpected error %v", err)
		}
		close(ch)
		complete <- struct{}{}
	}()

	// ensure the condition is opened
	<-ch

	// ensure channel sends events
	for i := 0; i < 10; i++ {
		select {
		case _, open := <-ch:
			if !open {
				t.Fatalf("did not expect channel to be closed")
			}
		case <-time.After(time.Second):
			t.Fatalf("channel did not return at least once within the poll interval")
		}
	}

	// at most two poll notifications should be sent once we return from the condition
	done <- struct{}{}
	go func() {
		for i := 0; i < 2; i++ {
			_, open := <-ch
			if open {
				<-complete
				return
			}
		}
		t.Fatalf("expected closed channel after two iterations")
	}()
	<-complete
}

type WaitForTestCase struct {
	F       ConditionFunc
	Ticks   int
	Invoked int
	Err     bool // set to false if you set the timeout
	Timeout bool
}

func runWaitForTestCase(t *testing.T, testCase WaitForTestCase, description string) {
	invocations := 0
	ticker := fakeTicker(testCase.Ticks)
	err := WaitFor(ticker, func() (bool, error) {
		invocations++
		return testCase.F()
	})
	switch {
	case testCase.Err:
		if err == nil {
			t.Errorf("WaitFor: %s: Expected error, got nil", description)
			return
		}
	case testCase.Timeout:
		if err != ErrWaitTimeout {
			t.Errorf("WaitFor: %s: Expected timeout error, got %#v", description, err)
			return
		}
	case true:
		if err != nil {
			t.Errorf("WaitFor: %s: Expected no error, got: %#v", description, err)
			return
		}
	}
	if invocations != testCase.Invoked {
		t.Errorf("%s: Expected %d invocations, called %d", description, testCase.Invoked, invocations)
	}
}

func runWaitForMsgTestCase(t *testing.T, testCase WaitForTestCase, description string) {
	invocations := 0
	ticker := fakeTicker(testCase.Ticks)
	err := WaitForMsg(description, ticker, func() (bool, error) {
		invocations++
		return testCase.F()
	})
	switch {
	case testCase.Err:
		if err == nil {
			t.Errorf("WaitForMsg: %s: Expected error, got nil", description)
			return
		}
	case testCase.Timeout:
		if err == nil || !strings.Contains(err.Error(), description) {
			t.Errorf("WaitForMsg: %s: Expected timeout error to contains '%s', got %#v", description, description, err)
			return
		}
	case true:
		if err != nil {
			t.Errorf("WaitForMsg: %s: Expected no error, got: %#v", description, err)
			return
		}
	}
	if invocations != testCase.Invoked {
		t.Errorf("WaitForMsg: %s: Expected %d invocations, called %d", description, testCase.Invoked, invocations)
	}
	if invocations != testCase.Invoked {
		t.Errorf("WaitForMsg: %s: Expected %d invocations, called %d", description, testCase.Invoked, invocations)
	}
}

func TestWaitFor(t *testing.T) {
	testCases := map[string]WaitForTestCase{
		"invoked once": {
			ConditionFunc(func() (bool, error) {
				return true, nil
			}),
			2,
			1,
			false,
			false,
		},
		"invoked and returns a timeout": {
			ConditionFunc(func() (bool, error) {
				return false, nil
			}),
			2,
			3,
			false,
			true,
		},
		"returns immediately on error": {
			ConditionFunc(func() (bool, error) {
				return false, errors.New("test")
			}),
			2,
			1,
			true,
			false,
		},
	}
	for k, c := range testCases {
		runWaitForTestCase(t, c, k)
		runWaitForMsgTestCase(t, c, k)
	}
}
