/*
Copyright 2023 The Kubernetes Authors.

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

package executor

import (
	"fmt"
	"sync/atomic"
	"testing"
)

func TestExecutor(t *testing.T) {
	t.Run("single worker, no panic", func(t *testing.T) {
		var invoked int32 = 0
		ch := Single().Execute(func() {
			atomic.AddInt32(&invoked, 1)
		})

		recovered, ok := <-ch
		if ok {
			t.Errorf("expected the channel to have no values")
		}
		if recovered != nil {
			t.Errorf("expected no recovered panic, but got: %v", recovered)
		}
		if invokedGot := atomic.LoadInt32(&invoked); invokedGot != 1 {
			t.Errorf("expected the function to be invoked once, but got: %d", invokedGot)
		}
	})

	t.Run("multiple workers, no panic", func(t *testing.T) {
		workers, invoked := int32(10), int32(0)
		ch := Pool(workers).Execute(func() {
			atomic.AddInt32(&invoked, 1)
		})

		recovered, ok := <-ch
		if ok {
			t.Errorf("expected the channel to have no values")
		}
		if recovered != nil {
			t.Errorf("expected no recovered panic, but got: %v", recovered)
			return
		}
		if invokedGot := atomic.LoadInt32(&invoked); invokedGot != workers {
			t.Errorf("expected the function to be invoked: %d times, but got: %d", workers, invokedGot)
		}
	})

	t.Run("single worker, panic should not crash the application", func(t *testing.T) {
		var recoveredErrExpected interface{} = fmt.Errorf("unexpected error")
		var unhandledPanicErrGot interface{}
		var ch <-chan RecoveredPanic
		func() {
			defer func() {
				if err := recover(); err != nil {
					unhandledPanicErrGot = err
				}
			}()
			ch = Single().Execute(func() {
				panic(recoveredErrExpected)
			})
		}()

		if unhandledPanicErrGot != nil {
			t.Errorf("expected panic to be handled: %v", unhandledPanicErrGot)
		}

		recovered, ok := <-ch
		if !ok {
			t.Errorf("expected the channel to have a RecoveredPanic value")
		}
		if recovered == nil {
			t.Errorf("expected a non nil RecoveredPanic object")
			return
		}
		// the channel should not have any more values
		if _, ok := <-ch; ok {
			t.Errorf("expected the channel to be closed")
		}

		recoveredPanicGot, ok := recovered.(ReasonAndStackTrace)
		if !ok {
			t.Errorf("expected a RecoveredPanic of type: %T", ReasonAndStackTrace{})
			return
		}
		if recoveredPanicGot.Reason != recoveredErrExpected {
			t.Errorf("expected panick error to be: %v, but got: %v", recoveredErrExpected, recoveredPanicGot.Reason)
		}
		if len(recoveredPanicGot.StackTrace) == 0 {
			t.Errorf("expected StackTrace to be non empty")
		}
	})

	t.Run("multiple workers, some panic", func(t *testing.T) {
		counter := int32(10)
		var unhandledPanicErrGot interface{}
		var ch <-chan RecoveredPanic
		func() {
			defer func() {
				if err := recover(); err != nil {
					unhandledPanicErrGot = err
				}
			}()
			ch = Pool(counter).Execute(func() {
				current := atomic.AddInt32(&counter, -1)
				if current%2 == 0 {
					panic(fmt.Errorf("worker(%d): unexpected panic", current))
				}
			})
		}()

		if unhandledPanicErrGot != nil {
			t.Errorf("expected panic to be handled: %v", unhandledPanicErrGot)
		}

		recovered, ok := <-ch
		if !ok {
			t.Errorf("expected the channel to have a RecoveredPanic value")
		}
		if recovered == nil {
			t.Errorf("expected a non nil RecoveredPanic value")
			return
		}
		// the channel should not have any more values
		if _, ok := <-ch; ok {
			t.Errorf("expected the channel to be closed")
		}

		allRecoveredGot, ok := recovered.(RecoveredPanics)
		if !ok {
			t.Errorf("expected an object of type: %T", RecoveredPanics{})
			return
		}
		if len(allRecoveredGot) != 5 {
			t.Errorf("expected 5 recovered panics, but got: %d", len(allRecoveredGot))
		}
	})

	t.Run("multiple workers, all panic", func(t *testing.T) {
		var recoveredErrExpected interface{} = fmt.Errorf("unexpected error")
		workers := int32(3)
		var unhandledPanicErrGot interface{}
		var ch <-chan RecoveredPanic
		func() {
			defer func() {
				if err := recover(); err != nil {
					unhandledPanicErrGot = err
				}
			}()
			ch = Pool(workers).Execute(func() {
				panic(recoveredErrExpected)
			})
		}()

		if unhandledPanicErrGot != nil {
			t.Errorf("expected panic to be handled: %v", unhandledPanicErrGot)
		}

		recovered, ok := <-ch
		if !ok {
			t.Errorf("expected the channel to have a RecoveredPanic value")
		}
		if recovered == nil {
			t.Errorf("expected a non nil RecoveredPanic object")
			return
		}
		// the channel should not have any more values
		if _, ok := <-ch; ok {
			t.Errorf("expected the channel to be closed")
		}

		allRecoveredGot, ok := recovered.(RecoveredPanics)
		if !ok {
			t.Errorf("expected an object of type: %T", RecoveredPanics{})
			return
		}
		if len(allRecoveredGot) != int(workers) {
			t.Errorf("expected %d recovered panics, but got: %d", workers, len(allRecoveredGot))
		}

		for _, got := range allRecoveredGot {
			recoveredGot, ok := got.(ReasonAndStackTrace)
			if !ok {
				t.Errorf("expected an object of type: %T", ReasonAndStackTrace{})
				continue
			}
			if len(recoveredGot.StackTrace) == 0 {
				t.Errorf("expected captured stack trace to be non empty: %v", recoveredGot)
			}
			if recoveredGot.Reason != recoveredErrExpected {
				t.Errorf("expected recover error: %v, but got: %v", recoveredErrExpected, recoveredGot.Reason)
			}
		}
	})

	t.Run("multiple workers, one panics", func(t *testing.T) {
		var recoveredErrExpected interface{} = fmt.Errorf("unexpected error")
		workers, counter := int32(3), int32(3)
		var unhandledPanicErrGot interface{}
		var ch <-chan RecoveredPanic
		func() {
			defer func() {
				if err := recover(); err != nil {
					unhandledPanicErrGot = err
				}
			}()
			ch = Pool(workers).Execute(func() {
				if current := atomic.AddInt32(&counter, -1); current == 2 {
					panic(recoveredErrExpected)
				}
			})
		}()

		if unhandledPanicErrGot != nil {
			t.Errorf("expected panic to be handled: %v", unhandledPanicErrGot)
		}

		recovered, ok := <-ch
		if !ok {
			t.Errorf("expected the channel to have a RecoveredPanic value")
		}
		if recovered == nil {
			t.Errorf("expected a non nil RecoveredPanic object")
			return
		}
		// the channel should not have any more values
		if _, ok := <-ch; ok {
			t.Errorf("expected the channel to be closed")
		}

		recoveredGot, ok := recovered.(ReasonAndStackTrace)
		if !ok {
			t.Errorf("expected an object of type: %T", ReasonAndStackTrace{})
			return
		}
		if len(recoveredGot.StackTrace) == 0 {
			t.Errorf("expected captured stack trace to be non empty: %v", recoveredGot)
		}
		if recoveredGot.Reason != recoveredErrExpected {
			t.Errorf("expected recover error: %v, but got: %v", recoveredErrExpected, recoveredGot.Reason)
		}
	})
}
