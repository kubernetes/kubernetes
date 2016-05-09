/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package goroutinemap

import (
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/util/wait"
)

func Test_NewGoRoutineMap_Positive_SingleOp(t *testing.T) {
	// Arrange
	grm := NewGoRoutineMap()
	operationName := "operation-name"
	operation := func() error { return nil }

	// Act
	err := grm.NewGoRoutine(operationName, operation)

	// Assert
	if err != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err)
	}
}

func Test_NewGoRoutineMap_Positive_SecondOpAfterFirstCompletes(t *testing.T) {
	// Arrange
	grm := NewGoRoutineMap()
	operationName := "operation-name"
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateCallbackFunc(operation1DoneCh)
	err1 := grm.NewGoRoutine(operationName, operation1)
	if err1 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()
	<-operation1DoneCh // Force operation1 to complete

	// Act
	err2 := retryWithExponentialBackOff(
		time.Duration(20*time.Millisecond),
		func() (bool, error) {
			err := grm.NewGoRoutine(operationName, operation2)
			if err != nil {
				t.Logf("Warning: NewGoRoutine failed. Expected: <no error> Actual: <%v>. Will retry.", err)
				return false, nil
			}
			return true, nil
		},
	)

	// Assert
	if err2 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err2)
	}
}

func Test_NewGoRoutineMap_Positive_SecondOpAfterFirstPanics(t *testing.T) {
	// Arrange
	grm := NewGoRoutineMap()
	operationName := "operation-name"
	operation1 := generatePanicFunc()
	err1 := grm.NewGoRoutine(operationName, operation1)
	if err1 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := retryWithExponentialBackOff(
		time.Duration(20*time.Millisecond),
		func() (bool, error) {
			err := grm.NewGoRoutine(operationName, operation2)
			if err != nil {
				t.Logf("Warning: NewGoRoutine failed. Expected: <no error> Actual: <%v>. Will retry.", err)
				return false, nil
			}
			return true, nil
		},
	)

	// Assert
	if err2 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err2)
	}
}

func Test_NewGoRoutineMap_Negative_SecondOpBeforeFirstCompletes(t *testing.T) {
	// Arrange
	grm := NewGoRoutineMap()
	operationName := "operation-name"
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.NewGoRoutine(operationName, operation1)
	if err1 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := grm.NewGoRoutine(operationName, operation2)

	// Assert
	if err2 == nil {
		t.Fatalf("NewGoRoutine did not fail. Expected: <Failed to create operation with name \"%s\". An operation with that name already exists.> Actual: <no error>", operationName)
	}
}

func Test_NewGoRoutineMap_Positive_ThirdOpAfterFirstCompletes(t *testing.T) {
	// Arrange
	grm := NewGoRoutineMap()
	operationName := "operation-name"
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.NewGoRoutine(operationName, operation1)
	if err1 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()
	operation3 := generateNoopFunc()

	// Act
	err2 := grm.NewGoRoutine(operationName, operation2)

	// Assert
	if err2 == nil {
		t.Fatalf("NewGoRoutine did not fail. Expected: <Failed to create operation with name \"%s\". An operation with that name already exists.> Actual: <no error>", operationName)
	}

	// Act
	operation1DoneCh <- true // Force operation1 to complete
	err3 := retryWithExponentialBackOff(
		time.Duration(20*time.Millisecond),
		func() (bool, error) {
			err := grm.NewGoRoutine(operationName, operation3)
			if err != nil {
				t.Logf("Warning: NewGoRoutine failed. Expected: <no error> Actual: <%v>. Will retry.", err)
				return false, nil
			}
			return true, nil
		},
	)

	// Assert
	if err3 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err3)
	}
}

func generateCallbackFunc(done chan<- interface{}) func() error {
	return func() error {
		done <- true
		return nil
	}
}

func generateWaitFunc(done <-chan interface{}) func() error {
	return func() error {
		<-done
		return nil
	}
}

func generatePanicFunc() func() error {
	return func() error {
		panic("testing panic")
	}
}

func generateNoopFunc() func() error {
	return func() error { return nil }
}

func retryWithExponentialBackOff(initialDuration time.Duration, fn wait.ConditionFunc) error {
	backoff := wait.Backoff{
		Duration: initialDuration,
		Factor:   3,
		Jitter:   0,
		Steps:    4,
	}
	return wait.ExponentialBackoff(backoff, fn)
}
