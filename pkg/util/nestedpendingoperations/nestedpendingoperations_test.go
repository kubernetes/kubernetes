/*
Copyright 2016 The Kubernetes Authors.

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

package nestedpendingoperations

import (
	"fmt"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/nestedpendingoperations/exponentialbackoff"
	"k8s.io/kubernetes/pkg/volume/util/types"
)

const (
	// testTimeout is a timeout of goroutines to finish. This _should_ be just a
	// "context switch" and it should take several ms, however, Clayton says "We
	// have had flakes due to tests that assumed that 15s is long enough to sleep")
	testTimeout time.Duration = 1 * time.Minute

	// initialOperationWaitTimeShort is the initial amount of time the test will
	// wait for an operation to complete (each successive failure results in
	// exponential backoff).
	initialOperationWaitTimeShort time.Duration = 20 * time.Millisecond

	// initialOperationWaitTimeLong is the initial amount of time the test will
	// wait for an operation to complete (each successive failure results in
	// exponential backoff).
	initialOperationWaitTimeLong time.Duration = 500 * time.Millisecond
)

func Test_NewNestedPendingOperations_Positive_SingleOp(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	operationName := "operation-name"
	operation := func() (error, error) { return nil, nil }

	// Act
	err := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation})

	// Assert
	if err != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err)
	}
}

func Test_NewNestedPendingOperations_Positive_TwoOps(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	operation1Name := "operation1-name"
	operation2Name := "operation2-name"
	operation := func() (error, error) { return nil, nil }

	// Act
	err1 := grm.Run(operation1Name, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation})
	err2 := grm.Run(operation2Name, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation})

	// Assert
	if err1 != nil {
		t.Fatalf("NewGoRoutine %q failed. Expected: <no error> Actual: <%v>", operation1Name, err1)
	}

	if err2 != nil {
		t.Fatalf("NewGoRoutine %q failed. Expected: <no error> Actual: <%v>", operation2Name, err2)
	}
}

func Test_NewNestedPendingOperations_Positive_TwoSubOps(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	volumeName := fmt.Sprintf("%v", v1.UniqueVolumeName("volume-name"))
	operation1PodName := fmt.Sprintf("%v", types.UniquePodName("operation1-podname"))
	operation2PodName := fmt.Sprintf("%v", types.UniquePodName("operation2-podname"))
	operation := func() (error, error) { return nil, nil }

	// Act
	err1 := grm.Run(volumeName, operation1PodName, GeneratedOperations{OperationFunc: operation})
	err2 := grm.Run(volumeName, operation2PodName, GeneratedOperations{OperationFunc: operation})

	// Assert
	if err1 != nil {
		t.Fatalf("NewGoRoutine %q failed. Expected: <no error> Actual: <%v>", operation1PodName, err1)
	}

	if err2 != nil {
		t.Fatalf("NewGoRoutine %q failed. Expected: <no error> Actual: <%v>", operation2PodName, err2)
	}
}

func Test_NewNestedPendingOperations_Positive_SingleOpWithExpBackoff(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	operationName := "operation-name"
	operation := func() (error, error) { return nil, nil }

	// Act
	err := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation})

	// Assert
	if err != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err)
	}
}

func Test_NewNestedPendingOperations_Positive_SecondOpAfterFirstCompletes(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	operationName := "operation-name"
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateCallbackFunc(operation1DoneCh)
	err1 := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()
	<-operation1DoneCh // Force operation1 to complete

	// Act
	err2 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeShort),
		func() (bool, error) {
			err := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation2})
			if err != nil {
				t.Logf("Warning: NewGoRoutine failed with %v. Will retry.", err)
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

func Test_NewNestedPendingOperations_Positive_SecondOpAfterFirstCompletesWithExpBackoff(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	operationName := "operation-name"
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateCallbackFunc(operation1DoneCh)
	err1 := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()
	<-operation1DoneCh // Force operation1 to complete

	// Act
	err2 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeShort),
		func() (bool, error) {
			err := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation2})
			if err != nil {
				t.Logf("Warning: NewGoRoutine failed with %v. Will retry.", err)
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

func Test_NewNestedPendingOperations_Positive_SecondOpAfterFirstPanics(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	operationName := "operation-name"
	operation1 := generatePanicFunc()
	err1 := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeShort),
		func() (bool, error) {
			err := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation2})
			if err != nil {
				t.Logf("Warning: NewGoRoutine failed with %v. Will retry.", err)
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

func Test_NewNestedPendingOperations_Positive_SecondOpAfterFirstPanicsWithExpBackoff(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	operationName := "operation-name"
	operation1 := generatePanicFunc()
	err1 := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeLong), // Longer duration to accommodate for backoff
		func() (bool, error) {
			err := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation2})
			if err != nil {
				t.Logf("Warning: NewGoRoutine failed with %v. Will retry.", err)
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

func Test_NewNestedPendingOperations_Negative_SecondOpBeforeFirstCompletes(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	operationName := "operation-name"
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 == nil {
		t.Fatalf("NewGoRoutine did not fail. Expected: <Failed to create operation with name \"%s\". An operation with that name already exists.> Actual: <no error>", operationName)
	}
	if !IsAlreadyExists(err2) {
		t.Fatalf("NewGoRoutine did not return alreadyExistsError, got: %v", err2)
	}
}

func Test_NewNestedPendingOperations_Negative_SecondThirdOpWithDifferentNames(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	volumeName := fmt.Sprintf("%v", v1.UniqueVolumeName("volume-name"))
	op1Name := "mount_volume"
	operation1 := generateErrorFunc()
	err1 := grm.Run(volumeName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation1, OperationName: op1Name})
	if err1 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err1)
	}
	// Shorter than exponential backoff period, so as to trigger exponential backoff error on second
	// operation.
	operation2 := generateErrorFunc()
	err2 := retryWithExponentialBackOff(
		initialOperationWaitTimeShort,
		func() (bool, error) {
			err := grm.Run(volumeName,
				"", /* operationSubName */
				GeneratedOperations{OperationFunc: operation2, OperationName: op1Name})

			if exponentialbackoff.IsExponentialBackoff(err) {
				return true, nil
			}
			return false, nil
		},
	)

	// Assert
	if err2 != nil {
		t.Fatalf("Expected NewGoRoutine to fail with exponential backoff for operationKey : %s and operationName : %s", volumeName, op1Name)
	}

	operation3 := generateNoopFunc()
	op3Name := "unmount_volume"
	// Act
	err3 := grm.Run(volumeName, "" /*operationKey2*/, GeneratedOperations{OperationFunc: operation3, OperationName: op3Name})
	if err3 != nil {
		t.Fatalf("NewGoRoutine failed. Expected <no error> Actual: <%v>", err3)
	}
}

func Test_NewNestedPendingOperations_Negative_SecondSubOpBeforeFirstCompletes2(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	volumeName := fmt.Sprintf("%v", v1.UniqueVolumeName("volume-name"))
	operationPodName := fmt.Sprintf("%v", types.UniquePodName("operation-podname"))
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, operationPodName, GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := grm.Run(volumeName, operationPodName, GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 == nil {
		t.Fatalf("NewGoRoutine did not fail. Expected: <Failed to create operation with name \"%s\". An operation with that name already exists.> Actual: <no error>", volumeName)
	}
	if !IsAlreadyExists(err2) {
		t.Fatalf("NewGoRoutine did not return alreadyExistsError, got: %v", err2)
	}
}

func Test_NewNestedPendingOperations_Negative_SecondSubOpBeforeFirstCompletes(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	volumeName := fmt.Sprintf("%v", v1.UniqueVolumeName("volume-name"))
	operationPodName := fmt.Sprintf("%v", types.UniquePodName("operation-podname"))
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, operationPodName, GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := grm.Run(volumeName, operationPodName, GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 == nil {
		t.Fatalf("NewGoRoutine did not fail. Expected: <Failed to create operation with name \"%s\". An operation with that name already exists.> Actual: <no error>", volumeName)
	}
	if !IsAlreadyExists(err2) {
		t.Fatalf("NewGoRoutine did not return alreadyExistsError, got: %v", err2)
	}
}

func Test_NewNestedPendingOperations_Negative_SecondOpBeforeFirstCompletesWithExpBackoff(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	operationName := "operation-name"
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 == nil {
		t.Fatalf("NewGoRoutine did not fail. Expected: <Failed to create operation with name \"%s\". An operation with that name already exists.> Actual: <no error>", operationName)
	}
	if !IsAlreadyExists(err2) {
		t.Fatalf("NewGoRoutine did not return alreadyExistsError, got: %v", err2)
	}
}

func Test_NewNestedPendingOperations_Positive_ThirdOpAfterFirstCompletes(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	operationName := "operation-name"
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()
	operation3 := generateNoopFunc()

	// Act
	err2 := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 == nil {
		t.Fatalf("NewGoRoutine did not fail. Expected: <Failed to create operation with name \"%s\". An operation with that name already exists.> Actual: <no error>", operationName)
	}
	if !IsAlreadyExists(err2) {
		t.Fatalf("NewGoRoutine did not return alreadyExistsError, got: %v", err2)
	}

	// Act
	operation1DoneCh <- true // Force operation1 to complete
	err3 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeShort),
		func() (bool, error) {
			err := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation3})
			if err != nil {
				t.Logf("Warning: NewGoRoutine failed with %v. Will retry.", err)
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

func Test_NewNestedPendingOperations_Positive_ThirdOpAfterFirstCompletesWithExpBackoff(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	operationName := "operation-name"
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()
	operation3 := generateNoopFunc()

	// Act
	err2 := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 == nil {
		t.Fatalf("NewGoRoutine did not fail. Expected: <Failed to create operation with name \"%s\". An operation with that name already exists.> Actual: <no error>", operationName)
	}
	if !IsAlreadyExists(err2) {
		t.Fatalf("NewGoRoutine did not return alreadyExistsError, got: %v", err2)
	}

	// Act
	operation1DoneCh <- true // Force operation1 to complete
	err3 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeShort),
		func() (bool, error) {
			err := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation3})
			if err != nil {
				t.Logf("Warning: NewGoRoutine failed with %v. Will retry.", err)
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

func Test_NewNestedPendingOperations_Positive_WaitEmpty(t *testing.T) {
	// Test than Wait() on empty NestedPendingOperations always succeeds without blocking
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)

	// Act
	waitDoneCh := make(chan interface{}, 1)
	go func() {
		grm.Wait()
		waitDoneCh <- true
	}()

	// Assert
	err := waitChannelWithTimeout(waitDoneCh, testTimeout)
	if err != nil {
		t.Errorf("Error waiting for NestedPendingOperations.Wait: %v", err)
	}
}

func Test_NewNestedPendingOperations_Positive_WaitEmptyWithExpBackoff(t *testing.T) {
	// Test than Wait() on empty NestedPendingOperations always succeeds without blocking
	// Arrange
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)

	// Act
	waitDoneCh := make(chan interface{}, 1)
	go func() {
		grm.Wait()
		waitDoneCh <- true
	}()

	// Assert
	err := waitChannelWithTimeout(waitDoneCh, testTimeout)
	if err != nil {
		t.Errorf("Error waiting for NestedPendingOperations.Wait: %v", err)
	}
}

func Test_NewNestedPendingOperations_Positive_Wait(t *testing.T) {
	// Test that Wait() really blocks until the last operation succeeds
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	operationName := "operation-name"
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation1})
	if err != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	waitDoneCh := make(chan interface{}, 1)
	go func() {
		grm.Wait()
		waitDoneCh <- true
	}()

	// Finish the operation
	operation1DoneCh <- true

	// Assert
	err = waitChannelWithTimeout(waitDoneCh, testTimeout)
	if err != nil {
		t.Fatalf("Error waiting for NestedPendingOperations.Wait: %v", err)
	}
}

func Test_NewNestedPendingOperations_Positive_WaitWithExpBackoff(t *testing.T) {
	// Test that Wait() really blocks until the last operation succeeds
	// Arrange
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	operationName := "operation-name"
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation1})
	if err != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	waitDoneCh := make(chan interface{}, 1)
	go func() {
		grm.Wait()
		waitDoneCh <- true
	}()

	// Finish the operation
	operation1DoneCh <- true

	// Assert
	err = waitChannelWithTimeout(waitDoneCh, testTimeout)
	if err != nil {
		t.Fatalf("Error waiting for NestedPendingOperations.Wait: %v", err)
	}
}

func Test_NewNestedPendingOperations_WaitForCompletionWithExpBackoff(t *testing.T) {
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	operationName := "operation-err"

	operation1 := generateErrorFunc()
	err := grm.Run(operationName, "" /* operationKey2 */, GeneratedOperations{OperationFunc: operation1})
	if err != nil {
		t.Fatalf("NewGoRoutine failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	waitDoneCh := make(chan interface{}, 1)
	go func() {
		grm.WaitForCompletion()
		waitDoneCh <- true
	}()

	// Assert that WaitForCompletion returns even if scheduled op had error
	err = waitChannelWithTimeout(waitDoneCh, testTimeout)
	if err != nil {
		t.Fatalf("Error waiting for NestedPendingOperations.Wait: %v", err)
	}
}

func generateCallbackFunc(done chan<- interface{}) func() (error, error) {
	return func() (error, error) {
		done <- true
		return nil, nil
	}
}

func generateWaitFunc(done <-chan interface{}) func() (error, error) {
	return func() (error, error) {
		<-done
		return nil, nil
	}
}

func generatePanicFunc() func() (error, error) {
	return func() (error, error) {
		panic("testing panic")
	}
}

func generateErrorFunc() func() (error, error) {
	return func() (error, error) {
		return fmt.Errorf("placholder1"), fmt.Errorf("placeholder2")
	}
}

func generateNoopFunc() func() (error, error) {
	return func() (error, error) { return nil, nil }
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

func waitChannelWithTimeout(ch <-chan interface{}, timeout time.Duration) error {
	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case <-ch:
		// Success!
		return nil
	case <-timer.C:
		return fmt.Errorf("timeout after %v", timeout)
	}
}
