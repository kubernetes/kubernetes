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
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
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

func Test_NestedPendingOperations_Positive_SingleOp(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operation := func() (error, error) { return nil, nil }

	// Act
	err := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation})

	// Assert
	if err != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err)
	}
}

func Test_NestedPendingOperations_Positive_TwoOps(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	volume1Name := v1.UniqueVolumeName("volume1-name")
	volume2Name := v1.UniqueVolumeName("volume2-name")
	operation := func() (error, error) { return nil, nil }

	// Act
	err1 := grm.Run(volume1Name, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation})
	err2 := grm.Run(volume2Name, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation})

	// Assert
	if err1 != nil {
		t.Fatalf("NestedPendingOperations %q failed. Expected: <no error> Actual: <%v>", volume1Name, err1)
	}

	if err2 != nil {
		t.Fatalf("NestedPendingOperations %q failed. Expected: <no error> Actual: <%v>", volume2Name, err2)
	}
}

func Test_NestedPendingOperations_Positive_TwoSubOps(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operation1PodName := volumetypes.UniquePodName("operation1-podname")
	operation2PodName := volumetypes.UniquePodName("operation2-podname")
	operation := func() (error, error) { return nil, nil }

	// Act
	err1 := grm.Run(volumeName, operation1PodName, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation})
	err2 := grm.Run(volumeName, operation2PodName, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation})

	// Assert
	if err1 != nil {
		t.Fatalf("NestedPendingOperations %q failed. Expected: <no error> Actual: <%v>", operation1PodName, err1)
	}

	if err2 != nil {
		t.Fatalf("NestedPendingOperations %q failed. Expected: <no error> Actual: <%v>", operation2PodName, err2)
	}
}

func Test_NestedPendingOperations_Positive_SingleOpWithExpBackoff(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operation := func() (error, error) { return nil, nil }

	// Act
	err := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation})

	// Assert
	if err != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err)
	}
}

func Test_NestedPendingOperations_Positive_SecondOpAfterFirstCompletes(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateCallbackFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()
	<-operation1DoneCh // Force operation1 to complete

	// Act
	err2 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeShort),
		func() (bool, error) {
			err := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation2})
			if err != nil {
				t.Logf("Warning: NestedPendingOperations failed with %v. Will retry.", err)
				return false, nil
			}
			return true, nil
		},
	)

	// Assert
	if err2 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err2)
	}
}

func Test_NestedPendingOperations_Positive_SecondOpAfterFirstCompletesWithExpBackoff(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateCallbackFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()
	<-operation1DoneCh // Force operation1 to complete

	// Act
	err2 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeShort),
		func() (bool, error) {
			err := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation2})
			if err != nil {
				t.Logf("Warning: NestedPendingOperations failed with %v. Will retry.", err)
				return false, nil
			}
			return true, nil
		},
	)

	// Assert
	if err2 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err2)
	}
}

func Test_NestedPendingOperations_Positive_SecondOpAfterFirstPanics(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operation1 := generatePanicFunc()
	err1 := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeShort),
		func() (bool, error) {
			err := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation2})
			if err != nil {
				t.Logf("Warning: NestedPendingOperations failed with %v. Will retry.", err)
				return false, nil
			}
			return true, nil
		},
	)

	// Assert
	if err2 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err2)
	}
}

func Test_NestedPendingOperations_Positive_SecondOpAfterFirstPanicsWithExpBackoff(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operation1 := generatePanicFunc()
	err1 := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeLong), // Longer duration to accommodate for backoff
		func() (bool, error) {
			err := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation2})
			if err != nil {
				t.Logf("Warning: NestedPendingOperations failed with %v. Will retry.", err)
				return false, nil
			}
			return true, nil
		},
	)

	// Assert
	if err2 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err2)
	}
}

func Test_NestedPendingOperations_Negative_SecondOpBeforeFirstCompletes(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 == nil {
		t.Fatalf("NestedPendingOperations did not fail. Expected: <Failed to create operation with name \"%s\". An operation with that name already exists.> Actual: <no error>", volumeName)
	}
	if !IsAlreadyExists(err2) {
		t.Fatalf("NestedPendingOperations did not return alreadyExistsError, got: %v", err2)
	}
}

func Test_NestedPendingOperations_Negative_SecondThirdOpWithDifferentNames(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	op1Name := "mount_volume"
	operation1 := generateErrorFunc()
	err1 := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1, OperationName: op1Name})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	// Shorter than exponential backoff period, so as to trigger exponential backoff error on second
	// operation.
	operation2 := generateErrorFunc()
	err2 := retryWithExponentialBackOff(
		initialOperationWaitTimeShort,
		func() (bool, error) {
			err := grm.Run(volumeName,
				"", /* operationSubName */
				"", /* nodeName */
				volumetypes.GeneratedOperations{OperationFunc: operation2, OperationName: op1Name})

			if exponentialbackoff.IsExponentialBackoff(err) {
				return true, nil
			}
			return false, nil
		},
	)

	// Assert
	if err2 != nil {
		t.Fatalf("Expected NestedPendingOperations to fail with exponential backoff for operationKey : %s and operationName : %s", volumeName, op1Name)
	}

	operation3 := generateNoopFunc()
	op3Name := "unmount_volume"
	// Act
	err3 := grm.Run(volumeName, "" /*pod name*/, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation3, OperationName: op3Name})
	if err3 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected <no error> Actual: <%v>", err3)
	}
}

func Test_NestedPendingOperations_Negative_SecondSubOpBeforeFirstCompletes2(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operationPodName := volumetypes.UniquePodName("operation-podname")
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, operationPodName, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := grm.Run(volumeName, operationPodName, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 == nil {
		t.Fatalf("NestedPendingOperations did not fail. Expected: <Failed to create operation with name \"%s\". An operation with that name already exists.> Actual: <no error>", volumeName)
	}
	if !IsAlreadyExists(err2) {
		t.Fatalf("NestedPendingOperations did not return alreadyExistsError, got: %v", err2)
	}
}

func Test_NestedPendingOperations_Negative_SecondSubOpBeforeFirstCompletes(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operationPodName := volumetypes.UniquePodName("operation-podname")
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, operationPodName, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := grm.Run(volumeName, operationPodName, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 == nil {
		t.Fatalf("NestedPendingOperations did not fail. Expected: <Failed to create operation with name \"%s\". An operation with that name already exists.> Actual: <no error>", volumeName)
	}
	if !IsAlreadyExists(err2) {
		t.Fatalf("NestedPendingOperations did not return alreadyExistsError, got: %v", err2)
	}
}

func Test_NestedPendingOperations_Negative_SecondOpBeforeFirstCompletesWithExpBackoff(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 == nil {
		t.Fatalf("NestedPendingOperations did not fail. Expected: <Failed to create operation with name \"%s\". An operation with that name already exists.> Actual: <no error>", volumeName)
	}
	if !IsAlreadyExists(err2) {
		t.Fatalf("NestedPendingOperations did not return alreadyExistsError, got: %v", err2)
	}
}

func Test_NestedPendingOperations_Positive_ThirdOpAfterFirstCompletes(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()
	operation3 := generateNoopFunc()

	// Act
	err2 := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 == nil {
		t.Fatalf("NestedPendingOperations did not fail. Expected: <Failed to create operation with name \"%s\". An operation with that name already exists.> Actual: <no error>", volumeName)
	}
	if !IsAlreadyExists(err2) {
		t.Fatalf("NestedPendingOperations did not return alreadyExistsError, got: %v", err2)
	}

	// Act
	operation1DoneCh <- true // Force operation1 to complete
	err3 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeShort),
		func() (bool, error) {
			err := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation3})
			if err != nil {
				t.Logf("Warning: NestedPendingOperations failed with %v. Will retry.", err)
				return false, nil
			}
			return true, nil
		},
	)

	// Assert
	if err3 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err3)
	}
}

func Test_NestedPendingOperations_Positive_ThirdOpAfterFirstCompletesWithExpBackoff(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()
	operation3 := generateNoopFunc()

	// Act
	err2 := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 == nil {
		t.Fatalf("NestedPendingOperations did not fail. Expected: <Failed to create operation with name \"%s\". An operation with that name already exists.> Actual: <no error>", volumeName)
	}
	if !IsAlreadyExists(err2) {
		t.Fatalf("NestedPendingOperations did not return alreadyExistsError, got: %v", err2)
	}

	// Act
	operation1DoneCh <- true // Force operation1 to complete
	err3 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeShort),
		func() (bool, error) {
			err := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation3})
			if err != nil {
				t.Logf("Warning: NestedPendingOperations failed with %v. Will retry.", err)
				return false, nil
			}
			return true, nil
		},
	)

	// Assert
	if err3 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err3)
	}
}

func Test_NestedPendingOperations_Positive_WaitEmpty(t *testing.T) {
	// Test than Wait() on empty GoRoutineMap always succeeds without blocking
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
		t.Errorf("Error waiting for GoRoutineMap.Wait: %v", err)
	}
}

func Test_NestedPendingOperations_Positive_WaitEmptyWithExpBackoff(t *testing.T) {
	// Test than Wait() on empty GoRoutineMap always succeeds without blocking
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
		t.Errorf("Error waiting for GoRoutineMap.Wait: %v", err)
	}
}

func Test_NestedPendingOperations_Positive_Wait(t *testing.T) {
	// Test that Wait() really blocks until the last operation succeeds
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err)
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
		t.Fatalf("Error waiting for GoRoutineMap.Wait: %v", err)
	}
}

func Test_NestedPendingOperations_Positive_WaitWithExpBackoff(t *testing.T) {
	// Test that Wait() really blocks until the last operation succeeds
	// Arrange
	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err := grm.Run(volumeName, "" /* operationSubName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err)
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
		t.Fatalf("Error waiting for GoRoutineMap.Wait: %v", err)
	}
}

/* Concurrent operations tests */

// Covered cases:
// FIRST OP    | SECOND OP   | RESULT
// None        | None        | Positive
// None        | Volume      | Positive
// None        | Volume Pod  | Positive
// None        | Volume Node | Positive
// Volume      | None        | Positive
// Volume      | Volume      | Negative (covered in Test_NestedPendingOperations_Negative_SecondOpBeforeFirstCompletes above)
// Volume      | Volume Pod  | Negative
// Volume      | Volume Node | Negative
// Volume Pod  | None        | Positive
// Volume Pod  | Volume      | Negative
// Volume Pod  | Volume Pod  | Negative (covered in Test_NestedPendingOperations_Negative_SecondSubOpBeforeFirstCompletes above)
// Volume Node | None        | Positive
// Volume Node | Volume      | Negative
// Volume Node | Volume Node | Negative

// These cases are not covered because they will never occur within the same
// binary, so either result works.
// Volume Pod  | Volume Node
// Volume Node | Volume Pod

func Test_NestedPendingOperations_Positive_SecondOpBeforeFirstCompletes_FirstNone_SecondNone(t *testing.T) {
	testConcurrentOperationsPositive(t,
		"", /* volumeName1 */
		"", /* podName1 */
		"", /* nodeName1 */
		"", /* volumeName1 */
		"", /* podName2 */
		"" /* nodeName2 */)
}

func Test_NestedPendingOperations_Positive_SecondOpBeforeFirstCompletes_FirstNone_SecondVolume(t *testing.T) {
	testConcurrentOperationsPositive(t,
		"", /* volumeName1 */
		"", /* podName1 */
		"", /* nodeName1 */
		v1.UniqueVolumeName("volume-name"),
		"", /* podName2 */
		"" /* nodeName2 */)
}

func Test_NestedPendingOperations_Positive_SecondOpBeforeFirstCompletes_FirstNone_SecondVolumePod(t *testing.T) {
	testConcurrentOperationsPositive(t,
		"", /* volumeName1 */
		"", /* podName1 */
		"", /* nodeName1 */
		v1.UniqueVolumeName("volume-name"),
		volumetypes.UniquePodName("operation-podname"),
		"" /* nodeName2 */)
}

func Test_NestedPendingOperations_Positive_SecondOpBeforeFirstCompletes_FirstNone_SecondVolumeNode(t *testing.T) {
	testConcurrentOperationsPositive(t,
		"", /* volumeName1 */
		"", /* podName1 */
		"", /* nodeName1 */
		v1.UniqueVolumeName("volume-name"),
		"", /* podName2 */
		types.NodeName("operation-nodename"))
}

func Test_NestedPendingOperations_Positive_SecondOpBeforeFirstCompletes_FirstVolume_SecondNone(t *testing.T) {
	testConcurrentOperationsPositive(t,
		v1.UniqueVolumeName("volume-name"),
		"", /* podName1 */
		"", /* nodeName1 */
		"", /* volumeName1 */
		"", /* podName2 */
		"" /* nodeName2 */)
}

func Test_NestedPendingOperations_Negative_SecondOpBeforeFirstCompletes_FirstVolume_SecondVolumePod(t *testing.T) {
	testConcurrentOperationsNegative(t,
		v1.UniqueVolumeName("volume-name"),
		"", /* podName1 */
		"", /* nodeName1 */
		v1.UniqueVolumeName("volume-name"),
		volumetypes.UniquePodName("operation-podname"),
		"" /* nodeName2 */)
}

func Test_NestedPendingOperations_Negative_SecondOpBeforeFirstCompletes_FirstVolume_SecondVolumeNode(t *testing.T) {
	testConcurrentOperationsNegative(t,
		v1.UniqueVolumeName("volume-name"),
		"", /* podName1 */
		"", /* nodeName1 */
		v1.UniqueVolumeName("volume-name"),
		"", /* podName2 */
		types.NodeName("operation-nodename"))
}

func Test_NestedPendingOperations_Positive_SecondOpBeforeFirstCompletes_FirstVolumePod_SecondNone(t *testing.T) {
	testConcurrentOperationsPositive(t,
		v1.UniqueVolumeName("volume-name"),
		volumetypes.UniquePodName("operation-podname"),
		"", /* nodeName1 */
		"", /* volumeName1 */
		"", /* podName2 */
		"" /* nodeName2 */)
}

func Test_NestedPendingOperations_Negative_SecondOpBeforeFirstCompletes_FirstVolumePod_SecondVolume(t *testing.T) {
	testConcurrentOperationsNegative(t,
		v1.UniqueVolumeName("volume-name"),
		volumetypes.UniquePodName("operation-podname"),
		"", /* nodeName1 */
		v1.UniqueVolumeName("volume-name"),
		"", /* podName2 */
		"" /* nodeName2 */)
}

func Test_NestedPendingOperations_Positive_SecondOpBeforeFirstCompletes_FirstVolumeNode_SecondNone(t *testing.T) {
	testConcurrentOperationsPositive(t,
		v1.UniqueVolumeName("volume-name"),
		"", /* podName1 */
		types.NodeName("operation-nodename"),
		"", /* volumeName1 */
		"", /* podName2 */
		"" /* nodeName2 */)
}

func Test_NestedPendingOperations_Negative_SecondOpBeforeFirstCompletes_FirstVolumeNode_SecondVolume(t *testing.T) {
	testConcurrentOperationsNegative(t,
		v1.UniqueVolumeName("volume-name"),
		"", /* podName1 */
		types.NodeName("operation-nodename"),
		v1.UniqueVolumeName("volume-name"),
		"", /* podName2 */
		"" /* nodeName2 */)
}

func Test_NestedPendingOperations_Negative_SecondOpBeforeFirstCompletes_FirstVolumeNode_SecondVolumeNode(t *testing.T) {
	testConcurrentOperationsNegative(t,
		v1.UniqueVolumeName("volume-name"),
		"", /* podName1 */
		types.NodeName("operation-nodename"),
		v1.UniqueVolumeName("volume-name"),
		"", /* podName2 */
		types.NodeName("operation-nodename"))
}

// testConcurrentOperationsPositive passes if the two operations keyed by the
// provided parameters are executed in parallel, and fails otherwise.
func testConcurrentOperationsPositive(
	t *testing.T,
	volumeName1 v1.UniqueVolumeName,
	podName1 volumetypes.UniquePodName,
	nodeName1 types.NodeName,
	volumeName2 v1.UniqueVolumeName,
	podName2 volumetypes.UniquePodName,
	nodeName2 types.NodeName) {

	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName1, podName1, nodeName1 /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := grm.Run(volumeName2, podName2, nodeName2, volumetypes.GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err2)
	}
}

// testConcurrentOperationsNegative passes if the creation of the second
// operation returns an alreadyExists error, and fails otherwise.
func testConcurrentOperationsNegative(
	t *testing.T,
	volumeName1 v1.UniqueVolumeName,
	podName1 volumetypes.UniquePodName,
	nodeName1 types.NodeName,
	volumeName2 v1.UniqueVolumeName,
	podName2 volumetypes.UniquePodName,
	nodeName2 types.NodeName) {

	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	operation1DoneCh := make(chan interface{}, 0 /* bufferSize */)
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName1, podName1, nodeName1 /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := generateNoopFunc()

	// Act
	err2 := grm.Run(volumeName2, podName2, nodeName2, volumetypes.GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 == nil {
		t.Fatalf("NestedPendingOperations did not fail. Expected an operation to already exist")
	}
	if !IsAlreadyExists(err2) {
		t.Fatalf("NestedPendingOperations did not return alreadyExistsError, got: %v", err2)
	}
}

/* END concurrent operations tests */

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
