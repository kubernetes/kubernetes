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

	// Act
	err := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: noopFunc})

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

	// Act
	err1 := grm.Run(volume1Name, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: noopFunc})
	err2 := grm.Run(volume2Name, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: noopFunc})

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

	// Act
	err1 := grm.Run(volumeName, operation1PodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: noopFunc})
	err2 := grm.Run(volumeName, operation2PodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: noopFunc})

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

	// Act
	err := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: noopFunc})

	// Assert
	if err != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err)
	}
}

func Test_NestedPendingOperations_Positive_SecondOpAfterFirstCompletes(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operation1DoneCh := make(chan interface{})
	operation1 := generateCallbackFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := noopFunc
	<-operation1DoneCh // Force operation1 to complete

	// Act
	err2 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeShort),
		func() (bool, error) {
			err := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation2})
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
	operation1DoneCh := make(chan interface{})
	operation1 := generateCallbackFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := noopFunc
	<-operation1DoneCh // Force operation1 to complete

	// Act
	err2 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeShort),
		func() (bool, error) {
			err := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation2})
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
	operation1 := panicFunc
	err1 := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := noopFunc

	// Act
	err2 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeShort),
		func() (bool, error) {
			err := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation2})
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
	operation1 := panicFunc
	err1 := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := noopFunc

	// Act
	err2 := retryWithExponentialBackOff(
		time.Duration(initialOperationWaitTimeLong), // Longer duration to accommodate for backoff
		func() (bool, error) {
			err := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation2})
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
	operation1DoneCh := make(chan interface{})
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := noopFunc

	// Act
	err2 := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation2})

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
	operation1 := errorFunc
	err1 := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation1, OperationName: op1Name})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	// Shorter than exponential backoff period, so as to trigger exponential backoff error on second
	// operation.
	operation2 := errorFunc
	err2 := retryWithExponentialBackOff(
		initialOperationWaitTimeShort,
		func() (bool, error) {
			err := grm.Run(volumeName,
				EmptyUniquePodName,
				EmptyNodeName,
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

	operation3 := noopFunc
	op3Name := "unmount_volume"
	// Act
	err3 := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation3, OperationName: op3Name})
	if err3 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected <no error> Actual: <%v>", err3)
	}
}

func Test_NestedPendingOperations_Negative_SecondSubOpBeforeFirstCompletes2(t *testing.T) {
	// Arrange
	grm := NewNestedPendingOperations(false /* exponentialBackOffOnError */)
	volumeName := v1.UniqueVolumeName("volume-name")
	operationPodName := volumetypes.UniquePodName("operation-podname")
	operation1DoneCh := make(chan interface{})
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, operationPodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := noopFunc

	// Act
	err2 := grm.Run(volumeName, operationPodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation2})

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
	operation1DoneCh := make(chan interface{})
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, operationPodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := noopFunc

	// Act
	err2 := grm.Run(volumeName, operationPodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation2})

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
	operation1DoneCh := make(chan interface{})
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := noopFunc

	// Act
	err2 := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation2})

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
	operation1DoneCh := make(chan interface{})
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := noopFunc
	operation3 := noopFunc

	// Act
	err2 := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation2})

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
			err := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation3})
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
	operation1DoneCh := make(chan interface{})
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := noopFunc
	operation3 := noopFunc

	// Act
	err2 := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation2})

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
			err := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation3})
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
	operation1DoneCh := make(chan interface{})
	operation1 := generateWaitFunc(operation1DoneCh)
	err := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation1})
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
	operation1DoneCh := make(chan interface{})
	operation1 := generateWaitFunc(operation1DoneCh)
	err := grm.Run(volumeName, EmptyUniquePodName, EmptyNodeName, volumetypes.GeneratedOperations{OperationFunc: operation1})
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

// "None" means volume, pod, and node names are all empty
// "Volume" means volume name is set, but pod name and node name are empty
// "Volume Pod" means volume and pod names are set, but the node name is empty
// "Volume Node" means volume and node names are set, but the pod name is empty

// The same volume, pod, and node names are used (where they are not empty).

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

func Test_NestedPendingOperations_SecondOpBeforeFirstCompletes(t *testing.T) {
	const (
		keyNone = iota
		keyVolume
		keyVolumePod
		keyVolumeNode
	)

	type testCase struct {
		testID     int
		keyTypes   []int // only 2 elements are supported
		expectPass bool
	}

	tests := []testCase{
		{testID: 1, keyTypes: []int{keyNone, keyNone}, expectPass: true},
		{testID: 2, keyTypes: []int{keyNone, keyVolume}, expectPass: true},
		{testID: 3, keyTypes: []int{keyNone, keyVolumePod}, expectPass: true},
		{testID: 4, keyTypes: []int{keyNone, keyVolumeNode}, expectPass: true},
		{testID: 5, keyTypes: []int{keyVolume, keyNone}, expectPass: true},
		{testID: 6, keyTypes: []int{keyVolume, keyVolumePod}, expectPass: false},
		{testID: 7, keyTypes: []int{keyVolume, keyVolumeNode}, expectPass: false},
		{testID: 8, keyTypes: []int{keyVolumePod, keyNone}, expectPass: true},
		{testID: 9, keyTypes: []int{keyVolumePod, keyVolume}, expectPass: false},
		{testID: 10, keyTypes: []int{keyVolumeNode, keyNone}, expectPass: true},
		{testID: 11, keyTypes: []int{keyVolumeNode, keyVolume}, expectPass: false},
		{testID: 12, keyTypes: []int{keyVolumeNode, keyVolumeNode}, expectPass: false},
	}

	for _, test := range tests {
		var (
			volumeNames []v1.UniqueVolumeName
			podNames    []volumetypes.UniquePodName
			nodeNames   []types.NodeName
		)
		for _, keyType := range test.keyTypes {
			var (
				v v1.UniqueVolumeName
				p volumetypes.UniquePodName
				n types.NodeName
			)
			switch keyType {
			case keyNone:
				v = EmptyUniqueVolumeName
				p = EmptyUniquePodName
				n = EmptyNodeName
			case keyVolume:
				v = v1.UniqueVolumeName("volume-name")
				p = EmptyUniquePodName
				n = EmptyNodeName
			case keyVolumePod:
				v = v1.UniqueVolumeName("volume-name")
				p = volumetypes.UniquePodName("operation-podname")
				n = EmptyNodeName
			case keyVolumeNode:
				v = v1.UniqueVolumeName("volume-name")
				p = EmptyUniquePodName
				n = types.NodeName("operation-nodename")
			}
			volumeNames = append(volumeNames, v)
			podNames = append(podNames, p)
			nodeNames = append(nodeNames, n)
		}

		t.Run(fmt.Sprintf("Test %d", test.testID), func(t *testing.T) {
			if test.expectPass {
				testConcurrentOperationsPositive(t,
					volumeNames[0], podNames[0], nodeNames[0],
					volumeNames[1], podNames[1], nodeNames[1],
				)
			} else {
				testConcurrentOperationsNegative(t,
					volumeNames[0], podNames[0], nodeNames[0],
					volumeNames[1], podNames[1], nodeNames[1],
				)
			}
		})

	}

}

func Test_NestedPendingOperations_Positive_Issue_88355(t *testing.T) {
	// This test reproduces the scenario that is likely to have caused
	// kubernetes/kubernetes issue #88355.
	// Please refer to the issue for more context:
	// https://github.com/kubernetes/kubernetes/issues/88355

	// Below, vx is a volume name, and nx is a node name.

	// Operation sequence:
	// opZ(v0) starts (operates on a different volume from all other operations)
	// op1(v1, n1) starts
	// op2(v1, n2) starts
	// opZ(v0) ends with success
	// op2(v1, n2) ends with an error (exponential backoff should be triggered)
	// op1(v1, n1) ends with success
	// op3(v1, n2) starts (continuously retried on exponential backoff error)
	// op3(v1, n2) ends with success
	// op4(v1, n2) starts
	// op4(v1, n2) ends with success

	const (
		mainVolumeName = "main-volume"
		opZVolumeName  = "other-volume"
		node1          = "node1"
		node2          = "node2"

		// delay after an operation is signaled to finish to ensure it actually
		// finishes before running the next operation.
		delay = 50 * time.Millisecond

		// Replicates the default AttachDetachController reconcile period
		reconcilerPeriod = 100 * time.Millisecond
	)

	grm := NewNestedPendingOperations(true /* exponentialBackOffOnError */)
	opZContinueCh := make(chan interface{})
	op1ContinueCh := make(chan interface{})
	op2ContinueCh := make(chan interface{})
	operationZ := generateWaitFunc(opZContinueCh)
	operation1 := generateWaitFunc(op1ContinueCh)
	operation2 := generateWaitWithErrorFunc(op2ContinueCh)
	operation3 := noopFunc
	operation4 := noopFunc

	errZ := grm.Run(opZVolumeName, "" /* podName */, "" /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operationZ})
	if errZ != nil {
		t.Fatalf("NestedPendingOperations failed for operationZ. Expected: <no error> Actual: <%v>", errZ)
	}

	err1 := grm.Run(mainVolumeName, "" /* podName */, node1, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Fatalf("NestedPendingOperations failed for operation1. Expected: <no error> Actual: <%v>", err1)
	}

	err2 := grm.Run(mainVolumeName, "" /* podName */, node2, volumetypes.GeneratedOperations{OperationFunc: operation2})
	if err2 != nil {
		t.Fatalf("NestedPendingOperations failed for operation2. Expected: <no error> Actual: <%v>", err2)
	}

	opZContinueCh <- true
	time.Sleep(delay)
	op2ContinueCh <- true
	time.Sleep(delay)
	op1ContinueCh <- true
	time.Sleep(delay)

	for {
		err3 := grm.Run(mainVolumeName, "" /* podName */, node2, volumetypes.GeneratedOperations{OperationFunc: operation3})
		if err3 == nil {
			break
		} else if !exponentialbackoff.IsExponentialBackoff(err3) {
			t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err3)
		}
		time.Sleep(reconcilerPeriod)
	}

	time.Sleep(delay)

	err4 := grm.Run(mainVolumeName, "" /* podName */, node2, volumetypes.GeneratedOperations{OperationFunc: operation4})
	if err4 != nil {
		t.Fatalf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err4)
	}
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
	operation1DoneCh := make(chan interface{})
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName1, podName1, nodeName1 /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Errorf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := noopFunc

	// Act
	err2 := grm.Run(volumeName2, podName2, nodeName2, volumetypes.GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 != nil {
		t.Errorf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err2)
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
	operation1DoneCh := make(chan interface{})
	operation1 := generateWaitFunc(operation1DoneCh)
	err1 := grm.Run(volumeName1, podName1, nodeName1 /* nodeName */, volumetypes.GeneratedOperations{OperationFunc: operation1})
	if err1 != nil {
		t.Errorf("NestedPendingOperations failed. Expected: <no error> Actual: <%v>", err1)
	}
	operation2 := noopFunc

	// Act
	err2 := grm.Run(volumeName2, podName2, nodeName2, volumetypes.GeneratedOperations{OperationFunc: operation2})

	// Assert
	if err2 == nil {
		t.Errorf("NestedPendingOperations did not fail. Expected an operation to already exist")
	}
	if !IsAlreadyExists(err2) {
		t.Errorf("NestedPendingOperations did not return alreadyExistsError, got: %v", err2)
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

func panicFunc() (error, error) {
	panic("testing panic")
}

func errorFunc() (error, error) {
	return fmt.Errorf("placeholder1"), fmt.Errorf("placeholder2")
}

func generateWaitWithErrorFunc(done <-chan interface{}) func() (error, error) {
	return func() (error, error) {
		<-done
		return fmt.Errorf("placeholder1"), fmt.Errorf("placeholder2")
	}
}

func noopFunc() (error, error) { return nil, nil }

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
