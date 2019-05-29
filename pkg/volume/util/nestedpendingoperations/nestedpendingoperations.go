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

/*
Package nestedpendingoperations is a modified implementation of
pkg/util/goroutinemap. It implements a data structure for managing go routines
by volume/pod name. It prevents the creation of new go routines if an existing
go routine for the volume already exists. It also allows multiple operations to
execute in parallel for the same volume as long as they are operating on
different pods.
*/
package nestedpendingoperations

import (
	"fmt"
	"sync"

	"k8s.io/api/core/v1"
	k8sRuntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
	"k8s.io/kubernetes/pkg/volume/util/types"
)

const (
	// EmptyUniquePodName is a UniquePodName for empty string.
	EmptyUniquePodName types.UniquePodName = types.UniquePodName("")

	// EmptyUniqueVolumeName is a UniqueVolumeName for empty string
	EmptyUniqueVolumeName v1.UniqueVolumeName = v1.UniqueVolumeName("")
)

// NestedPendingOperations defines the supported set of operations.
type NestedPendingOperations interface {
	// Run adds the concatenation of volumeName and podName to the list of
	// running operations and spawns a new go routine to execute operationFunc.
	// If an operation with the same volumeName, same or empty podName
	// and same operationName exits, an AlreadyExists or ExponentialBackoff
	// error is returned. If an operation with same volumeName and podName
	// has ExponentialBackoff error but operationName is different, exponential
	// backoff is reset and operation is allowed to proceed.
	// This enables multiple operations to execute in parallel for the same
	// volumeName as long as they have different podName.
	// Once the operation is complete, the go routine is terminated and the
	// concatenation of volumeName and podName is removed from the list of
	// executing operations allowing a new operation to be started with the
	// volumeName without error.
	Run(volumeName v1.UniqueVolumeName, podName types.UniquePodName, generatedOperations types.GeneratedOperations) error

	// Wait blocks until all operations are completed. This is typically
	// necessary during tests - the test should wait until all operations finish
	// and evaluate results after that.
	Wait()

	// IsOperationPending returns true if an operation for the given volumeName and podName is pending,
	// otherwise it returns false
	IsOperationPending(volumeName v1.UniqueVolumeName, podName types.UniquePodName) bool
}

// NewNestedPendingOperations returns a new instance of NestedPendingOperations.
func NewNestedPendingOperations(exponentialBackOffOnError bool) NestedPendingOperations {
	g := &nestedPendingOperations{
		operations:                []operation{},
		exponentialBackOffOnError: exponentialBackOffOnError,
	}
	g.cond = sync.NewCond(&g.lock)
	return g
}

type nestedPendingOperations struct {
	operations                []operation
	exponentialBackOffOnError bool
	cond                      *sync.Cond
	lock                      sync.RWMutex
}

type operation struct {
	volumeName       v1.UniqueVolumeName
	podName          types.UniquePodName
	operationName    string
	operationPending bool
	expBackoff       exponentialbackoff.ExponentialBackoff
}

func (grm *nestedPendingOperations) Run(
	volumeName v1.UniqueVolumeName,
	podName types.UniquePodName,
	generatedOperations types.GeneratedOperations) error {
	grm.lock.Lock()
	defer grm.lock.Unlock()
	opExists, previousOpIndex := grm.isOperationExists(volumeName, podName)
	if opExists {
		previousOp := grm.operations[previousOpIndex]
		// Operation already exists
		if previousOp.operationPending {
			// Operation is pending
			operationKey := getOperationKey(volumeName, podName)
			return NewAlreadyExistsError(operationKey)
		}

		operationKey := getOperationKey(volumeName, podName)
		backOffErr := previousOp.expBackoff.SafeToRetry(operationKey)
		if backOffErr != nil {
			if previousOp.operationName == generatedOperations.OperationName {
				return backOffErr
			}
			// previous operation and new operation are different. reset op. name and exp. backoff
			grm.operations[previousOpIndex].operationName = generatedOperations.OperationName
			grm.operations[previousOpIndex].expBackoff = exponentialbackoff.ExponentialBackoff{}
		}

		// Update existing operation to mark as pending.
		grm.operations[previousOpIndex].operationPending = true
		grm.operations[previousOpIndex].volumeName = volumeName
		grm.operations[previousOpIndex].podName = podName
	} else {
		// Create a new operation
		grm.operations = append(grm.operations,
			operation{
				operationPending: true,
				volumeName:       volumeName,
				podName:          podName,
				operationName:    generatedOperations.OperationName,
				expBackoff:       exponentialbackoff.ExponentialBackoff{},
			})
	}

	go func() (eventErr, detailedErr error) {
		// Handle unhandled panics (very unlikely)
		defer k8sRuntime.HandleCrash()
		// Handle completion of and error, if any, from operationFunc()
		defer grm.operationComplete(volumeName, podName, &detailedErr)
		return generatedOperations.Run()
	}()

	return nil
}

func (grm *nestedPendingOperations) IsOperationPending(
	volumeName v1.UniqueVolumeName,
	podName types.UniquePodName) bool {

	grm.lock.RLock()
	defer grm.lock.RUnlock()

	exist, previousOpIndex := grm.isOperationExists(volumeName, podName)
	if exist && grm.operations[previousOpIndex].operationPending {
		return true
	}
	return false
}

// This is an internal function and caller should acquire and release the lock
func (grm *nestedPendingOperations) isOperationExists(
	volumeName v1.UniqueVolumeName,
	podName types.UniquePodName) (bool, int) {

	// If volumeName is empty, operation can be executed concurrently
	if volumeName == EmptyUniqueVolumeName {
		return false, -1
	}

	for previousOpIndex, previousOp := range grm.operations {
		if previousOp.volumeName != volumeName {
			// No match, keep searching
			continue
		}

		if previousOp.podName != EmptyUniquePodName &&
			podName != EmptyUniquePodName &&
			previousOp.podName != podName {
			// No match, keep searching
			continue
		}

		// Match
		return true, previousOpIndex
	}
	return false, -1
}

func (grm *nestedPendingOperations) getOperation(
	volumeName v1.UniqueVolumeName,
	podName types.UniquePodName) (uint, error) {
	// Assumes lock has been acquired by caller.

	for i, op := range grm.operations {
		if op.volumeName == volumeName &&
			op.podName == podName {
			return uint(i), nil
		}
	}

	logOperationKey := getOperationKey(volumeName, podName)
	return 0, fmt.Errorf("Operation %q not found", logOperationKey)
}

func (grm *nestedPendingOperations) deleteOperation(
	// Assumes lock has been acquired by caller.
	volumeName v1.UniqueVolumeName,
	podName types.UniquePodName) {

	opIndex := -1
	for i, op := range grm.operations {
		if op.volumeName == volumeName &&
			op.podName == podName {
			opIndex = i
			break
		}
	}

	// Delete index without preserving order
	grm.operations[opIndex] = grm.operations[len(grm.operations)-1]
	grm.operations = grm.operations[:len(grm.operations)-1]
}

func (grm *nestedPendingOperations) operationComplete(
	volumeName v1.UniqueVolumeName, podName types.UniquePodName, err *error) {
	// Defer operations are executed in Last-In is First-Out order. In this case
	// the lock is acquired first when operationCompletes begins, and is
	// released when the method finishes, after the lock is released cond is
	// signaled to wake waiting goroutine.
	defer grm.cond.Signal()
	grm.lock.Lock()
	defer grm.lock.Unlock()

	if *err == nil || !grm.exponentialBackOffOnError {
		// Operation completed without error, or exponentialBackOffOnError disabled
		grm.deleteOperation(volumeName, podName)
		if *err != nil {
			// Log error
			logOperationKey := getOperationKey(volumeName, podName)
			klog.Errorf("operation %s failed with: %v",
				logOperationKey,
				*err)
		}
		return
	}

	// Operation completed with error and exponentialBackOffOnError Enabled
	existingOpIndex, getOpErr := grm.getOperation(volumeName, podName)
	if getOpErr != nil {
		// Failed to find existing operation
		logOperationKey := getOperationKey(volumeName, podName)
		klog.Errorf("Operation %s completed. error: %v. exponentialBackOffOnError is enabled, but failed to get operation to update.",
			logOperationKey,
			*err)
		return
	}

	grm.operations[existingOpIndex].expBackoff.Update(err)
	grm.operations[existingOpIndex].operationPending = false

	// Log error
	operationKey :=
		getOperationKey(volumeName, podName)
	klog.Errorf("%v", grm.operations[existingOpIndex].expBackoff.
		GenerateNoRetriesPermittedMsg(operationKey))
}

func (grm *nestedPendingOperations) Wait() {
	grm.lock.Lock()
	defer grm.lock.Unlock()

	for len(grm.operations) > 0 {
		grm.cond.Wait()
	}
}

func getOperationKey(
	volumeName v1.UniqueVolumeName, podName types.UniquePodName) string {
	podNameStr := ""
	if podName != EmptyUniquePodName {
		podNameStr = fmt.Sprintf(" (%q)", podName)
	}

	return fmt.Sprintf("%q%s",
		volumeName,
		podNameStr)
}

// NewAlreadyExistsError returns a new instance of AlreadyExists error.
func NewAlreadyExistsError(operationKey string) error {
	return alreadyExistsError{operationKey}
}

// IsAlreadyExists returns true if an error returned from
// NestedPendingOperations indicates a new operation can not be started because
// an operation with the same operation name is already executing.
func IsAlreadyExists(err error) bool {
	switch err.(type) {
	case alreadyExistsError:
		return true
	default:
		return false
	}
}

// alreadyExistsError is the error returned by NestedPendingOperations when a
// new operation can not be started because an operation with the same operation
// name is already executing.
type alreadyExistsError struct {
	operationKey string
}

var _ error = alreadyExistsError{}

func (err alreadyExistsError) Error() string {
	return fmt.Sprintf(
		"Failed to create operation with name %q. An operation with that name is already executing.",
		err.operationKey)
}
