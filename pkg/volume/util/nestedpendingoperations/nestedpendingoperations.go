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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	k8sRuntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

const (
	// EmptyUniquePodName is a UniquePodName for empty string.
	EmptyUniquePodName volumetypes.UniquePodName = volumetypes.UniquePodName("")

	// EmptyUniqueVolumeName is a UniqueVolumeName for empty string
	EmptyUniqueVolumeName v1.UniqueVolumeName = v1.UniqueVolumeName("")

	// EmptyNodeName is a NodeName for empty string
	EmptyNodeName types.NodeName = types.NodeName("")
)

// NestedPendingOperations defines the supported set of operations.
type NestedPendingOperations interface {

	// Run adds the concatenation of volumeName, podName, and nodeName to the list
	// of running operations and spawns a new go routine to run
	// generatedOperations.

	// volumeName, podName, and nodeName collectively form the operation key.
	// The following forms of operation keys are supported (two keys are designed
	// to be "matched" if we want to serialize their operations):
	// - volumeName empty, podName and nodeName could be anything
	//   This key does not match with any keys.
	// - volumeName exists, podName empty, nodeName empty
	//   This key matches all other keys with the same volumeName.
	// - volumeName exists, podName exists, nodeName empty
	//   This key matches with:
	//   - the same volumeName and podName
	//   - the same volumeName, but empty podName
	// - volumeName exists, podName empty, nodeName exists
	//   This key matches with:
	//   - the same volumeName and nodeName
	//   - the same volumeName but empty nodeName

	// If there is no operation with a matching key, the operation is allowed to
	// proceed.
	// If an operation with a matching key exists and the previous operation is
	// running, an AlreadyExists error is returned.
	// If an operation with a matching key exists and the previous operation
	// failed:
	// - If the previous operation has the same
	//   generatedOperations.operationName:
	//   - If the full exponential backoff period is satisfied, the operation is
	//     allowed to proceed.
	//   - Otherwise, an ExponentialBackoff error is returned.
	// - Otherwise, exponential backoff is reset and operation is allowed to
	//   proceed.

	// Once the operation is complete, the go routine is terminated. If the
	// operation succeeded, its corresponding key is removed from the list of
	// executing operations, allowing a new operation to be started with the key
	// without error. If it failed, the key remains and the exponential
	// backoff status is updated.
	Run(
		volumeName v1.UniqueVolumeName,
		podName volumetypes.UniquePodName,
		nodeName types.NodeName,
		generatedOperations volumetypes.GeneratedOperations) error

	// Wait blocks until all operations are completed. This is typically
	// necessary during tests - the test should wait until all operations finish
	// and evaluate results after that.
	Wait()

	// IsOperationPending returns true if an operation for the given volumeName
	// and one of podName or nodeName is pending, otherwise it returns false
	IsOperationPending(
		volumeName v1.UniqueVolumeName,
		podName volumetypes.UniquePodName,
		nodeName types.NodeName) bool

	// IsOperationSafeToRetry returns false if an operation for the given volumeName
	// and one of podName or nodeName is pending or in exponential backoff, otherwise it returns true
	IsOperationSafeToRetry(
		volumeName v1.UniqueVolumeName,
		podName volumetypes.UniquePodName,
		nodeName types.NodeName, operationName string) bool
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
	key              operationKey
	operationName    string
	operationPending bool
	expBackoff       exponentialbackoff.ExponentialBackoff
}

func (grm *nestedPendingOperations) Run(
	volumeName v1.UniqueVolumeName,
	podName volumetypes.UniquePodName,
	nodeName types.NodeName,
	generatedOperations volumetypes.GeneratedOperations) error {
	grm.lock.Lock()
	defer grm.lock.Unlock()

	opKey := operationKey{volumeName, podName, nodeName}

	opExists, previousOpIndex := grm.isOperationExists(opKey)
	if opExists {
		previousOp := grm.operations[previousOpIndex]
		// Operation already exists
		if previousOp.operationPending {
			// Operation is pending
			return NewAlreadyExistsError(opKey)
		}

		backOffErr := previousOp.expBackoff.SafeToRetry(fmt.Sprintf("%+v", opKey))
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
		grm.operations[previousOpIndex].key = opKey
	} else {
		// Create a new operation
		grm.operations = append(grm.operations,
			operation{
				key:              opKey,
				operationPending: true,
				operationName:    generatedOperations.OperationName,
				expBackoff:       exponentialbackoff.ExponentialBackoff{},
			})
	}

	go func() (eventErr, detailedErr error) {
		// Handle unhandled panics (very unlikely)
		defer k8sRuntime.HandleCrash()
		// Handle completion of and error, if any, from operationFunc()
		defer grm.operationComplete(opKey, &detailedErr)
		return generatedOperations.Run()
	}()

	return nil
}
func (grm *nestedPendingOperations) IsOperationSafeToRetry(
	volumeName v1.UniqueVolumeName,
	podName volumetypes.UniquePodName,
	nodeName types.NodeName,
	operationName string) bool {

	grm.lock.RLock()
	defer grm.lock.RUnlock()

	opKey := operationKey{volumeName, podName, nodeName}
	exist, previousOpIndex := grm.isOperationExists(opKey)
	if !exist {
		return true
	}
	previousOp := grm.operations[previousOpIndex]
	if previousOp.operationPending {
		return false
	}
	backOffErr := previousOp.expBackoff.SafeToRetry(fmt.Sprintf("%+v", opKey))
	if backOffErr != nil {
		if previousOp.operationName == operationName {
			return false
		}
	}

	return true
}

func (grm *nestedPendingOperations) IsOperationPending(
	volumeName v1.UniqueVolumeName,
	podName volumetypes.UniquePodName,
	nodeName types.NodeName) bool {

	grm.lock.RLock()
	defer grm.lock.RUnlock()

	opKey := operationKey{volumeName, podName, nodeName}
	exist, previousOpIndex := grm.isOperationExists(opKey)
	if exist && grm.operations[previousOpIndex].operationPending {
		return true
	}
	return false
}

// This is an internal function and caller should acquire and release the lock
func (grm *nestedPendingOperations) isOperationExists(key operationKey) (bool, int) {

	// If volumeName is empty, operation can be executed concurrently
	if key.volumeName == EmptyUniqueVolumeName {
		return false, -1
	}

	opIndex := -1
	for previousOpIndex, previousOp := range grm.operations {
		volumeNameMatch := previousOp.key.volumeName == key.volumeName

		podNameMatch := previousOp.key.podName == EmptyUniquePodName ||
			key.podName == EmptyUniquePodName ||
			previousOp.key.podName == key.podName

		podNameExactMatch := previousOp.key.podName == key.podName

		nodeNameMatch := previousOp.key.nodeName == EmptyNodeName ||
			key.nodeName == EmptyNodeName ||
			previousOp.key.nodeName == key.nodeName

		nodeNameExactMatch := previousOp.key.nodeName == key.nodeName

		if volumeNameMatch && podNameMatch && nodeNameMatch {
			// nonExactMatch pending first
			if previousOp.operationPending {
				return true, previousOpIndex
			}
			// nonExactMatch with no pending, set opIndex to the first nonExactMatch
			// exactMatch can override opIndex to expected
			if opIndex == -1 || (podNameExactMatch && nodeNameExactMatch) {
				opIndex = previousOpIndex
			}
		}
	}
	return opIndex != -1, opIndex

}

func (grm *nestedPendingOperations) getOperation(key operationKey) (uint, error) {
	// Assumes lock has been acquired by caller.

	for i, op := range grm.operations {
		if op.key.volumeName == key.volumeName &&
			op.key.podName == key.podName &&
			op.key.nodeName == key.nodeName {
			return uint(i), nil
		}
	}

	return 0, fmt.Errorf("operation %+v not found", key)
}

func (grm *nestedPendingOperations) deleteOperation(key operationKey) {
	// Assumes lock has been acquired by caller.

	opIndex := -1
	for i, op := range grm.operations {
		if op.key.volumeName == key.volumeName &&
			op.key.podName == key.podName &&
			op.key.nodeName == key.nodeName {
			opIndex = i
			break
		}
	}

	if opIndex < 0 {
		return
	}

	// Delete index without preserving order
	grm.operations[opIndex] = grm.operations[len(grm.operations)-1]
	grm.operations = grm.operations[:len(grm.operations)-1]
}

func (grm *nestedPendingOperations) operationComplete(key operationKey, err *error) {
	// Defer operations are executed in Last-In is First-Out order. In this case
	// the lock is acquired first when operationCompletes begins, and is
	// released when the method finishes, after the lock is released cond is
	// signaled to wake waiting goroutine.
	defer grm.cond.Signal()
	grm.lock.Lock()
	defer grm.lock.Unlock()

	if *err == nil || !grm.exponentialBackOffOnError {
		// Operation completed without error, or exponentialBackOffOnError disabled
		grm.deleteOperation(key)
		if *err != nil {
			// Log error
			klog.Errorf("operation %+v failed with: %v", key, *err)
		}
		return
	}

	// Operation completed with error and exponentialBackOffOnError Enabled
	existingOpIndex, getOpErr := grm.getOperation(key)
	if getOpErr != nil {
		// Failed to find existing operation
		klog.Errorf("Operation %+v completed. error: %v. exponentialBackOffOnError is enabled, but failed to get operation to update.",
			key,
			*err)
		return
	}

	grm.operations[existingOpIndex].expBackoff.Update(err)
	grm.operations[existingOpIndex].operationPending = false

	// Log error
	klog.Errorf("%v", grm.operations[existingOpIndex].expBackoff.
		GenerateNoRetriesPermittedMsg(fmt.Sprintf("%+v", key)))
}

func (grm *nestedPendingOperations) Wait() {
	grm.lock.Lock()
	defer grm.lock.Unlock()

	for len(grm.operations) > 0 {
		grm.cond.Wait()
	}
}

type operationKey struct {
	volumeName v1.UniqueVolumeName
	podName    volumetypes.UniquePodName
	nodeName   types.NodeName
}

// NewAlreadyExistsError returns a new instance of AlreadyExists error.
func NewAlreadyExistsError(key operationKey) error {
	return alreadyExistsError{key}
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
	operationKey operationKey
}

var _ error = alreadyExistsError{}

func (err alreadyExistsError) Error() string {
	return fmt.Sprintf(
		"Failed to create operation with name %+v. An operation with that name is already executing.",
		err.operationKey)
}
