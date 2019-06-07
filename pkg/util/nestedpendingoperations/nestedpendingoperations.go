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
Package nestedpendingoperations implements a data structure for managing go routines
by 2 operation keys. It prevents the creation of new go routines if an existing go routine
with the same first key already exists. It also allows multiple operations to
execute in parallel for the same first key as long as they are operating on
different second keys.
*/
package nestedpendingoperations

import (
	"fmt"
	"sync"

	"k8s.io/api/core/v1"
	k8sRuntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/util/nestedpendingoperations/exponentialbackoff"
	"k8s.io/kubernetes/pkg/volume/util/types"
)

var (
	// EmptyUniquePodName is a UniquePodName for empty string.
	EmptyUniquePodName = types.UniquePodName("")

	// EmptyUniqueVolumeName is a UniqueVolumeName for empty string
	EmptyUniqueVolumeName = v1.UniqueVolumeName("")
)

// NestedPendingOperations defines a type that can run named goroutines and track their
// state.  It prevents the creation of multiple goroutines with the same name
// and may prevent recreation of a goroutine until after the a backoff time
// has elapsed after the last goroutine with that name finished.  TODOTARA NEED TO UDPATE THIS
type NestedPendingOperations interface {
	// Run adds the concatenation of operation key 1 and key 2 to the list of running
	// operations and spawns a new go routine to execute the operation.
	// If an operation with the same first key, same or empty second key already exists
	// and same operationName exists, an AlreadyExists or ExponentialBackoff error is
	// returned.
	// If an operation with same first key and second key has ExponentialBackoff error
	// but generated operation's operationName is different, exponential backoff is reset
	// and operation is allowed to proceed.
	// This enables multiple operations to execute in parallel for the same
	// first key as long as they have different second keys.
	// Once the operation is complete, the go routine is terminated and the
	// concatenation of key 1 and key 2 is removed from the list of executing operations
	// allowing a new operation to be started with the same first key without error.
	Run(operationKey1, operationKey2 string, generatedOperations GeneratedOperations) error

	// Wait blocks until operations map is empty. This is typically
	// necessary during tests - the test should wait until all operations finish
	// and evaluate results after that.
	Wait()

	// WaitForCompletion blocks until either all operations have successfully completed
	// or have failed but are not pending. The test should wait until operations are either
	// complete or have failed.
	WaitForCompletion()

	// IsOperationPending returns true if the operation for the given key 1 and key 2
	// is pending (currently running), otherwise returns false.
	IsOperationPending(operationKey1, operationKey2 string) bool
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

// operation holds the state of a single goroutine.
type operation struct {
	operationPending bool
	expBackoff       exponentialbackoff.ExponentialBackoff
	operationKey1    string
	operationKey2    string
	operationName    string
}

// GeneratedOperations contains the operation that is created as well as
// supporting functions required for the operation executor
type GeneratedOperations struct {
	// Name of operation - could be used for resetting shared exponential backoff
	OperationName     string
	OperationFunc     func() (eventErr error, detailedErr error)
	EventRecorderFunc func(*error)
	CompleteFunc      func(*error)
}

// Run executes the operations and its supporting functions
func (o *GeneratedOperations) Run() (eventErr, detailedErr error) {
	if o.CompleteFunc != nil {
		defer o.CompleteFunc(&detailedErr)
	}
	if o.EventRecorderFunc != nil {
		defer o.EventRecorderFunc(&eventErr)
	}
	// Handle panic, if any, from operationFunc()
	defer k8sRuntime.RecoverFromPanic(&detailedErr)
	return o.OperationFunc()
}

func (grm *nestedPendingOperations) Run(
	operationKey1, operationKey2 string,
	generatedOperations GeneratedOperations) error {
	grm.lock.Lock()
	defer grm.lock.Unlock()

	opExists, previousOpIndex := grm.isOperationExists(operationKey1, operationKey2)
	if opExists {
		previousOp := grm.operations[previousOpIndex]
		// Operation already exists
		if previousOp.operationPending {
			// Operation is pending
			operationKey := getOperationKey(operationKey1, operationKey2)
			return NewAlreadyExistsError(operationKey)
		}

		operationKey := getOperationKey(operationKey1, operationKey2)
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
		grm.operations[previousOpIndex].operationKey1 = operationKey1
		grm.operations[previousOpIndex].operationKey2 = operationKey2
	} else {
		// Create a new operation
		grm.operations = append(grm.operations,
			operation{
				operationPending: true,
				operationKey1:    operationKey1,
				operationKey2:    operationKey2,
				operationName:    generatedOperations.OperationName,
				expBackoff:       exponentialbackoff.ExponentialBackoff{},
			})
	}

	// existingOp, exists := grm.operations[operationName]
	// if exists {
	// 	// Operation with name exists
	// 	if existingOp.operationPending {
	// 		return NewAlreadyExistsError(operationName)
	// 	}

	// 	if err := existingOp.expBackoff.SafeToRetry(operationName); err != nil {
	// 		return err
	// 	}
	// }

	// grm.operations[operationName] = operation{
	// 	operationPending: true,
	// 	expBackoff:       existingOp.expBackoff,
	// }
	go func() (eventErr, detailedErr error) {
		// Handle unhandled panics (very unlikely)
		defer k8sRuntime.HandleCrash()
		// Handle completion of and error, if any, from operationFunc()
		defer grm.operationComplete(operationKey1, operationKey2, &detailedErr)
		// Handle panic, if any, from operationFunc()
		defer k8sRuntime.RecoverFromPanic(&detailedErr)
		return generatedOperations.Run()
	}()

	return nil
}

// This is an internal function and caller should acquire and release the lock
func (grm *nestedPendingOperations) isOperationExists(
	operationKey1, operationKey2 string) (bool, int) {

	// If volumeName is empty, operation can be executed concurrently
	if operationKey1 == fmt.Sprintf("%v", EmptyUniqueVolumeName) || operationKey1 == "" {
		return false, -1
	}

	for previousOpIndex, previousOp := range grm.operations {
		if previousOp.operationKey1 != operationKey1 {
			// No match, keep searching
			continue
		}

		if (previousOp.operationKey2 != fmt.Sprintf("%v", EmptyUniquePodName) &&
			operationKey2 != fmt.Sprintf("%v", EmptyUniquePodName) &&
			previousOp.operationKey2 != operationKey2) ||
			(previousOp.operationKey2 != "" &&
				operationKey2 != "" &&
				previousOp.operationKey2 != operationKey2) {
			// No match, keep searching
			continue
		}

		// Match
		return true, previousOpIndex
	}
	return false, -1
}

func getOperationKey(
	operationKey1, operationKey2 string) string {
	operationKey2Str := ""
	if operationKey2 != fmt.Sprintf("%v", EmptyUniquePodName) && operationKey2 != "" {
		operationKey2Str = fmt.Sprintf(" (%q)", operationKey2)
	}

	return fmt.Sprintf("%q%s",
		operationKey1,
		operationKey2Str)
}

func (grm *nestedPendingOperations) getOperation(
	operationKey1, operationKey2 string) (uint, error) {
	// Assumes lock has been acquired by caller.

	for i, op := range grm.operations {
		if op.operationKey1 == operationKey1 &&
			op.operationKey2 == operationKey2 {
			return uint(i), nil
		}
	}

	logOperationKey := getOperationKey(operationKey1, operationKey2)
	return 0, fmt.Errorf("Operation %q not found", logOperationKey)
}

func (grm *nestedPendingOperations) deleteOperation(
	// Assumes lock has been acquired by caller.
	operationKey1, operationKey2 string) {

	opIndex := -1
	for i, op := range grm.operations {
		if op.operationKey1 == operationKey1 &&
			op.operationKey2 == operationKey2 {
			opIndex = i
			break
		}
	}

	// Delete index without preserving order
	grm.operations[opIndex] = grm.operations[len(grm.operations)-1]
	grm.operations = grm.operations[:len(grm.operations)-1]
}

// operationComplete handles the completion of a goroutine run in the
// nestedPendingOperations.
func (grm *nestedPendingOperations) operationComplete(
	operation1, operation2 string, err *error) {
	// Defer operations are executed in Last-In is First-Out order. In this case
	// the lock is acquired first when operationCompletes begins, and is
	// released when the method finishes, after the lock is released cond is
	// signaled to wake waiting goroutine.
	defer grm.cond.Signal()
	grm.lock.Lock()
	defer grm.lock.Unlock()

	if *err == nil || !grm.exponentialBackOffOnError {
		// Operation completed without error, or exponentialBackOffOnError disabled
		grm.deleteOperation(operation1, operation2)
		if *err != nil {
			// Log error
			logOperationKey := getOperationKey(operation1, operation2)
			klog.Errorf("operation %s failed with: %v",
				logOperationKey,
				*err)
		}
		return
	}

	// Operation completed with error and exponentialBackOffOnError Enabled
	existingOpIndex, getOpErr := grm.getOperation(operation1, operation2)
	if getOpErr != nil {
		// Failed to find existing operation
		logOperationKey := getOperationKey(operation1, operation2)
		klog.Errorf("Operation %s completed. error: %v. exponentialBackOffOnError is enabled, but failed to get operation to update.",
			logOperationKey,
			*err)
		return
	}

	grm.operations[existingOpIndex].expBackoff.Update(err)
	grm.operations[existingOpIndex].operationPending = false

	// Log error
	operationKey :=
		getOperationKey(operation1, operation2)
	klog.Errorf("%v", grm.operations[existingOpIndex].expBackoff.
		GenerateNoRetriesPermittedMsg(operationKey))
}

func (grm *nestedPendingOperations) IsOperationPending(
	operationKey1, operationKey2 string) bool {

	grm.lock.RLock()
	defer grm.lock.RUnlock()

	exist, previousOpIndex := grm.isOperationExists(operationKey1, operationKey2)
	if exist && grm.operations[previousOpIndex].operationPending {
		return true
	}
	return false
}

func (grm *nestedPendingOperations) Wait() {
	grm.lock.Lock()
	defer grm.lock.Unlock()

	for len(grm.operations) > 0 {
		grm.cond.Wait()
	}
}

func (grm *nestedPendingOperations) WaitForCompletion() {
	grm.lock.Lock()
	defer grm.lock.Unlock()

	for {
		if len(grm.operations) == 0 || grm.nothingPending() {
			break
		} else {
			grm.cond.Wait()
		}
	}
}

// Check if any operation is pending. Already assumes caller has the
// necessary locks
func (grm *nestedPendingOperations) nothingPending() bool {
	nothingIsPending := true
	for _, operation := range grm.operations {
		if operation.operationPending {
			nothingIsPending = false
			break
		}
	}
	return nothingIsPending
}

// NewAlreadyExistsError returns a new instance of AlreadyExists error.
func NewAlreadyExistsError(operationName string) error {
	return alreadyExistsError{operationName}
}

// IsAlreadyExists returns true if an error returned from nestedPendingOperations indicates
// a new operation can not be started because an operation with the same
// operation name is already executing.
func IsAlreadyExists(err error) bool {
	switch err.(type) {
	case alreadyExistsError:
		return true
	default:
		return false
	}
}

// alreadyExistsError is the error returned by nestedPendingOperations when a new operation
// can not be started because an operation with the same operation name is
// already executing.
type alreadyExistsError struct {
	operationName string
}

var _ error = alreadyExistsError{}

func (err alreadyExistsError) Error() string {
	return fmt.Sprintf(
		"Failed to create operation with name %q. An operation with that name is already executing.",
		err.operationName)
}
