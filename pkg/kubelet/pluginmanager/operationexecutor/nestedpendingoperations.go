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

package operationexecutor

import (
	"fmt"
	"sync"

	k8sRuntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
)

type generatedOperations struct {
	// Name of operation - could be used for resetting shared exponential backoff
	operationName string
	operationFunc func() error
	completeFunc  func(*error)
}

// Run executes the operations and its supporting functions
func (o *generatedOperations) Run() (err error) {
	if o.completeFunc != nil {
		defer o.completeFunc(&err)
	}
	// Handle panic, if any, from operationFunc()
	defer k8sRuntime.RecoverFromPanic(&err)
	return o.operationFunc()
}

// newNestedPendingOperations returns a new instance of nestedPendingOperations.
func newNestedPendingOperations(exponentialBackOffOnError bool) *nestedPendingOperations {
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
	socketPath       string
	operationName    string
	operationPending bool
	expBackoff       exponentialbackoff.ExponentialBackoff
}

func (grm *nestedPendingOperations) Run(
	socketPath string,
	generatedOperations generatedOperations) error {
	grm.lock.Lock()
	defer grm.lock.Unlock()
	opExists, previousOpIndex := grm.isOperationExists(socketPath)
	if opExists {
		previousOp := grm.operations[previousOpIndex]
		// Operation already exists
		if previousOp.operationPending {
			// Operation is pending
			return NewAlreadyExistsError(socketPath)
		}

		backOffErr := previousOp.expBackoff.SafeToRetry(socketPath)
		if backOffErr != nil {
			if previousOp.operationName == generatedOperations.operationName {
				return backOffErr
			}
			// previous operation and new operation are different. reset op. name and exp. backoff
			grm.operations[previousOpIndex].operationName = generatedOperations.operationName
			grm.operations[previousOpIndex].expBackoff = exponentialbackoff.ExponentialBackoff{}
		}

		// Update existing operation to mark as pending.
		grm.operations[previousOpIndex].operationPending = true
		grm.operations[previousOpIndex].socketPath = socketPath
	} else {
		// Create a new operation
		grm.operations = append(grm.operations,
			operation{
				operationPending: true,
				socketPath:       socketPath,
				operationName:    generatedOperations.operationName,
				expBackoff:       exponentialbackoff.ExponentialBackoff{},
			})
	}

	go func() (err error) {
		// Handle unhandled panics (very unlikely)
		defer k8sRuntime.HandleCrash()
		// Handle completion of and error, if any, from operationFunc()
		defer grm.operationComplete(socketPath, &err)
		return generatedOperations.Run()
	}()

	return nil
}

func (grm *nestedPendingOperations) IsOperationPending(socketPath string) bool {

	grm.lock.RLock()
	defer grm.lock.RUnlock()

	exist, previousOpIndex := grm.isOperationExists(socketPath)
	if exist && grm.operations[previousOpIndex].operationPending {
		return true
	}
	return false
}

// This is an internal function and caller should acquire and release the lock
func (grm *nestedPendingOperations) isOperationExists(socketPath string) (bool, int) {

	for previousOpIndex, previousOp := range grm.operations {
		if previousOp.socketPath != socketPath {
			// No match, keep searching
			continue
		}

		// Match
		return true, previousOpIndex
	}
	return false, -1
}

func (grm *nestedPendingOperations) getOperation(socketPath string) (uint, error) {
	// Assumes lock has been acquired by caller.

	for i, op := range grm.operations {
		if op.socketPath == socketPath {
			return uint(i), nil
		}
	}

	return 0, fmt.Errorf("Operation %q not found", socketPath)
}

func (grm *nestedPendingOperations) deleteOperation(
	// Assumes lock has been acquired by caller.
	socketPath string) {

	opIndex := -1
	for i, op := range grm.operations {
		if op.socketPath == socketPath {
			opIndex = i
			break
		}
	}

	// Delete index without preserving order
	grm.operations[opIndex] = grm.operations[len(grm.operations)-1]
	grm.operations = grm.operations[:len(grm.operations)-1]
}

func (grm *nestedPendingOperations) operationComplete(socketPath string, err *error) {
	// Defer operations are executed in Last-In is First-Out order. In this case
	// the lock is acquired first when operationCompletes begins, and is
	// released when the method finishes, after the lock is released cond is
	// signaled to wake waiting goroutine.
	defer grm.cond.Signal()
	grm.lock.Lock()
	defer grm.lock.Unlock()

	if *err == nil || !grm.exponentialBackOffOnError {
		// Operation completed without error, or exponentialBackOffOnError disabled
		grm.deleteOperation(socketPath)
		if *err != nil {
			// Log error
			klog.Errorf("operation %s failed with: %v",
				socketPath,
				*err)
		}
		return
	}

	// Operation completed with error and exponentialBackOffOnError Enabled
	existingOpIndex, getOpErr := grm.getOperation(socketPath)
	if getOpErr != nil {
		// Failed to find existing operation
		klog.Errorf("Operation %s completed. error: %v. exponentialBackOffOnError is enabled, but failed to get operation to update.",
			socketPath,
			*err)
		return
	}

	grm.operations[existingOpIndex].expBackoff.Update(err)
	grm.operations[existingOpIndex].operationPending = false

	// Log error
	klog.Errorf("%v", grm.operations[existingOpIndex].expBackoff.
		GenerateNoRetriesPermittedMsg(socketPath))
}

func (grm *nestedPendingOperations) Wait() {
	grm.lock.Lock()
	defer grm.lock.Unlock()

	for len(grm.operations) > 0 {
		grm.cond.Wait()
	}
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
