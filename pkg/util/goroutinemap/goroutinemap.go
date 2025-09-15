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
Package goroutinemap implements a data structure for managing go routines
by name. It prevents the creation of new go routines if an existing go routine
with the same name exists.
*/
package goroutinemap

import (
	"fmt"
	"sync"

	k8sRuntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
)

// GoRoutineMap defines a type that can run named goroutines and track their
// state.  It prevents the creation of multiple goroutines with the same name
// and may prevent recreation of a goroutine until after the a backoff time
// has elapsed after the last goroutine with that name finished.
type GoRoutineMap interface {
	// Run adds operation name to the list of running operations and spawns a
	// new go routine to execute the operation.
	// If an operation with the same operation name already exists, an
	// AlreadyExists or ExponentialBackoff error is returned.
	// Once the operation is complete, the go routine is terminated and the
	// operation name is removed from the list of executing operations allowing
	// a new operation to be started with the same operation name without error.
	Run(operationName string, operationFunc func() error) error

	// Wait blocks until operations map is empty. This is typically
	// necessary during tests - the test should wait until all operations finish
	// and evaluate results after that.
	Wait()

	// WaitForCompletion blocks until either all operations have successfully completed
	// or have failed but are not pending. The test should wait until operations are either
	// complete or have failed.
	WaitForCompletion()

	// IsOperationPending returns true if the operation is pending (currently
	// running), otherwise returns false.
	IsOperationPending(operationName string) bool
}

// NewGoRoutineMap returns a new instance of GoRoutineMap.
func NewGoRoutineMap(exponentialBackOffOnError bool) GoRoutineMap {
	g := &goRoutineMap{
		operations:                make(map[string]operation),
		exponentialBackOffOnError: exponentialBackOffOnError,
	}

	g.cond = sync.NewCond(&g.lock)
	return g
}

type goRoutineMap struct {
	operations                map[string]operation
	exponentialBackOffOnError bool
	cond                      *sync.Cond
	lock                      sync.RWMutex
}

// operation holds the state of a single goroutine.
type operation struct {
	operationPending bool
	expBackoff       exponentialbackoff.ExponentialBackoff
}

func (grm *goRoutineMap) Run(
	operationName string,
	operationFunc func() error) error {
	grm.lock.Lock()
	defer grm.lock.Unlock()

	existingOp, exists := grm.operations[operationName]
	if exists {
		// Operation with name exists
		if existingOp.operationPending {
			return NewAlreadyExistsError(operationName)
		}

		if err := existingOp.expBackoff.SafeToRetry(operationName); err != nil {
			return err
		}
	}

	grm.operations[operationName] = operation{
		operationPending: true,
		expBackoff:       existingOp.expBackoff,
	}
	go func() (err error) {
		// Handle unhandled panics (very unlikely)
		defer k8sRuntime.HandleCrash()
		// Handle completion of and error, if any, from operationFunc()
		defer grm.operationComplete(operationName, &err)
		// Handle panic, if any, from operationFunc()
		defer k8sRuntime.RecoverFromPanic(&err)
		return operationFunc()
	}()

	return nil
}

// operationComplete handles the completion of a goroutine run in the
// goRoutineMap.
func (grm *goRoutineMap) operationComplete(
	operationName string, err *error) {
	// Defer operations are executed in Last-In is First-Out order. In this case
	// the lock is acquired first when operationCompletes begins, and is
	// released when the method finishes, after the lock is released cond is
	// signaled to wake waiting goroutine.
	defer grm.cond.Signal()
	grm.lock.Lock()
	defer grm.lock.Unlock()

	if *err == nil || !grm.exponentialBackOffOnError {
		// Operation completed without error, or exponentialBackOffOnError disabled
		delete(grm.operations, operationName)
		if *err != nil {
			// Log error
			klog.Errorf("operation for %q failed with: %v",
				operationName,
				*err)
		}
	} else {
		// Operation completed with error and exponentialBackOffOnError Enabled
		existingOp := grm.operations[operationName]
		existingOp.expBackoff.Update(err)
		existingOp.operationPending = false
		grm.operations[operationName] = existingOp

		// Log error
		klog.Errorf("%v",
			existingOp.expBackoff.GenerateNoRetriesPermittedMsg(operationName))
	}
}

func (grm *goRoutineMap) IsOperationPending(operationName string) bool {
	grm.lock.RLock()
	defer grm.lock.RUnlock()
	existingOp, exists := grm.operations[operationName]
	if exists && existingOp.operationPending {
		return true
	}
	return false
}

func (grm *goRoutineMap) Wait() {
	grm.lock.Lock()
	defer grm.lock.Unlock()

	for len(grm.operations) > 0 {
		grm.cond.Wait()
	}
}

func (grm *goRoutineMap) WaitForCompletion() {
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
func (grm *goRoutineMap) nothingPending() bool {
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

// IsAlreadyExists returns true if an error returned from GoRoutineMap indicates
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

// alreadyExistsError is the error returned by GoRoutineMap when a new operation
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
