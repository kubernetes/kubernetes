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

/*
Package goroutinemap implements a data structure for managing go routines
by name. It prevents the creation of new go routines if an existing go routine
with the same name exists.
*/
package goroutinemap

import (
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/golang/glog"
	k8sRuntime "k8s.io/kubernetes/pkg/util/runtime"
)

const (
	// initialDurationBeforeRetry is the amount of time after an error occurs
	// that GoRoutineMap will refuse to allow another operation to start with
	// the same operationName (if exponentialBackOffOnError is enabled). Each
	// successive error results in a wait 2x times the previous.
	initialDurationBeforeRetry time.Duration = 500 * time.Millisecond

	// maxDurationBeforeRetry is the maximum amount of time that
	// durationBeforeRetry will grow to due to exponential backoff.
	maxDurationBeforeRetry time.Duration = 2 * time.Minute
)

// GoRoutineMap defines the supported set of operations.
type GoRoutineMap interface {
	// Run adds operationName to the list of running operations and spawns a new
	// go routine to execute the operation. If an operation with the same name
	// already exists, an error is returned. Once the operation is complete, the
	// go routine is terminated and the operationName is removed from the list
	// of executing operations allowing a new operation to be started with the
	// same name without error.
	Run(operationName string, operationFunc func() error) error

	// Wait blocks until all operations are completed. This is typically
	// necessary during tests - the test should wait until all operations finish
	// and evaluate results after that.
	Wait()
}

// NewGoRoutineMap returns a new instance of GoRoutineMap.
func NewGoRoutineMap(exponentialBackOffOnError bool) GoRoutineMap {
	g := &goRoutineMap{
		operations:                make(map[string]operation),
		exponentialBackOffOnError: exponentialBackOffOnError,
	}
	g.cond = sync.NewCond(g)
	return g
}

type goRoutineMap struct {
	operations                map[string]operation
	exponentialBackOffOnError bool
	cond                      *sync.Cond
	sync.Mutex
}

type operation struct {
	operationPending    bool
	lastError           error
	lastErrorTime       time.Time
	durationBeforeRetry time.Duration
}

func (grm *goRoutineMap) Run(operationName string, operationFunc func() error) error {
	grm.Lock()
	defer grm.Unlock()
	existingOp, exists := grm.operations[operationName]
	if exists {
		// Operation with name exists
		if existingOp.operationPending {
			return newAlreadyExistsError(operationName)
		}

		if time.Since(existingOp.lastErrorTime) <= existingOp.durationBeforeRetry {
			return newExponentialBackoffError(operationName, existingOp)
		}
	}

	grm.operations[operationName] = operation{
		operationPending:    true,
		lastError:           existingOp.lastError,
		lastErrorTime:       existingOp.lastErrorTime,
		durationBeforeRetry: existingOp.durationBeforeRetry,
	}
	go func() (err error) {
		// Handle unhandled panics (very unlikely)
		defer k8sRuntime.HandleCrash()
		// Handle completion of and error, if any, from operationFunc()
		defer grm.operationComplete(operationName, &err)
		// Handle panic, if any, from operationFunc()
		defer recoverFromPanic(operationName, &err)
		return operationFunc()
	}()

	return nil
}

func (grm *goRoutineMap) operationComplete(operationName string, err *error) {
	defer grm.cond.Signal()
	grm.Lock()
	defer grm.Unlock()

	if *err == nil || !grm.exponentialBackOffOnError {
		// Operation completed without error, or exponentialBackOffOnError disabled
		delete(grm.operations, operationName)
		if *err != nil {
			// Log error
			glog.Errorf("operation for %q failed with: %v",
				operationName,
				*err)
		}
	} else {
		// Operation completed with error and exponentialBackOffOnError Enabled
		existingOp := grm.operations[operationName]
		if existingOp.durationBeforeRetry == 0 {
			existingOp.durationBeforeRetry = initialDurationBeforeRetry
		} else {
			existingOp.durationBeforeRetry = 2 * existingOp.durationBeforeRetry
			if existingOp.durationBeforeRetry > maxDurationBeforeRetry {
				existingOp.durationBeforeRetry = maxDurationBeforeRetry
			}
		}
		existingOp.lastError = *err
		existingOp.lastErrorTime = time.Now()
		existingOp.operationPending = false

		grm.operations[operationName] = existingOp

		// Log error
		glog.Errorf("Operation for %q failed. No retries permitted until %v (durationBeforeRetry %v). error: %v",
			operationName,
			existingOp.lastErrorTime.Add(existingOp.durationBeforeRetry),
			existingOp.durationBeforeRetry,
			*err)
	}
}

func (grm *goRoutineMap) Wait() {
	grm.Lock()
	defer grm.Unlock()

	for len(grm.operations) > 0 {
		grm.cond.Wait()
	}
}

func recoverFromPanic(operationName string, err *error) {
	if r := recover(); r != nil {
		callers := ""
		for i := 0; true; i++ {
			_, file, line, ok := runtime.Caller(i)
			if !ok {
				break
			}
			callers = callers + fmt.Sprintf("%v:%v\n", file, line)
		}
		*err = fmt.Errorf(
			"operation for %q recovered from panic %q. (err=%v) Call stack:\n%v",
			operationName,
			r,
			*err,
			callers)
	}
}

// alreadyExistsError is the error returned when NewGoRoutine() detects that
// an operation with the given name is already running.
type alreadyExistsError struct {
	operationName string
}

var _ error = alreadyExistsError{}

func (err alreadyExistsError) Error() string {
	return fmt.Sprintf("Failed to create operation with name %q. An operation with that name is already executing.", err.operationName)
}

func newAlreadyExistsError(operationName string) error {
	return alreadyExistsError{operationName}
}

// IsAlreadyExists returns true if an error returned from NewGoRoutine indicates
// that operation with the same name already exists.
func IsAlreadyExists(err error) bool {
	switch err.(type) {
	case alreadyExistsError:
		return true
	default:
		return false
	}
}

// exponentialBackoffError is the error returned when NewGoRoutine() detects
// that the previous operation for given name failed less then
// durationBeforeRetry.
type exponentialBackoffError struct {
	operationName string
	failedOp      operation
}

var _ error = exponentialBackoffError{}

func (err exponentialBackoffError) Error() string {
	return fmt.Sprintf(
		"Failed to create operation with name %q. An operation with that name failed at %v. No retries permitted until %v (%v). Last error: %q.",
		err.operationName,
		err.failedOp.lastErrorTime,
		err.failedOp.lastErrorTime.Add(err.failedOp.durationBeforeRetry),
		err.failedOp.durationBeforeRetry,
		err.failedOp.lastError)
}

func newExponentialBackoffError(
	operationName string, failedOp operation) error {
	return exponentialBackoffError{
		operationName: operationName,
		failedOp:      failedOp,
	}
}

// IsExponentialBackoff returns true if an error returned from NewGoRoutine()
// indicates that the previous operation for given name failed less then
// durationBeforeRetry.
func IsExponentialBackoff(err error) bool {
	switch err.(type) {
	case exponentialBackoffError:
		return true
	default:
		return false
	}
}
