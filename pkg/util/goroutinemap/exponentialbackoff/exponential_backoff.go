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

// Package exponentialbackoff contains logic for implementing exponential
// backoff for GoRoutineMap and NestedPendingOperations.
package exponentialbackoff

import (
	"fmt"
	"time"
)

const (
	// initialDurationBeforeRetry is the amount of time after an error occurs
	// that GoroutineMap will refuse to allow another operation to start with
	// the same target (if exponentialBackOffOnError is enabled). Each
	// successive error results in a wait 2x times the previous.
	initialDurationBeforeRetry time.Duration = 500 * time.Millisecond

	// maxDurationBeforeRetry is the maximum amount of time that
	// durationBeforeRetry will grow to due to exponential backoff.
	maxDurationBeforeRetry time.Duration = 2 * time.Minute
)

// ExponentialBackoff contains the last occurrence of an error and the duration
// that retries are not permitted.
type ExponentialBackoff struct {
	lastError           error
	lastErrorTime       time.Time
	durationBeforeRetry time.Duration
}

// SafeToRetry returns an error if the durationBeforeRetry period for the given
// lastErrorTime has not yet expired. Otherwise it returns nil.
func (expBackoff *ExponentialBackoff) SafeToRetry(operationName string) error {
	if time.Since(expBackoff.lastErrorTime) <= expBackoff.durationBeforeRetry {
		return NewExponentialBackoffError(operationName, *expBackoff)
	}

	return nil
}

func (expBackoff *ExponentialBackoff) Update(err *error) {
	if expBackoff.durationBeforeRetry == 0 {
		expBackoff.durationBeforeRetry = initialDurationBeforeRetry
	} else {
		expBackoff.durationBeforeRetry = 2 * expBackoff.durationBeforeRetry
		if expBackoff.durationBeforeRetry > maxDurationBeforeRetry {
			expBackoff.durationBeforeRetry = maxDurationBeforeRetry
		}
	}

	expBackoff.lastError = *err
	expBackoff.lastErrorTime = time.Now()
}

func (expBackoff *ExponentialBackoff) GenerateNoRetriesPermittedMsg(
	operationName string) string {
	return fmt.Sprintf("Operation for %q failed. No retries permitted until %v (durationBeforeRetry %v). Error: %v",
		operationName,
		expBackoff.lastErrorTime.Add(expBackoff.durationBeforeRetry),
		expBackoff.durationBeforeRetry,
		expBackoff.lastError)
}

// NewExponentialBackoffError returns a new instance of ExponentialBackoff error.
func NewExponentialBackoffError(
	operationName string, expBackoff ExponentialBackoff) error {
	return exponentialBackoffError{
		operationName: operationName,
		expBackoff:    expBackoff,
	}
}

// IsExponentialBackoff returns true if an error returned from GoroutineMap
// indicates that a new operation can not be started because
// exponentialBackOffOnError is enabled and a previous operation with the same
// operation failed within the durationBeforeRetry period.
func IsExponentialBackoff(err error) bool {
	switch err.(type) {
	case exponentialBackoffError:
		return true
	default:
		return false
	}
}

// exponentialBackoffError is the error returned returned from GoroutineMap when
// a new operation can not be started because exponentialBackOffOnError is
// enabled and a previous operation with the same operation failed within the
// durationBeforeRetry period.
type exponentialBackoffError struct {
	operationName string
	expBackoff    ExponentialBackoff
}

var _ error = exponentialBackoffError{}

func (err exponentialBackoffError) Error() string {
	return fmt.Sprintf(
		"Failed to create operation with name %q. An operation with that name failed at %v. No retries permitted until %v (%v). Last error: %q.",
		err.operationName,
		err.expBackoff.lastErrorTime,
		err.expBackoff.lastErrorTime.Add(err.expBackoff.durationBeforeRetry),
		err.expBackoff.durationBeforeRetry,
		err.expBackoff.lastError)
}
