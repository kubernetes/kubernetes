/*
Copyright 2023 The Kubernetes Authors.

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

package api

import (
	"time"
)

// RetryError indicates that a service reconciliation should be retried after a
// fixed duration (as opposed to backing off exponentially).
type RetryError struct {
	msg        string
	retryAfter time.Duration
}

// NewRetryError returns a RetryError.
func NewRetryError(msg string, retryAfter time.Duration) *RetryError {
	return &RetryError{
		msg:        msg,
		retryAfter: retryAfter,
	}
}

// Error shows the details of the retry reason.
func (re *RetryError) Error() string {
	return re.msg
}

// RetryAfter returns the defined retry-after duration.
func (re *RetryError) RetryAfter() time.Duration {
	return re.retryAfter
}
