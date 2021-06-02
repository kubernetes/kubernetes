/*
Copyright 2020 The Kubernetes Authors.

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

package exec

import (
	"time"
)

// NewTimeoutError returns a new TimeoutError.
func NewTimeoutError(err error, timeout time.Duration) *TimeoutError {
	return &TimeoutError{
		err:     err,
		timeout: timeout,
	}
}

// TimeoutError is an error returned on exec probe timeouts. It should be returned by CRI implementations
// in order for the exec prober to interpret exec timeouts as failed probes.
// TODO: this error type can likely be removed when we support CRI errors.
type TimeoutError struct {
	err     error
	timeout time.Duration
}

// Error returns the error string.
func (t *TimeoutError) Error() string {
	return t.err.Error()
}

// Timeout returns the timeout duration of the exec probe.
func (t *TimeoutError) Timeout() time.Duration {
	return t.timeout
}
