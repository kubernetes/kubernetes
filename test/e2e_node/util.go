/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e_node

import (
	"time"
)

// RetryFn represents a retryable test condition.  It returns an error if the condition is not met
// otherwise returns nil for success.
type RetryFn func() error

// Retry retries the RetryFn for a maximum of maxWait time.  The wait duration is waited between
// retries.  If the success condition is not met in maxWait time, the list of encountered errors
// is returned.  If successful returns an empty list.
// Example:
// Expect(Retry(time.Minute*1, time.Second*2, func() error {
//    if success {
//      return nil
//    } else {
//      return errors.New("Failed")
//    }
// }).To(BeNil(), fmt.Sprintf("Failed"))
func Retry(maxWait time.Duration, wait time.Duration, retry RetryFn) []error {
	errs := []error{}
	for start := time.Now(); time.Now().Before(start.Add(maxWait)); {
		if err := retry(); err != nil {
			errs = append(errs, err)
		} else {
			return []error{}
		}
	}
	return errs
}
