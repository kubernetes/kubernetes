/*
Copyright 2015 The Kubernetes Authors.

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

package assert

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// EventuallyTrue asserts that the given predicate becomes true within the given timeout. It
// checks the predicate regularly each 100ms.
func EventuallyTrue(t *testing.T, timeout time.Duration, fn func() bool, msgAndArgs ...interface{}) bool {
	start := time.Now()
	for {
		if fn() {
			return true
		}
		if time.Now().Sub(start) > timeout {
			if len(msgAndArgs) > 0 {
				return assert.Fail(t, msgAndArgs[0].(string), msgAndArgs[1:]...)
			} else {
				return assert.Fail(t, "predicate fn has not been true after %v", timeout.String())
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
}
