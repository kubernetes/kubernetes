/*
Copyright The Kubernetes Authors.

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

package ktesting_test

import (
	"testing"
	"testing/synctest"

	"k8s.io/kubernetes/test/utils/ktesting"
)

// TestSyncTestInit matches the corresponding test in the main package. It
// exists here as the only test inside this package because signal.Notify fails
// inside a synctest bubble when called for the first time in a process and we
// want to enforce that situation.
func TestSyncTestInit(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		// This must work inside a synctest bubble, despite Deadline panicking there.
		// We then don't have a deadline.
		tCtx := ktesting.Init(t)
		deadline, ok := tCtx.Deadline()
		if ok {
			tCtx.Errorf("Expected no deadline, got %s", deadline)
		}
		if !tCtx.IsSyncTest() {
			tCtx.Errorf("Expected to run as synctest")
		}
	})
}
