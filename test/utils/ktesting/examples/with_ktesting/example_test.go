//go:build example
// +build example

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

package withktesting

// The tests below will fail and therefore are excluded from
// normal "make test" via the "example" build tag. To run
// the tests and check the output, use "go test -tags example ."

import (
	"context"
	"testing"
	"time"

	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestTimeout(t *testing.T) {
	tCtx := ktesting.Init(t)
	tmp := t.TempDir()
	tCtx.Logf("Using %q as temporary directory.", tmp)
	tCtx.Cleanup(func() {
		t.Log("Cleaning up...")
	})
	if deadline, ok := t.Deadline(); ok {
		t.Logf("Will fail shortly before the test suite deadline at %s.", deadline)
	}
	select {
	case <-time.After(1000 * time.Hour):
		// This should not be reached.
		tCtx.Log("Huh?! I shouldn't be that old.")
	case <-tCtx.Done():
		// But this will before the test suite timeout.
		tCtx.Errorf("need to stop: %v", context.Cause(tCtx))
	}
}
