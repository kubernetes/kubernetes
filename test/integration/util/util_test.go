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

package util

import (
	"strings"
	"testing"
	"time"
)

// TestInitTestAPIServerContextCanceledBeforeCleanup verifies that the returned
// ktesting.TContext gets canceled when the test completes and cleanup runs.
// The cancellation itself runs via context.AfterFunc, in parallel with (not
// necessarily before) any t.Cleanup callback, so a cleanup callback which
// depends on it must wait for it instead of checking it just once.
func TestInitTestAPIServerContextCanceledBeforeCleanup(t *testing.T) {
	testCtx := InitTestAPIServer(t, "normal-shutdown", nil)

	t.Cleanup(func() {
		select {
		case <-testCtx.Done():
			// Canceled, as expected.
		case <-time.After(30 * time.Second):
			t.Error("testCtx was not canceled by the time cleanup ran")
		}
	})
}

// TestInitTestAPIServerTContext verifies that InitTestAPIServer returns a ktesting.TContext
// with clients, REST config, and namespace properly initialized.
func TestInitTestAPIServerTContext(t *testing.T) {
	testCtx := InitTestAPIServer(t, "tcontext", nil)

	if testCtx.Client() == nil {
		t.Errorf("testCtx.Client() is nil")
	}
	if testCtx.RESTConfig() == nil {
		t.Errorf("testCtx.RESTConfig() is nil")
	}
	if !strings.HasPrefix(testCtx.Namespace(), "tcontext") {
		t.Errorf("testCtx.Namespace() = %q, expected prefix %q", testCtx.Namespace(), "tcontext")
	}
}
