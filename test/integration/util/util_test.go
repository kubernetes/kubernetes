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
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
)

// TestInitTestAPIServerContextCanceledBeforeCleanup verifies the normal
// (non-explicit) shutdown path: testCtx.Ctx gets canceled as soon as the
// test function returns. The cancellation itself runs via
// context.AfterFunc, in parallel with (not necessarily before) any
// t.Cleanup callback, so a cleanup callback which depends on it must wait
// for it instead of checking it just once.
func TestInitTestAPIServerContextCanceledBeforeCleanup(t *testing.T) {
	testCtx := InitTestAPIServer(t, "normal-shutdown", nil)

	t.Cleanup(func() {
		select {
		case <-testCtx.Ctx.Done():
			// Canceled, as expected.
		case <-time.After(30 * time.Second):
			t.Error("testCtx.Ctx was not canceled by the time cleanup ran")
		}
	})
}

// TestInitTestAPIServerCloseFnCancelsCallerContext verifies that calling
// testCtx.CloseFn explicitly (i.e. before the test itself ends and the
// top-level context gets canceled through test cleanup) cancels
// testCtx.Ctx. Callers rely on that context to know when to stop any
// goroutines they started with it, so CloseFn must cancel it directly
// instead of only canceling the apiserver's own, detached context.
//
// Note that this is not properly supported by InitTestAPIServer: the
// automatic cleanup in CleanupTest fails when tests call CloseFn (API server
// shuts down before cleanup) or CleanupTest itself (cleanup runs twice,
// second invocation fails). The test works around that with a fake client.
func TestInitTestAPIServerCloseFnCancelsCallerContext(t *testing.T) {
	testCtx := InitTestAPIServer(t, "close-fn-cancel", nil)

	// Explicit shutdown, simulating a caller which stops the server
	// before its test function returns.
	testCtx.CloseFn()

	select {
	case <-testCtx.Ctx.Done():
		// Canceled, as expected.
	case <-time.After(30 * time.Second):
		t.Fatal("testCtx.Ctx was not canceled by CloseFn")
	}

	// Prevent failures when CleanupTest runs as cleanup callback.
	testCtx.ClientSet = fake.NewClientset(&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testCtx.NS.Name}})
}
