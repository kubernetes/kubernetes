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

// This file is in package ktesting (not ktesting_test) so that it can access
// unexported fields of TContext.
package ktesting

import (
	"context"
	"maps"
	"slices"
	"testing"
	"testing/synctest"
	"time"

	"github.com/onsi/gomega"

	"k8s.io/kubernetes/test/utils/ktesting/initoption"
)

func TestSyncTestInit(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		// This must work inside a synctest bubble, despite Deadline panicking there.
		// We then don't have a deadline.
		tCtx := Init(t)
		deadline, ok := tCtx.Deadline()
		if ok {
			tCtx.Errorf("Expected no deadline, got %s", deadline)
		}
		if !tCtx.IsSyncTest() {
			tCtx.Errorf("Expected to run as synctest")
		}
		tCtx.Expect(getRunningTests()).To(gomega.ContainElement(gomega.Equal(t.Name())))
	})
}

func TestNormalInit(t *testing.T) {
	// The outcome depends on how the unit test was started.
	// See below for deterministic deadline/no deadline testing.
	expectDeadline, expectOK := t.Deadline()
	expectDeadline = expectDeadline.Add(-DefaultCleanupGracePeriod)
	tCtx := Init(t)
	actualDeadline, actualOK := tCtx.Deadline()
	tCtx.Expect(actualOK).To(gomega.Equal(expectOK), "have deadline")
	if expectOK {
		tCtx.Expect(actualDeadline).To(gomega.BeTemporally("~", expectDeadline, 2*time.Second), "deadline")
	}
	if tCtx.IsSyncTest() {
		tCtx.Errorf("Expected to not run as synctest")
	}
	tCtx.Expect(getRunningTests()).To(gomega.ContainElement(gomega.Equal(t.Name())))
}

// deadlineT2 mirrors the deadlineT helper in ktesting_test.
type deadlineT2 struct {
	TB
	deadline *time.Time
}

func (t *deadlineT2) Deadline() (time.Time, bool) {
	if t.deadline == nil {
		return time.Time{}, false
	}
	return *t.deadline, true
}

// TestDefaultCleanupGracePeriod verifies that cleanupGracePeriod is set to
// DefaultCleanupGracePeriod when no override is given.
func TestDefaultCleanupGracePeriod(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		mockDeadline := time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC)
		mockT := &deadlineT2{TB: t, deadline: &mockDeadline}
		tCtx := Init(mockT)
		if tCtx.cleanupGracePeriod != DefaultCleanupGracePeriod {
			t.Errorf("expected cleanupGracePeriod %v, got %v", DefaultCleanupGracePeriod, tCtx.cleanupGracePeriod)
		}
		tCtx.Expect(getRunningTests()).To(gomega.ContainElement(gomega.Equal(t.Name())))
	})
}

// TestCustomCleanupGracePeriod verifies that WithCleanupGracePeriod stores the
// override in cleanupGracePeriod and shifts the effective deadline accordingly.
func TestCustomCleanupGracePeriod(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		mockDeadline := time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC)
		mockT := &deadlineT2{TB: t, deadline: &mockDeadline}
		custom := 30 * time.Second
		tCtx := Init(mockT, initoption.WithCleanupGracePeriod(custom))

		if tCtx.cleanupGracePeriod != custom {
			t.Errorf("expected cleanupGracePeriod %v, got %v", custom, tCtx.cleanupGracePeriod)
		}

		actualDeadline, ok := tCtx.Deadline()
		if !ok {
			t.Fatal("expected a deadline but got none")
		}
		expect := mockDeadline.Add(-custom)
		tCtx.Expect(actualDeadline).To(
			gomega.BeTemporally("==", expect),
			"context deadline should be shifted by the custom grace period")
		tCtx.Expect(getRunningTests()).To(gomega.ContainElement(gomega.Equal(t.Name())))
	})
}

// TestInitCtxCleanupGracePeriod verifies that InitCtx applies
// WithCleanupGracePeriod via the previously ignored opts parameter.
func TestInitCtxCleanupGracePeriod(t *testing.T) {
	custom := 42 * time.Second
	tCtx := InitCtx(context.Background(), t, initoption.WithCleanupGracePeriod(custom))
	if tCtx.cleanupGracePeriod != custom {
		t.Errorf("expected cleanupGracePeriod %v, got %v", custom, tCtx.cleanupGracePeriod)
	}
	tCtx.Expect(getRunningTests()).To(gomega.ContainElement(gomega.Equal(t.Name())))
}

// TestInitCtxDefaultCleanupGracePeriod verifies that InitCtx falls back to
// DefaultCleanupGracePeriod when no option is given.
func TestInitCtxDefaultCleanupGracePeriod(t *testing.T) {
	tCtx := InitCtx(context.Background(), t)
	if tCtx.cleanupGracePeriod != DefaultCleanupGracePeriod {
		t.Errorf("expected cleanupGracePeriod %v, got %v", DefaultCleanupGracePeriod, tCtx.cleanupGracePeriod)
	}
	tCtx.Expect(getRunningTests()).To(gomega.ContainElement(gomega.Equal(t.Name())))
}

// getRunningTests reports all currently running tests, sorted by name.
func getRunningTests() []string {
	defaultProgressReporter.reportMutex.Lock()
	defer defaultProgressReporter.reportMutex.Unlock()

	return slices.Sorted(maps.Keys(defaultProgressReporter.runningTests))
}
