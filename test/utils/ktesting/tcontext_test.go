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

package ktesting_test

import (
	"context"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/onsi/gomega"

	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestCancelManual(t *testing.T) {
	tCtx := ktesting.Init(t)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		// Blocks until tCtx.Cancel is called below.
		<-tCtx.Done()
	}()
	tCtx.Cancel("manually canceled")
	wg.Wait()
}

func TestCancelAutomatic(t *testing.T) {
	var wg sync.WaitGroup
	// This callback gets registered first and thus
	// gets invoked last.
	t.Cleanup(wg.Wait)
	tCtx := ktesting.Init(t)
	wg.Add(1)
	go func() {
		defer wg.Done()
		// Blocks until the context gets canceled automatically.
		<-tCtx.Done()
	}()
}

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

func TestNormalInit(t *testing.T) {
	// The outcome depends on how the unit test was started.
	// See below for deterministic deadline/no deadline testing.
	expectDeadline, expectOK := t.Deadline()
	expectDeadline = expectDeadline.Add(-ktesting.CleanupGracePeriod)
	tCtx := ktesting.Init(t)
	actualDeadline, actualOK := tCtx.Deadline()
	tCtx.Expect(actualOK).To(gomega.Equal(expectOK), "have deadline")
	if expectOK {
		tCtx.Expect(actualDeadline).To(gomega.BeTemporally("~", expectDeadline, 2*time.Second), "deadline")
	}
	if tCtx.IsSyncTest() {
		tCtx.Errorf("Expected to not run as synctest")
	}
}

func TestNoDeadline(t *testing.T) {
	mockT := &deadlineT{T: t, deadline: nil}
	tCtx := ktesting.Init(mockT)
	deadline, ok := tCtx.Deadline()
	if ok {
		tCtx.Errorf("Expected no deadline, got %s", deadline)
	}
}

func TestDeadline(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		// Inside a synctest bubble this is always in the future.
		mockDeadline := time.Date(2000, 01, 01, 0, 0, 0, 0, time.UTC)
		mockT := &deadlineT{T: t, deadline: &mockDeadline}
		tCtx := ktesting.Init(mockT)
		actualDeadline, ok := tCtx.Deadline()
		if ok {
			expectDeadline := mockDeadline.Add(-ktesting.CleanupGracePeriod)
			tCtx.Expect(actualDeadline).To(gomega.BeTemporally("==", expectDeadline), "deadline")
		} else {
			tCtx.Error("Expected a deadline, got none")
		}
	})
}

// deadlineT overrides Deadline, returning false if no
// deadline is configured and the deadline otherwise.
type deadlineT struct {
	*testing.T
	deadline *time.Time
}

func (t *deadlineT) Deadline() (time.Time, bool) {
	if t.deadline == nil {
		return time.Time{}, false
	}
	return *t.deadline, true
}

func TestCancelCtx(t *testing.T) {
	tCtx := ktesting.Init(t)
	var discardLogger klog.Logger
	tCtx = tCtx.WithLogger(discardLogger)
	baseCtx := tCtx

	tCtx.Cleanup(func() {
		if tCtx.Err() == nil {
			t.Error("context should be canceled but isn't")
		}
	})
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.Assert(tCtx.Err()).To(gomega.Succeed(), "context should not be canceled but is")
		tCtx.Assert(tCtx.Logger()).To(gomega.Equal(baseCtx.Logger()), "Logger()")
	})

	// Cancel, then let testing.T invoke test cleanup.
	tCtx.Cancel("test is complete")
}

func TestParallel(t *testing.T) {
	var wg sync.WaitGroup
	wg.Add(3)

	tCtx := ktesting.Init(t)

	// Each sub-test runs in parallel to the others and waits for the other two.
	test := func(tCtx ktesting.TContext) {
		tCtx.Parallel()
		wg.Done()
		wg.Wait()
	}
	tCtx.Run("one", test)
	tCtx.Run("two", test)
	tCtx.Run("three", test)
}

func TestRun(t *testing.T) {
	tCtx := ktesting.Init(t)

	key := 42
	value := "fish"
	tCtx = tCtx.WithValue(key, value)

	tCtx.Run("sub", func(tCtx ktesting.TContext) {
		tCtx.Assert(tCtx.Value(key)).To(gomega.Equal(value))

		tCtx.Cancel("test is complete")
		<-tCtx.Done()
	})

	if err := tCtx.Err(); err != nil {
		t.Errorf("parent TContext should not have been cancelled: %v", err)
	}
}

func TestWithNamespace(t *testing.T) {
	tCtx := ktesting.Init(t)
	namespace := "foo"
	tCtxWithNamespace := tCtx.WithNamespace(namespace)
	tCtx.Expect(tCtxWithNamespace.Namespace()).To(gomega.Equal(namespace))
}

func TestWithContext(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx.Cancel("done")
	tCtx = tCtx.WithValue("foo", "bar")
	deadline := time.Now().Add(-time.Minute)
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()
	newCtx := tCtx.WithContext(ctx)
	tCtx.Expect(context.Cause(tCtx)).To(gomega.MatchError(gomega.ContainSubstring("done")))
	tCtx.Expect(newCtx.Err()).To(gomega.MatchError(context.DeadlineExceeded))
	tCtx.Expect(newCtx.Value("foo")).To(gomega.Equal("bar"))
}
