/*
Copyright 2016 The Kubernetes Authors.

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

package retry

import (
	"context"
	"errors"
	"fmt"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
)

func TestRetryOnConflict(t *testing.T) {
	opts := wait.Backoff{Factor: 1.0, Steps: 3}
	conflictErr := apierrors.NewConflict(schema.GroupResource{Resource: "test"}, "other", nil)

	// never returns
	err := RetryOnConflict(opts, func() error {
		return conflictErr
	})
	if !errors.Is(err, conflictErr) {
		t.Errorf("unexpected error: %v", err)
	}

	// returns immediately
	i := 0
	err = RetryOnConflict(opts, func() error {
		i++
		return nil
	})
	if err != nil || i != 1 {
		t.Errorf("unexpected error: %v", err)
	}

	// returns immediately on error
	testErr := fmt.Errorf("some other error")
	err = RetryOnConflict(opts, func() error {
		return testErr
	})
	if !errors.Is(err, testErr) {
		t.Errorf("unexpected error: %v", err)
	}

	// keeps retrying
	i = 0
	err = RetryOnConflict(opts, func() error {
		if i < 2 {
			i++
			return apierrors.NewConflict(schema.GroupResource{Resource: "test"}, "other", nil)
		}
		return nil
	})
	if err != nil || i != 2 {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRetryOnConflictPreservesContextErrorWithoutLastError(t *testing.T) {
	err := RetryOnConflict(DefaultBackoff, func() error {
		return context.Canceled
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestOnErrorWithContext(t *testing.T) {
	opts := wait.Backoff{Factor: 1.0, Steps: 3}
	testErr := fmt.Errorf("retriable error")
	retriable := func(err error) bool { return errors.Is(err, testErr) }

	// returns nil when fn succeeds on first call
	callCount := 0
	err := OnErrorWithContext(context.Background(), opts, retriable, func(_ context.Context) error {
		callCount++
		return nil
	})
	if err != nil || callCount != 1 {
		t.Errorf("expected nil error and 1 call, got err=%v calls=%d", err, callCount)
	}

	// retries and eventually succeeds
	callCount = 0
	err = OnErrorWithContext(context.Background(), opts, retriable, func(_ context.Context) error {
		callCount++
		if callCount < 3 {
			return testErr
		}
		return nil
	})
	if err != nil || callCount != 3 {
		t.Errorf("expected nil error and 3 calls, got err=%v calls=%d", err, callCount)
	}

	// returns last retriable error after exhausting steps
	err = OnErrorWithContext(context.Background(), opts, retriable, func(_ context.Context) error {
		return testErr
	})
	if !errors.Is(err, testErr) {
		t.Errorf("expected testErr, got %v", err)
	}

	// stops immediately on non-retriable error
	otherErr := fmt.Errorf("non-retriable")
	err = OnErrorWithContext(context.Background(), opts, retriable, func(_ context.Context) error {
		return otherErr
	})
	if !errors.Is(err, otherErr) {
		t.Errorf("expected otherErr, got %v", err)
	}

	// respects context cancellation
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately
	err = OnErrorWithContext(ctx, opts, retriable, func(_ context.Context) error {
		return testErr
	})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestRetryOnConflictWithContext(t *testing.T) {
	opts := wait.Backoff{Factor: 1.0, Steps: 3}
	conflictErr := apierrors.NewConflict(schema.GroupResource{Resource: "test"}, "other", nil)

	// never resolves — exhausts retries and returns conflict error
	err := RetryOnConflictWithContext(context.Background(), opts, func(_ context.Context) error {
		return conflictErr
	})
	if !errors.Is(err, conflictErr) {
		t.Errorf("expected conflictErr, got %v", err)
	}

	// succeeds on first call
	i := 0
	err = RetryOnConflictWithContext(context.Background(), opts, func(_ context.Context) error {
		i++
		return nil
	})
	if err != nil || i != 1 {
		t.Errorf("expected nil error and 1 call, got err=%v calls=%d", err, i)
	}

	// stops immediately on a non-conflict error
	testErr := fmt.Errorf("some other error")
	err = RetryOnConflictWithContext(context.Background(), opts, func(_ context.Context) error {
		return testErr
	})
	if !errors.Is(err, testErr) {
		t.Errorf("expected testErr, got %v", err)
	}

	// retries then succeeds
	i = 0
	err = RetryOnConflictWithContext(context.Background(), opts, func(_ context.Context) error {
		if i < 2 {
			i++
			return apierrors.NewConflict(schema.GroupResource{Resource: "test"}, "other", nil)
		}
		return nil
	})
	if err != nil || i != 2 {
		t.Errorf("expected nil error and 2 retries, got err=%v retries=%d", err, i)
	}

	// respects context cancellation
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err = RetryOnConflictWithContext(ctx, opts, func(_ context.Context) error {
		return conflictErr
	})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}
