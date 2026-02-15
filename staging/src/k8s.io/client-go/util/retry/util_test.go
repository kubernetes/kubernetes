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
	stderrors "errors"
	"fmt"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
)

func TestRetryOnConflict(t *testing.T) {
	opts := wait.Backoff{Factor: 1.0, Steps: 3}
	conflictErr := errors.NewConflict(schema.GroupResource{Resource: "test"}, "other", nil)

	// never returns
	err := RetryOnConflict(opts, func() error {
		return conflictErr
	})
	if err != conflictErr {
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
	if err != testErr {
		t.Errorf("unexpected error: %v", err)
	}

	// keeps retrying
	i = 0
	err = RetryOnConflict(opts, func() error {
		if i < 2 {
			i++
			return errors.NewConflict(schema.GroupResource{Resource: "test"}, "other", nil)
		}
		return nil
	})
	if err != nil || i != 2 {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRetryOnConflictWithContext(t *testing.T) {
	opts := wait.Backoff{Factor: 1.0, Steps: 3}
	conflictErr := errors.NewConflict(schema.GroupResource{Resource: "test"}, "other", nil)

	// context cancelled before first attempt - returns context.Canceled immediately
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := RetryOnConflictWithContext(ctx, opts, func(context.Context) error {
		return conflictErr
	})
	if !stderrors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got: %v", err)
	}

	// context cancelled during retry - returns context.Canceled
	ctx, cancel = context.WithCancel(context.Background())
	i := 0
	err = RetryOnConflictWithContext(ctx, wait.Backoff{Factor: 1.0, Steps: 100, Duration: 50 * time.Millisecond}, func(context.Context) error {
		i++
		if i == 2 {
			cancel()
		}
		return conflictErr
	})
	if !stderrors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got: %v", err)
	}

	// context timeout - returns context.DeadlineExceeded
	ctx, cancel = context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	i = 0
	err = RetryOnConflictWithContext(ctx, wait.Backoff{Factor: 1.0, Steps: 100, Duration: time.Hour}, func(context.Context) error {
		i++
		return conflictErr
	})
	if !stderrors.Is(err, context.DeadlineExceeded) {
		t.Errorf("expected context.DeadlineExceeded, got: %v", err)
	}
	if i < 1 {
		t.Errorf("expected at least one retry attempt, got: %d", i)
	}

	// max retries
	i = 0
	err = RetryOnConflictWithContext(context.Background(), opts, func(context.Context) error {
		i++
		return conflictErr
	})
	if !stderrors.Is(err, conflictErr) {
		t.Errorf("expected conflictErr, got: %v", err)
	}
	if i != opts.Steps {
		t.Errorf("expected %d attempts, got: %d", opts.Steps, i)
	}

	// returns immediately on success
	i = 0
	err = RetryOnConflictWithContext(context.Background(), opts, func(context.Context) error {
		i++
		return nil
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if i != 1 {
		t.Errorf("expected 1 attempt, got: %d", i)
	}

	// returns immediately on non-retriable error
	testErr := fmt.Errorf("some other error")
	i = 0
	err = RetryOnConflictWithContext(context.Background(), opts, func(context.Context) error {
		i++
		return testErr
	})
	if !stderrors.Is(err, testErr) {
		t.Errorf("expected testErr, got: %v", err)
	}
	if i != 1 {
		t.Errorf("expected 1 attempt, got: %d", i)
	}

	// retries until success
	i = 0
	err = RetryOnConflictWithContext(context.Background(), opts, func(context.Context) error {
		if i < 2 {
			i++
			return conflictErr
		}
		return nil
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if i != 2 {
		t.Errorf("expected 2 attempts, got: %d", i)
	}

	// context is passed to callback
	ctx, cancel = context.WithTimeout(context.Background(), time.Hour)
	defer cancel()
	var receivedCtx context.Context
	_ = RetryOnConflictWithContext(ctx, opts, func(ctx context.Context) error {
		receivedCtx = ctx
		return nil
	})
	if receivedCtx != ctx {
		t.Error("context was not passed to callback")
	}
}

func TestOnErrorWithContext(t *testing.T) {
	opts := wait.Backoff{Factor: 1.0, Steps: 3}
	testErr := fmt.Errorf("test error")
	retriableErr := fmt.Errorf("retriable error")
	retriable := func(err error) bool {
		return stderrors.Is(err, retriableErr)
	}

	// context cancelled before first attempt - returns context.Canceled immediately
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := OnErrorWithContext(ctx, opts, retriable, func(context.Context) error {
		return testErr
	})
	if !stderrors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got: %v", err)
	}

	// context cancelled during retry - returns context.Canceled
	ctx, cancel = context.WithCancel(context.Background())
	i := 0
	err = OnErrorWithContext(ctx, wait.Backoff{Factor: 1.0, Steps: 100, Duration: 50 * time.Millisecond}, retriable, func(context.Context) error {
		i++
		if i == 2 {
			cancel()
		}
		return retriableErr
	})
	if !stderrors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got: %v", err)
	}

	// context timeout - returns context.DeadlineExceeded
	ctx, cancel = context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	i = 0
	err = OnErrorWithContext(ctx, wait.Backoff{Factor: 1.0, Steps: 100, Duration: time.Hour}, retriable, func(context.Context) error {
		i++
		return retriableErr
	})
	if !stderrors.Is(err, context.DeadlineExceeded) {
		t.Errorf("expected context.DeadlineExceeded, got: %v", err)
	}
	if i < 1 {
		t.Errorf("expected at least one retry attempt, got: %d", i)
	}

	// max retries
	i = 0
	err = OnErrorWithContext(context.Background(), opts, retriable, func(context.Context) error {
		i++
		return retriableErr
	})
	if !stderrors.Is(err, retriableErr) {
		t.Errorf("expected retriableErr, got: %v", err)
	}
	if i != opts.Steps {
		t.Errorf("expected %d attempts, got: %d", opts.Steps, i)
	}

	// returns immediately on success
	i = 0
	err = OnErrorWithContext(context.Background(), opts, retriable, func(context.Context) error {
		i++
		return nil
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if i != 1 {
		t.Errorf("expected 1 attempt, got: %d", i)
	}

	// returns immediately on non-retriable error
	i = 0
	err = OnErrorWithContext(context.Background(), opts, retriable, func(context.Context) error {
		i++
		return testErr
	})
	if !stderrors.Is(err, testErr) {
		t.Errorf("expected testErr, got: %v", err)
	}
	if i != 1 {
		t.Errorf("expected 1 attempt, got: %d", i)
	}

	// retries until success
	i = 0
	err = OnErrorWithContext(context.Background(), opts, retriable, func(context.Context) error {
		if i < 2 {
			i++
			return retriableErr
		}
		return nil
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if i != 2 {
		t.Errorf("expected 2 attempts, got: %d", i)
	}

	// context is passed to callback
	ctx, cancel = context.WithTimeout(context.Background(), time.Hour)
	defer cancel()
	var receivedCtx context.Context
	_ = OnErrorWithContext(ctx, opts, retriable, func(ctx context.Context) error {
		receivedCtx = ctx
		return nil
	})
	if receivedCtx != ctx {
		t.Error("context was not passed to callback")
	}
}
