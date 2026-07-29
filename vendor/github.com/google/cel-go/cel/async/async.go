// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package async provides helpers for configuring and executing asynchronous CEL functions,
// including drain strategies, retry, timeout, concurrency limiting, and caching wrappers.
package async

import (
	"context"
	"errors"
	"time"

	"github.com/google/cel-go/common/decls"
	"github.com/google/cel-go/common/functions"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter"
)

// Call describes a pending or completed asynchronous function call.
// This interface exposes a safe, read-only view of the internal interpreter state.
type Call = interpreter.AsyncCall

// Observer provides callbacks for monitoring the lifecycle of asynchronous function calls.
//
// Implementations must be safe for concurrent use: the start and finish callbacks run on different
// goroutines, and finish callbacks for distinct calls may run concurrently. See
// interpreter.AsyncObserver for details.
type Observer = interpreter.AsyncObserver

// BlockingOp is a blocking asynchronous function operation.
type BlockingOp = functions.BlockingAsyncOp

// DrainAction dictates what ConcurrentEval should do after inspecting completions.
type DrainAction struct {
	// Reevaluate indicates that the AST should be re-evaluated immediately.
	// If true, WaitDuration is ignored.
	Reevaluate bool
	// WaitDuration indicates how long the evaluator should wait for additional
	// completions before deciding to re-evaluate. A duration of 0 means wait
	// indefinitely (block on the next completion).
	WaitDuration time.Duration
}

// DrainStrategy controls when ConcurrentEval re-evaluates after async completions.
//
// The evaluator consults the strategy each time a completion is received.
type DrainStrategy interface {
	// NextAction evaluates the current state of asynchronous evaluation and
	// determines the next step.
	//
	// - completed: The set of completions accumulated in the current batch.
	// - active: The number of async calls currently launched but unresolved.
	NextAction(completed []Call, active int) DrainAction
}

// DrainNone returns a strategy that re-evaluates after every single completion.
// This is the default strategy.
func DrainNone() DrainStrategy {
	return drainNone{}
}

type drainNone struct{}

func (drainNone) NextAction(completed []Call, active int) DrainAction {
	return DrainAction{Reevaluate: active == 0 || len(completed) > 0}
}

// DrainReady returns a strategy that waits for a short duration after the first
// completion to batch any other functions that complete at roughly the same time.
func DrainReady(debounce time.Duration) DrainStrategy {
	return drainReady{debounce: debounce}
}

type drainReady struct {
	debounce time.Duration
}

func (d drainReady) NextAction(completed []Call, active int) DrainAction {
	if active == 0 {
		return DrainAction{Reevaluate: true} // Nothing left to wait for
	}
	if len(completed) == 0 {
		return DrainAction{Reevaluate: false, WaitDuration: 0} // Wait indefinitely for first
	}
	return DrainAction{Reevaluate: false, WaitDuration: d.debounce} // Wait for debounce period
}

// DrainAll returns a strategy that waits for all currently pending calls to
// complete before re-evaluating.
//
// Note: This strategy is optimal for independent async calls, but will over-wait
// if some calls depend on the results of others.
func DrainAll() DrainStrategy {
	return drainAll{}
}

type drainAll struct{}

func (drainAll) NextAction(completed []Call, active int) DrainAction {
	return DrainAction{Reevaluate: active == 0}
}

// Timeout wraps a BlockingAsyncOp with a per-call timeout.
//
// The timeout is enforced even when the wrapped function ignores its context: the function runs on
// its own goroutine and Timeout selects on the deadline, returning a timeout error when it
// fires. A function that ignores cancellation cannot be forcibly stopped (Go cannot kill a
// goroutine), so its goroutine continues running in the background until it returns on its own;
// only its result is abandoned. This is the recommended way to bound functions that may hang or
// are not under the caller's control. The extra goroutine is incurred only by Timeout-wrapped
// calls, not by async evaluation in general.
func Timeout(fn functions.BlockingAsyncOp, timeout time.Duration) functions.BlockingAsyncOp {
	return func(ctx context.Context, args ...ref.Val) ref.Val {
		tCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()
		resCh := make(chan ref.Val, 1)
		go func() { resCh <- fn(tCtx, args...) }()
		select {
		case res := <-resCh:
			return res
		case <-tCtx.Done():
			return types.NewErr("operation timed out after %v: %v", timeout, tCtx.Err())
		}
	}
}

// TimeoutBinding wraps a BlockingAsyncOp with a per-call timeout and returns an OverloadOpt.
func TimeoutBinding(fn functions.BlockingAsyncOp, timeout time.Duration) decls.OverloadOpt {
	return decls.AsyncBinding(Timeout(fn, timeout))
}

// RetryOption configures the behavior of RetryBinding.
type RetryOption func(*retryConfig)

type retryConfig struct {
	maxAttempts int
	backoff     time.Duration
}

// RetryAttempts sets the maximum number of attempts (including the first one).
func RetryAttempts(attempts int) RetryOption {
	return func(c *retryConfig) {
		c.maxAttempts = attempts
	}
}

// RetryBackoff sets the fixed backoff duration between attempts.
func RetryBackoff(backoff time.Duration) RetryOption {
	return func(c *retryConfig) {
		c.backoff = backoff
	}
}

// RetryableError is an interface that errors can implement to signal whether they are retryable.
type RetryableError interface {
	error
	IsRetryable() bool
}

// Retry wraps a BlockingAsyncOp with a retry policy.
// It will retry the operation if it returns a types.Err that wraps a RetryableError returning true for IsRetryable.
func Retry(fn functions.BlockingAsyncOp, opts ...RetryOption) functions.BlockingAsyncOp {
	config := &retryConfig{
		maxAttempts: 3,
		backoff:     100 * time.Millisecond,
	}
	for _, opt := range opts {
		opt(config)
	}

	return func(ctx context.Context, args ...ref.Val) ref.Val {
		var lastErr ref.Val
		var backoff *time.Timer
		defer func() {
			if backoff != nil {
				backoff.Stop()
			}
		}()
		for i := 0; i < config.maxAttempts; i++ {
			if i > 0 {
				// Reuse a single timer across attempts and stop it on cancellation so the
				// pending timer is not left to fire after the call returns.
				if backoff == nil {
					backoff = time.NewTimer(config.backoff)
				} else {
					backoff.Reset(config.backoff)
				}
				select {
				case <-backoff.C:
				case <-ctx.Done():
					backoff.Stop()
					return types.NewErr("operation cancelled during retry: %v", ctx.Err())
				}
			}

			res := fn(ctx, args...)
			if !types.IsError(res) {
				return res
			}

			err := res.(*types.Err)
			lastErr = res

			if !isRetryable(err) {
				return res
			}
		}
		return lastErr
	}
}

// RetryBinding wraps a BlockingAsyncOp with a retry policy and returns an OverloadOpt.
func RetryBinding(fn functions.BlockingAsyncOp, opts ...RetryOption) decls.OverloadOpt {
	return decls.AsyncBinding(Retry(fn, opts...))
}

func isRetryable(err *types.Err) bool {
	var re RetryableError
	if errors.As(err, &re) {
		return re.IsRetryable()
	}
	return false
}
