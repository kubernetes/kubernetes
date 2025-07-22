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

package wait

import (
	"context"
	"time"
)

// PollUntilContextCancel tries a condition func until it returns true, an error, or the context
// is cancelled or hits a deadline. condition will be invoked after the first interval if the
// context is not cancelled first. The returned error will be from ctx.Err(), the condition's
// err return value, or nil. If invoking condition takes longer than interval the next condition
// will be invoked immediately. When using very short intervals, condition may be invoked multiple
// times before a context cancellation is detected. If immediate is true, condition will be
// invoked before waiting and guarantees that condition is invoked at least once, regardless of
// whether the context has been cancelled.
func PollUntilContextCancel(ctx context.Context, interval time.Duration, immediate bool, condition ConditionWithContextFunc) error {
	return loopConditionUntilContext(ctx, Backoff{Duration: interval}.Timer(), immediate, false, condition)
}

// PollUntilContextTimeout will terminate polling after timeout duration by setting a context
// timeout. This is provided as a convenience function for callers not currently executing under
// a deadline and is equivalent to:
//
//	deadlineCtx, deadlineCancel := context.WithTimeout(ctx, timeout)
//	err := PollUntilContextCancel(deadlineCtx, interval, immediate, condition)
//
// The deadline context will be cancelled if the Poll succeeds before the timeout, simplifying
// inline usage. All other behavior is identical to PollUntilContextCancel.
func PollUntilContextTimeout(ctx context.Context, interval, timeout time.Duration, immediate bool, condition ConditionWithContextFunc) error {
	deadlineCtx, deadlineCancel := context.WithTimeout(ctx, timeout)
	defer deadlineCancel()
	return loopConditionUntilContext(deadlineCtx, Backoff{Duration: interval}.Timer(), immediate, false, condition)
}

// Poll tries a condition func until it returns true, an error, or the timeout
// is reached.
//
// Poll always waits the interval before the run of 'condition'.
// 'condition' will always be invoked at least once.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
//
// If you want to Poll something forever, see PollInfinite.
//
// Deprecated: This method does not return errors from context, use PollUntilContextTimeout.
// Note that the new method will no longer return ErrWaitTimeout and instead return errors
// defined by the context package. Will be removed in a future release.
func Poll(interval, timeout time.Duration, condition ConditionFunc) error {
	return PollWithContext(context.Background(), interval, timeout, condition.WithContext())
}

// PollWithContext tries a condition func until it returns true, an error,
// or when the context expires or the timeout is reached, whichever
// happens first.
//
// PollWithContext always waits the interval before the run of 'condition'.
// 'condition' will always be invoked at least once.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
//
// If you want to Poll something forever, see PollInfinite.
//
// Deprecated: This method does not return errors from context, use PollUntilContextTimeout.
// Note that the new method will no longer return ErrWaitTimeout and instead return errors
// defined by the context package. Will be removed in a future release.
func PollWithContext(ctx context.Context, interval, timeout time.Duration, condition ConditionWithContextFunc) error {
	return poll(ctx, false, poller(interval, timeout), condition)
}

// PollUntil tries a condition func until it returns true, an error or stopCh is
// closed.
//
// PollUntil always waits interval before the first run of 'condition'.
// 'condition' will always be invoked at least once.
//
// Deprecated: This method does not return errors from context, use PollUntilContextCancel.
// Note that the new method will no longer return ErrWaitTimeout and instead return errors
// defined by the context package. Will be removed in a future release.
func PollUntil(interval time.Duration, condition ConditionFunc, stopCh <-chan struct{}) error {
	return PollUntilWithContext(ContextForChannel(stopCh), interval, condition.WithContext())
}

// PollUntilWithContext tries a condition func until it returns true,
// an error or the specified context is cancelled or expired.
//
// PollUntilWithContext always waits interval before the first run of 'condition'.
// 'condition' will always be invoked at least once.
//
// Deprecated: This method does not return errors from context, use PollUntilContextCancel.
// Note that the new method will no longer return ErrWaitTimeout and instead return errors
// defined by the context package. Will be removed in a future release.
func PollUntilWithContext(ctx context.Context, interval time.Duration, condition ConditionWithContextFunc) error {
	return poll(ctx, false, poller(interval, 0), condition)
}

// PollInfinite tries a condition func until it returns true or an error
//
// PollInfinite always waits the interval before the run of 'condition'.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
//
// Deprecated: This method does not return errors from context, use PollUntilContextCancel.
// Note that the new method will no longer return ErrWaitTimeout and instead return errors
// defined by the context package. Will be removed in a future release.
func PollInfinite(interval time.Duration, condition ConditionFunc) error {
	return PollInfiniteWithContext(context.Background(), interval, condition.WithContext())
}

// PollInfiniteWithContext tries a condition func until it returns true or an error
//
// PollInfiniteWithContext always waits the interval before the run of 'condition'.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
//
// Deprecated: This method does not return errors from context, use PollUntilContextCancel.
// Note that the new method will no longer return ErrWaitTimeout and instead return errors
// defined by the context package. Will be removed in a future release.
func PollInfiniteWithContext(ctx context.Context, interval time.Duration, condition ConditionWithContextFunc) error {
	return poll(ctx, false, poller(interval, 0), condition)
}

// PollImmediate tries a condition func until it returns true, an error, or the timeout
// is reached.
//
// PollImmediate always checks 'condition' before waiting for the interval. 'condition'
// will always be invoked at least once.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
//
// If you want to immediately Poll something forever, see PollImmediateInfinite.
//
// Deprecated: This method does not return errors from context, use PollUntilContextTimeout.
// Note that the new method will no longer return ErrWaitTimeout and instead return errors
// defined by the context package. Will be removed in a future release.
func PollImmediate(interval, timeout time.Duration, condition ConditionFunc) error {
	return PollImmediateWithContext(context.Background(), interval, timeout, condition.WithContext())
}

// PollImmediateWithContext tries a condition func until it returns true, an error,
// or the timeout is reached or the specified context expires, whichever happens first.
//
// PollImmediateWithContext always checks 'condition' before waiting for the interval.
// 'condition' will always be invoked at least once.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
//
// If you want to immediately Poll something forever, see PollImmediateInfinite.
//
// Deprecated: This method does not return errors from context, use PollUntilContextTimeout.
// Note that the new method will no longer return ErrWaitTimeout and instead return errors
// defined by the context package. Will be removed in a future release.
func PollImmediateWithContext(ctx context.Context, interval, timeout time.Duration, condition ConditionWithContextFunc) error {
	return poll(ctx, true, poller(interval, timeout), condition)
}

// PollImmediateUntil tries a condition func until it returns true, an error or stopCh is closed.
//
// PollImmediateUntil runs the 'condition' before waiting for the interval.
// 'condition' will always be invoked at least once.
//
// Deprecated: This method does not return errors from context, use PollUntilContextCancel.
// Note that the new method will no longer return ErrWaitTimeout and instead return errors
// defined by the context package. Will be removed in a future release.
func PollImmediateUntil(interval time.Duration, condition ConditionFunc, stopCh <-chan struct{}) error {
	return PollImmediateUntilWithContext(ContextForChannel(stopCh), interval, condition.WithContext())
}

// PollImmediateUntilWithContext tries a condition func until it returns true,
// an error or the specified context is cancelled or expired.
//
// PollImmediateUntilWithContext runs the 'condition' before waiting for the interval.
// 'condition' will always be invoked at least once.
//
// Deprecated: This method does not return errors from context, use PollUntilContextCancel.
// Note that the new method will no longer return ErrWaitTimeout and instead return errors
// defined by the context package. Will be removed in a future release.
func PollImmediateUntilWithContext(ctx context.Context, interval time.Duration, condition ConditionWithContextFunc) error {
	return poll(ctx, true, poller(interval, 0), condition)
}

// PollImmediateInfinite tries a condition func until it returns true or an error
//
// PollImmediateInfinite runs the 'condition' before waiting for the interval.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
//
// Deprecated: This method does not return errors from context, use PollUntilContextCancel.
// Note that the new method will no longer return ErrWaitTimeout and instead return errors
// defined by the context package. Will be removed in a future release.
func PollImmediateInfinite(interval time.Duration, condition ConditionFunc) error {
	return PollImmediateInfiniteWithContext(context.Background(), interval, condition.WithContext())
}

// PollImmediateInfiniteWithContext tries a condition func until it returns true
// or an error or the specified context gets cancelled or expired.
//
// PollImmediateInfiniteWithContext runs the 'condition' before waiting for the interval.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
//
// Deprecated: This method does not return errors from context, use PollUntilContextCancel.
// Note that the new method will no longer return ErrWaitTimeout and instead return errors
// defined by the context package. Will be removed in a future release.
func PollImmediateInfiniteWithContext(ctx context.Context, interval time.Duration, condition ConditionWithContextFunc) error {
	return poll(ctx, true, poller(interval, 0), condition)
}

// Internally used, each of the public 'Poll*' function defined in this
// package should invoke this internal function with appropriate parameters.
// ctx: the context specified by the caller, for infinite polling pass
// a context that never gets cancelled or expired.
// immediate: if true, the 'condition' will be invoked before waiting for the interval,
// in this case 'condition' will always be invoked at least once.
// wait: user specified WaitFunc function that controls at what interval the condition
// function should be invoked periodically and whether it is bound by a timeout.
// condition: user specified ConditionWithContextFunc function.
//
// Deprecated: will be removed in favor of loopConditionUntilContext.
func poll(ctx context.Context, immediate bool, wait waitWithContextFunc, condition ConditionWithContextFunc) error {
	if immediate {
		done, err := runConditionWithCrashProtectionWithContext(ctx, condition)
		if err != nil {
			return err
		}
		if done {
			return nil
		}
	}

	select {
	case <-ctx.Done():
		// returning ctx.Err() will break backward compatibility, use new PollUntilContext*
		// methods instead
		return ErrWaitTimeout
	default:
		return waitForWithContext(ctx, wait, condition)
	}
}

// poller returns a WaitFunc that will send to the channel every interval until
// timeout has elapsed and then closes the channel.
//
// Over very short intervals you may receive no ticks before the channel is
// closed. A timeout of 0 is interpreted as an infinity, and in such a case
// it would be the caller's responsibility to close the done channel.
// Failure to do so would result in a leaked goroutine.
//
// Output ticks are not buffered. If the channel is not ready to receive an
// item, the tick is skipped.
//
// Deprecated: Will be removed in a future release.
func poller(interval, timeout time.Duration) waitWithContextFunc {
	return waitWithContextFunc(func(ctx context.Context) <-chan struct{} {
		ch := make(chan struct{})

		go func() {
			defer close(ch)

			tick := time.NewTicker(interval)
			defer tick.Stop()

			var after <-chan time.Time
			if timeout != 0 {
				// time.After is more convenient, but it
				// potentially leaves timers around much longer
				// than necessary if we exit early.
				timer := time.NewTimer(timeout)
				after = timer.C
				defer timer.Stop()
			}

			for {
				select {
				case <-tick.C:
					// If the consumer isn't ready for this signal drop it and
					// check the other channels.
					select {
					case ch <- struct{}{}:
					default:
					}
				case <-after:
					return
				case <-ctx.Done():
					return
				}
			}
		}()

		return ch
	})
}
