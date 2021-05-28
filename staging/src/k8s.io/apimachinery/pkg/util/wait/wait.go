/*
Copyright 2014 The Kubernetes Authors.

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
	"errors"
	"math/rand"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/utils/clock"
)

// For any test of the style:
//
//	...
//	<- time.After(timeout):
//	   t.Errorf("Timed out")
//
// The value for timeout should effectively be "forever." Obviously we don't want our tests to truly lock up forever, but 30s
// is long enough that it is effectively forever for the things that can slow down a run on a heavily contended machine
// (GC, seeks, etc), but not so long as to make a developer ctrl-c a test run if they do happen to break that test.
var ForeverTestTimeout = time.Second * 30

// NeverStop may be passed to Until to make it never stop.
var NeverStop <-chan struct{} = make(chan struct{})

// Group allows to start a group of goroutines and wait for their completion.
type Group struct {
	wg sync.WaitGroup
}

func (g *Group) Wait() {
	g.wg.Wait()
}

// StartWithChannel starts f in a new goroutine in the group.
// stopCh is passed to f as an argument. f should stop when stopCh is available.
func (g *Group) StartWithChannel(stopCh <-chan struct{}, f func(stopCh <-chan struct{})) {
	g.Start(func() {
		f(stopCh)
	})
}

// StartWithContext starts f in a new goroutine in the group.
// ctx is passed to f as an argument. f should stop when ctx.Done() is available.
func (g *Group) StartWithContext(ctx context.Context, f func(context.Context)) {
	g.Start(func() {
		f(ctx)
	})
}

// Start starts f in a new goroutine in the group.
func (g *Group) Start(f func()) {
	g.wg.Add(1)
	go func() {
		defer g.wg.Done()
		f()
	}()
}

// Forever calls f every period for ever.
//
// Forever is syntactic sugar on top of Until.
func Forever(f func(), period time.Duration) {
	Until(f, period, NeverStop)
}

// Until loops until stop channel is closed, running f every period.
//
// Until is syntactic sugar on top of JitterUntil with zero jitter factor and
// with sliding = true (which means the timer for period starts after the f
// completes).
func Until(f func(), period time.Duration, stopCh <-chan struct{}) {
	JitterUntil(f, period, 0.0, true, stopCh)
}

// UntilWithContext loops until context is done, running f every period.
//
// UntilWithContext is syntactic sugar on top of JitterUntilWithContext
// with zero jitter factor and with sliding = true (which means the timer
// for period starts after the f completes).
func UntilWithContext(ctx context.Context, f func(context.Context), period time.Duration) {
	JitterUntilWithContext(ctx, f, period, 0.0, true)
}

// NonSlidingUntil loops until stop channel is closed, running f every
// period.
//
// NonSlidingUntil is syntactic sugar on top of JitterUntil with zero jitter
// factor, with sliding = false (meaning the timer for period starts at the same
// time as the function starts).
func NonSlidingUntil(f func(), period time.Duration, stopCh <-chan struct{}) {
	JitterUntil(f, period, 0.0, false, stopCh)
}

// NonSlidingUntilWithContext loops until context is done, running f every
// period.
//
// NonSlidingUntilWithContext is syntactic sugar on top of JitterUntilWithContext
// with zero jitter factor, with sliding = false (meaning the timer for period
// starts at the same time as the function starts).
func NonSlidingUntilWithContext(ctx context.Context, f func(context.Context), period time.Duration) {
	JitterUntilWithContext(ctx, f, period, 0.0, false)
}

// JitterUntil loops until stop channel is closed, running f every period.
//
// If jitterFactor is positive, the period is jittered before every run of f.
// If jitterFactor is not positive, the period is unchanged and not jittered.
//
// If sliding is true, the period is computed after f runs. If it is false then
// period includes the runtime for f.
//
// Close stopCh to stop. f may not be invoked if stop channel is already
// closed. Pass NeverStop to if you don't want it stop.
func JitterUntil(f func(), period time.Duration, jitterFactor float64, sliding bool, stopCh <-chan struct{}) {
	b := Backoff{Duration: period, Jitter: jitterFactor}
	BackoffUntil(f, internalClock.NewTimer, b.Step, sliding, stopCh)
}

var (
	// RealTimer can be passed to methods that need a TimerFunc.
	RealTimer = clock.RealClock{}.NewTimer
)

var (
	// internalClock is used for test injection of clocks
	internalClock = clock.RealClock{}
	// internalNewTicker is used for injecting behavior into tests
	// Deprecated: Will be removed when poller() is removed.
	internalNewTicker func(time.Duration) *time.Ticker = time.NewTicker
	// internalNewTimer is used for injecting behavior into tests
	// Deprecated: Will be removed when poller() is removed.
	internalNewTimer func(time.Duration) *time.Timer = time.NewTimer
)

// TimerFunc allows the caller to inject a timer for testing. Most callers should use RealTimer.
type TimerFunc func(time.Duration) clock.Timer

// BackoffUntil loops until stop channel is closed, run f every duration given by delayFn.
// An appropriately delayFn is provided by the Backoff.Step method. If sliding is true, the
// period is computed after f runs. If it is false then period includes the runtime for f.
func BackoffUntil(f func(), timerFn TimerFunc, delayFn DelayFunc, sliding bool, stopCh <-chan struct{}) {
	ctx, cancel := ContextForChannel(stopCh)
	defer cancel()
	loopConditionUntilContext(ctx, timerFn, delayFn, true, sliding, func(_ context.Context) (bool, error) {
		f()
		return false, nil
	})
}

// JitterUntilWithContext loops until context is done, running f every period.
//
// If jitterFactor is positive, the period is jittered before every run of f.
// If jitterFactor is not positive, the period is unchanged and not jittered.
//
// If sliding is true, the period is computed after f runs. If it is false then
// period includes the runtime for f.
//
// Cancel context to stop. f may not be invoked if context is already expired.
func JitterUntilWithContext(ctx context.Context, f func(context.Context), period time.Duration, jitterFactor float64, sliding bool) {
	b := Backoff{Duration: period, Jitter: jitterFactor}
	loopConditionUntilContext(ctx, internalClock.NewTimer, b.Step, true, sliding, func(ctx context.Context) (bool, error) {
		f(ctx)
		return false, nil
	})
}

// Jitter returns a time.Duration between duration and duration + maxFactor *
// duration.
//
// This allows clients to avoid converging on periodic behavior. If maxFactor
// is 0.0, a suggested default value will be chosen.
func Jitter(duration time.Duration, maxFactor float64) time.Duration {
	if maxFactor <= 0.0 {
		maxFactor = 1.0
	}
	wait := duration + time.Duration(rand.Float64()*maxFactor*float64(duration))
	return wait
}

// ErrWaitTimeout is returned when the condition was not satisfied in time.
//
// Deprecated: This type will be made private in 1.28 in favor of WaitEndedEarly
// for checking errors or WrapEndedEarly(err) for returning a typed error.
var ErrWaitTimeout = ErrorEndedEarly(errors.New("timed out waiting for the condition"))

// EndedEarly returns true if the error returned by Poll or ExponentialBackoff
// methods indicates the condition was not successful within the method execution.
// Callers should use this method instead of comparing the error value directly to
// ErrWaitTimeout, as methods that cancel a context may not return that error.
//
// Instead of:
//
//	err := wait.Poll(...)
//	if err == wait.ErrWaitTimeout {
//	    log.Infof("Wait for operation exceeded")
//	} else ...
//
// Use:
//
//	err := wait.Poll(...)
//	if wait.EndedEarly(err) {
//	    log.Infof("Wait for operation exceeded")
//	} else ...
func EndedEarly(err error) bool {
	switch {
	case errors.Is(err, errErrWaitTimeout),
		errors.Is(err, context.Canceled),
		errors.Is(err, context.DeadlineExceeded):
		return true
	default:
		return false
	}
}

// errEndedEarly
type errEndedEarly struct {
	cause error
}

// ErrorEndedEarly returns an error that indicates the wait was ended
// early for a given reason. If no cause is provided a generic error
// will be used but callers are encouraged to provide a real cause for
// clarity in debugging.
func ErrorEndedEarly(cause error) error {
	switch cause.(type) {
	case errEndedEarly:
		// no need to wrap twice since errEndedEarly is only needed
		// once in a chain
		return cause
	default:
		return errEndedEarly{cause}
	}
}

// errErrWaitTimeout is the private version of the previous ErrWaitTimeout
// and is private to prevent direct comparison. Use ErrorEndedEarly(...)
// instead.
var errErrWaitTimeout = errEndedEarly{}

func (e errEndedEarly) Unwrap() error        { return e.cause }
func (e errEndedEarly) Is(target error) bool { return target == errErrWaitTimeout }
func (e errEndedEarly) Error() string {
	if e.cause == nil {
		// returns the same error message as before
		return "timed out waiting for the condition"
	}
	return e.cause.Error()
}

// ConditionFunc returns true if the condition is satisfied, or an error
// if the loop should be aborted.
type ConditionFunc func() (done bool, err error)

// ConditionWithContextFunc returns true if the condition is satisfied, or an error
// if the loop should be aborted.
//
// The caller passes along a context that can be used by the condition function.
type ConditionWithContextFunc func(context.Context) (done bool, err error)

// WithContext converts a ConditionFunc into a ConditionWithContextFunc
func (cf ConditionFunc) WithContext() ConditionWithContextFunc {
	return func(context.Context) (done bool, err error) {
		return cf()
	}
}

// ContextForChannel derives a child context from a parent channel.
//
// The derived context's Done channel is closed when the returned cancel function
// is called or when the parent channel is closed, whichever happens first.
//
// Note the caller must *always* call the CancelFunc, otherwise resources may be leaked.
func ContextForChannel(parentCh <-chan struct{}) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(context.Background())
	select {
	case <-parentCh:
		// already closed, cancel now and no goroutine necessary
		cancel()
	default:
		go func() {
			select {
			case <-parentCh:
				cancel()
			case <-ctx.Done():
			}
		}()
	}
	return ctx, cancel
}

// runConditionWithCrashProtection runs a ConditionFunc with crash protection
func runConditionWithCrashProtection(condition ConditionFunc) (bool, error) {
	return runConditionWithCrashProtectionWithContext(context.TODO(), condition.WithContext())
}

// runConditionWithCrashProtectionWithContext runs a
// ConditionWithContextFunc with crash protection.
func runConditionWithCrashProtectionWithContext(ctx context.Context, condition ConditionWithContextFunc) (bool, error) {
	defer runtime.HandleCrash()
	return condition(ctx)
}

// DelayFunc returns the next time interval to wait.
type DelayFunc func() time.Duration

// Until takes an arbitrary delay function and runs until cancelled or the condition indicates exit. This
// offers all of the functionality of the methods in this package.
func (fn DelayFunc) Until(ctx context.Context, immediate, sliding bool, condition ConditionWithContextFunc) error {
	return loopConditionUntilContext(ctx, internalClock.NewTimer, fn, immediate, sliding, condition)
}

// Concurrent returns a version of this DelayFunc that is safe for use by multiple goroutines that
// wish to share a single delay timer.
func (fn DelayFunc) Concurrent() DelayFunc {
	var lock sync.Mutex
	return func() time.Duration {
		lock.Lock()
		defer lock.Unlock()
		return fn()
	}
}

// Backoff holds parameters applied to a Backoff function.
type Backoff struct {
	// The initial duration.
	Duration time.Duration
	// Duration is multiplied by factor each iteration, if factor is not zero
	// and the limits imposed by Steps and Cap have not been reached.
	// Should not be negative.
	// The jitter does not contribute to the updates to the duration parameter.
	Factor float64
	// The sleep at each iteration is the duration plus an additional
	// amount chosen uniformly at random from the interval between
	// zero and `jitter*duration`.
	Jitter float64
	// The remaining number of iterations in which the duration
	// parameter may change (but progress can be stopped earlier by
	// hitting the cap). If not positive, the duration is not
	// changed. Used for exponential backoff in combination with
	// Factor and Cap.
	Steps int
	// A limit on revised values of the duration parameter. If a
	// multiplication by the factor parameter would make the duration
	// exceed the cap then the duration is set to the cap and the
	// steps parameter is set to zero.
	Cap time.Duration
}

// Step returns an amount of time to sleep determined by the
// original Duration and Jitter. The backoff is mutated to update its
// Steps and Duration.
func (b *Backoff) Step() time.Duration {
	var nextDuration time.Duration
	nextDuration, b.Duration, b.Steps = delay(b.Steps, b.Duration, b.Cap, b.Factor, b.Jitter)
	return nextDuration
}

// DelayFunc returns a function that will compute the next interval to
// wait given the arguments in b. It does not mutate the original backoff
// but the function is safe to use only from a single goroutine.
func (b Backoff) DelayFunc() DelayFunc {
	steps := b.Steps
	duration := b.Duration
	cap := b.Cap
	factor := b.Factor
	jitter := b.Jitter

	return func() time.Duration {
		var nextDuration time.Duration
		// jitter is applied per step and is not cumulative over multiple steps
		nextDuration, duration, steps = delay(steps, duration, cap, factor, jitter)
		return nextDuration
	}
}

// delay implements the core delay algorithm used in this package.
func delay(steps int, duration, cap time.Duration, factor, jitter float64) (_ time.Duration, next time.Duration, nextSteps int) {
	// when steps is non-positive, do not alter the base duration
	if steps < 1 {
		if jitter > 0 {
			return Jitter(duration, jitter), duration, 0
		}
		return duration, duration, 0
	}
	steps--

	// calculate the next step's interval
	if factor != 0 {
		next = time.Duration(float64(duration) * factor)
		if cap > 0 && next > cap {
			next = cap
			steps = 0
		}
	} else {
		next = duration
	}

	// add jitter for this step
	if jitter > 0 {
		duration = Jitter(duration, jitter)
	}

	return duration, next, steps

}

// StepWithReset returns a DelayFunc that will return the appropriate next interval to
// wait. Every resetInterval the backoff parameters are reset to their initial state.
// This method is safe to invoke from multiple goroutines.
func (b Backoff) StepWithReset(c clock.Clock, resetInterval time.Duration) DelayFunc {
	return (&backoffManager{
		backoff:        b,
		initialBackoff: b,
		resetInterval:  resetInterval,

		clock:     c,
		lastStart: c.Now(),
		timer:     nil,
	}).Step
}

type backoffManager struct {
	backoff        Backoff
	initialBackoff Backoff
	resetInterval  time.Duration

	clock clock.Clock

	lock      sync.Mutex
	lastStart time.Time
	timer     clock.Timer
}

// Step returns the expected next duration to wait.
func (b *backoffManager) Step() time.Duration {
	b.lock.Lock()
	defer b.lock.Unlock()

	switch {
	case b.resetInterval == 0:
		b.backoff = b.initialBackoff
	case b.clock.Now().Sub(b.lastStart) > b.resetInterval:
		b.backoff = b.initialBackoff
		b.lastStart = b.clock.Now()
	}
	return b.backoff.Step()
}

// ExponentialBackoff repeats a condition check with exponential backoff.
//
// It repeatedly checks the condition and then sleeps, using `backoff.Step()`
// to determine the length of the sleep and adjust Duration and Steps.
// Stops and returns as soon as:
// 1. the condition check returns true or an error,
// 2. `backoff.Steps` checks of the condition have been done, or
// 3. a sleep truncated by the cap on duration has been completed.
// In case (1) the returned error is what the condition function returned.
// In all other cases, ErrWaitTimeout is returned.
//
// Since backoffs are often subject to cancellation, we recommend using
// ExponentialBackoffWithContext and passing a context to the method.
func ExponentialBackoff(backoff Backoff, condition ConditionFunc) error {
	return ExponentialBackoffWithContext(context.Background(), backoff, condition.WithContext())
}

// ExponentialBackoffWithContext repeats a condition check with exponential backoff.
// It immediately returns an error if the condition returns an error, the context is cancelled
// or hits the deadline, or if the maximum attempts defined in backoff is exceeded (ErrWaitTimeout).
// If an error is returned by the condition the backoff stops immediately. The condition will
// never be invoked more than backoff.Steps times.
func ExponentialBackoffWithContext(ctx context.Context, backoff Backoff, condition ConditionWithContextFunc) error {
	steps := backoff.Steps
	return loopConditionUntilContext(ctx, internalClock.NewTimer, backoff.DelayFunc(), true, true, func(ctx context.Context) (bool, error) {
		if steps < 1 {
			return true, ErrWaitTimeout
		}
		ok, err := condition(ctx)
		if err != nil || ok {
			return ok, err
		}
		if steps == 1 {
			return false, ErrWaitTimeout
		}
		steps--
		return false, nil
	})
}

// PollUntilContextCancel tries a condition func until it returns true, an error, or the context
// is cancelled or hits a deadline. condition will be invoked after the first interval if the
// context is not cancelled first. The returned error will be from ctx.Err(), the condition's
// err return value, or nil. If invoking condition takes longer than interval the next condition
// will be invoked immediately. When using very short intervals, condition may be invoked multiple
// times before a context cancellation is detected. If immediate is true, condition will be
// invoked before waiting and guarantees that condition is invoked at least once, regardless of
// whether the context has been cancelled.
func PollUntilContextCancel(ctx context.Context, interval time.Duration, immediate bool, condition ConditionWithContextFunc) error {
	return loopConditionUntilContext(ctx, internalClock.NewTimer, Backoff{Duration: interval}.DelayFunc(), immediate, false, condition)
}

// PollUntilContextTimeout will terminate polling after timeout duration by setting a context
// timeout. This is provided as a convenience function for callers not currently executing under
// a deadline and is equivalent to:
//
//	deadlineCtx, deadlineCancel := context.WithTimeout(ctx, timeout)
//	err := PollUntilContextCancel(ctx, interval, immediate, condition)
//
// The deadline context will be cancelled if the Poll succeeds before the timeout, simplifying
// inline usage. All other behavior is identical to PollWithContextTimeout.
func PollUntilContextTimeout(ctx context.Context, interval, timeout time.Duration, immediate bool, condition ConditionWithContextFunc) error {
	deadlineCtx, deadlineCancel := context.WithTimeout(ctx, timeout)
	defer deadlineCancel()
	return loopConditionUntilContext(deadlineCtx, internalClock.NewTimer, Backoff{Duration: interval}.DelayFunc(), immediate, false, condition)
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
// Deprecated: Use PollWithContextCancel with a deadline context. Note that
// the new method will no longer return ErrWaitTimeout and instead return errors
// defined by the context package. Will be removed in 1.28.
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
// Deprecated: This method does not return errors from context, use
// PollWithContextCancel with a deadline context. Note that the new method
// will no longer return ErrWaitTimeout and instead return errors defined by the
// context package. Will be removed in 1.28.
func PollWithContext(ctx context.Context, interval, timeout time.Duration, condition ConditionWithContextFunc) error {
	return poll(ctx, false, poller(interval, timeout), condition)
}

// PollUntil tries a condition func until it returns true, an error or stopCh is
// closed.
//
// PollUntil always waits interval before the first run of 'condition'.
// 'condition' will always be invoked at least once.
//
// Deprecated: Use PollWithContextCancel instead. Note that
// the new method will no longer return ErrWaitTimeout and instead return errors
// defined by the context package. Will be removed in 1.28.
func PollUntil(interval time.Duration, condition ConditionFunc, stopCh <-chan struct{}) error {
	ctx, cancel := ContextForChannel(stopCh)
	defer cancel()
	return PollUntilWithContext(ctx, interval, condition.WithContext())
}

// PollUntilWithContext tries a condition func until it returns true,
// an error or the specified context is cancelled or expired.
//
// PollUntilWithContext always waits interval before the first run of 'condition'.
// 'condition' will always be invoked at least once.
//
// Deprecated: This method does not return errors from context, use
// PollWithContextCancel. Note that the new method will no longer return ErrWaitTimeout
// and instead return errors defined by the context package. Will be removed in 1.28.
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
// Deprecated: Use PollWithContextCancel without a deadline. Note that
// the new method will no longer return ErrWaitTimeout and instead return errors
// defined by the context package. Will be removed in 1.28.
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
// Deprecated: This method does not return errors from context, use
// PollWithContextCancel without a deadline. Note that the new method will no longer return
// ErrWaitTimeout and instead return errors defined by the context package. Will be
// removed in 1.28.
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
// Deprecated: This method does not return errors from context, use
// PollImmediateWithContextCancel without a deadline. Note that the new method will no longer
// return ErrWaitTimeout and instead return errors defined by the context package. Will
// be removed in 1.28.
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
// Deprecated: This method does not return errors from context, use
// PollImmediateWithContextCancel without a deadline. Note that the new method will no longer
// return ErrWaitTimeout and instead return errors defined by the context package. Will
// be removed in 1.28.
func PollImmediateWithContext(ctx context.Context, interval, timeout time.Duration, condition ConditionWithContextFunc) error {
	return poll(ctx, true, poller(interval, timeout), condition)
}

// PollImmediateUntil tries a condition func until it returns true, an error or stopCh is closed.
//
// PollImmediateUntil runs the 'condition' before waiting for the interval.
// 'condition' will always be invoked at least once.
//
// Deprecated: This method does not return errors from context, use
// PollImmediateWithContextCancel without a deadline. Note that the new method will no longer
// return ErrWaitTimeout and instead return errors defined by the context package. Will
// be removed in 1.28.
func PollImmediateUntil(interval time.Duration, condition ConditionFunc, stopCh <-chan struct{}) error {
	ctx, cancel := ContextForChannel(stopCh)
	defer cancel()
	return PollImmediateUntilWithContext(ctx, interval, condition.WithContext())
}

// PollImmediateUntilWithContext tries a condition func until it returns true,
// an error or the specified context is cancelled or expired.
//
// PollImmediateUntilWithContext runs the 'condition' before waiting for the interval.
// 'condition' will always be invoked at least once.
//
// Deprecated: This method does not return errors from context, use
// PollImmediateWithContextCancel without a deadline. Note that the new method will no longer
// return ErrWaitTimeout and instead return errors defined by the context package. Will
// be removed in 1.28.
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
// Deprecated: This method does not return errors from context, use
// PollImmediateWithContextCancel without a deadline. Note that the new method will no longer
// return ErrWaitTimeout and instead return errors defined by the context package. Will
// be removed in 1.28.
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
// Deprecated: This method does not return errors from context, use
// PollImmediateWithContextCancel without a deadline. Note that the new method will no longer
// return ErrWaitTimeout and instead return errors defined by the context package. Will
// be removed in 1.28.
func PollImmediateInfiniteWithContext(ctx context.Context, interval time.Duration, condition ConditionWithContextFunc) error {
	return poll(ctx, true, poller(interval, 0), condition)
}

// poll invokes condition until it is satisfied, the context is cancelled, or an
// error occurs. It returns ErrWaitWithTimeout on ANY loop error (including context
// cancellation) unless returnContextErr is true. If immediate is true, the condition
// will be invoked before beginning the wait loop, otherwise there is no guarantee that
// condition will be invoked before returning. The wait function will be invoked between
// each execution of condition.
//
// Deprecated: Will be removed in 1.28.
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
		// returning ctx.Err() will break backward compatibility, use new Poll*ContextCancel
		// methods instead
		return ErrWaitTimeout
	default:
		return waitForWithContext(ctx, wait, condition)
	}
}

// waitFunc creates a channel that receives an item every time a test
// should be executed and is closed when the last test should be invoked.
//
// Deprecated: Will be removed in 1.28.
type waitFunc func(done <-chan struct{}) <-chan struct{}

// WithContext converts the WaitFunc to an equivalent WaitWithContextFunc
func (w waitFunc) WithContext() waitWithContextFunc {
	return func(ctx context.Context) <-chan struct{} {
		return w(ctx.Done())
	}
}

// waitWithContextFunc creates a channel that receives an item every time a test
// should be executed and is closed when the last test should be invoked.
//
// When the specified context gets cancelled or expires the function
// stops sending item and returns immediately.
//
// Deprecated: Will be removed in 1.28.
type waitWithContextFunc func(ctx context.Context) <-chan struct{}

// waitForWithContext continually checks 'fn' as driven by 'wait'.
//
// waitForWithContext gets a channel from 'wait()”, and then invokes 'fn'
// once for every value placed on the channel and once more when the
// channel is closed. If the channel is closed and 'fn'
// returns false without error, waitForWithContext returns ErrWaitTimeout.
//
// If 'fn' returns an error the loop ends and that error is returned. If
// 'fn' returns true the loop ends and nil is returned.
//
// context.Canceled will be returned if the ctx.Done() channel is closed
// without fn ever returning true.
//
// When the ctx.Done() channel is closed, because the golang `select` statement is
// "uniform pseudo-random", the `fn` might still run one or multiple times,
// though eventually `waitForWithContext` will return.
//
// Deprecated: Will be removed in 1.28.
func waitForWithContext(ctx context.Context, wait waitWithContextFunc, fn ConditionWithContextFunc) error {
	waitCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c := wait(waitCtx)
	for {
		select {
		case _, open := <-c:
			ok, err := runConditionWithCrashProtectionWithContext(ctx, fn)
			if err != nil {
				return err
			}
			if ok {
				return nil
			}
			if !open {
				return ErrWaitTimeout
			}
		case <-ctx.Done():
			// returning ctx.Err() will break backward compatibility, use new Poll*ContextCancel
			// methods instead
			return ErrWaitTimeout
		}
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
// Deprecated: Will be removed in 1.28.
func poller(interval, timeout time.Duration) waitWithContextFunc {
	return waitWithContextFunc(func(ctx context.Context) <-chan struct{} {
		ch := make(chan struct{})

		go func() {
			defer close(ch)

			tick := internalNewTicker(interval)
			defer tick.Stop()

			var after <-chan time.Time
			if timeout != 0 {
				// time.After is more convenient, but it
				// potentially leaves timers around much longer
				// than necessary if we exit early.
				timer := internalNewTimer(timeout)
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

// loopConditionUntilContext executes the provided condition at intervals defined by
// the provided delayFn until the provided context is cancelled, the condition returns
// true, or the condition returns an error. If sliding is true, the period is computed
// after condition runs. If it is false then period includes the runtime for condition.
// If immediate is false the first delay happens before any call to condition. Use timerFn to
// provide a test timer for verification. The returned error is the error returned by the
// last condition or the context error if the context was terminated.
//
// This is the common loop construct for all polling in the wait package.
func loopConditionUntilContext(ctx context.Context, timerFn TimerFunc, delayFn DelayFunc, immediate, sliding bool, condition ConditionWithContextFunc) error {
	var t clock.Timer
	defer func() {
		if t != nil {
			t.Stop()
		}
	}()

	doneCh := ctx.Done()

	// if we haven't requested immediate execution, delay once
	if !immediate {
		t = timerFn(delayFn())
		select {
		case <-doneCh:
			return ctx.Err()
		case <-t.C():
		}
	}

	for {
		// checking ctx.Err() is slightly faster than checking a select
		if err := ctx.Err(); err != nil {
			return err
		}

		var interval time.Duration
		if !sliding {
			interval = delayFn()
		}
		if ok, err := func() (bool, error) {
			defer runtime.HandleCrash()
			return condition(ctx)
		}(); err != nil || ok {
			return err
		}
		if sliding {
			interval = delayFn()
		}

		// no interval requested, continue immediately
		if interval == 0 {
			continue
		}

		if t == nil {
			t = timerFn(interval)
		} else {
			t.Reset(interval)
		}

		// NOTE: b/c there is no priority selection in golang
		// it is possible for this to race, meaning we could
		// trigger t.C and doneCh, and t.C select falls through.
		// In order to mitigate we re-check doneCh at the beginning
		// of every loop to guarantee at-most one extra execution
		// of condition.
		select {
		case <-doneCh:
			return ctx.Err()
		case <-t.C():
		}
	}
}
