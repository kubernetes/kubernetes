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
	"math"
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
	BackoffUntil(f, Backoff{Duration: period, Jitter: jitterFactor}.Timer(), sliding, stopCh)
}

var (
	// RealTimer can be passed to methods that need a clock.Timer.
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

// BackoffUntil loops until stop channel is closed, run f every duration given by delayFn.
// An appropriately delayFn is provided by the Backoff.Step method. If sliding is true, the
// period is computed after f runs. If it is false then period includes the runtime for f.
func BackoffUntil(f func(), timer Timer, sliding bool, stopCh <-chan struct{}) {
	loopConditionUntilContext(ContextForChannel(stopCh), timer, true, sliding, func(_ context.Context) (bool, error) {
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
	loopConditionUntilContext(ctx, Backoff{Duration: period, Jitter: jitterFactor}.Timer(), true, sliding, func(ctx context.Context) (bool, error) {
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

// ErrWaitTimeout is returned when the condition exited without success.
var ErrWaitTimeout = errors.New("timed out waiting for the condition")

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

// ContextForChannel provides a context that will be treated as cancelled
// when the provided parentCh is closed. The implementation returns
// context.Canceled for Err() if and only if the parentCh is closed.
func ContextForChannel(parentCh <-chan struct{}) context.Context {
	return channelContext{stopCh: parentCh}
}

var _ context.Context = channelContext{}

// channelContext will behave as if the context were cancelled when stopCh is
// closed.
type channelContext struct {
	stopCh <-chan struct{}
}

func (c channelContext) Done() <-chan struct{} { return c.stopCh }
func (c channelContext) Err() error {
	select {
	case <-c.stopCh:
		return context.Canceled
	default:
		return nil
	}
}
func (c channelContext) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c channelContext) Value(key any) any           { return nil }

// runConditionWithCrashProtectionWithContext runs a
// ConditionWithContextFunc with crash protection.
//
// Deprecated: Will be removed when the legacy polling methods are removed.
func runConditionWithCrashProtectionWithContext(ctx context.Context, condition ConditionWithContextFunc) (bool, error) {
	defer runtime.HandleCrash()
	return condition(ctx)
}

// DelayFunc returns the next time interval to wait.
type DelayFunc func() time.Duration

// Timer takes an arbitrary delay function and returns a timer that can handle arbitrary interval changes.
// Use Backoff{...}.Timer() for simple delays and more efficient timers.
func (fn DelayFunc) Timer(c clock.Clock) Timer {
	return &variableTimer{fn: fn, new: c.NewTimer}
}

// Until takes an arbitrary delay function and runs until cancelled or the condition indicates exit. This
// offers all of the functionality of the methods in this package.
func (fn DelayFunc) Until(ctx context.Context, immediate, sliding bool, condition ConditionWithContextFunc) error {
	return loopConditionUntilContext(ctx, &variableTimer{fn: fn, new: internalClock.NewTimer}, immediate, sliding, condition)
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

// Step returns an amount of time to sleep determined by the original
// Duration and Jitter. The backoff is mutated to update its Steps and
// Duration. A nil Backoff always has a zero-duration step.
func (b *Backoff) Step() time.Duration {
	if b == nil {
		return 0
	}
	var nextDuration time.Duration
	nextDuration, b.Duration, b.Steps = delay(b.Steps, b.Duration, b.Cap, b.Factor, b.Jitter)
	return nextDuration
}

// delayFunc returns a function that will compute the next interval to
// wait given the arguments in b. It does not mutate the original backoff
// but the function is safe to use only from a single goroutine.
func (b Backoff) delayFunc() DelayFunc {
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

// Timer returns a timer implementation appropriate to this backoff's parameters
// for use with wait functions.
func (b Backoff) Timer() Timer {
	if b.Steps > 1 || b.Jitter != 0 {
		return &variableTimer{new: internalClock.NewTimer, fn: b.delayFunc()}
	}
	if b.Duration > 0 {
		return &fixedTimer{new: internalClock.NewTicker, interval: b.Duration}
	}
	return newNoopTimer()
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

// DelayWithReset returns a DelayFunc that will return the appropriate next interval to
// wait. Every resetInterval the backoff parameters are reset to their initial state.
// This method is safe to invoke from multiple goroutines, but all calls will advance
// the backoff state when Factor is set. If Factor is zero, this method is the same as
// invoking b.DelayFunc() since Steps has no impact without Factor. If resetInterval is
// zero no backoff will be performed as the same calling DelayFunc with a zero factor
// and steps.
func (b Backoff) DelayWithReset(c clock.Clock, resetInterval time.Duration) DelayFunc {
	if b.Factor <= 0 {
		return b.delayFunc()
	}
	if resetInterval <= 0 {
		b.Steps = 0
		b.Factor = 0
		return b.delayFunc()
	}
	return (&backoffManager{
		backoff:        b,
		initialBackoff: b,
		resetInterval:  resetInterval,

		clock:     c,
		lastStart: c.Now(),
		timer:     nil,
	}).Step
}

// BackoffManager manages backoff with a particular scheme based on its underlying implementation.
type BackoffManager interface {
	// Backoff returns a shared clock.Timer that is Reset on every invocation. This method is not
	// safe for use from multiple threads. It returns a timer for backoff, and caller shall backoff
	// until Timer.C() drains. If the second Backoff() is called before the timer from the first
	// Backoff() call finishes, the first timer will NOT be drained and result in undetermined
	// behavior.
	Backoff() clock.Timer
	// Timer returns a new Timer for use in a loop function in this package. Each timer retrieves its
	// next interval from the manager, and so all Timers from this manager will participate in shared
	// backoff.
	Timer() Timer
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

// Backoff implements BackoffManager.Backoff, it returns a timer so caller can block on the timer
// for exponential backoff. The returned timer must be drained before calling Backoff() the second
// time.
func (b *backoffManager) Backoff() clock.Timer {
	b.lock.Lock()
	defer b.lock.Unlock()
	if b.timer == nil {
		b.timer = b.clock.NewTimer(b.Step())
	} else {
		b.timer.Reset(b.Step())
	}
	return b.timer
}

// Timer returns a new Timer instance that shares the clock and the reset behavior with all other
// timers.
func (b *backoffManager) Timer() Timer {
	return DelayFunc(b.Step).Timer(b.clock)
}

// NewExponentialBackoffManager returns a manager for managing exponential backoff. Each backoff is jittered and
// backoff will not exceed the given max. If the backoff is not called within resetDuration, the backoff is reset.
// This backoff manager is used to reduce load during upstream unhealthiness.
//
// Deprecated: Will be removed when the legacy Poll methods are removed. Callers should construct a
// Backoff struct, use DelayWithReset() to get a DelayFunc that periodically resets itself, and then
// invoke Timer() when calling wait.BackoffUntil.
//
// Instead of:
//
//	bm := wait.NewExponentialBackoffManager(init, max, reset, factor, jitter, clock)
//	...
//	wait.BackoffUntil(..., bm.Backoff, ...)
//
// Use:
//
//	delayFn := wait.Backoff{
//	  Duration: init,
//	  Cap:      max,
//	  Steps:    int(math.Ceil(float64(max) / float64(init))), // now a required argument
//	  Factor:   factor,
//	  Jitter:   jitter,
//	}.DelayWithReset(reset, clock)
//	wait.BackoffUntil(..., delayFn.Timer(), ...)
func NewExponentialBackoffManager(initBackoff, maxBackoff, resetDuration time.Duration, backoffFactor, jitter float64, c clock.Clock) BackoffManager {
	b := Backoff{
		Duration: initBackoff,
		Cap:      maxBackoff,
		Factor:   backoffFactor,
		Jitter:   jitter,
	}
	// ensure a sufficient number of steps is provided
	if maxBackoff > initBackoff {
		b.Steps = int(math.Ceil(float64(maxBackoff)/float64(initBackoff))) + 1
	}
	return &backoffManager{
		backoff:        b,
		initialBackoff: b,
		resetInterval:  resetDuration,
		clock:          c,
	}
}

// NewJitteredBackoffManager returns a BackoffManager that backoffs with given duration plus given jitter. If the jitter
// is negative, backoff will not be jittered.
//
// Deprecated: Will be removed when the legacy Poll methods are removed. Callers should construct a
// Backoff struct and invoke Timer() when calling wait.BackoffUntil.
//
// Instead of:
//
//	bm := wait.NewJitteredBackoffManager(duration, jitter, clock)
//	...
//	wait.BackoffUntil(..., bm.Backoff, ...)
//
// Use:
//
//	wait.BackoffUntil(..., wait.Backoff{Duration: duration, Jitter: jitter}.Timer(), ...)
func NewJitteredBackoffManager(duration time.Duration, jitter float64, c clock.Clock) BackoffManager {
	b := Backoff{
		Duration: duration,
		Jitter:   jitter,
	}
	return &backoffManager{
		backoff:        b,
		initialBackoff: b,
		clock:          c,
	}
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
func ExponentialBackoff(backoff Backoff, condition ConditionFunc) error {
	return ExponentialBackoffWithContext(context.Background(), backoff, condition.WithContext())
}

// ExponentialBackoffWithContext works with a request context and a Backoff. It ensures that the retry wait never
// exceeds the deadline specified by the request context.
func ExponentialBackoffWithContext(ctx context.Context, backoff Backoff, condition ConditionWithContextFunc) error {
	steps := backoff.Steps
	return loopConditionUntilContext(ctx, backoff.Timer(), true, true, func(ctx context.Context) (bool, error) {
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
func PollWithContext(ctx context.Context, interval, timeout time.Duration, condition ConditionWithContextFunc) error {
	return poll(ctx, false, poller(interval, timeout), condition)
}

// PollUntil tries a condition func until it returns true, an error or stopCh is
// closed.
//
// PollUntil always waits interval before the first run of 'condition'.
// 'condition' will always be invoked at least once.
func PollUntil(interval time.Duration, condition ConditionFunc, stopCh <-chan struct{}) error {
	return PollUntilWithContext(ContextForChannel(stopCh), interval, condition.WithContext())
}

// PollUntilWithContext tries a condition func until it returns true,
// an error or the specified context is cancelled or expired.
//
// PollUntilWithContext always waits interval before the first run of 'condition'.
// 'condition' will always be invoked at least once.
func PollUntilWithContext(ctx context.Context, interval time.Duration, condition ConditionWithContextFunc) error {
	return poll(ctx, false, poller(interval, 0), condition)
}

// PollInfinite tries a condition func until it returns true or an error
//
// PollInfinite always waits the interval before the run of 'condition'.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
func PollInfinite(interval time.Duration, condition ConditionFunc) error {
	return PollInfiniteWithContext(context.Background(), interval, condition.WithContext())
}

// PollInfiniteWithContext tries a condition func until it returns true or an error
//
// PollInfiniteWithContext always waits the interval before the run of 'condition'.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
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
func PollImmediateWithContext(ctx context.Context, interval, timeout time.Duration, condition ConditionWithContextFunc) error {
	return poll(ctx, true, poller(interval, timeout), condition)
}

// PollImmediateUntil tries a condition func until it returns true, an error or stopCh is closed.
//
// PollImmediateUntil runs the 'condition' before waiting for the interval.
// 'condition' will always be invoked at least once.
func PollImmediateUntil(interval time.Duration, condition ConditionFunc, stopCh <-chan struct{}) error {
	return PollImmediateUntilWithContext(ContextForChannel(stopCh), interval, condition.WithContext())
}

// PollImmediateUntilWithContext tries a condition func until it returns true,
// an error or the specified context is cancelled or expired.
//
// PollImmediateUntilWithContext runs the 'condition' before waiting for the interval.
// 'condition' will always be invoked at least once.
func PollImmediateUntilWithContext(ctx context.Context, interval time.Duration, condition ConditionWithContextFunc) error {
	return poll(ctx, true, poller(interval, 0), condition)
}

// PollImmediateInfinite tries a condition func until it returns true or an error
//
// PollImmediateInfinite runs the 'condition' before waiting for the interval.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
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
		// returning ctx.Err() will break backward compatibility
		return ErrWaitTimeout
	default:
		return waitForWithContext(ctx, wait, condition)
	}
}

// waitFunc creates a channel that receives an item every time a test
// should be executed and is closed when the last test should be invoked.
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
// Deprecated: Will be removed when the legacy Poll methods are removed.
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
// Deprecated: Will be removed when the legacy Poll methods are removed.
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
			// returning ctx.Err() will break backward compatibility
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
// the provided timer until the provided context is cancelled, the condition returns
// true, or the condition returns an error. If sliding is true, the period is computed
// after condition runs. If it is false then period includes the runtime for condition.
// If immediate is false the first delay happens before any call to condition. The
// returned error is the error returned by the last condition or the context error if
// the context was terminated.
//
// This is the common loop construct for all polling in the wait package.
func loopConditionUntilContext(ctx context.Context, t Timer, immediate, sliding bool, condition ConditionWithContextFunc) error {
	defer t.Stop()

	var timeCh <-chan time.Time
	doneCh := ctx.Done()

	// if we haven't requested immediate execution, delay once
	if !immediate {
		timeCh = t.C()
		select {
		case <-doneCh:
			return ctx.Err()
		case <-timeCh:
		}
	}

	for {
		// checking ctx.Err() is slightly faster than checking a select
		if err := ctx.Err(); err != nil {
			return err
		}

		if !sliding {
			t.Next()
		}
		if ok, err := func() (bool, error) {
			defer runtime.HandleCrash()
			return condition(ctx)
		}(); err != nil || ok {
			return err
		}
		if sliding {
			t.Next()
		}

		if timeCh == nil {
			timeCh = t.C()
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
		case <-timeCh:
		}
	}
}
