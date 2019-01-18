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
)

// For any test of the style:
//   ...
//   <- time.After(timeout):
//      t.Errorf("Timed out")
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

// NonSlidingUntil loops until stop channel is closed, running f every
// period.
//
// NonSlidingUntil is syntactic sugar on top of JitterUntil with zero jitter
// factor, with sliding = false (meaning the timer for period starts at the same
// time as the function starts).
func NonSlidingUntil(f func(), period time.Duration, stopCh <-chan struct{}) {
	JitterUntil(f, period, 0.0, false, stopCh)
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
	var t *time.Timer
	var sawTimeout bool

	for {
		select {
		case <-stopCh:
			return
		default:
		}

		jitteredPeriod := period
		if jitterFactor > 0.0 {
			jitteredPeriod = Jitter(period, jitterFactor)
		}

		if !sliding {
			t = resetOrReuseTimer(t, jitteredPeriod, sawTimeout)
		}

		func() {
			defer runtime.HandleCrash()
			f()
		}()

		if sliding {
			t = resetOrReuseTimer(t, jitteredPeriod, sawTimeout)
		}

		// NOTE: b/c there is no priority selection in golang
		// it is possible for this to race, meaning we could
		// trigger t.C and stopCh, and t.C select falls through.
		// In order to mitigate we re-check stopCh at the beginning
		// of every loop to prevent extra executions of f().
		select {
		case <-stopCh:
			return
		case <-t.C:
			sawTimeout = true
		}
	}
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

// Backoff holds parameters applied to a Backoff function.
type Backoff struct {
	// The initial duration.
	Duration time.Duration
	// Duration is multiplied by factor each iteration. Must be greater
	// than or equal to zero.
	Factor float64
	// The amount of jitter applied each iteration. Jitter is applied after
	// cap.
	Jitter float64
	// The number of steps before duration stops changing. If zero, initial
	// duration is always used. Used for exponential backoff in combination
	// with Factor.
	Steps int
	// The returned duration will never be greater than cap *before* jitter
	// is applied. The actual maximum cap is `cap * (1.0 + jitter)`.
	Cap time.Duration
}

// Step returns the next interval in the exponential backoff. This method
// will mutate the provided backoff.
func (b *Backoff) Step() time.Duration {
	if b.Steps < 1 {
		if b.Jitter > 0 {
			return Jitter(b.Duration, b.Jitter)
		}
		return b.Duration
	}
	b.Steps--

	duration := b.Duration

	// calculate the next step
	if b.Factor != 0 {
		b.Duration = time.Duration(float64(b.Duration) * b.Factor)
		if b.Cap > 0 && b.Duration > b.Cap {
			b.Duration = b.Cap
			b.Steps = 0
		}
	}

	if b.Jitter > 0 {
		duration = Jitter(duration, b.Jitter)
	}
	return duration
}

// ExponentialBackoff repeats a condition check with exponential backoff.
//
// It checks the condition up to Steps times, increasing the wait by multiplying
// the previous duration by Factor.
//
// If Jitter is greater than zero, a random amount of each duration is added
// (between duration and duration*(1+jitter)).
//
// If the condition never returns true, ErrWaitTimeout is returned. All other
// errors terminate immediately.
func ExponentialBackoff(backoff Backoff, condition ConditionFunc) error {
	for backoff.Steps > 0 {
		if ok, err := condition(); err != nil || ok {
			return err
		}
		if backoff.Steps == 1 {
			break
		}
		time.Sleep(backoff.Step())
	}
	return ErrWaitTimeout
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
	return pollInternal(poller(interval, timeout), condition)
}

func pollInternal(wait WaitFunc, condition ConditionFunc) error {
	done := make(chan struct{})
	defer close(done)
	return WaitFor(wait, condition, done)
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
	return pollImmediateInternal(poller(interval, timeout), condition)
}

func pollImmediateInternal(wait WaitFunc, condition ConditionFunc) error {
	done, err := condition()
	if err != nil {
		return err
	}
	if done {
		return nil
	}
	return pollInternal(wait, condition)
}

// PollInfinite tries a condition func until it returns true or an error
//
// PollInfinite always waits the interval before the run of 'condition'.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
func PollInfinite(interval time.Duration, condition ConditionFunc) error {
	done := make(chan struct{})
	defer close(done)
	return PollUntil(interval, condition, done)
}

// PollImmediateInfinite tries a condition func until it returns true or an error
//
// PollImmediateInfinite runs the 'condition' before waiting for the interval.
//
// Some intervals may be missed if the condition takes too long or the time
// window is too short.
func PollImmediateInfinite(interval time.Duration, condition ConditionFunc) error {
	done, err := condition()
	if err != nil {
		return err
	}
	if done {
		return nil
	}
	return PollInfinite(interval, condition)
}

// PollUntil tries a condition func until it returns true, an error or stopCh is
// closed.
//
// PollUntil always waits interval before the first run of 'condition'.
// 'condition' will always be invoked at least once.
func PollUntil(interval time.Duration, condition ConditionFunc, stopCh <-chan struct{}) error {
	return WaitFor(poller(interval, 0), condition, stopCh)
}

// PollImmediateUntil tries a condition func until it returns true, an error or stopCh is closed.
//
// PollImmediateUntil runs the 'condition' before waiting for the interval.
// 'condition' will always be invoked at least once.
func PollImmediateUntil(interval time.Duration, condition ConditionFunc, stopCh <-chan struct{}) error {
	done, err := condition()
	if err != nil {
		return err
	}
	if done {
		return nil
	}
	select {
	case <-stopCh:
		return ErrWaitTimeout
	default:
		return PollUntil(interval, condition, stopCh)
	}
}

// WaitFunc creates a channel that receives an item every time a test
// should be executed and is closed when the last test should be invoked.
type WaitFunc func(done <-chan struct{}) <-chan struct{}

// WaitFor continually checks 'fn' as driven by 'wait'.
//
// WaitFor gets a channel from 'wait()'', and then invokes 'fn' once for every value
// placed on the channel and once more when the channel is closed. If the channel is closed
// and 'fn' returns false without error, WaitFor returns ErrWaitTimeout.
//
// If 'fn' returns an error the loop ends and that error is returned. If
// 'fn' returns true the loop ends and nil is returned.
//
// ErrWaitTimeout will be returned if the 'done' channel is closed without fn ever
// returning true.
//
// When the done channel is closed, because the golang `select` statement is
// "uniform pseudo-random", the `fn` might still run one or multiple time,
// though eventually `WaitFor` will return.
func WaitFor(wait WaitFunc, fn ConditionFunc, done <-chan struct{}) error {
	stopCh := make(chan struct{})
	defer close(stopCh)
	c := wait(stopCh)
	for {
		select {
		case _, open := <-c:
			ok, err := fn()
			if err != nil {
				return err
			}
			if ok {
				return nil
			}
			if !open {
				return ErrWaitTimeout
			}
		case <-done:
			return ErrWaitTimeout
		}
	}
}

// poller returns a WaitFunc that will send to the channel every interval until
// timeout has elapsed and then closes the channel.
//
// Over very short intervals you may receive no ticks before the channel is
// closed. A timeout of 0 is interpreted as an infinity.
//
// Output ticks are not buffered. If the channel is not ready to receive an
// item, the tick is skipped.
func poller(interval, timeout time.Duration) WaitFunc {
	return WaitFunc(func(done <-chan struct{}) <-chan struct{} {
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
				case <-done:
					return
				}
			}
		}()

		return ch
	})
}

// resetOrReuseTimer avoids allocating a new timer if one is already in use.
// Not safe for multiple threads.
func resetOrReuseTimer(t *time.Timer, d time.Duration, sawTimeout bool) *time.Timer {
	if t == nil {
		return time.NewTimer(d)
	}
	if !t.Stop() && !sawTimeout {
		<-t.C
	}
	t.Reset(d)
	return t
}
