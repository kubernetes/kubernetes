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
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"
)

func timerWithClock(t Timer, c clock.WithTicker) Timer {
	switch t := t.(type) {
	case *fixedTimer:
		t.new = c.NewTicker
	case *variableTimer:
		t.new = c.NewTimer
	default:
		panic("unrecognized timer type, cannot inject clock")
	}
	return t
}

func Test_loopConditionWithContextImmediateDelay(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Time{})
	backoff := Backoff{Duration: time.Second}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	expectedError := errors.New("Expected error")
	var attempt int
	f := ConditionFunc(func() (bool, error) {
		attempt++
		return false, expectedError
	})

	doneCh := make(chan struct{})
	go func() {
		defer close(doneCh)
		if err := loopConditionUntilContext(ctx, timerWithClock(backoff.Timer(), fakeClock), false, true, f.WithContext()); err == nil || err != expectedError {
			t.Errorf("unexpected error: %v", err)
		}
	}()

	for !fakeClock.HasWaiters() {
		time.Sleep(time.Microsecond)
	}

	fakeClock.Step(time.Second - time.Millisecond)
	if attempt != 0 {
		t.Fatalf("should still be waiting for condition")
	}
	fakeClock.Step(2 * time.Millisecond)

	select {
	case <-doneCh:
	case <-time.After(time.Second):
		t.Fatalf("should have exited after a single loop")
	}
	if attempt != 1 {
		t.Fatalf("expected attempt")
	}
}

func Test_loopConditionUntilContext_semantic(t *testing.T) {
	defaultCallback := func(_ int) (bool, error) {
		return false, nil
	}

	conditionErr := errors.New("condition failed")

	tests := []struct {
		name               string
		immediate          bool
		sliding            bool
		context            func() (context.Context, context.CancelFunc)
		callback           func(calls int) (bool, error)
		cancelContextAfter int
		attemptsExpected   int
		errExpected        error
		timer              Timer
	}{
		{
			name: "condition successful is only one attempt",
			callback: func(attempts int) (bool, error) {
				return true, nil
			},
			attemptsExpected: 1,
		},
		{
			name: "delayed condition successful causes return and attempts",
			callback: func(attempts int) (bool, error) {
				return attempts > 1, nil
			},
			attemptsExpected: 2,
		},
		{
			name: "delayed condition successful causes return and attempts many times",
			callback: func(attempts int) (bool, error) {
				return attempts >= 100, nil
			},
			attemptsExpected: 100,
		},
		{
			name: "condition returns error even if ok is true",
			callback: func(_ int) (bool, error) {
				return true, conditionErr
			},
			attemptsExpected: 1,
			errExpected:      conditionErr,
		},
		{
			name: "condition exits after an error",
			callback: func(_ int) (bool, error) {
				return false, conditionErr
			},
			attemptsExpected: 1,
			errExpected:      conditionErr,
		},
		{
			name:             "context already canceled no attempts expected",
			context:          cancelledContext,
			callback:         defaultCallback,
			attemptsExpected: 0,
			errExpected:      context.Canceled,
		},
		{
			name:    "context already canceled condition success and immediate 1 attempt expected",
			context: cancelledContext,
			callback: func(_ int) (bool, error) {
				return true, nil
			},
			immediate:        true,
			attemptsExpected: 1,
		},
		{
			name:    "context already canceled condition fail and immediate 1 attempt expected",
			context: cancelledContext,
			callback: func(_ int) (bool, error) {
				return false, conditionErr
			},
			immediate:        true,
			attemptsExpected: 1,
			errExpected:      conditionErr,
		},
		{
			name:             "context already canceled and immediate 1 attempt expected",
			context:          cancelledContext,
			callback:         defaultCallback,
			immediate:        true,
			attemptsExpected: 1,
			errExpected:      context.Canceled,
		},
		{
			name:               "context cancelled after 5 attempts",
			context:            defaultContext,
			callback:           defaultCallback,
			cancelContextAfter: 5,
			attemptsExpected:   5,
			errExpected:        context.Canceled,
		},
		{
			name:               "context cancelled and immediate after 5 attempts",
			context:            defaultContext,
			callback:           defaultCallback,
			immediate:          true,
			cancelContextAfter: 5,
			attemptsExpected:   5,
			errExpected:        context.Canceled,
		},
		{
			name:             "context at deadline and immediate 1 attempt expected",
			context:          deadlinedContext,
			callback:         defaultCallback,
			immediate:        true,
			attemptsExpected: 1,
			errExpected:      context.DeadlineExceeded,
		},
		{
			name:             "context at deadline no attempts expected",
			context:          deadlinedContext,
			callback:         defaultCallback,
			attemptsExpected: 0,
			errExpected:      context.DeadlineExceeded,
		},
		{
			name:      "context canceled before the second execution and immediate",
			immediate: true,
			context: func() (context.Context, context.CancelFunc) {
				return context.WithTimeout(context.Background(), time.Second)
			},
			callback: func(attempts int) (bool, error) {
				return false, nil
			},
			attemptsExpected: 1,
			errExpected:      context.DeadlineExceeded,
			timer:            Backoff{Duration: 2 * time.Second}.Timer(),
		},
		{
			name:      "immediate and long duration of condition and sliding false",
			immediate: true,
			sliding:   false,
			context: func() (context.Context, context.CancelFunc) {
				return context.WithTimeout(context.Background(), time.Second)
			},
			callback: func(attempts int) (bool, error) {
				if attempts >= 4 {
					return true, nil
				}
				time.Sleep(time.Second / 5)
				return false, nil
			},
			attemptsExpected: 4,
			timer:            Backoff{Duration: time.Second / 5, Jitter: 0.001}.Timer(),
		},
		{
			name:      "immediate and long duration of condition and sliding true",
			immediate: true,
			sliding:   true,
			context: func() (context.Context, context.CancelFunc) {
				return context.WithTimeout(context.Background(), time.Second)
			},
			callback: func(attempts int) (bool, error) {
				if attempts >= 4 {
					return true, nil
				}
				time.Sleep(time.Second / 5)
				return false, nil
			},
			errExpected:      context.DeadlineExceeded,
			attemptsExpected: 3,
			timer:            Backoff{Duration: time.Second / 5, Jitter: 0.001}.Timer(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			contextFn := test.context
			if contextFn == nil {
				contextFn = defaultContext
			}
			ctx, cancel := contextFn()
			defer cancel()

			timer := test.timer
			if timer == nil {
				timer = Backoff{Duration: time.Microsecond}.Timer()
			}
			attempts := 0
			err := loopConditionUntilContext(ctx, timer, test.immediate, test.sliding, func(_ context.Context) (bool, error) {
				attempts++
				defer func() {
					if test.cancelContextAfter > 0 && test.cancelContextAfter == attempts {
						cancel()
					}
				}()
				return test.callback(attempts)
			})

			if test.errExpected != err {
				t.Errorf("expected error: %v but got: %v", test.errExpected, err)
			}

			if test.attemptsExpected != attempts {
				t.Errorf("expected attempts count: %d but got: %d", test.attemptsExpected, attempts)
			}
		})
	}
}

type timerWrapper struct {
	timer   clock.Timer
	resets  []time.Duration
	onReset func(d time.Duration)
}

func (w *timerWrapper) C() <-chan time.Time { return w.timer.C() }
func (w *timerWrapper) Stop() bool          { return w.timer.Stop() }
func (w *timerWrapper) Reset(d time.Duration) bool {
	w.resets = append(w.resets, d)
	b := w.timer.Reset(d)
	if w.onReset != nil {
		w.onReset(d)
	}
	return b
}

func Test_loopConditionUntilContext_timings(t *testing.T) {
	// Verify that timings returned by the delay func are passed to the timer, and that
	// the timer advancing is enough to drive the state machine. Not a deep verification
	// of the behavior of the loop, but tests that we drive the scenario to completion.
	tests := []struct {
		name               string
		delayFn            DelayFunc
		immediate          bool
		sliding            bool
		context            func() (context.Context, context.CancelFunc)
		callback           func(calls int, lastInterval time.Duration) (bool, error)
		cancelContextAfter int
		attemptsExpected   int
		errExpected        error
		expectedIntervals  func(t *testing.T, delays []time.Duration, delaysRequested []time.Duration)
	}{
		{
			name:    "condition success",
			delayFn: Backoff{Duration: time.Second, Steps: 2, Factor: 2.0, Jitter: 0}.DelayFunc(),
			callback: func(attempts int, _ time.Duration) (bool, error) {
				return true, nil
			},
			attemptsExpected: 1,
			expectedIntervals: func(t *testing.T, delays []time.Duration, delaysRequested []time.Duration) {
				if reflect.DeepEqual(delays, []time.Duration{time.Second, 2 * time.Second}) {
					return
				}
				if reflect.DeepEqual(delaysRequested, []time.Duration{time.Second}) {
					return
				}
			},
		},
		{
			name:      "condition success and immediate",
			immediate: true,
			delayFn:   Backoff{Duration: time.Second, Steps: 2, Factor: 2.0, Jitter: 0}.DelayFunc(),
			callback: func(attempts int, _ time.Duration) (bool, error) {
				return true, nil
			},
			attemptsExpected: 1,
			expectedIntervals: func(t *testing.T, delays []time.Duration, delaysRequested []time.Duration) {
				if reflect.DeepEqual(delays, []time.Duration{time.Second}) {
					return
				}
				if reflect.DeepEqual(delaysRequested, []time.Duration{}) {
					return
				}
			},
		},
		{
			name:    "condition success and sliding",
			sliding: true,
			delayFn: Backoff{Duration: time.Second, Steps: 2, Factor: 2.0, Jitter: 0}.DelayFunc(),
			callback: func(attempts int, _ time.Duration) (bool, error) {
				return true, nil
			},
			attemptsExpected: 1,
			expectedIntervals: func(t *testing.T, delays []time.Duration, delaysRequested []time.Duration) {
				if reflect.DeepEqual(delays, []time.Duration{time.Second}) {
					return
				}
				if !reflect.DeepEqual(delays, delaysRequested) {
					t.Fatalf("sliding non-immediate should have equal delays: %v", cmp.Diff(delays, delaysRequested))
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf("%s/sliding=%t/immediate=%t", test.name, test.sliding, test.immediate), func(t *testing.T) {
			contextFn := test.context
			if contextFn == nil {
				contextFn = defaultContext
			}
			ctx, cancel := contextFn()
			defer cancel()

			fakeClock := &testingclock.FakeClock{}
			var fakeTimers []*timerWrapper
			timerFn := func(d time.Duration) clock.Timer {
				t := fakeClock.NewTimer(d)
				fakeClock.Step(d + 1)
				w := &timerWrapper{timer: t, resets: []time.Duration{d}, onReset: func(d time.Duration) {
					fakeClock.Step(d + 1)
				}}
				fakeTimers = append(fakeTimers, w)
				return w
			}

			delayFn := test.delayFn
			if delayFn == nil {
				delayFn = Backoff{Duration: time.Microsecond}.DelayFunc()
			}
			var delays []time.Duration
			wrappedDelayFn := func() time.Duration {
				d := delayFn()
				delays = append(delays, d)
				return d
			}
			timer := &variableTimer{fn: wrappedDelayFn, new: timerFn}

			attempts := 0
			err := loopConditionUntilContext(ctx, timer, test.immediate, test.sliding, func(_ context.Context) (bool, error) {
				attempts++
				defer func() {
					if test.cancelContextAfter > 0 && test.cancelContextAfter == attempts {
						cancel()
					}
				}()
				lastInterval := time.Duration(-1)
				if len(delays) > 0 {
					lastInterval = delays[len(delays)-1]
				}
				return test.callback(attempts, lastInterval)
			})

			if test.errExpected != err {
				t.Errorf("expected error: %v but got: %v", test.errExpected, err)
			}

			if test.attemptsExpected != attempts {
				t.Errorf("expected attempts count: %d but got: %d", test.attemptsExpected, attempts)
			}
			switch len(fakeTimers) {
			case 0:
				test.expectedIntervals(t, delays, nil)
			case 1:
				test.expectedIntervals(t, delays, fakeTimers[0].resets)
			default:
				t.Fatalf("expected zero or one timers: %#v", fakeTimers)
			}
		})
	}
}

// Test_loopConditionUntilContext_timings runs actual timing loops and calculates the delta. This
// test depends on high precision wakeups which depends on low CPU contention so it is not a
// candidate to run during normal unit test execution (nor is it a benchmark or example). Instead,
// it can be run manually if there is a scenario where we suspect the timings are off and other
// tests haven't caught it. A final sanity test that would have to be run serially in isolation.
func Test_loopConditionUntilContext_Elapsed(t *testing.T) {
	const maxAttempts = 10
	// TODO: this may be too aggressive, but the overhead should be minor
	const estimatedLoopOverhead = time.Millisecond
	// estimate how long this delay can be
	intervalMax := func(backoff Backoff) time.Duration {
		d := backoff.Duration
		if backoff.Jitter > 0 {
			d += time.Duration(backoff.Jitter * float64(d))
		}
		return d
	}
	// estimate how short this delay can be
	intervalMin := func(backoff Backoff) time.Duration {
		d := backoff.Duration
		return d
	}

	// Because timing is dependent other factors in test environments, such as
	// whether the OS or go runtime scheduler wake the timers, excess duration
	// is logged by default and can be converted to a fatal error for testing.
	// fail := t.Fatalf
	fail := t.Logf

	for _, test := range []struct {
		name    string
		backoff Backoff
		t       reflect.Type
	}{
		{name: "variable timer with jitter", backoff: Backoff{Duration: time.Millisecond, Jitter: 1.0}, t: reflect.TypeOf(&variableTimer{})},
		{name: "fixed timer", backoff: Backoff{Duration: time.Millisecond}, t: reflect.TypeOf(&fixedTimer{})},
		{name: "no-op timer", backoff: Backoff{}, t: reflect.TypeOf(noopTimer{})},
	} {
		t.Run(test.name, func(t *testing.T) {
			var attempts int
			start := time.Now()
			timer := test.backoff.Timer()
			if test.t != reflect.ValueOf(timer).Type() {
				t.Fatalf("unexpected timer type %T: expected %v", timer, test.t)
			}
			if err := loopConditionUntilContext(context.Background(), timer, false, false, func(_ context.Context) (bool, error) {
				attempts++
				if attempts > maxAttempts {
					t.Fatalf("should not reach %d attempts", maxAttempts+1)
				}
				return attempts >= maxAttempts, nil
			}); err != nil {
				t.Fatal(err)
			}
			duration := time.Since(start)
			if min := maxAttempts * intervalMin(test.backoff); duration < min {
				fail("elapsed duration %v < expected min duration %v", duration, min)
			}
			if max := maxAttempts * (intervalMax(test.backoff) + estimatedLoopOverhead); duration > max {
				fail("elapsed duration %v > expected max duration %v", duration, max)
			}
		})
	}
}

func Benchmark_loopConditionUntilContext_ZeroDuration(b *testing.B) {
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		attempts := 0
		if err := loopConditionUntilContext(ctx, Backoff{Duration: 0}.Timer(), true, false, func(_ context.Context) (bool, error) {
			attempts++
			return attempts >= 100, nil
		}); err != nil {
			b.Fatalf("unexpected err: %v", err)
		}
	}
}

func Benchmark_loopConditionUntilContext_ShortDuration(b *testing.B) {
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		attempts := 0
		if err := loopConditionUntilContext(ctx, Backoff{Duration: time.Microsecond}.Timer(), true, false, func(_ context.Context) (bool, error) {
			attempts++
			return attempts >= 100, nil
		}); err != nil {
			b.Fatalf("unexpected err: %v", err)
		}
	}
}
