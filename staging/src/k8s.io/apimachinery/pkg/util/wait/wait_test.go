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
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/runtime"
)

func TestUntil(t *testing.T) {
	ch := make(chan struct{})
	close(ch)
	Until(func() {
		t.Fatal("should not have been invoked")
	}, 0, ch)

	ch = make(chan struct{})
	called := make(chan struct{})
	go func() {
		Until(func() {
			called <- struct{}{}
		}, 0, ch)
		close(called)
	}()
	<-called
	close(ch)
	<-called
}

func TestUntilWithContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.TODO())
	cancel()
	UntilWithContext(ctx, func(context.Context) {
		t.Fatal("should not have been invoked")
	}, 0)

	ctx, cancel = context.WithCancel(context.TODO())
	called := make(chan struct{})
	go func() {
		UntilWithContext(ctx, func(context.Context) {
			called <- struct{}{}
		}, 0)
		close(called)
	}()
	<-called
	cancel()
	<-called
}

func TestNonSlidingUntil(t *testing.T) {
	ch := make(chan struct{})
	close(ch)
	NonSlidingUntil(func() {
		t.Fatal("should not have been invoked")
	}, 0, ch)

	ch = make(chan struct{})
	called := make(chan struct{})
	go func() {
		NonSlidingUntil(func() {
			called <- struct{}{}
		}, 0, ch)
		close(called)
	}()
	<-called
	close(ch)
	<-called
}

func TestNonSlidingUntilWithContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.TODO())
	cancel()
	NonSlidingUntilWithContext(ctx, func(context.Context) {
		t.Fatal("should not have been invoked")
	}, 0)

	ctx, cancel = context.WithCancel(context.TODO())
	called := make(chan struct{})
	go func() {
		NonSlidingUntilWithContext(ctx, func(context.Context) {
			called <- struct{}{}
		}, 0)
		close(called)
	}()
	<-called
	cancel()
	<-called
}

func TestUntilReturnsImmediately(t *testing.T) {
	now := time.Now()
	ch := make(chan struct{})
	Until(func() {
		close(ch)
	}, 30*time.Second, ch)
	if now.Add(25 * time.Second).Before(time.Now()) {
		t.Errorf("Until did not return immediately when the stop chan was closed inside the func")
	}
}

func TestJitterUntil(t *testing.T) {
	ch := make(chan struct{})
	// if a channel is closed JitterUntil never calls function f
	// and returns immediately
	close(ch)
	JitterUntil(func() {
		t.Fatal("should not have been invoked")
	}, 0, 1.0, true, ch)

	ch = make(chan struct{})
	called := make(chan struct{})
	go func() {
		JitterUntil(func() {
			called <- struct{}{}
		}, 0, 1.0, true, ch)
		close(called)
	}()
	<-called
	close(ch)
	<-called
}

func TestJitterUntilWithContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.TODO())
	cancel()
	JitterUntilWithContext(ctx, func(context.Context) {
		t.Fatal("should not have been invoked")
	}, 0, 1.0, true)

	ctx, cancel = context.WithCancel(context.TODO())
	called := make(chan struct{})
	go func() {
		JitterUntilWithContext(ctx, func(context.Context) {
			called <- struct{}{}
		}, 0, 1.0, true)
		close(called)
	}()
	<-called
	cancel()
	<-called
}

func TestJitterUntilReturnsImmediately(t *testing.T) {
	now := time.Now()
	ch := make(chan struct{})
	JitterUntil(func() {
		close(ch)
	}, 30*time.Second, 1.0, true, ch)
	if now.Add(25 * time.Second).Before(time.Now()) {
		t.Errorf("JitterUntil did not return immediately when the stop chan was closed inside the func")
	}
}

func TestJitterUntilRecoversPanic(t *testing.T) {
	// Save and restore crash handlers
	originalReallyCrash := runtime.ReallyCrash
	originalHandlers := runtime.PanicHandlers
	defer func() {
		runtime.ReallyCrash = originalReallyCrash
		runtime.PanicHandlers = originalHandlers
	}()

	called := 0
	handled := 0

	// Hook up a custom crash handler to ensure it is called when a jitter function panics
	runtime.ReallyCrash = false
	runtime.PanicHandlers = []func(interface{}){
		func(p interface{}) {
			handled++
		},
	}

	ch := make(chan struct{})
	JitterUntil(func() {
		called++
		if called > 2 {
			close(ch)
			return
		}
		panic("TestJitterUntilRecoversPanic")
	}, time.Millisecond, 1.0, true, ch)

	if called != 3 {
		t.Errorf("Expected panic recovers")
	}
}

func TestJitterUntilNegativeFactor(t *testing.T) {
	now := time.Now()
	ch := make(chan struct{})
	called := make(chan struct{})
	received := make(chan struct{})
	go func() {
		JitterUntil(func() {
			called <- struct{}{}
			<-received
		}, time.Second, -30.0, true, ch)
	}()
	// first loop
	<-called
	received <- struct{}{}
	// second loop
	<-called
	close(ch)
	received <- struct{}{}

	// it should take at most 2 seconds + some overhead, not 3
	if now.Add(3 * time.Second).Before(time.Now()) {
		t.Errorf("JitterUntil did not returned after predefined period with negative jitter factor when the stop chan was closed inside the func")
	}

}

func TestExponentialBackoff(t *testing.T) {
	opts := Backoff{Factor: 1.0, Steps: 3}

	// waits up to steps
	i := 0
	err := ExponentialBackoff(opts, func() (bool, error) {
		i++
		return false, nil
	})
	if err != ErrWaitTimeout || i != opts.Steps {
		t.Errorf("unexpected error: %v", err)
	}

	// returns immediately
	i = 0
	err = ExponentialBackoff(opts, func() (bool, error) {
		i++
		return true, nil
	})
	if err != nil || i != 1 {
		t.Errorf("unexpected error: %v", err)
	}

	// returns immediately on error
	testErr := fmt.Errorf("some other error")
	err = ExponentialBackoff(opts, func() (bool, error) {
		return false, testErr
	})
	if err != testErr {
		t.Errorf("unexpected error: %v", err)
	}

	// invoked multiple times
	i = 1
	err = ExponentialBackoff(opts, func() (bool, error) {
		if i < opts.Steps {
			i++
			return false, nil
		}
		return true, nil
	})
	if err != nil || i != opts.Steps {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestPoller(t *testing.T) {
	done := make(chan struct{})
	defer close(done)
	w := poller(time.Millisecond, 2*time.Millisecond)
	ch := w(done)
	count := 0
DRAIN:
	for {
		select {
		case _, open := <-ch:
			if !open {
				break DRAIN
			}
			count++
		case <-time.After(ForeverTestTimeout):
			t.Errorf("unexpected timeout after poll")
		}
	}
	if count > 3 {
		t.Errorf("expected up to three values, got %d", count)
	}
}

type fakePoller struct {
	max  int
	used int32 // accessed with atomics
	wg   sync.WaitGroup
}

func fakeTicker(max int, used *int32, doneFunc func()) WaitFunc {
	return func(done <-chan struct{}) <-chan struct{} {
		ch := make(chan struct{})
		go func() {
			defer doneFunc()
			defer close(ch)
			for i := 0; i < max; i++ {
				select {
				case ch <- struct{}{}:
				case <-done:
					return
				}
				if used != nil {
					atomic.AddInt32(used, 1)
				}
			}
		}()
		return ch
	}
}

func (fp *fakePoller) GetWaitFunc() WaitFunc {
	fp.wg.Add(1)
	return fakeTicker(fp.max, &fp.used, fp.wg.Done)
}

func TestPoll(t *testing.T) {
	invocations := 0
	f := ConditionFunc(func() (bool, error) {
		invocations++
		return true, nil
	})
	fp := fakePoller{max: 1}
	if err := pollInternal(fp.GetWaitFunc(), f); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	fp.wg.Wait()
	if invocations != 1 {
		t.Errorf("Expected exactly one invocation, got %d", invocations)
	}
	used := atomic.LoadInt32(&fp.used)
	if used != 1 {
		t.Errorf("Expected exactly one tick, got %d", used)
	}
}

func TestPollError(t *testing.T) {
	expectedError := errors.New("Expected error")
	f := ConditionFunc(func() (bool, error) {
		return false, expectedError
	})
	fp := fakePoller{max: 1}
	if err := pollInternal(fp.GetWaitFunc(), f); err == nil || err != expectedError {
		t.Fatalf("Expected error %v, got none %v", expectedError, err)
	}
	fp.wg.Wait()
	used := atomic.LoadInt32(&fp.used)
	if used != 1 {
		t.Errorf("Expected exactly one tick, got %d", used)
	}
}

func TestPollImmediate(t *testing.T) {
	invocations := 0
	f := ConditionFunc(func() (bool, error) {
		invocations++
		return true, nil
	})
	fp := fakePoller{max: 0}
	if err := pollImmediateInternal(fp.GetWaitFunc(), f); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	// We don't need to wait for fp.wg, as pollImmediate shouldn't call WaitFunc at all.
	if invocations != 1 {
		t.Errorf("Expected exactly one invocation, got %d", invocations)
	}
	used := atomic.LoadInt32(&fp.used)
	if used != 0 {
		t.Errorf("Expected exactly zero ticks, got %d", used)
	}
}

func TestPollImmediateError(t *testing.T) {
	expectedError := errors.New("Expected error")
	f := ConditionFunc(func() (bool, error) {
		return false, expectedError
	})
	fp := fakePoller{max: 0}
	if err := pollImmediateInternal(fp.GetWaitFunc(), f); err == nil || err != expectedError {
		t.Fatalf("Expected error %v, got none %v", expectedError, err)
	}
	// We don't need to wait for fp.wg, as pollImmediate shouldn't call WaitFunc at all.
	used := atomic.LoadInt32(&fp.used)
	if used != 0 {
		t.Errorf("Expected exactly zero ticks, got %d", used)
	}
}

func TestPollForever(t *testing.T) {
	ch := make(chan struct{})
	done := make(chan struct{}, 1)
	complete := make(chan struct{})
	go func() {
		f := ConditionFunc(func() (bool, error) {
			ch <- struct{}{}
			select {
			case <-done:
				return true, nil
			default:
			}
			return false, nil
		})

		if err := PollInfinite(time.Microsecond, f); err != nil {
			t.Fatalf("unexpected error %v", err)
		}

		close(ch)
		complete <- struct{}{}
	}()

	// ensure the condition is opened
	<-ch

	// ensure channel sends events
	for i := 0; i < 10; i++ {
		select {
		case _, open := <-ch:
			if !open {
				t.Fatalf("did not expect channel to be closed")
			}
		case <-time.After(ForeverTestTimeout):
			t.Fatalf("channel did not return at least once within the poll interval")
		}
	}

	// at most one poll notification should be sent once we return from the condition
	done <- struct{}{}
	go func() {
		for i := 0; i < 2; i++ {
			_, open := <-ch
			if !open {
				return
			}
		}
		t.Fatalf("expected closed channel after two iterations")
	}()
	<-complete
}

func TestWaitFor(t *testing.T) {
	var invocations int
	testCases := map[string]struct {
		F       ConditionFunc
		Ticks   int
		Invoked int
		Err     bool
	}{
		"invoked once": {
			ConditionFunc(func() (bool, error) {
				invocations++
				return true, nil
			}),
			2,
			1,
			false,
		},
		"invoked and returns a timeout": {
			ConditionFunc(func() (bool, error) {
				invocations++
				return false, nil
			}),
			2,
			3, // the contract of WaitFor() says the func is called once more at the end of the wait
			true,
		},
		"returns immediately on error": {
			ConditionFunc(func() (bool, error) {
				invocations++
				return false, errors.New("test")
			}),
			2,
			1,
			true,
		},
	}
	for k, c := range testCases {
		invocations = 0
		ticker := fakeTicker(c.Ticks, nil, func() {})
		err := func() error {
			done := make(chan struct{})
			defer close(done)
			return WaitFor(ticker, c.F, done)
		}()
		switch {
		case c.Err && err == nil:
			t.Errorf("%s: Expected error, got nil", k)
			continue
		case !c.Err && err != nil:
			t.Errorf("%s: Expected no error, got: %#v", k, err)
			continue
		}
		if invocations != c.Invoked {
			t.Errorf("%s: Expected %d invocations, got %d", k, c.Invoked, invocations)
		}
	}
}

// TestWaitForWithEarlyClosingWaitFunc tests WaitFor when the WaitFunc closes its channel. The WaitFor should
// always return ErrWaitTimeout.
func TestWaitForWithEarlyClosingWaitFunc(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)

	start := time.Now()
	err := WaitFor(func(done <-chan struct{}) <-chan struct{} {
		c := make(chan struct{})
		close(c)
		return c
	}, func() (bool, error) {
		return false, nil
	}, stopCh)
	duration := time.Since(start)

	// The WaitFor should return immediately, so the duration is close to 0s.
	if duration >= ForeverTestTimeout/2 {
		t.Errorf("expected short timeout duration")
	}
	if err != ErrWaitTimeout {
		t.Errorf("expected ErrWaitTimeout from WaitFunc")
	}
}

// TestWaitForWithClosedChannel tests WaitFor when it receives a closed channel. The WaitFor should
// always return ErrWaitTimeout.
func TestWaitForWithClosedChannel(t *testing.T) {
	stopCh := make(chan struct{})
	close(stopCh)
	c := make(chan struct{})
	defer close(c)
	start := time.Now()
	err := WaitFor(func(done <-chan struct{}) <-chan struct{} {
		return c
	}, func() (bool, error) {
		return false, nil
	}, stopCh)
	duration := time.Since(start)
	// The WaitFor should return immediately, so the duration is close to 0s.
	if duration >= ForeverTestTimeout/2 {
		t.Errorf("expected short timeout duration")
	}
	// The interval of the poller is ForeverTestTimeout, so the WaitFor should always return ErrWaitTimeout.
	if err != ErrWaitTimeout {
		t.Errorf("expected ErrWaitTimeout from WaitFunc")
	}
}

// TestWaitForClosesStopCh verifies that after the condition func returns true, WaitFor() closes the stop channel it supplies to the WaitFunc.
func TestWaitForClosesStopCh(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)
	waitFunc := poller(time.Millisecond, ForeverTestTimeout)
	var doneCh <-chan struct{}

	WaitFor(func(done <-chan struct{}) <-chan struct{} {
		doneCh = done
		return waitFunc(done)
	}, func() (bool, error) {
		time.Sleep(10 * time.Millisecond)
		return true, nil
	}, stopCh)
	// The polling goroutine should be closed after WaitFor returning.
	select {
	case _, ok := <-doneCh:
		if ok {
			t.Errorf("expected closed channel after WaitFunc returning")
		}
	default:
		t.Errorf("expected an ack of the done signal")
	}
}

func TestPollUntil(t *testing.T) {
	stopCh := make(chan struct{})
	called := make(chan bool)
	pollDone := make(chan struct{})

	go func() {
		PollUntil(time.Microsecond, ConditionFunc(func() (bool, error) {
			called <- true
			return false, nil
		}), stopCh)

		close(pollDone)
	}()

	// make sure we're called once
	<-called
	// this should trigger a "done"
	close(stopCh)

	go func() {
		// release the condition func  if needed
		for {
			<-called
		}
	}()

	// make sure we finished the poll
	<-pollDone
}

func TestBackoff_Step(t *testing.T) {
	tests := []struct {
		initial *Backoff
		want    []time.Duration
	}{
		{initial: &Backoff{Duration: time.Second, Steps: 0}, want: []time.Duration{time.Second, time.Second, time.Second}},
		{initial: &Backoff{Duration: time.Second, Steps: 1}, want: []time.Duration{time.Second, time.Second, time.Second}},
		{initial: &Backoff{Duration: time.Second, Factor: 1.0, Steps: 1}, want: []time.Duration{time.Second, time.Second, time.Second}},
		{initial: &Backoff{Duration: time.Second, Factor: 2, Steps: 3}, want: []time.Duration{1 * time.Second, 2 * time.Second, 4 * time.Second}},
		{initial: &Backoff{Duration: time.Second, Factor: 2, Steps: 3, Cap: 3 * time.Second}, want: []time.Duration{1 * time.Second, 2 * time.Second, 3 * time.Second}},
		{initial: &Backoff{Duration: time.Second, Factor: 2, Steps: 2, Cap: 3 * time.Second, Jitter: 0.5}, want: []time.Duration{2 * time.Second, 3 * time.Second, 3 * time.Second}},
		{initial: &Backoff{Duration: time.Second, Factor: 2, Steps: 6, Jitter: 4}, want: []time.Duration{1 * time.Second, 2 * time.Second, 4 * time.Second, 8 * time.Second, 16 * time.Second, 32 * time.Second}},
	}
	for seed := int64(0); seed < 5; seed++ {
		for _, tt := range tests {
			initial := *tt.initial
			t.Run(fmt.Sprintf("%#v seed=%d", initial, seed), func(t *testing.T) {
				rand.Seed(seed)
				for i := 0; i < len(tt.want); i++ {
					got := initial.Step()
					t.Logf("[%d]=%s", i, got)
					if initial.Jitter > 0 {
						if got == tt.want[i] {
							// this is statistically unlikely to happen by chance
							t.Errorf("Backoff.Step(%d) = %v, no jitter", i, got)
							continue
						}
						diff := float64(tt.want[i]-got) / float64(tt.want[i])
						if diff > initial.Jitter {
							t.Errorf("Backoff.Step(%d) = %v, want %v, outside range", i, got, tt.want)
							continue
						}
					} else {
						if got != tt.want[i] {
							t.Errorf("Backoff.Step(%d) = %v, want %v", i, got, tt.want)
							continue
						}
					}
				}
			})
		}
	}
}

func TestContextForChannel(t *testing.T) {
	var wg sync.WaitGroup
	parentCh := make(chan struct{})
	done := make(chan struct{})

	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := contextForChannel(parentCh)
			defer cancel()
			<-ctx.Done()
		}()
	}

	go func() {
		wg.Wait()
		close(done)
	}()

	// Closing parent channel should cancel all children contexts
	close(parentCh)

	select {
	case <-done:
	case <-time.After(ForeverTestTimeout):
		t.Errorf("unexepcted timeout waiting for parent to cancel child contexts")
	}
}

func TestExponentialBackoffManagerGetNextBackoff(t *testing.T) {
	fc := clock.NewFakeClock(time.Now())
	backoff := NewExponentialBackoffManager(1, 10, 10, 2.0, 0.0, fc)
	durations := []time.Duration{1, 2, 4, 8, 10, 10, 10}
	for i := 0; i < len(durations); i++ {
		generatedBackoff := backoff.(*exponentialBackoffManagerImpl).getNextBackoff()
		if generatedBackoff != durations[i] {
			t.Errorf("unexpected %d-th backoff: %d, expecting %d", i, generatedBackoff, durations[i])
		}
	}

	fc.Step(11)
	resetDuration := backoff.(*exponentialBackoffManagerImpl).getNextBackoff()
	if resetDuration != 1 {
		t.Errorf("after reset, backoff should be 1, but got %d", resetDuration)
	}
}

func TestJitteredBackoffManagerGetNextBackoff(t *testing.T) {
	// positive jitter
	backoffMgr := NewJitteredBackoffManager(1, 1, clock.NewFakeClock(time.Now()))
	for i := 0; i < 5; i++ {
		backoff := backoffMgr.(*jitteredBackoffManagerImpl).getNextBackoff()
		if backoff < 1 || backoff > 2 {
			t.Errorf("backoff out of range: %d", backoff)
		}
	}

	// negative jitter, shall be a fixed backoff
	backoffMgr = NewJitteredBackoffManager(1, -1, clock.NewFakeClock(time.Now()))
	backoff := backoffMgr.(*jitteredBackoffManagerImpl).getNextBackoff()
	if backoff != 1 {
		t.Errorf("backoff should be 1, but got %d", backoff)
	}
}

func TestJitterBackoffManagerWithRealClock(t *testing.T) {
	backoffMgr := NewJitteredBackoffManager(1*time.Millisecond, 0, &clock.RealClock{})
	for i := 0; i < 5; i++ {
		start := time.Now()
		<-backoffMgr.Backoff().C()
		passed := time.Now().Sub(start)
		if passed < 1*time.Millisecond {
			t.Errorf("backoff should be at least 1ms, but got %s", passed.String())
		}
	}
}

func TestExponentialBackoffManagerWithRealClock(t *testing.T) {
	// backoff at least 1ms, 2ms, 4ms, 8ms, 10ms, 10ms, 10ms
	durationFactors := []time.Duration{1, 2, 4, 8, 10, 10, 10}
	backoffMgr := NewExponentialBackoffManager(1*time.Millisecond, 10*time.Millisecond, 1*time.Hour, 2.0, 0.0, &clock.RealClock{})

	for i := range durationFactors {
		start := time.Now()
		<-backoffMgr.Backoff().C()
		passed := time.Now().Sub(start)
		if passed < durationFactors[i]*time.Millisecond {
			t.Errorf("backoff should be at least %d ms, but got %s", durationFactors[i], passed.String())
		}
	}
}

func TestExponentialBackoffWithContext(t *testing.T) {
	defaultCtx := func() context.Context {
		return context.Background()
	}

	defaultCallback := func(_ int) (bool, error) {
		return false, nil
	}

	conditionErr := errors.New("condition failed")

	tests := []struct {
		name             string
		steps            int
		ctxGetter        func() context.Context
		callback         func(calls int) (bool, error)
		attemptsExpected int
		errExpected      error
	}{
		{
			name:             "no attempts expected with zero backoff steps",
			steps:            0,
			ctxGetter:        defaultCtx,
			callback:         defaultCallback,
			attemptsExpected: 0,
			errExpected:      ErrWaitTimeout,
		},
		{
			name:             "condition returns false with single backoff step",
			steps:            1,
			ctxGetter:        defaultCtx,
			callback:         defaultCallback,
			attemptsExpected: 1,
			errExpected:      ErrWaitTimeout,
		},
		{
			name:      "condition returns true with single backoff step",
			steps:     1,
			ctxGetter: defaultCtx,
			callback: func(_ int) (bool, error) {
				return true, nil
			},
			attemptsExpected: 1,
			errExpected:      nil,
		},
		{
			name:             "condition always returns false with multiple backoff steps",
			steps:            5,
			ctxGetter:        defaultCtx,
			callback:         defaultCallback,
			attemptsExpected: 5,
			errExpected:      ErrWaitTimeout,
		},
		{
			name:      "condition returns true after certain attempts with multiple backoff steps",
			steps:     5,
			ctxGetter: defaultCtx,
			callback: func(attempts int) (bool, error) {
				if attempts == 3 {
					return true, nil
				}
				return false, nil
			},
			attemptsExpected: 3,
			errExpected:      nil,
		},
		{
			name:      "condition returns error no further attempts expected",
			steps:     5,
			ctxGetter: defaultCtx,
			callback: func(_ int) (bool, error) {
				return true, conditionErr
			},
			attemptsExpected: 1,
			errExpected:      conditionErr,
		},
		{
			name:  "context already canceled no attempts expected",
			steps: 5,
			ctxGetter: func() context.Context {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				return ctx
			},
			callback:         defaultCallback,
			attemptsExpected: 0,
			errExpected:      context.Canceled,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			backoff := Backoff{
				Duration: 1 * time.Millisecond,
				Factor:   1.0,
				Steps:    test.steps,
			}

			attempts := 0
			err := ExponentialBackoffWithContext(test.ctxGetter(), backoff, func() (bool, error) {
				attempts++
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
