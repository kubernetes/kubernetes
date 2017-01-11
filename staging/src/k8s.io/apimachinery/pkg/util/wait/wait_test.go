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
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

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

func TestWaitForWithDelay(t *testing.T) {
	done := make(chan struct{})
	defer close(done)
	WaitFor(poller(time.Millisecond, ForeverTestTimeout), func() (bool, error) {
		time.Sleep(10 * time.Millisecond)
		return true, nil
	}, done)
	// If polling goroutine doesn't see the done signal it will leak timers.
	select {
	case done <- struct{}{}:
	case <-time.After(ForeverTestTimeout):
		t.Errorf("expected an ack of the done signal.")
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
