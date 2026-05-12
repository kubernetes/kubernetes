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

package waitgroup

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/time/rate"
	"k8s.io/apimachinery/pkg/util/wait"
)

func TestRateLimitedSafeWaitGroup(t *testing.T) {
	// we want to keep track of how many times rate limiter Wait method is
	// being invoked, both before and after the wait group is in waiting mode.
	limiter := &limiterWrapper{}

	// we expect the context passed by the factory to be used
	var cancelInvoked int
	factory := &factory{
		limiter: limiter,
		grace:   2 * time.Second,
		ctx:     context.Background(),
		cancel: func() {
			cancelInvoked++
		},
	}
	target := &rateLimitedSafeWaitGroupWrapper{
		RateLimitedSafeWaitGroup: &RateLimitedSafeWaitGroup{limiter: limiter},
	}

	// two set of requests
	//  - n1: this set will finish using this waitgroup before Wait is invoked
	//  - n2: this set will be in flight after Wait is invoked
	n1, n2 := 100, 101

	// so we know when all requests in n1 are done using the waitgroup
	n1DoneWG := sync.WaitGroup{}

	// so we know when all requests in n2 have called Add,
	// but not finished with the waitgroup yet.
	// this will allow the test to invoke 'Wait' once all requests
	// in n2 have called `Add`, but none has called `Done` yet.
	n2BeforeWaitWG := sync.WaitGroup{}
	// so we know when all requests in n2 have called Done and
	// are finished using the waitgroup
	n2DoneWG := sync.WaitGroup{}

	startCh, blockedCh := make(chan struct{}), make(chan struct{})
	n1DoneWG.Add(n1)
	for i := 0; i < n1; i++ {
		go func() {
			defer n1DoneWG.Done()
			<-startCh

			target.Add(1)
			// let's finish using the waitgroup immediately
			target.Done()
		}()
	}

	n2BeforeWaitWG.Add(n2)
	n2DoneWG.Add(n2)
	for i := 0; i < n2; i++ {
		go func() {
			func() {
				defer n2BeforeWaitWG.Done()
				<-startCh

				target.Add(1)
			}()

			func() {
				defer n2DoneWG.Done()
				// let's wait for the test to instruct the requests in n2
				// that it is time to finish using the waitgroup.
				<-blockedCh

				target.Done()
			}()
		}()
	}

	// initially the count should be zero
	if count := target.Count(); count != 0 {
		t.Errorf("expected count to be zero, but got: %d", count)
	}
	// start the test
	close(startCh)
	// wait for the first set of requests (n1) to be done
	n1DoneWG.Wait()

	// after the first set of requests (n1) are done, the count should be zero
	if invoked := limiter.invoked(); invoked != 0 {
		t.Errorf("expected no call to rate limiter before Wait is called, but got: %d", invoked)
	}

	// make sure all requetss in the second group (n2) have started using the
	// waitgroup (Add invoked) but no request is done using the waitgroup yet.
	n2BeforeWaitWG.Wait()

	// count should be n2, since every request in n2 is still using the waitgroup
	if count := target.Count(); count != n2 {
		t.Errorf("expected count to be: %d, but got: %d", n2, count)
	}

	// time for us to mark the waitgroup as `Waiting`
	waitDoneCh := make(chan waitResult)
	go func() {
		factory.grace = 2 * time.Second
		before, after, err := target.Wait(factory.NewRateLimiter)
		waitDoneCh <- waitResult{before: before, after: after, err: err}
	}()

	// make sure there is no flake in the test due to this race condition
	var waitingGot bool
	wait.PollImmediate(500*time.Millisecond, wait.ForeverTestTimeout, func() (done bool, err error) {
		if waiting := target.Waiting(); waiting {
			waitingGot = true
			return true, nil
		}
		return false, nil
	})
	// verify that the waitgroup is in 'Waiting' mode
	if !waitingGot {
		t.Errorf("expected to be in waiting")
	}

	// we should not allow any new request to use this waitgroup any longer
	if err := target.Add(1); err == nil ||
		!strings.Contains(err.Error(), "add with positive delta after Wait is forbidden") {
		t.Errorf("expected Add to return error while in waiting mode: %v", err)
	}

	// make sure that RateLimitedSafeWaitGroup passes the right
	// request count to the limiter factory.
	if factory.countGot != n2 {
		t.Errorf("expected count passed to factory to be: %d, but got: %d", n2, factory.countGot)
	}

	// indicate to all requests (each request in n2) that are
	// currently using this waitgroup that they can go ahead
	// and invoke 'Done' to finish using this waitgroup.
	close(blockedCh)
	n2DoneWG.Wait()

	if invoked := limiter.invoked(); invoked != n2 {
		t.Errorf("expected rate limiter to be called %d times, but got: %d", n2, invoked)
	}

	waitResult := <-waitDoneCh
	if count := target.Count(); count != 0 {
		t.Errorf("expected count to be zero, but got: %d", count)
	}
	if waitResult.before != n2 {
		t.Errorf("expected count before Wait to be: %d, but got: %d", n2, waitResult.before)
	}
	if waitResult.after != 0 {
		t.Errorf("expected count after Wait to be zero, but got: %d", waitResult.after)
	}
	if cancelInvoked != 1 {
		t.Errorf("expected context cancel to be invoked once, but got: %d", cancelInvoked)
	}
}

func TestRateLimitedSafeWaitGroupWithHardTimeout(t *testing.T) {
	target := &rateLimitedSafeWaitGroupWrapper{
		RateLimitedSafeWaitGroup: &RateLimitedSafeWaitGroup{},
	}
	n := 10
	wg := sync.WaitGroup{}
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			target.Add(1)
		}()
	}

	wg.Wait()
	if count := target.Count(); count != n {
		t.Errorf("expected count to be: %d, but got: %d", n, count)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	activeAt, activeNow, err := target.Wait(func(count int) (RateLimiter, context.Context, context.CancelFunc) {
		return nil, ctx, cancel
	})
	if activeAt != n {
		t.Errorf("expected active at Wait to be: %d, but got: %d", n, activeAt)
	}
	if activeNow != n {
		t.Errorf("expected active after Wait to be: %d, but got: %d", n, activeNow)
	}
	if err != context.Canceled {
		t.Errorf("expected error: %v, but got: %v", context.Canceled, err)
	}
}

func TestRateLimitedSafeWaitGroupWithBurstOfOne(t *testing.T) {
	target := &rateLimitedSafeWaitGroupWrapper{
		RateLimitedSafeWaitGroup: &RateLimitedSafeWaitGroup{},
	}
	n := 200
	grace := 5 * time.Second
	wg := sync.WaitGroup{}
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			target.Add(1)
		}()
	}
	wg.Wait()

	waitingCh := make(chan struct{})
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()

			<-waitingCh
			target.Done()
		}()
	}
	defer wg.Wait()

	now := time.Now()
	t.Logf("Wait starting, N=%d, grace: %s, at: %s", n, grace, now)
	activeAt, activeNow, err := target.Wait(func(count int) (RateLimiter, context.Context, context.CancelFunc) {
		defer close(waitingCh)
		// no deadline in context, Wait will wait forever, we want to measure
		// how long it takes for the requests to drain.
		return rate.NewLimiter(rate.Limit(n/int(grace.Seconds())), 1), context.Background(), func() {}
	})
	took := time.Since(now)
	t.Logf("Wait finished, count(before): %d, count(after): %d, took: %s, err: %v", activeAt, activeNow, took, err)

	// in CPU starved environment, the go routines may not finish in time
	if took > 2*grace {
		t.Errorf("expected Wait to take: %s, but it took: %s", grace, took)
	}
}

type waitResult struct {
	before, after int
	err           error
}

type rateLimitedSafeWaitGroupWrapper struct {
	*RateLimitedSafeWaitGroup
}

// used by test only
func (wg *rateLimitedSafeWaitGroupWrapper) Count() int {
	wg.mu.Lock()
	defer wg.mu.Unlock()

	return wg.count
}
func (wg *rateLimitedSafeWaitGroupWrapper) Waiting() bool {
	wg.mu.Lock()
	defer wg.mu.Unlock()

	return wg.wait
}

type limiterWrapper struct {
	delegate RateLimiter
	lock     sync.Mutex
	invokedN int
}

func (w *limiterWrapper) invoked() int {
	w.lock.Lock()
	defer w.lock.Unlock()
	return w.invokedN
}
func (w *limiterWrapper) Wait(ctx context.Context) error {
	w.lock.Lock()
	w.invokedN++
	w.lock.Unlock()

	if w.delegate != nil {
		w.delegate.Wait(ctx)
	}
	return nil
}

type factory struct {
	limiter  *limiterWrapper
	grace    time.Duration
	ctx      context.Context
	cancel   context.CancelFunc
	countGot int
}

func (f *factory) NewRateLimiter(count int) (RateLimiter, context.Context, context.CancelFunc) {
	f.countGot = count
	f.limiter.delegate = rate.NewLimiter(rate.Limit(count/int(f.grace.Seconds())), 20)
	return f.limiter, f.ctx, f.cancel
}
