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
	"math/rand"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/runtime"
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

// jitterRand is a dedicated random source for jitter calculations.
// It defaults to rand.Float64, but is a package variable so it can be overridden to make unit tests deterministic.
var jitterRand = rand.Float64

// Jitter returns a time.Duration between duration and duration + maxFactor *
// duration.
//
// This allows clients to avoid converging on periodic behavior. If maxFactor
// is 0.0, a suggested default value will be chosen.
func Jitter(duration time.Duration, maxFactor float64) time.Duration {
	if maxFactor <= 0.0 {
		maxFactor = 1.0
	}
	wait := duration + time.Duration(jitterRand()*maxFactor*float64(duration))
	return wait
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

// runConditionWithCrashProtection runs a ConditionFunc with crash protection.
//
// Deprecated: Will be removed when the legacy polling methods are removed.
func runConditionWithCrashProtection(condition ConditionFunc) (bool, error) {
	//nolint:logcheck // Already deprecated.
	defer runtime.HandleCrash()
	return condition()
}

// runConditionWithCrashProtectionWithContext runs a ConditionWithContextFunc
// with crash protection.
//
// Deprecated: Will be removed when the legacy polling methods are removed.
func runConditionWithCrashProtectionWithContext(ctx context.Context, condition ConditionWithContextFunc) (bool, error) {
	defer runtime.HandleCrashWithContext(ctx)
	return condition(ctx)
}

// waitFunc creates a channel that receives an item every time a test
// should be executed and is closed when the last test should be invoked.
//
// Deprecated: Will be removed in a future release in favor of
// loopConditionUntilContext.
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
// Deprecated: Will be removed in a future release in favor of
// loopConditionUntilContext.
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
// Deprecated: Will be removed in a future release in favor of
// loopConditionUntilContext.
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
			// returning ctx.Err() will break backward compatibility, use new PollUntilContext*
			// methods instead
			return ErrWaitTimeout
		}
	}
}
