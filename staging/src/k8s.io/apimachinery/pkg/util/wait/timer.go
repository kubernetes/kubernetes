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
	"time"

	"k8s.io/utils/clock"
)

// Timer abstracts how wait functions interact with time runtime efficiently. Test
// code may implement this interface directly but package consumers are encouraged
// to use the Backoff type as the primary mechanism for acquiring a Timer. The
// interface is a simplification of clock.Timer to prevent misuse. Timers are not
// expected to be safe for calls from multiple goroutines.
type Timer interface {
	// C returns a channel that will receive a struct{} each time the timer fires.
	// The channel should not be waited on after Stop() is invoked. It is allowed
	// to cache the returned value of C() for the lifetime of the Timer.
	C() <-chan time.Time
	// Next is invoked by wait functions to signal timers that the next interval
	// should begin. You may only use Next() if you have drained the channel C().
	// You should not call Next() after Stop() is invoked.
	Next()
	// Stop releases the timer. It is safe to invoke if no other methods have been
	// called.
	Stop()
}

type noopTimer struct {
	closedCh <-chan time.Time
}

// newNoopTimer creates a timer with a unique channel to avoid contention
// for the channel's lock across multiple unrelated timers.
func newNoopTimer() noopTimer {
	ch := make(chan time.Time)
	close(ch)
	return noopTimer{closedCh: ch}
}

func (t noopTimer) C() <-chan time.Time {
	return t.closedCh
}
func (noopTimer) Next() {}
func (noopTimer) Stop() {}

type variableTimer struct {
	fn  DelayFunc
	t   clock.Timer
	new func(time.Duration) clock.Timer
}

func (t *variableTimer) C() <-chan time.Time {
	if t.t == nil {
		d := t.fn()
		t.t = t.new(d)
	}
	return t.t.C()
}
func (t *variableTimer) Next() {
	if t.t == nil {
		return
	}
	d := t.fn()
	t.t.Reset(d)
}
func (t *variableTimer) Stop() {
	if t.t == nil {
		return
	}
	t.t.Stop()
	t.t = nil
}

type fixedTimer struct {
	interval time.Duration
	t        clock.Ticker
	new      func(time.Duration) clock.Ticker
}

func (t *fixedTimer) C() <-chan time.Time {
	if t.t == nil {
		t.t = t.new(t.interval)
	}
	return t.t.C()
}
func (t *fixedTimer) Next() {
	// no-op for fixed timers
}
func (t *fixedTimer) Stop() {
	if t.t == nil {
		return
	}
	t.t.Stop()
	t.t = nil
}

var (
	// RealTimer can be passed to methods that need a clock.Timer.
	RealTimer = clock.RealClock{}.NewTimer
)

var (
	// internalClock is used for test injection of clocks
	internalClock = clock.RealClock{}
)
