/*
Copyright 2015 The Kubernetes Authors.

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

package testing

import (
	"testing"
	"time"

	"k8s.io/utils/clock"
)

type SettablePassiveClock interface {
	clock.PassiveClock
	SetTime(time.Time)
}

func exercisePassiveClock(t *testing.T, pc SettablePassiveClock) {
	t1 := time.Now()
	t2 := t1.Add(time.Hour)
	pc.SetTime(t1)
	tx := pc.Now()
	if tx != t1 {
		t.Errorf("SetTime(%#+v); Now() => %#+v", t1, tx)
	}
	dx := pc.Since(t1)
	if dx != 0 {
		t.Errorf("Since() => %v", dx)
	}
	pc.SetTime(t2)
	dx = pc.Since(t1)
	if dx != time.Hour {
		t.Errorf("Since() => %v", dx)
	}
	tx = pc.Now()
	if tx != t2 {
		t.Errorf("Now() => %#+v", tx)
	}
}

func TestFakePassiveClock(t *testing.T) {
	startTime := time.Now()
	tc := NewFakePassiveClock(startTime)
	exercisePassiveClock(t, tc)
}

func TestFakeClock(t *testing.T) {
	startTime := time.Now()
	tc := NewFakeClock(startTime)
	exercisePassiveClock(t, tc)
	tc.SetTime(startTime)
	tc.Step(time.Second)
	now := tc.Now()
	if now.Sub(startTime) != time.Second {
		t.Errorf("input: %s now=%s gap=%s expected=%s", startTime, now, now.Sub(startTime), time.Second)
	}
}

func TestFakeClockSleep(t *testing.T) {
	startTime := time.Now()
	tc := NewFakeClock(startTime)
	tc.Sleep(time.Duration(1) * time.Hour)
	now := tc.Now()
	if now.Sub(startTime) != time.Hour {
		t.Errorf("Fake sleep failed, expected time to advance by one hour, instead, its %v", now.Sub(startTime))
	}
}

func TestFakeAfter(t *testing.T) {
	tc := NewFakeClock(time.Now())
	if tc.HasWaiters() {
		t.Errorf("unexpected waiter?")
	}
	oneSec := tc.After(time.Second)
	if !tc.HasWaiters() {
		t.Errorf("unexpected lack of waiter?")
	}

	oneOhOneSec := tc.After(time.Second + time.Millisecond)
	twoSec := tc.After(2 * time.Second)
	select {
	case <-oneSec:
		t.Errorf("unexpected channel read")
	case <-oneOhOneSec:
		t.Errorf("unexpected channel read")
	case <-twoSec:
		t.Errorf("unexpected channel read")
	default:
	}

	tc.Step(999 * time.Millisecond)
	select {
	case <-oneSec:
		t.Errorf("unexpected channel read")
	case <-oneOhOneSec:
		t.Errorf("unexpected channel read")
	case <-twoSec:
		t.Errorf("unexpected channel read")
	default:
	}

	tc.Step(time.Millisecond)
	select {
	case <-oneSec:
		// Expected!
	case <-oneOhOneSec:
		t.Errorf("unexpected channel read")
	case <-twoSec:
		t.Errorf("unexpected channel read")
	default:
		t.Errorf("unexpected non-channel read")
	}
	tc.Step(time.Millisecond)
	select {
	case <-oneSec:
		// should not double-trigger!
		t.Errorf("unexpected channel read")
	case <-oneOhOneSec:
		// Expected!
	case <-twoSec:
		t.Errorf("unexpected channel read")
	default:
		t.Errorf("unexpected non-channel read")
	}
}

func TestFakeTick(t *testing.T) {
	tc := NewFakeClock(time.Now())
	if tc.HasWaiters() {
		t.Errorf("unexpected waiter?")
	}
	oneSec := tc.Tick(time.Second)
	if !tc.HasWaiters() {
		t.Errorf("unexpected lack of waiter?")
	}

	oneOhOneSec := tc.Tick(time.Second + time.Millisecond)
	twoSec := tc.Tick(2 * time.Second)
	select {
	case <-oneSec:
		t.Errorf("unexpected channel read")
	case <-oneOhOneSec:
		t.Errorf("unexpected channel read")
	case <-twoSec:
		t.Errorf("unexpected channel read")
	default:
	}

	tc.Step(999 * time.Millisecond) // t=.999
	select {
	case <-oneSec:
		t.Errorf("unexpected channel read")
	case <-oneOhOneSec:
		t.Errorf("unexpected channel read")
	case <-twoSec:
		t.Errorf("unexpected channel read")
	default:
	}

	tc.Step(time.Millisecond) // t=1.000
	select {
	case <-oneSec:
		// Expected!
	case <-oneOhOneSec:
		t.Errorf("unexpected channel read")
	case <-twoSec:
		t.Errorf("unexpected channel read")
	default:
		t.Errorf("unexpected non-channel read")
	}
	tc.Step(time.Millisecond) // t=1.001
	select {
	case <-oneSec:
		// should not double-trigger!
		t.Errorf("unexpected channel read")
	case <-oneOhOneSec:
		// Expected!
	case <-twoSec:
		t.Errorf("unexpected channel read")
	default:
		t.Errorf("unexpected non-channel read")
	}

	tc.Step(time.Second) // t=2.001
	tc.Step(time.Second) // t=3.001
	tc.Step(time.Second) // t=4.001
	tc.Step(time.Second) // t=5.001

	// The one second ticker should not accumulate ticks
	accumulatedTicks := 0
	drained := false
	for !drained {
		select {
		case <-oneSec:
			accumulatedTicks++
		default:
			drained = true
		}
	}
	if accumulatedTicks != 1 {
		t.Errorf("unexpected number of accumulated ticks: %d", accumulatedTicks)
	}
}

func TestFakeStop(t *testing.T) {
	tc := NewFakeClock(time.Now())
	timer := tc.NewTimer(time.Second)
	if !tc.HasWaiters() {
		t.Errorf("expected a waiter to be present, but it is not")
	}
	timer.Stop()
	if tc.HasWaiters() {
		t.Errorf("expected existing waiter to be cleaned up, but it is still present")
	}
}

// This tests the pattern documented in the go docs here: https://golang.org/pkg/time/#Timer.Stop
// This pattern is required to safely reset a timer, so should be common.
// This also tests resetting the timer
func TestFakeStopDrain(t *testing.T) {
	start := time.Time{}
	tc := NewFakeClock(start)
	timer := tc.NewTimer(time.Second)
	tc.Step(1 * time.Second)
	// Effectively `if !timer.Stop { <-t.C }` but with more asserts
	if timer.Stop() {
		t.Errorf("stop should report the timer had triggered")
	}
	if readTime := assertReadTime(t, timer.C()); !readTime.Equal(start.Add(1 * time.Second)) {
		t.Errorf("timer should have ticked after 1 second, got %v", readTime)
	}

	timer.Reset(time.Second)
	if !tc.HasWaiters() {
		t.Errorf("expected a waiter to be present, but it is not")
	}
	select {
	case <-timer.C():
		t.Fatal("got time early on clock; haven't stepped yet")
	default:
	}
	tc.Step(1 * time.Second)
	if readTime := assertReadTime(t, timer.C()); !readTime.Equal(start.Add(2 * time.Second)) {
		t.Errorf("timer should have ticked again after reset + 1 more second, got %v", readTime)
	}
}

func TestTimerNegative(t *testing.T) {
	tc := NewFakeClock(time.Now())
	timer := tc.NewTimer(-1 * time.Second)
	if !tc.HasWaiters() {
		t.Errorf("expected a waiter to be present, but it is not")
	}
	// force waiters to be called
	tc.Step(0)
	tick := assertReadTime(t, timer.C())
	if tick != tc.Now() {
		t.Errorf("expected -1s to turn into now: %v != %v", tick, tc.Now())
	}
}

func TestTickNegative(t *testing.T) {
	// The stdlib 'Tick' returns nil for negative and zero values, so our fake
	// should too.
	tc := NewFakeClock(time.Now())
	if tick := tc.Tick(-1 * time.Second); tick != nil {
		t.Errorf("expected negative tick to be nil: %v", tick)
	}
	if tick := tc.Tick(0); tick != nil {
		t.Errorf("expected negative tick to be nil: %v", tick)
	}
}

// assertReadTime asserts that the channel can be read and returns the time it
// reads from the channel.
func assertReadTime(t testing.TB, c <-chan time.Time) time.Time {
	type helper interface {
		Helper()
	}
	if h, ok := t.(helper); ok {
		h.Helper()
	}
	select {
	case ti, ok := <-c:
		if !ok {
			t.Fatalf("expected to read time from channel, but it was closed")
		}
		return ti
	default:
		t.Fatalf("expected to read time from channel, but couldn't")
	}
	panic("unreachable")
}
