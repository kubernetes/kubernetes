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

package clock

import (
	"testing"
	"time"
)

var (
	_ = Clock(RealClock{})
	_ = Clock(&FakeClock{})
	_ = Clock(&IntervalClock{})

	_ = Timer(&realTimer{})
	_ = Timer(&fakeTimer{})

	_ = Ticker(&realTicker{})
	_ = Ticker(&fakeTicker{})
)

type SettablePassiveClock interface {
	PassiveClock
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

func TestFakeTimer(t *testing.T) {
	tc := NewFakeClock(time.Now())
	if tc.HasWaiters() {
		t.Errorf("unexpected waiter?")
	}
	oneSec := tc.NewTimer(time.Second)
	twoSec := tc.NewTimer(time.Second * 2)
	treSec := tc.NewTimer(time.Second * 3)
	if !tc.HasWaiters() {
		t.Errorf("unexpected lack of waiter?")
	}
	select {
	case <-oneSec.C():
		t.Errorf("unexpected channel read")
	case <-twoSec.C():
		t.Errorf("unexpected channel read")
	case <-treSec.C():
		t.Errorf("unexpected channel read")
	default:
	}
	tc.Step(999999999 * time.Nanosecond) // t=.999,999,999
	select {
	case <-oneSec.C():
		t.Errorf("unexpected channel read")
	case <-twoSec.C():
		t.Errorf("unexpected channel read")
	case <-treSec.C():
		t.Errorf("unexpected channel read")
	default:
	}
	tc.Step(time.Nanosecond) // t=1
	select {
	case <-twoSec.C():
		t.Errorf("unexpected channel read")
	case <-treSec.C():
		t.Errorf("unexpected channel read")
	default:
	}
	select {
	case <-oneSec.C():
		// Expected!
	default:
		t.Errorf("unexpected channel non-read")
	}
	tc.Step(time.Nanosecond) // t=1.000,000,001
	select {
	case <-oneSec.C():
		t.Errorf("unexpected channel read")
	case <-twoSec.C():
		t.Errorf("unexpected channel read")
	case <-treSec.C():
		t.Errorf("unexpected channel read")
	default:
	}
	if oneSec.Stop() {
		t.Errorf("Expected oneSec.Stop() to return false")
	}
	if !twoSec.Stop() {
		t.Errorf("Expected twoSec.Stop() to return true")
	}
	tc.Step(time.Second) // t=2.000,000,001
	select {
	case <-oneSec.C():
		t.Errorf("unexpected channel read")
	case <-twoSec.C():
		t.Errorf("unexpected channel read")
	case <-treSec.C():
		t.Errorf("unexpected channel read")
	default:
	}
	if twoSec.Reset(time.Second) {
		t.Errorf("Expected twoSec.Reset() to return false")
	}
	if !treSec.Reset(time.Second) {
		t.Errorf("Expected treSec.Reset() to return true")
	}
	tc.Step(time.Nanosecond * 999999999) // t=3.0
	select {
	case <-oneSec.C():
		t.Errorf("unexpected channel read")
	case <-twoSec.C():
		t.Errorf("unexpected channel read")
	case <-treSec.C():
		t.Errorf("unexpected channel read")
	default:
	}
	tc.Step(time.Nanosecond) // t=3.000,000,001
	select {
	case <-oneSec.C():
		t.Errorf("unexpected channel read")
	case <-twoSec.C():
		t.Errorf("unexpected channel read")
	default:
	}
	select {
	case <-treSec.C():
		// Expected!
	default:
		t.Errorf("unexpected channel non-read")
	}
}

func TestFakeTick(t *testing.T) {
	tc := NewFakeClock(time.Now())
	if tc.HasWaiters() {
		t.Errorf("unexpected waiter?")
	}
	oneSec := tc.NewTicker(time.Second).C()
	if !tc.HasWaiters() {
		t.Errorf("unexpected lack of waiter?")
	}

	oneOhOneSec := tc.NewTicker(time.Second + time.Millisecond).C()
	twoSec := tc.NewTicker(2 * time.Second).C()
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
