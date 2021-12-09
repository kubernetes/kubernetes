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

package clock

import (
	"time"

	clocks "k8s.io/utils/clock"
	testclocks "k8s.io/utils/clock/testing"
)

// PassiveClock allows for injecting fake or real clocks into code
// that needs to read the current time but does not support scheduling
// activity in the future.
//
// Deprecated: Use k8s.io/utils/clock.PassiveClock instead.
type PassiveClock = clocks.PassiveClock

// Clock allows for injecting fake or real clocks into code that
// needs to do arbitrary things based on time.
//
// Deprecated: Use k8s.io/utils/clock.WithTickerAndDelayedExecution instead.
type Clock = clocks.WithTickerAndDelayedExecution

// Deprecated: Use k8s.io/utils/clock.RealClock instead.
type RealClock = clocks.RealClock

// FakePassiveClock implements PassiveClock, but returns an arbitrary time.
//
// Deprecated: Use k8s.io/utils/clock/testing.FakePassiveClock instead.
type FakePassiveClock = testclocks.FakePassiveClock

// FakeClock implements Clock, but returns an arbitrary time.
//
// Deprecated: Use k8s.io/utils/clock/testing.FakeClock instead.
type FakeClock = testclocks.FakeClock

// NewFakePassiveClock returns a new FakePassiveClock.
//
// Deprecated: Use k8s.io/utils/clock/testing.NewFakePassiveClock instead.
func NewFakePassiveClock(t time.Time) *testclocks.FakePassiveClock {
	return testclocks.NewFakePassiveClock(t)
}

// NewFakeClock returns a new FakeClock.
//
// Deprecated: Use k8s.io/utils/clock/testing.NewFakeClock instead.
func NewFakeClock(t time.Time) *testclocks.FakeClock {
	return testclocks.NewFakeClock(t)
}

// IntervalClock implements Clock, but each invocation of Now steps
// the clock forward the specified duration.
//
// WARNING: most of the Clock methods just `panic`;
// only PassiveClock is honestly implemented.
// The alternative, SimpleIntervalClock, has only the
// PassiveClock methods.
//
// Deprecated: Use k8s.io/utils/clock/testing.SimpleIntervalClock instead.
type IntervalClock = testclocks.IntervalClock

// Timer allows for injecting fake or real timers into code that
// needs to do arbitrary things based on time.
//
// Deprecated: Use k8s.io/utils/clock.Timer instead.
type Timer = clocks.Timer

// Ticker defines the Ticker interface.
//
// Deprecated: Use k8s.io/utils/clock.Ticker instead.
type Ticker = clocks.Ticker
