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

package flowcontrol

import (
	"testing"
	"time"

	testingclock "k8s.io/utils/clock/testing"
)

func TestSlowBackoff(t *testing.T) {
	id := "_idSlow"
	tc := testingclock.NewFakeClock(time.Now())
	step := time.Second
	maxDuration := 50 * step

	b := NewFakeBackOff(step, maxDuration, tc)
	cases := []time.Duration{0, 1, 2, 4, 8, 16, 32, 50, 50, 50}
	for ix, c := range cases {
		tc.Step(step)
		w := b.Get(id)
		if w != c*step {
			t.Errorf("input: '%d': expected %s, got %s", ix, c*step, w)
		}
		b.Next(id, tc.Now())
	}

	//Now confirm that the Reset cancels backoff.
	b.Next(id, tc.Now())
	b.Reset(id)
	if b.Get(id) != 0 {
		t.Errorf("Reset didn't clear the backoff.")
	}

}

func TestBackoffReset(t *testing.T) {
	id := "_idReset"
	tc := testingclock.NewFakeClock(time.Now())
	step := time.Second
	maxDuration := step * 5
	b := NewFakeBackOff(step, maxDuration, tc)
	startTime := tc.Now()

	// get to backoff = maxDuration
	for i := 0; i <= int(maxDuration/step); i++ {
		tc.Step(step)
		b.Next(id, tc.Now())
	}

	// backoff should be capped at maxDuration
	if !b.IsInBackOffSince(id, tc.Now()) {
		t.Errorf("expected to be in Backoff got %s", b.Get(id))
	}

	lastUpdate := tc.Now()
	tc.Step(2*maxDuration + step) // time += 11s, 11 > 2*maxDuration
	if b.IsInBackOffSince(id, lastUpdate) {
		t.Errorf("expected to not be in Backoff after reset (start=%s, now=%s, lastUpdate=%s), got %s", startTime, tc.Now(), lastUpdate, b.Get(id))
	}
}

func TestBackoffHighWaterMark(t *testing.T) {
	id := "_idHiWaterMark"
	tc := testingclock.NewFakeClock(time.Now())
	step := time.Second
	maxDuration := 5 * step
	b := NewFakeBackOff(step, maxDuration, tc)

	// get to backoff = maxDuration
	for i := 0; i <= int(maxDuration/step); i++ {
		tc.Step(step)
		b.Next(id, tc.Now())
	}

	// backoff high watermark expires after 2*maxDuration
	tc.Step(maxDuration + step)
	b.Next(id, tc.Now())

	if b.Get(id) != maxDuration {
		t.Errorf("expected Backoff to stay at high watermark %s got %s", maxDuration, b.Get(id))
	}
}

func TestBackoffGC(t *testing.T) {
	id := "_idGC"
	tc := testingclock.NewFakeClock(time.Now())
	step := time.Second
	maxDuration := 5 * step

	b := NewFakeBackOff(step, maxDuration, tc)

	for i := 0; i <= int(maxDuration/step); i++ {
		tc.Step(step)
		b.Next(id, tc.Now())
	}
	lastUpdate := tc.Now()
	tc.Step(maxDuration + step)
	b.GC()
	_, found := b.perItemBackoff[id]
	if !found {
		t.Errorf("expected GC to skip entry, elapsed time=%s maxDuration=%s", tc.Since(lastUpdate), maxDuration)
	}

	tc.Step(maxDuration + step)
	b.GC()
	r, found := b.perItemBackoff[id]
	if found {
		t.Errorf("expected GC of entry after %s got entry %v", tc.Since(lastUpdate), r)
	}
}

func TestAlternateBackoffGC(t *testing.T) {
	cases := []struct {
		name           string
		hasExpiredFunc func(time.Time, time.Time, time.Duration) bool
		maxDuration    time.Duration
		nonExpiredTime time.Duration
		expiredTime    time.Duration
	}{
		{
			name:           "default GC",
			maxDuration:    time.Duration(50 * time.Second),
			nonExpiredTime: time.Duration(5 * time.Second),
			expiredTime:    time.Duration(101 * time.Second),
		},
		{
			name: "GC later than 2*maxDuration",
			hasExpiredFunc: func(eventTime time.Time, lastUpdate time.Time, maxDuration time.Duration) bool {
				return eventTime.Sub(lastUpdate) >= 200*time.Second
			},
			maxDuration:    time.Duration(50 * time.Second),
			nonExpiredTime: time.Duration(101 * time.Second),
			expiredTime:    time.Duration(501 * time.Second),
		},
	}

	for _, tt := range cases {
		clock := testingclock.NewFakeClock(time.Now())
		base := time.Second
		maxDuration := tt.maxDuration
		id := tt.name

		b := NewFakeBackOff(base, maxDuration, clock)
		if tt.hasExpiredFunc != nil {
			b.HasExpiredFunc = tt.hasExpiredFunc
		}

		// initialize backoff
		lastUpdate := clock.Now()
		b.Next(id, lastUpdate)

		// increment to a time within GC expiration
		clock.Step(tt.nonExpiredTime)
		b.GC()

		// confirm we did not GC this entry
		_, found := b.perItemBackoff[id]
		if !found {
			t.Errorf("[%s] expected GC to skip entry, elapsed time=%s", tt.name, clock.Since(lastUpdate))
		}

		// increment to a time beyond GC expiration
		clock.Step(tt.expiredTime)
		b.GC()
		r, found := b.perItemBackoff[id]
		if found {
			t.Errorf("[%s] expected GC of entry after %s got entry %v", tt.name, clock.Since(lastUpdate), r)
		}

	}
}

func TestIsInBackOffSinceUpdate(t *testing.T) {
	id := "_idIsInBackOffSinceUpdate"
	tc := testingclock.NewFakeClock(time.Now())
	step := time.Second
	maxDuration := 10 * step
	b := NewFakeBackOff(step, maxDuration, tc)
	startTime := tc.Now()

	cases := []struct {
		tick      time.Duration
		inBackOff bool
		value     int
	}{
		{tick: 0, inBackOff: false, value: 0},
		{tick: 1, inBackOff: false, value: 1},
		{tick: 2, inBackOff: true, value: 2},
		{tick: 3, inBackOff: false, value: 2},
		{tick: 4, inBackOff: true, value: 4},
		{tick: 5, inBackOff: true, value: 4},
		{tick: 6, inBackOff: true, value: 4},
		{tick: 7, inBackOff: false, value: 4},
		{tick: 8, inBackOff: true, value: 8},
		{tick: 9, inBackOff: true, value: 8},
		{tick: 10, inBackOff: true, value: 8},
		{tick: 11, inBackOff: true, value: 8},
		{tick: 12, inBackOff: true, value: 8},
		{tick: 13, inBackOff: true, value: 8},
		{tick: 14, inBackOff: true, value: 8},
		{tick: 15, inBackOff: false, value: 8},
		{tick: 16, inBackOff: true, value: 10},
		{tick: 17, inBackOff: true, value: 10},
		{tick: 18, inBackOff: true, value: 10},
		{tick: 19, inBackOff: true, value: 10},
		{tick: 20, inBackOff: true, value: 10},
		{tick: 21, inBackOff: true, value: 10},
		{tick: 22, inBackOff: true, value: 10},
		{tick: 23, inBackOff: true, value: 10},
		{tick: 24, inBackOff: true, value: 10},
		{tick: 25, inBackOff: false, value: 10},
		{tick: 26, inBackOff: true, value: 10},
		{tick: 27, inBackOff: true, value: 10},
		{tick: 28, inBackOff: true, value: 10},
		{tick: 29, inBackOff: true, value: 10},
		{tick: 30, inBackOff: true, value: 10},
		{tick: 31, inBackOff: true, value: 10},
		{tick: 32, inBackOff: true, value: 10},
		{tick: 33, inBackOff: true, value: 10},
		{tick: 34, inBackOff: true, value: 10},
		{tick: 35, inBackOff: false, value: 10},
		{tick: 56, inBackOff: false, value: 0},
		{tick: 57, inBackOff: false, value: 1},
	}

	for _, c := range cases {
		tc.SetTime(startTime.Add(c.tick * step))
		if c.inBackOff != b.IsInBackOffSinceUpdate(id, tc.Now()) {
			t.Errorf("expected IsInBackOffSinceUpdate %v got %v at tick %s", c.inBackOff, b.IsInBackOffSinceUpdate(id, tc.Now()), c.tick*step)
		}

		if c.inBackOff && (time.Duration(c.value)*step != b.Get(id)) {
			t.Errorf("expected backoff value=%s got %s at tick %s", time.Duration(c.value)*step, b.Get(id), c.tick*step)
		}

		if !c.inBackOff {
			b.Next(id, tc.Now())
		}
	}
}

func TestBackoffWithJitter(t *testing.T) {
	id := "_idJitter"
	tc := testingclock.NewFakeClock(time.Now())

	// test setup: we show 11 iterations, series of delays we expect with
	// a jitter factor of zero each time:
	// 100ms  200ms  400ms  800ms  1.6s  3.2s  06.4s  12.8s  25.6s  51.2s  1m42s
	// and with jitter factor of 0.1 (max) each time:
	// 110ms  231ms  485ms  1.0s   2.1s  4.4s  09.4s  19.8s  41.6s  1m27s  2m6s
	//
	// with the following configuration, it is guaranteed that the maximum delay
	// will be reached even though we are unlucky and get jitter factor of zero.
	// This ensures that this test covers the code path for checking whether
	// maximum delay has been reached with jitter enabled.
	initial := 100 * time.Millisecond
	maxDuration := time.Minute
	maxJitterFactor := 0.1
	attempts := 10

	b := NewFakeBackOffWithJitter(initial, maxDuration, tc, maxJitterFactor)

	assert := func(t *testing.T, factor int, prevDelayGot, curDelayGot time.Duration) {
		low := time.Duration((float64(prevDelayGot) * float64(factor)))
		high := low + time.Duration(maxJitterFactor*float64(prevDelayGot))
		if !((curDelayGot > low && curDelayGot <= high) || curDelayGot == maxDuration) {
			t.Errorf("jittered delay not within range: (%s - %s], but got %s", low, high, curDelayGot)
		}
	}

	delays := make([]time.Duration, 0)
	next := func() time.Duration {
		tc.Step(initial)
		b.Next(id, tc.Now())

		delay := b.Get(id)
		delays = append(delays, delay)
		return delay
	}

	if got := b.Get(id); got != 0 {
		t.Errorf("expected a zero wait durtion, but got: %s", got)
	}

	delayGot := next()
	assert(t, 1, initial, delayGot)

	prevDelayGot := delayGot
	for i := 0; i < attempts; i++ {
		delayGot = next()
		assert(t, 2, prevDelayGot, delayGot)

		prevDelayGot = delayGot
	}

	t.Logf("exponentially backed off jittered delays: %v", delays)
}

func TestAlternateHasExpiredFunc(t *testing.T) {
	cases := []struct {
		name           string
		hasExpiredFunc func(time.Time, time.Time, time.Duration) bool
		maxDuration    time.Duration
		nonExpiredTime time.Duration
		expiredTime    time.Duration
	}{
		{
			name:           "default expiration",
			maxDuration:    time.Duration(50 * time.Second),
			nonExpiredTime: time.Duration(5 * time.Second),
			expiredTime:    time.Duration(101 * time.Second),
		},
		{
			name: "expires faster than maxDuration",
			hasExpiredFunc: func(eventTime time.Time, lastUpdate time.Time, maxDuration time.Duration) bool {
				return eventTime.Sub(lastUpdate) >= 8*time.Second
			},
			maxDuration:    time.Duration(50 * time.Second),
			nonExpiredTime: time.Duration(5 * time.Second),
			expiredTime:    time.Duration(9 * time.Second),
		},
	}

	for _, tt := range cases {
		clock := testingclock.NewFakeClock(time.Now())
		base := time.Second
		maxDuration := tt.maxDuration
		id := tt.name

		b := NewFakeBackOff(base, maxDuration, clock)

		if tt.hasExpiredFunc != nil {
			b.HasExpiredFunc = tt.hasExpiredFunc
		}
		// initialize backoff
		b.Next(id, clock.Now())

		// increment to a time within expiration
		clock.Step(tt.nonExpiredTime)
		b.Next(id, clock.Now())

		// confirm we did a backoff
		w := b.Get(id)
		if w < base*2 {
			t.Errorf("case %v: backoff object has not incremented like expected: want %s, got %s", tt.name, base*2, w)
		}

		// increment to a time beyond expiration
		clock.Step(tt.expiredTime)
		b.Next(id, clock.Now())

		// confirm we have reset the backoff to base
		w = b.Get(id)
		if w != base {
			t.Errorf("case %v: hasexpired value: expected %s (backoff to be reset to initial), got %s", tt.name, base, w)
		}

		clock.SetTime(time.Now())
		b.Reset(id)
	}
}
