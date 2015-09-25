/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package util

import (
	"testing"
	"time"
)

func NewFakeBackOff(initial, max time.Duration, tc *FakeClock) *Backoff {
	return &Backoff{
		perItemBackoff:  map[string]*backoffEntry{},
		Clock:           tc,
		defaultDuration: initial,
		maxDuration:     max,
	}
}

func TestSlowBackoff(t *testing.T) {
	id := "_idSlow"
	tc := &FakeClock{Time: time.Now()}
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
}

func TestBackoffReset(t *testing.T) {
	id := "_idReset"
	tc := &FakeClock{Time: time.Now()}
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
		t.Errorf("now=%s lastUpdate=%s (%s) expected Backoff reset got %s b.lastUpdate=%s", tc.Now(), startTime, tc.Now().Sub(lastUpdate), b.Get(id))
	}
}

func TestBackoffHightWaterMark(t *testing.T) {
	id := "_idHiWaterMark"
	tc := &FakeClock{Time: time.Now()}
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
	tc := &FakeClock{Time: time.Now()}
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
		t.Errorf("expected GC to skip entry, elapsed time=%s maxDuration=%s", tc.Now().Sub(lastUpdate), maxDuration)
	}

	tc.Step(maxDuration + step)
	b.GC()
	r, found := b.perItemBackoff[id]
	if found {
		t.Errorf("expected GC of entry after %s got entry %v", tc.Now().Sub(lastUpdate), r)
	}
}
