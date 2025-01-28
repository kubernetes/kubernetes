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
	"math/rand"
	"sync"
	"time"

	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"
)

type backoffEntry struct {
	backoff    time.Duration
	lastUpdate time.Time
}

type Backoff struct {
	sync.RWMutex
	Clock clock.Clock
	// HasExpiredFunc controls the logic that determines whether the backoff
	// counter should be reset, and when to GC old backoff entries. If nil, the
	// default hasExpired function will restart the backoff factor to the
	// beginning after observing time has passed at least equal to 2*maxDuration
	HasExpiredFunc  func(eventTime time.Time, lastUpdate time.Time, maxDuration time.Duration) bool
	defaultDuration time.Duration
	maxDuration     time.Duration
	perItemBackoff  map[string]*backoffEntry
	rand            *rand.Rand

	// maxJitterFactor adds jitter to the exponentially backed off delay.
	// if maxJitterFactor is zero, no jitter is added to the delay in
	// order to maintain current behavior.
	maxJitterFactor float64
}

func NewFakeBackOff(initial, max time.Duration, tc *testingclock.FakeClock) *Backoff {
	return newBackoff(tc, initial, max, 0.0)
}

func NewBackOff(initial, max time.Duration) *Backoff {
	return NewBackOffWithJitter(initial, max, 0.0)
}

func NewFakeBackOffWithJitter(initial, max time.Duration, tc *testingclock.FakeClock, maxJitterFactor float64) *Backoff {
	return newBackoff(tc, initial, max, maxJitterFactor)
}

func NewBackOffWithJitter(initial, max time.Duration, maxJitterFactor float64) *Backoff {
	clock := clock.RealClock{}
	return newBackoff(clock, initial, max, maxJitterFactor)
}

func newBackoff(clock clock.Clock, initial, max time.Duration, maxJitterFactor float64) *Backoff {
	var random *rand.Rand
	if maxJitterFactor > 0 {
		random = rand.New(rand.NewSource(clock.Now().UnixNano()))
	}
	return &Backoff{
		perItemBackoff:  map[string]*backoffEntry{},
		Clock:           clock,
		defaultDuration: initial,
		maxDuration:     max,
		maxJitterFactor: maxJitterFactor,
		rand:            random,
	}
}

// Get the current backoff Duration
func (p *Backoff) Get(id string) time.Duration {
	p.RLock()
	defer p.RUnlock()
	var delay time.Duration
	entry, ok := p.perItemBackoff[id]
	if ok {
		delay = entry.backoff
	}
	return delay
}

// move backoff to the next mark, capping at maxDuration
func (p *Backoff) Next(id string, eventTime time.Time) {
	p.Lock()
	defer p.Unlock()
	entry, ok := p.perItemBackoff[id]
	if !ok || p.hasExpired(eventTime, entry.lastUpdate, p.maxDuration) {
		entry = p.initEntryUnsafe(id)
		entry.backoff += p.jitter(entry.backoff)
	} else {
		delay := entry.backoff * 2       // exponential
		delay += p.jitter(entry.backoff) // add some jitter to the delay
		entry.backoff = min(delay, p.maxDuration)
	}
	entry.lastUpdate = p.Clock.Now()
}

// Reset forces clearing of all backoff data for a given key.
func (p *Backoff) Reset(id string) {
	p.Lock()
	defer p.Unlock()
	delete(p.perItemBackoff, id)
}

// Returns True if the elapsed time since eventTime is smaller than the current backoff window
func (p *Backoff) IsInBackOffSince(id string, eventTime time.Time) bool {
	p.RLock()
	defer p.RUnlock()
	entry, ok := p.perItemBackoff[id]
	if !ok {
		return false
	}
	if p.hasExpired(eventTime, entry.lastUpdate, p.maxDuration) {
		return false
	}
	return p.Clock.Since(eventTime) < entry.backoff
}

// Returns True if time since lastupdate is less than the current backoff window.
func (p *Backoff) IsInBackOffSinceUpdate(id string, eventTime time.Time) bool {
	p.RLock()
	defer p.RUnlock()
	entry, ok := p.perItemBackoff[id]
	if !ok {
		return false
	}
	if p.hasExpired(eventTime, entry.lastUpdate, p.maxDuration) {
		return false
	}
	return eventTime.Sub(entry.lastUpdate) < entry.backoff
}

// Garbage collect records that have aged past their expiration, which defaults
// to 2*maxDuration (see hasExpired godoc). Backoff users are expected to invoke
// this periodically.
func (p *Backoff) GC() {
	p.Lock()
	defer p.Unlock()
	now := p.Clock.Now()
	for id, entry := range p.perItemBackoff {
		if p.hasExpired(now, entry.lastUpdate, p.maxDuration) {
			delete(p.perItemBackoff, id)
		}
	}
}

func (p *Backoff) DeleteEntry(id string) {
	p.Lock()
	defer p.Unlock()
	delete(p.perItemBackoff, id)
}

// Take a lock on *Backoff, before calling initEntryUnsafe
func (p *Backoff) initEntryUnsafe(id string) *backoffEntry {
	entry := &backoffEntry{backoff: p.defaultDuration}
	p.perItemBackoff[id] = entry
	return entry
}

func (p *Backoff) jitter(delay time.Duration) time.Duration {
	if p.rand == nil {
		return 0
	}

	return time.Duration(p.rand.Float64() * p.maxJitterFactor * float64(delay))
}

// Unless an alternate function is provided, after 2*maxDuration we restart the backoff factor to the beginning
func (p *Backoff) hasExpired(eventTime time.Time, lastUpdate time.Time, maxDuration time.Duration) bool {
	if p.HasExpiredFunc != nil {
		return p.HasExpiredFunc(eventTime, lastUpdate, maxDuration)
	}
	return eventTime.Sub(lastUpdate) > maxDuration*2 // consider stable if it's ok for twice the maxDuration
}
