/*
Copyright 2021 The Kubernetes Authors.

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

package request

import (
	"context"
	"sync"
	"time"

	"k8s.io/utils/clock"
)

func sumDuration(d1 time.Duration, d2 time.Duration) time.Duration {
	return d1 + d2
}

func maxDuration(d1 time.Duration, d2 time.Duration) time.Duration {
	if d1 > d2 {
		return d1
	}
	return d2
}

// DurationTracker is a simple interface for tracking functions duration
type DurationTracker interface {
	Track(func())
	GetLatency() time.Duration
}

// durationTracker implements DurationTracker by measuring function time
// using given clock and aggregates the duration using given aggregate function
type durationTracker struct {
	clock             clock.Clock
	latency           time.Duration
	mu                sync.Mutex
	aggregateFunction func(time.Duration, time.Duration) time.Duration
}

// Track measures time spent in given function and aggregates measured
// duration using aggregateFunction
func (t *durationTracker) Track(f func()) {
	startedAt := t.clock.Now()
	defer func() {
		duration := t.clock.Since(startedAt)
		t.mu.Lock()
		defer t.mu.Unlock()
		t.latency = t.aggregateFunction(t.latency, duration)
	}()

	f()
}

// GetLatency returns aggregated latency tracked by a tracker
func (t *durationTracker) GetLatency() time.Duration {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.latency
}

func newSumLatencyTracker(c clock.Clock) DurationTracker {
	return &durationTracker{
		clock:             c,
		aggregateFunction: sumDuration,
	}
}

func newMaxLatencyTracker(c clock.Clock) DurationTracker {
	return &durationTracker{
		clock:             c,
		aggregateFunction: maxDuration,
	}
}

// LatencyTrackers stores trackers used to measure latecny incurred in
// components within the apiserver.
type LatencyTrackers struct {
	// MutatingWebhookTracker tracks the latency incurred in mutating webhook(s).
	// Since mutating webhooks are done sequentially, latency
	// is aggregated using sum function.
	MutatingWebhookTracker DurationTracker

	// ValidatingWebhookTracker tracks the latency incurred in validating webhook(s).
	// Validate webhooks are done in parallel, so max function is used.
	ValidatingWebhookTracker DurationTracker
}

type latencyTrackersKeyType int

// latencyTrackersKey is the key that associates a LatencyTrackers
// instance with the request context.
const latencyTrackersKey latencyTrackersKeyType = iota

// WithLatencyTrackers returns a copy of parent context to which an
// instance of LatencyTrackers is added.
func WithLatencyTrackers(parent context.Context) context.Context {
	return WithLatencyTrackersAndCustomClock(parent, clock.RealClock{})
}

// WithLatencyTrackersAndCustomClock returns a copy of parent context to which
// an instance of LatencyTrackers is added. Tracers use given clock.
func WithLatencyTrackersAndCustomClock(parent context.Context, c clock.Clock) context.Context {
	return WithValue(parent, latencyTrackersKey, &LatencyTrackers{
		MutatingWebhookTracker:   newSumLatencyTracker(c),
		ValidatingWebhookTracker: newMaxLatencyTracker(c),
	})
}

// LatencyTrackersFrom returns the associated LatencyTrackers instance
// from the specified context.
func LatencyTrackersFrom(ctx context.Context) (*LatencyTrackers, bool) {
	wd, ok := ctx.Value(latencyTrackersKey).(*LatencyTrackers)
	return wd, ok && wd != nil
}
