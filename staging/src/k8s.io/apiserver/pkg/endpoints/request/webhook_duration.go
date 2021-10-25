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

// WebhookDuration stores trackers used to measure webhook request durations.
// Since admit webhooks are done sequentially duration is aggregated using
// sum function. Validate webhooks are done in parallel so max function
// is used.
type WebhookDuration struct {
	AdmitTracker    DurationTracker
	ValidateTracker DurationTracker
}

type webhookDurationKeyType int

// webhookDurationKey is the WebhookDuration (the time the request spent waiting
// for the webhooks to finish) key for the context.
const webhookDurationKey webhookDurationKeyType = iota

// WithWebhookDuration returns a copy of parent context to which the
// WebhookDuration trackers are added.
func WithWebhookDuration(parent context.Context) context.Context {
	return WithWebhookDurationAndCustomClock(parent, clock.RealClock{})
}

// WithWebhookDurationAndCustomClock returns a copy of parent context to which
// the WebhookDuration trackers are added. Tracers use given clock.
func WithWebhookDurationAndCustomClock(parent context.Context, c clock.Clock) context.Context {
	return WithValue(parent, webhookDurationKey, &WebhookDuration{
		AdmitTracker:    newSumLatencyTracker(c),
		ValidateTracker: newMaxLatencyTracker(c),
	})
}

// WebhookDurationFrom returns the value of the WebhookDuration key from the specified context.
func WebhookDurationFrom(ctx context.Context) (*WebhookDuration, bool) {
	wd, ok := ctx.Value(webhookDurationKey).(*WebhookDuration)
	return wd, ok && wd != nil
}
