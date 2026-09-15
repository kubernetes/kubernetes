/*
Copyright The Kubernetes Authors.

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

package scheduler

import (
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

var token = struct{}{}

// statusPatchLimiter caps how many failure handler status patches may be dispatched
// concurrently.
//
// Without it, a burst of unschedulable pods dispatches one status patch each. The API
// dispatcher executes every queued call in its own goroutine, so all of them race for
// client-go rate limiter tokens at once. Tokens are handed out in request order, which
// pushes a subsequent Bind or preemption call behind the whole burst. Capping the
// failure handler keeps that budget available for the calls that make progress.
//
// Bind never passes through here. Preemption does, but only to record the node it freed,
// and patchPodStatusLimited exempts nominations from the limit; see isNomination.
//
// The synchronous path needs no equivalent because the scheduling cycle blocks on its
// own patch, which already allows at most one in flight.
//
// A nil limiter performs no limiting.
type statusPatchLimiter struct {
	tokens chan struct{}
}

func newStatusPatchLimiter(maxInFlight int) *statusPatchLimiter {
	if maxInFlight <= 0 {
		return nil
	}
	return &statusPatchLimiter{
		tokens: make(chan struct{}, maxInFlight),
	}
}

// acquire reserves a slot, blocking until one is free. It reports whether the slot was
// reserved; it returns false only when stopCh is closed first, in which case no slot is
// held and release must not be called. A nil stopCh means wait indefinitely.
func (l *statusPatchLimiter) acquire(stopCh <-chan struct{}) bool {
	if l == nil {
		return true
	}
	// Fast path: avoid touching the metric when there is no contention.
	select {
	case l.tokens <- token:
		return true
	default:
	}

	metrics.FailureHandlerThrottledPatches.Inc()
	defer metrics.FailureHandlerThrottledPatches.Dec()
	select {
	case l.tokens <- token:
		return true
	case <-stopCh:
		return false
	}
}

// release returns a slot previously taken by acquire.
func (l *statusPatchLimiter) release() {
	if l == nil {
		return
	}
	<-l.tokens
}
