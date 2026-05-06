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

package podautoscaler

import (
	"sync"
	"time"

	"k8s.io/client-go/util/workqueue"
)

// PerItemIntervalRateLimiter allows per-item interval overrides with a
// default fallback. Items without an explicit override use defaultInterval.
type PerItemIntervalRateLimiter struct {
	defaultInterval time.Duration
	mu              sync.RWMutex
	intervals       map[string]time.Duration
}

var _ workqueue.TypedRateLimiter[string] = &PerItemIntervalRateLimiter{}

// NewPerItemIntervalRateLimiter creates a rate limiter that supports per-item
// interval overrides. Items without an override use defaultInterval.
func NewPerItemIntervalRateLimiter(defaultInterval time.Duration) *PerItemIntervalRateLimiter {
	return &PerItemIntervalRateLimiter{
		defaultInterval: defaultInterval,
		intervals:       make(map[string]time.Duration),
	}
}

// When returns the per-item interval if set, otherwise the default.
func (r *PerItemIntervalRateLimiter) When(item string) time.Duration {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if d, ok := r.intervals[item]; ok {
		return d
	}
	return r.defaultInterval
}

// NumRequeues returns back how many failures the item has had
func (r *PerItemIntervalRateLimiter) NumRequeues(item string) int {
	return 1
}

// Forget indicates that an item is finished being retried.
func (r *PerItemIntervalRateLimiter) Forget(item string) {
}

// SetItemInterval sets a per-item interval override. Pass the default interval
// (or call RemoveItem) to revert to the global default.
func (r *PerItemIntervalRateLimiter) SetItemInterval(item string, interval time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.intervals[item] = interval
}

// RemoveItem removes the per-item interval override, reverting to the default.
func (r *PerItemIntervalRateLimiter) RemoveItem(item string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.intervals, item)
}

// NewDefaultHPARateLimiter creates a rate limiter which limits overall (as per the
// default controller rate limiter), as well as per the resync interval
func NewDefaultHPARateLimiter(interval time.Duration) *PerItemIntervalRateLimiter {
	return NewPerItemIntervalRateLimiter(interval)
}
