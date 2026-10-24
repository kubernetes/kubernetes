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

// PerItemIntervalRateLimiter limits items to a fixed-rate interval, which can
// be overridden for individual items.
type PerItemIntervalRateLimiter struct {
	defaultInterval time.Duration

	mu        sync.RWMutex
	intervals map[string]time.Duration
}

var _ workqueue.TypedRateLimiter[string] = &PerItemIntervalRateLimiter{}

// NewPerItemIntervalRateLimiter creates a rate limiter that returns defaultInterval
// for every item without an override.
func NewPerItemIntervalRateLimiter(defaultInterval time.Duration) *PerItemIntervalRateLimiter {
	return &PerItemIntervalRateLimiter{
		defaultInterval: defaultInterval,
		intervals:       map[string]time.Duration{},
	}
}

// When returns the interval of the item.
func (r *PerItemIntervalRateLimiter) When(item string) time.Duration {
	interval, _ := r.Interval(item)
	return interval
}

// Interval returns the interval of the item and whether it comes from an override.
func (r *PerItemIntervalRateLimiter) Interval(item string) (time.Duration, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if interval, ok := r.intervals[item]; ok {
		return interval, true
	}
	return r.defaultInterval, false
}

// SetInterval overrides the interval of the item.
func (r *PerItemIntervalRateLimiter) SetInterval(item string, interval time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.intervals[item] = interval
}

// RemoveInterval removes the override of the item, so it uses the default interval again.
func (r *PerItemIntervalRateLimiter) RemoveInterval(item string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.intervals, item)
}

// NumRequeues returns back how many failures the item has had
func (r *PerItemIntervalRateLimiter) NumRequeues(item string) int {
	return 1
}

// Forget indicates that an item is finished being retried. Overrides are kept
// until RemoveInterval is called.
func (r *PerItemIntervalRateLimiter) Forget(item string) {
}

// NewDefaultHPARateLimiter creates a rate limiter which limits overall (as per the
// default controller rate limiter), as well as per the resync interval
func NewDefaultHPARateLimiter(interval time.Duration) *PerItemIntervalRateLimiter {
	return NewPerItemIntervalRateLimiter(interval)
}
