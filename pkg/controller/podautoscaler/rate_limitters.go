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
	"time"

	"k8s.io/client-go/util/workqueue"
)

// FixedItemIntervalRateLimiter limits items to a fixed-rate interval
type FixedItemIntervalRateLimiter struct {
	interval time.Duration
}

var _ workqueue.RateLimiter = &FixedItemIntervalRateLimiter{}

func NewFixedItemIntervalRateLimiter(interval time.Duration) workqueue.RateLimiter {
	return &FixedItemIntervalRateLimiter{
		interval: interval,
	}
}

func (r *FixedItemIntervalRateLimiter) When(item interface{}) time.Duration {
	return r.interval
}

func (r *FixedItemIntervalRateLimiter) NumRequeues(item interface{}) int {
	return 1
}

func (r *FixedItemIntervalRateLimiter) Forget(item interface{}) {
}

// NewDefaultHPARateLimitter creates a rate limitter which limits overall (as per the
// default controller rate limiter), as well as per the resync interval
func NewDefaultHPARateLimiter(interval time.Duration) workqueue.RateLimiter {
	return NewFixedItemIntervalRateLimiter(interval)
}
