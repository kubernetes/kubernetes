/*
Copyright 2022 The Kubernetes Authors.

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

package cache

import (
	"k8s.io/utils/clock"
	"time"
)

type RetryWithDeadline interface {
	After(error)
	ShouldRetry() bool
}

type retryWithDeadlineImpl struct {
	firstErrorTime   time.Time
	lastErrorTime    time.Time
	maxRetryDuration time.Duration
	minResetPeriod   time.Duration
	isRetryable      func(error) bool
	clock            clock.Clock
}

func NewRetryWithDeadline(maxRetryDuration, minResetPeriod time.Duration, isRetryable func(error) bool, clock clock.Clock) RetryWithDeadline {
	return &retryWithDeadlineImpl{
		firstErrorTime:   time.Time{},
		lastErrorTime:    time.Time{},
		maxRetryDuration: maxRetryDuration,
		minResetPeriod:   minResetPeriod,
		isRetryable:      isRetryable,
		clock:            clock,
	}
}

func (r *retryWithDeadlineImpl) reset() {
	r.firstErrorTime = time.Time{}
	r.lastErrorTime = time.Time{}
}

func (r *retryWithDeadlineImpl) After(err error) {
	if r.isRetryable(err) {
		if r.clock.Now().Sub(r.lastErrorTime) >= r.minResetPeriod {
			r.reset()
		}

		if r.firstErrorTime.IsZero() {
			r.firstErrorTime = r.clock.Now()
		}
		r.lastErrorTime = r.clock.Now()
	}
}

func (r *retryWithDeadlineImpl) ShouldRetry() bool {
	if r.maxRetryDuration <= time.Duration(0) {
		return false
	}

	if r.clock.Now().Sub(r.firstErrorTime) <= r.maxRetryDuration {
		return true
	}

	r.reset()
	return false
}
