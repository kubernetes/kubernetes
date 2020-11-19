/*
Copyright 2016 The Kubernetes Authors.

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

package cacher

import (
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
)

const (
	refreshPerSecond = 50 * time.Millisecond
	maxBudget        = 100 * time.Millisecond
)

// timeBudget implements a budget of time that you can use and is
// periodically being refreshed. The pattern to use it is:
//   budget := newTimeBudget(...)
//   ...
//   timeout := budget.takeAvailable()
//   // Now you can spend at most timeout on doing stuff
//   ...
//   // If you didn't use all timeout, return what you didn't use
//   budget.returnUnused(<unused part of timeout>)
//
// NOTE: It's not recommended to be used concurrently from multiple threads -
// if first user takes the whole timeout, the second one will get 0 timeout
// even though the first one may return something later.
type timeBudget interface {
	takeAvailable() time.Duration
	returnUnused(unused time.Duration)
}

type timeBudgetImpl struct {
	sync.Mutex
	budget time.Duration

	refresh         time.Duration
	maxBudget       time.Duration
	clock           clock.Clock
	lastRefreshTime time.Time
}

func newTimeBudget(inputClock clock.Clock) timeBudget {
	return &timeBudgetImpl{
		budget:          time.Duration(0),
		refresh:         refreshPerSecond,
		maxBudget:       maxBudget,
		clock:           inputClock,
		lastRefreshTime: inputClock.Now(), // set the last refresh time to now
	}
}

func (t *timeBudgetImpl) takeAvailable() time.Duration {
	t.Lock()
	defer t.Unlock()

	// freeze the time of refresh
	now := t.clock.Now()

	// number of elapsed seconds since last refresh
	// it is rounded down to the previous integral second
	elapsedSeconds := int(now.Sub(t.lastRefreshTime).Seconds())

	// calculate the available budget as
	// minimum of
	// 1. existingBudget + number of seconds elapsed since last take * refresh duration per second
	// 2. maxBudget
	available := t.budget + time.Duration(elapsedSeconds)*t.refresh
	if available > t.maxBudget {
		available = t.maxBudget
	}

	// reset the lastRefreshTime only if the
	// last time takeAvailable called was more than a second earlier
	if elapsedSeconds >= 1 {
		t.lastRefreshTime = now
	}

	t.budget = time.Duration(0)

	return available
}

func (t *timeBudgetImpl) returnUnused(unused time.Duration) {
	t.Lock()
	defer t.Unlock()
	if unused < 0 {
		// We used more than allowed.
		return
	}
	if t.budget = t.budget + unused; t.budget > t.maxBudget {
		t.budget = t.maxBudget
	}
}
