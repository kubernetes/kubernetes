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

	"golang.org/x/time/rate"
)

const (
	// timeBudget has 50ms available in every 1s, and its maxBudget is 100ms,
	// use rate.limiter with 50 rate per second and 100 burst to implement timeBudget.
	refreshPerSecond = 50
	maxBudget        = 100
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

	maxBudget time.Duration
	clock     clock.Clock
	limiter   *rate.Limiter
}

func newTimeBudget() timeBudget {
	return &timeBudgetImpl{
		clock:     clock.RealClock{},
		budget:    time.Duration(0),
		maxBudget: maxBudget * time.Millisecond,
		limiter:   rate.NewLimiter(refreshPerSecond, maxBudget),
	}
}

func (t *timeBudgetImpl) takeAvailable() time.Duration {
	tokens := 0
	for t.limiter.AllowN(t.clock.Now(), refreshPerSecond) && tokens < maxBudget {
		tokens += refreshPerSecond
	}
	result := t.budget + time.Duration(tokens)*time.Millisecond
	if result > t.maxBudget {
		result = t.maxBudget
	}
	t.budget = time.Duration(0)
	return result
}

func (t *timeBudgetImpl) returnUnused(unused time.Duration) {
	if unused < 0 {
		// We used more than allowed.
		return
	}
	if t.budget = t.budget + unused; t.budget > t.maxBudget {
		t.budget = t.maxBudget
	}
}
