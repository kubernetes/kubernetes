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

	refresh   time.Duration
	maxBudget time.Duration
}

func newTimeBudget(stopCh <-chan struct{}) timeBudget {
	result := &timeBudgetImpl{
		budget:    time.Duration(0),
		refresh:   refreshPerSecond,
		maxBudget: maxBudget,
	}
	go result.periodicallyRefresh(stopCh)
	return result
}

func (t *timeBudgetImpl) periodicallyRefresh(stopCh <-chan struct{}) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			t.Lock()
			if t.budget = t.budget + t.refresh; t.budget > t.maxBudget {
				t.budget = t.maxBudget
			}
			t.Unlock()
		case <-stopCh:
			return
		}
	}
}

func (t *timeBudgetImpl) takeAvailable() time.Duration {
	t.Lock()
	defer t.Unlock()
	result := t.budget
	t.budget = time.Duration(0)
	return result
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
