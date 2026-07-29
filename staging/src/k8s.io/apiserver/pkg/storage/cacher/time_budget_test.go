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
	"testing"
	"time"

	testingclock "k8s.io/utils/clock/testing"
)

func TestTimeBudget(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())

	budget := &timeBudgetImpl{
		clock:     fakeClock,
		budget:    time.Duration(0),
		maxBudget: 200 * time.Millisecond,
		refresh:   50 * time.Millisecond,
		last:      fakeClock.Now(),
	}
	if res := budget.takeAvailable(); res != time.Duration(0) {
		t.Errorf("Expected: %v, got: %v", time.Duration(0), res)
	}

	// wait for longer than the maxBudget
	nextTime := time.Now().Add(10 * time.Second)
	fakeClock.SetTime(nextTime)
	if res := budget.takeAvailable(); res != budget.maxBudget {
		t.Errorf("Expected: %v, got: %v", budget.maxBudget, res)
	}
	// add two refresh intervals to accumulate 2*refresh durations
	nextTime = nextTime.Add(2 * time.Second)
	fakeClock.SetTime(nextTime)
	if res := budget.takeAvailable(); res != 2*budget.refresh {
		t.Errorf("Expected: %v, got: %v", 2*budget.refresh, res)
	}
	// return one refresh duration to have only one refresh duration available
	// we didn't advanced on time yet
	budget.returnUnused(budget.refresh)
	if res := budget.takeAvailable(); res != budget.refresh {
		t.Errorf("Expected: %v, got: %v", budget.refresh, res)
	}

	// return a negative value to the budget
	// we didn't advanced on time yet
	budget.returnUnused(-time.Duration(50))
	if res := budget.takeAvailable(); res != time.Duration(0) {
		t.Errorf("Expected: %v, got: %v", time.Duration(0), res)
	}
	// handle back in time problem with an empty budget
	nextTime = nextTime.Add(-2 * time.Minute)
	fakeClock.SetTime(nextTime)
	if res := budget.takeAvailable(); res != time.Duration(0) {
		t.Errorf("Expected: %v, got: %v", time.Duration(0), res)
	}
	// wait for longer than the maxBudget
	// verify that adding a negative value didn't affected
	nextTime = nextTime.Add(10 * time.Minute)
	fakeClock.SetTime(nextTime)
	if res := budget.takeAvailable(); res != budget.maxBudget {
		t.Errorf("Expected: %v, got: %v", budget.maxBudget, res)
	}

	// handle back in time problem with time on the budget
	budget.returnUnused(10 * time.Second)
	nextTime = nextTime.Add(-2 * time.Minute)
	fakeClock.SetTime(nextTime)
	if res := budget.takeAvailable(); res != budget.maxBudget {
		t.Errorf("Expected: %v, got: %v", budget.maxBudget, res)
	}
}
