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

	"k8s.io/apimachinery/pkg/util/clock"
)

func TestTimeBudget(t *testing.T) {
	inputClock := clock.NewFakeClock(time.Now())

	budget := &timeBudgetImpl{
		budget:          time.Duration(0),
		maxBudget:       time.Duration(200),
		refresh:         time.Duration(50),
		clock:           inputClock,
		lastRefreshTime: inputClock.Now(),
	}

	if res := budget.takeAvailable(); res != time.Duration(0) {
		t.Errorf("Expected: %v, got: %v", time.Duration(0), res)
	}
	budget.budget = time.Duration(100)
	if res := budget.takeAvailable(); res != time.Duration(100) {
		t.Errorf("Expected: %v, got: %v", time.Duration(100), res)
	}
	if res := budget.takeAvailable(); res != time.Duration(0) {
		t.Errorf("Expected: %v, got: %v", time.Duration(0), res)
	}
	budget.returnUnused(time.Duration(50))
	if res := budget.takeAvailable(); res != time.Duration(50) {
		t.Errorf("Expected: %v, got: %v", time.Duration(50), res)
	}
	budget.budget = time.Duration(100)
	budget.returnUnused(-time.Duration(50))
	if res := budget.takeAvailable(); res != time.Duration(100) {
		t.Errorf("Expected: %v, got: %v", time.Duration(100), res)
	}
	// test overflow.
	budget.returnUnused(time.Duration(500))
	if res := budget.takeAvailable(); res != time.Duration(200) {
		t.Errorf("Expected: %v, got: %v", time.Duration(200), res)
	}

	// test replenishment of budget over a time duration
	// take all available budget to reset budget and lastRefreshTime
	budget.takeAvailable()
	// fake wait for 2 seconds
	waitSeconds := 2
	inputClock.Step(time.Duration(waitSeconds) * time.Second)
	// expect (2 seconds * 50ms/second)=100ms of budget
	expectedBudget := time.Duration(waitSeconds) * budget.refresh
	if res := budget.takeAvailable(); res != expectedBudget {
		t.Errorf("Expected: %v, got: %v", expectedBudget, res)
	}

	// calling takeAvailable in intervals of less than a second
	// should not reset the budget
	budget.budget = time.Duration(0)
	budget.lastRefreshTime = inputClock.Now()
	// fake wait for 0.5 seconds
	inputClock.Step(500 * time.Millisecond)
	expectedBudget = 0 * budget.refresh
	if res := budget.takeAvailable(); res != expectedBudget {
		t.Errorf("Expected: %v, got: %v", expectedBudget, res)
	}
	// fake wait for 0.25 seconds
	inputClock.Step(250 * time.Millisecond)
	expectedBudget = 0 * budget.refresh
	if res := budget.takeAvailable(); res != expectedBudget {
		t.Errorf("Expected: %v, got: %v", expectedBudget, res)
	}
	inputClock.Step(500 * time.Millisecond)
	// expect (1 second * 50ms/second)=50s of budget
	expectedBudget = 1 * budget.refresh
	if res := budget.takeAvailable(); res != expectedBudget {
		t.Errorf("Expected: %v, got: %v", expectedBudget, res)
	}
}
