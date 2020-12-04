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

	"golang.org/x/time/rate"
)

func TestTimeBudget(t *testing.T) {
	fakeClock := clock.NewFakeClock(time.Now())
	budget := &timeBudgetImpl{
		budget:    time.Duration(0),
		maxBudget: time.Duration(200),
		clock:     fakeClock,
		limiter:   rate.NewLimiter(refreshPerSecond, maxBudget),
	}
	// bucket has full token so takeAvailable returns budget.maxBudget.
	if res := budget.takeAvailable(); res != budget.maxBudget {
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

	budget.maxBudget = maxBudget * time.Millisecond
	nextTime := time.Now().Add(3 * time.Second)
	fakeClock.SetTime(nextTime)
	if res := budget.takeAvailable(); res != budget.maxBudget {
		t.Errorf("Expected: %v, got: %v", budget.maxBudget, res)
	}

	nextTime = nextTime.Add(time.Duration(1500) * time.Millisecond)
	fakeClock.SetTime(nextTime)
	if res := budget.takeAvailable(); res != refreshPerSecond*time.Millisecond {
		t.Errorf("Expected: %v, got: %v", refreshPerSecond*time.Millisecond, res)
	}

	nextTime = nextTime.Add(time.Duration(1500) * time.Millisecond)
	fakeClock.SetTime(nextTime)
	if res := budget.takeAvailable(); res != budget.maxBudget {
		t.Errorf("Expected: %v, got: %v", budget.maxBudget, res)
	}
}
