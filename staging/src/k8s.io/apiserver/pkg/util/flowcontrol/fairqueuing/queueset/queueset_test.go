/*
Copyright 2019 The Kubernetes Authors.

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

package queueset

import (
	"context"
	"math"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apiserver/pkg/util/flowcontrol/counter"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	test "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing"
	"k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing/clock"
	"k8s.io/klog"
)

type uniformScenario []uniformClient

type uniformClient struct {
	hash     uint64
	nThreads int
	nCalls   int
	// duration for a simulated synchronous call
	execDuration time.Duration
	// duration for simulated "other work"
	thinkDuration time.Duration
}

// exerciseQueueSetUniformScenario runs a scenario based on the given set of uniform clients.
// Each uniform client specifies a number of threads, each of which alternates between thinking
// and making a synchronous request through the QueueSet.
// This function measures how much concurrency each client got, on average, over
// the initial evalDuration and tests to see whether they all got about the same amount.
// Each client needs to be demanding enough to use this amount, otherwise the fair result
// is not equal amounts and the simple test in this function would not accurately test fairness.
// expectPass indicates whether the QueueSet is expected to be fair.
// expectedAllRequests indicates whether all requests are expected to get dispatched.
func exerciseQueueSetUniformScenario(t *testing.T, name string, qs fq.QueueSet, sc uniformScenario,
	evalDuration time.Duration, expectPass bool, expectedAllRequests bool,
	clk *clock.FakeEventClock, counter counter.GoRoutineCounter) {

	now := time.Now()
	t.Logf("%s: Start %s, clk=%p, grc=%p", clk.Now().Format(nsTimeFmt), name, clk, counter)
	integrators := make([]test.Integrator, len(sc))
	var failedCount uint64
	for i, uc := range sc {
		integrators[i] = test.NewIntegrator(clk)
		for j := 0; j < uc.nThreads; j++ {
			counter.Add(1)
			go func(i, j int, uc uniformClient, igr test.Integrator) {
				for k := 0; k < uc.nCalls; k++ {
					ClockWait(clk, counter, uc.thinkDuration)
					for {
						tryAnother, execute, afterExecute := qs.Wait(context.Background(), uc.hash, name, []int{i, j, k})
						t.Logf("%s: %d, %d, %d got a=%v, e=%v", clk.Now().Format(nsTimeFmt), i, j, k, tryAnother, execute)
						if tryAnother {
							continue
						}
						if !execute {
							atomic.AddUint64(&failedCount, 1)
							break
						}
						igr.Add(1)
						ClockWait(clk, counter, uc.execDuration)
						afterExecute()
						igr.Add(-1)
						break
					}
				}
				counter.Add(-1)
			}(i, j, uc, integrators[i])
		}
	}
	lim := now.Add(evalDuration)
	clk.Run(&lim)
	clk.SetTime(lim)
	t.Logf("%s: End", clk.Now().Format(nsTimeFmt))
	results := make([]test.IntegratorResults, len(sc))
	var sumOfAvg float64
	for i := range sc {
		results[i] = integrators[i].GetResults()
		sumOfAvg += results[i].Average
	}
	idealAverage := sumOfAvg / float64(len(sc))
	passes := make([]bool, len(sc))
	allPass := true
	for i := range sc {
		relDiff := (results[i].Average - idealAverage) / idealAverage
		passes[i] = math.Abs(relDiff) <= 0.1
		allPass = allPass && passes[i]
	}
	for i := range sc {
		if allPass != expectPass {
			t.Errorf("Class %d got an Average of %v but the ideal was %v", i, results[i].Average, idealAverage)
		} else {
			t.Logf("Class %d got an Average of %v and the ideal was %v", i, results[i].Average, idealAverage)
		}
	}

	clk.Run(nil)
	if expectedAllRequests && failedCount > 0 {
		t.Errorf("Expected all requests to be successful but got %v failed requests", failedCount)
	} else if !expectedAllRequests && failedCount == 0 {
		t.Errorf("Expected failed requests but all requests succeeded")
	}
}

func ClockWait(clk *clock.FakeEventClock, counter counter.GoRoutineCounter, duration time.Duration) {
	dunch := make(chan struct{})
	clk.EventAfterDuration(func(time.Time) {
		counter.Add(1)
		close(dunch)
	}, duration)
	counter.Add(-1)
	select {
	case <-dunch:
	}
}

func init() {
	klog.InitFlags(nil)
}

// TestNoRestraint should fail because the dummy QueueSet exercises no control
func TestNoRestraint(t *testing.T) {
	now := time.Now()
	clk, counter := clock.NewFakeEventClock(now, 0, nil)
	nrf := test.NewNoRestraintFactory()
	config := fq.QueueSetConfig{}
	nr, err := nrf.NewQueueSet(config)
	if err != nil {
		t.Fatalf("QueueSet creation failed with %v", err)
	}
	exerciseQueueSetUniformScenario(t, "NoRestraint", nr, []uniformClient{
		{1001001001, 5, 10, time.Second, time.Second},
		{2002002002, 2, 10, time.Second, time.Second / 2},
	}, time.Second*10, false, true, clk, counter)
}

func TestUniformFlows(t *testing.T) {
	now := time.Now()

	clk, counter := clock.NewFakeEventClock(now, 0, nil)
	qsf := NewQueueSetFactory(clk, counter)
	config := fq.QueueSetConfig{
		Name:             "TestUniformFlows",
		ConcurrencyLimit: 4,
		DesiredNumQueues: 8,
		QueueLengthLimit: 6,
		HandSize:         3,
		RequestWaitLimit: 10 * time.Minute,
	}
	qs, err := qsf.NewQueueSet(config)
	if err != nil {
		t.Fatalf("QueueSet creation failed with %v", err)
	}

	exerciseQueueSetUniformScenario(t, "UniformFlows", qs, []uniformClient{
		{1001001001, 5, 10, time.Second, time.Second},
		{2002002002, 5, 10, time.Second, time.Second},
	}, time.Second*20, true, true, clk, counter)
}

func TestDifferentFlows(t *testing.T) {
	now := time.Now()

	clk, counter := clock.NewFakeEventClock(now, 0, nil)
	qsf := NewQueueSetFactory(clk, counter)
	config := fq.QueueSetConfig{
		Name:             "TestDifferentFlows",
		ConcurrencyLimit: 4,
		DesiredNumQueues: 8,
		QueueLengthLimit: 6,
		HandSize:         3,
		RequestWaitLimit: 10 * time.Minute,
	}
	qs, err := qsf.NewQueueSet(config)
	if err != nil {
		t.Fatalf("QueueSet creation failed with %v", err)
	}

	exerciseQueueSetUniformScenario(t, "DifferentFlows", qs, []uniformClient{
		{1001001001, 6, 10, time.Second, time.Second},
		{2002002002, 5, 15, time.Second, time.Second / 2},
	}, time.Second*20, true, true, clk, counter)
}

func TestDifferentFlowsWithoutQueuing(t *testing.T) {
	now := time.Now()

	clk, counter := clock.NewFakeEventClock(now, 0, nil)
	qsf := NewQueueSetFactory(clk, counter)
	config := fq.QueueSetConfig{
		Name:             "TestDifferentFlowsWithoutQueuing",
		ConcurrencyLimit: 4,
		DesiredNumQueues: 0,
		QueueLengthLimit: 6,
		HandSize:         3,
		RequestWaitLimit: 10 * time.Minute,
	}
	qs, err := qsf.NewQueueSet(config)
	if err != nil {
		t.Fatalf("QueueSet creation failed with %v", err)
	}

	exerciseQueueSetUniformScenario(t, "DifferentFlowsWithoutQueuing", qs, []uniformClient{
		{1001001001, 6, 10, time.Second, 57 * time.Millisecond},
		{2002002002, 4, 15, time.Second, 750 * time.Millisecond},
	}, time.Second*13, false, false, clk, counter)
}

func TestTimeout(t *testing.T) {
	now := time.Now()

	clk, counter := clock.NewFakeEventClock(now, 0, nil)
	qsf := NewQueueSetFactory(clk, counter)
	config := fq.QueueSetConfig{
		Name:             "TestTimeout",
		ConcurrencyLimit: 1,
		DesiredNumQueues: 128,
		QueueLengthLimit: 128,
		HandSize:         1,
		RequestWaitLimit: 0,
	}
	qs, err := qsf.NewQueueSet(config)
	if err != nil {
		t.Fatalf("QueueSet creation failed with %v", err)
	}

	exerciseQueueSetUniformScenario(t, "Timeout", qs, []uniformClient{
		{1001001001, 5, 100, time.Second, time.Second},
	}, time.Second*10, true, false, clk, counter)
}
