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
)

type uniformScenario []uniformClient

type uniformClient struct {
	hash          uint64
	nThreads      int
	nCalls        int
	execDuration  time.Duration
	thinkDuration time.Duration
}

// exerciseQueueSetUniformScenario.  Simple logic, only works if each
// client's offered load is at least as large as its fair share of
// capacity.
func exerciseQueueSetUniformScenario(t *testing.T, qs fq.QueueSet, sc uniformScenario,
	totalDuration time.Duration, expectPass bool, expectedAllRequests bool,
	clk *clock.FakeEventClock, counter counter.GoRoutineCounter) {

	now := time.Now()
	t.Logf("%s: Start", clk.Now().Format("2006-01-02 15:04:05.000000000"))
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
						tryAnother, execute, afterExecute := qs.Wait(context.Background(), uc.hash)
						t.Logf("%s: %d, %d, %d got q=%v, e=%v", clk.Now().Format("2006-01-02 15:04:05.000000000"), i, j, k, tryAnother, execute)
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
	lim := now.Add(totalDuration)
	clk.Run(&lim)
	clk.SetTime(lim)
	t.Logf("%s: End", clk.Now().Format("2006-01-02 15:04:05.000000000"))
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
	exerciseQueueSetUniformScenario(t, nr, []uniformClient{
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
		ConcurrencyLimit: 100,
		DesiredNumQueues: 128,
		QueueLengthLimit: 128,
		HandSize:         1,
		RequestWaitLimit: 10 * time.Minute,
	}
	qs, err := qsf.NewQueueSet(config)
	if err != nil {
		t.Fatalf("QueueSet creation failed with %v", err)
	}

	exerciseQueueSetUniformScenario(t, qs, []uniformClient{
		{1001001001, 5, 10, time.Second, time.Second},
		{2002002002, 5, 10, time.Second, time.Second},
	}, time.Second*10, true, true, clk, counter)
}

func TestDifferentFlows(t *testing.T) {
	now := time.Now()

	clk, counter := clock.NewFakeEventClock(now, 0, nil)
	qsf := NewQueueSetFactory(clk, counter)
	config := fq.QueueSetConfig{
		Name:             "TestDifferentFlows",
		ConcurrencyLimit: 1,
		DesiredNumQueues: 128,
		QueueLengthLimit: 128,
		HandSize:         1,
		RequestWaitLimit: 10 * time.Minute,
	}
	qs, err := qsf.NewQueueSet(config)
	if err != nil {
		t.Fatalf("QueueSet creation failed with %v", err)
	}

	exerciseQueueSetUniformScenario(t, qs, []uniformClient{
		{1001001001, 5, 10, time.Second, time.Second},
		{2002002002, 2, 5, time.Second, time.Second / 2},
	}, time.Second*10, true, true, clk, counter)
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

	exerciseQueueSetUniformScenario(t, qs, []uniformClient{
		{1001001001, 5, 100, time.Second, time.Second},
	}, time.Second*10, true, false, clk, counter)
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
