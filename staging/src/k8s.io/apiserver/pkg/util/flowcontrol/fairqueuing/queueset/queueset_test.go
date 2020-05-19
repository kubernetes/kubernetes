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
	"fmt"
	"math"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apiserver/pkg/util/flowcontrol/counter"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	test "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing"
	"k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing/clock"
	"k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	"k8s.io/klog/v2"
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
	evalDuration time.Duration,
	expectPass, expectedAllRequests, expectInqueueMetrics, expectExecutingMetrics bool,
	rejectReason string,
	clk *clock.FakeEventClock, counter counter.GoRoutineCounter) {

	now := time.Now()
	t.Logf("%s: Start %s, clk=%p, grc=%p", clk.Now().Format(nsTimeFmt), name, clk, counter)
	integrators := make([]test.Integrator, len(sc))
	var failedCount uint64
	expectedInqueue := ""
	expectedExecuting := ""
	if expectInqueueMetrics || expectExecutingMetrics {
		metrics.Reset()
	}
	executions := make([]int32, len(sc))
	rejects := make([]int32, len(sc))
	for i, uc := range sc {
		integrators[i] = test.NewIntegrator(clk)
		fsName := fmt.Sprintf("client%d", i)
		expectedInqueue = expectedInqueue + fmt.Sprintf(`				apiserver_flowcontrol_current_inqueue_requests{flowSchema=%q,priorityLevel=%q} 0%s`, fsName, name, "\n")
		for j := 0; j < uc.nThreads; j++ {
			counter.Add(1)
			go func(i, j int, uc uniformClient, igr test.Integrator) {
				for k := 0; k < uc.nCalls; k++ {
					ClockWait(clk, counter, uc.thinkDuration)
					req, idle := qs.StartRequest(context.Background(), uc.hash, fsName, name, []int{i, j, k})
					t.Logf("%s: %d, %d, %d got req=%p, idle=%v", clk.Now().Format(nsTimeFmt), i, j, k, req, idle)
					if req == nil {
						atomic.AddUint64(&failedCount, 1)
						atomic.AddInt32(&rejects[i], 1)
						break
					}
					if idle {
						t.Error("got request but QueueSet reported idle")
					}
					var executed bool
					idle2 := req.Finish(func() {
						executed = true
						t.Logf("%s: %d, %d, %d executing", clk.Now().Format(nsTimeFmt), i, j, k)
						atomic.AddInt32(&executions[i], 1)
						igr.Add(1)
						ClockWait(clk, counter, uc.execDuration)
						igr.Add(-1)
					})
					t.Logf("%s: %d, %d, %d got executed=%v, idle2=%v", clk.Now().Format(nsTimeFmt), i, j, k, executed, idle2)
					if !executed {
						atomic.AddUint64(&failedCount, 1)
						atomic.AddInt32(&rejects[i], 1)
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
	if expectInqueueMetrics {
		e := `
				# HELP apiserver_flowcontrol_current_inqueue_requests [ALPHA] Number of requests currently pending in queues of the API Priority and Fairness system
				# TYPE apiserver_flowcontrol_current_inqueue_requests gauge
` + expectedInqueue
		err := metrics.GatherAndCompare(e, "apiserver_flowcontrol_current_inqueue_requests")
		if err != nil {
			t.Error(err)
		} else {
			t.Log("Success with" + e)
		}
	}
	expectedRejects := ""
	for i := range sc {
		fsName := fmt.Sprintf("client%d", i)
		if atomic.AddInt32(&executions[i], 0) > 0 {
			expectedExecuting = expectedExecuting + fmt.Sprintf(`				apiserver_flowcontrol_current_executing_requests{flowSchema=%q,priorityLevel=%q} 0%s`, fsName, name, "\n")
		}
		if atomic.AddInt32(&rejects[i], 0) > 0 {
			expectedRejects = expectedRejects + fmt.Sprintf(`				apiserver_flowcontrol_rejected_requests_total{flowSchema=%q,priorityLevel=%q,reason=%q} %d%s`, fsName, name, rejectReason, rejects[i], "\n")
		}
	}
	if expectExecutingMetrics && len(expectedExecuting) > 0 {
		e := `
				# HELP apiserver_flowcontrol_current_executing_requests [ALPHA] Number of requests currently executing in the API Priority and Fairness system
				# TYPE apiserver_flowcontrol_current_executing_requests gauge
` + expectedExecuting
		err := metrics.GatherAndCompare(e, "apiserver_flowcontrol_current_executing_requests")
		if err != nil {
			t.Error(err)
		} else {
			t.Log("Success with" + e)
		}
	}
	if expectExecutingMetrics && len(expectedRejects) > 0 {
		e := `
				# HELP apiserver_flowcontrol_rejected_requests_total [ALPHA] Number of requests rejected by API Priority and Fairness system
				# TYPE apiserver_flowcontrol_rejected_requests_total counter
` + expectedRejects
		err := metrics.GatherAndCompare(e, "apiserver_flowcontrol_rejected_requests_total")
		if err != nil {
			t.Error(err)
		} else {
			t.Log("Success with" + e)
		}
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
	metrics.Register()
	now := time.Now()
	clk, counter := clock.NewFakeEventClock(now, 0, nil)
	nrc, err := test.NewNoRestraintFactory().BeginConstruction(fq.QueuingConfig{})
	if err != nil {
		t.Fatal(err)
	}
	nr := nrc.Complete(fq.DispatchingConfig{})
	exerciseQueueSetUniformScenario(t, "NoRestraint", nr, []uniformClient{
		{1001001001, 5, 10, time.Second, time.Second},
		{2002002002, 2, 10, time.Second, time.Second / 2},
	}, time.Second*10, false, true, false, false, "", clk, counter)
}

func TestUniformFlows(t *testing.T) {
	metrics.Register()
	now := time.Now()

	clk, counter := clock.NewFakeEventClock(now, 0, nil)
	qsf := NewQueueSetFactory(clk, counter)
	qCfg := fq.QueuingConfig{
		Name:             "TestUniformFlows",
		DesiredNumQueues: 8,
		QueueLengthLimit: 6,
		HandSize:         3,
		RequestWaitLimit: 10 * time.Minute,
	}
	qsc, err := qsf.BeginConstruction(qCfg)
	if err != nil {
		t.Fatal(err)
	}
	qs := qsc.Complete(fq.DispatchingConfig{ConcurrencyLimit: 4})

	exerciseQueueSetUniformScenario(t, qCfg.Name, qs, []uniformClient{
		{1001001001, 5, 10, time.Second, time.Second},
		{2002002002, 5, 10, time.Second, time.Second},
	}, time.Second*20, true, true, true, true, "", clk, counter)
}

func TestDifferentFlows(t *testing.T) {
	metrics.Register()
	now := time.Now()

	clk, counter := clock.NewFakeEventClock(now, 0, nil)
	qsf := NewQueueSetFactory(clk, counter)
	qCfg := fq.QueuingConfig{
		Name:             "TestDifferentFlows",
		DesiredNumQueues: 8,
		QueueLengthLimit: 6,
		HandSize:         3,
		RequestWaitLimit: 10 * time.Minute,
	}
	qsc, err := qsf.BeginConstruction(qCfg)
	if err != nil {
		t.Fatal(err)
	}
	qs := qsc.Complete(fq.DispatchingConfig{ConcurrencyLimit: 4})

	exerciseQueueSetUniformScenario(t, qCfg.Name, qs, []uniformClient{
		{1001001001, 6, 10, time.Second, time.Second},
		{2002002002, 5, 15, time.Second, time.Second / 2},
	}, time.Second*20, true, true, true, true, "", clk, counter)
}

func TestDifferentFlowsWithoutQueuing(t *testing.T) {
	metrics.Register()
	now := time.Now()

	clk, counter := clock.NewFakeEventClock(now, 0, nil)
	qsf := NewQueueSetFactory(clk, counter)
	qCfg := fq.QueuingConfig{
		Name:             "TestDifferentFlowsWithoutQueuing",
		DesiredNumQueues: 0,
	}
	qsc, err := qsf.BeginConstruction(qCfg)
	if err != nil {
		t.Fatal(err)
	}
	qs := qsc.Complete(fq.DispatchingConfig{ConcurrencyLimit: 4})

	exerciseQueueSetUniformScenario(t, qCfg.Name, qs, []uniformClient{
		{1001001001, 6, 10, time.Second, 57 * time.Millisecond},
		{2002002002, 4, 15, time.Second, 750 * time.Millisecond},
	}, time.Second*13, false, false, false, true, "concurrency-limit", clk, counter)
	err = metrics.GatherAndCompare(`
				# HELP apiserver_flowcontrol_rejected_requests_total [ALPHA] Number of requests rejected by API Priority and Fairness system
				# TYPE apiserver_flowcontrol_rejected_requests_total counter
				apiserver_flowcontrol_rejected_requests_total{flowSchema="client0",priorityLevel="TestDifferentFlowsWithoutQueuing",reason="concurrency-limit"} 2
				apiserver_flowcontrol_rejected_requests_total{flowSchema="client1",priorityLevel="TestDifferentFlowsWithoutQueuing",reason="concurrency-limit"} 4
				`,
		"apiserver_flowcontrol_rejected_requests_total")
	if err != nil {
		t.Fatal(err)
	}
}

func TestTimeout(t *testing.T) {
	metrics.Register()
	now := time.Now()

	clk, counter := clock.NewFakeEventClock(now, 0, nil)
	qsf := NewQueueSetFactory(clk, counter)
	qCfg := fq.QueuingConfig{
		Name:             "TestTimeout",
		DesiredNumQueues: 128,
		QueueLengthLimit: 128,
		HandSize:         1,
		RequestWaitLimit: 0,
	}
	qsc, err := qsf.BeginConstruction(qCfg)
	if err != nil {
		t.Fatal(err)
	}
	qs := qsc.Complete(fq.DispatchingConfig{ConcurrencyLimit: 1})

	exerciseQueueSetUniformScenario(t, qCfg.Name, qs, []uniformClient{
		{1001001001, 5, 100, time.Second, time.Second},
	}, time.Second*10, true, false, true, true, "time-out", clk, counter)
}

func TestContextCancel(t *testing.T) {
	metrics.Register()
	metrics.Reset()
	now := time.Now()
	clk, counter := clock.NewFakeEventClock(now, 0, nil)
	qsf := NewQueueSetFactory(clk, counter)
	qCfg := fq.QueuingConfig{
		Name:             "TestContextCancel",
		DesiredNumQueues: 11,
		QueueLengthLimit: 11,
		HandSize:         1,
		RequestWaitLimit: 15 * time.Second,
	}
	qsc, err := qsf.BeginConstruction(qCfg)
	if err != nil {
		t.Fatal(err)
	}
	qs := qsc.Complete(fq.DispatchingConfig{ConcurrencyLimit: 1})
	counter.Add(1) // account for the goroutine running this test
	ctx1 := context.Background()
	req1, _ := qs.StartRequest(ctx1, 1, "fs1", "test", "one")
	if req1 == nil {
		t.Error("Request rejected")
		return
	}
	var executed1 bool
	idle1 := req1.Finish(func() {
		executed1 = true
		ctx2, cancel2 := context.WithCancel(context.Background())
		tBefore := time.Now()
		go func() {
			time.Sleep(time.Second)
			// account for unblocking the goroutine that waits on cancelation
			counter.Add(1)
			cancel2()
		}()
		req2, idle2a := qs.StartRequest(ctx2, 2, "fs2", "test", "two")
		if idle2a {
			t.Error("2nd StartRequest returned idle")
		}
		if req2 != nil {
			idle2b := req2.Finish(func() {
				t.Error("Executing req2")
			})
			if idle2b {
				t.Error("2nd Finish returned idle")
			}
		}
		tAfter := time.Now()
		dt := tAfter.Sub(tBefore)
		if dt < time.Second || dt > 2*time.Second {
			t.Errorf("Unexpected: dt=%d", dt)
		}
	})
	if !executed1 {
		t.Errorf("Unexpected: executed1=%v", executed1)
	}
	if !idle1 {
		t.Error("Not idle at the end")
	}
}
