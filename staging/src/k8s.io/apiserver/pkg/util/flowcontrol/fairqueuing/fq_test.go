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

package fairqueuing

import (
	"math"
	"sync"
	"testing"
	"time"

	"k8s.io/apiserver/pkg/util/clock"
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
func exerciseQueueSetUniformScenario(t *testing.T, qs QueueSet, sc uniformScenario, handSize int32, totalDuration time.Duration, expectPass bool) {
	wg := new(sync.WaitGroup)
	now := time.Now()
	clk := clock.NewFakeEventClock(now, wg, 0, nil)
	t.Logf("%s: Start", clk.Now().Format("2006-01-02 15:04:05.000000000"))
	integrators := make([]Integrator, len(sc))
	for i, uc := range sc {
		integrators[i] = NewIntegrator(clk)
		for j := 0; j < uc.nThreads; j++ {
			wg.Add(1)
			go func(i, j int, uc uniformClient, igr Integrator) {
				for k := 0; k < uc.nCalls; k++ {
					ClockWait(clk, wg, uc.thinkDuration)
					for {
						quiescent, execute, _ := qs.Wait(uc.hash, handSize)
						t.Logf("%s: %d, %d, %d got q=%v, e=%v", clk.Now().Format("2006-01-02 15:04:05.000000000"), i, j, k, quiescent, execute)
						if quiescent {
							continue
						}
						if !execute {
							break
						}
						igr.Add(1)
						ClockWait(clk, wg, uc.execDuration)
						igr.Add(-1)
						break
					}
				}
				wg.Done()
			}(i, j, uc, integrators[i])
		}
	}
	lim := now.Add(totalDuration)
	clk.Run(&lim)
	clk.SetTime(lim)
	t.Logf("%s: End", clk.Now().Format("2006-01-02 15:04:05.000000000"))
	results := make([]IntegratorResults, len(sc))
	var sumOfAvg float64
	for i := range sc {
		results[i] = integrators[i].GetResults()
		sumOfAvg += results[i].average
	}
	idealAverage := sumOfAvg / float64(len(sc))
	passes := make([]bool, len(sc))
	allPass := true
	for i := range sc {
		relDiff := (results[i].average - idealAverage) / idealAverage
		passes[i] = math.Abs(relDiff) <= 0.1
		allPass = allPass && passes[i]
	}
	for i := range sc {
		if allPass != expectPass {
			t.Errorf("Class %d got an average of %v but the ideal was %v", i, results[i].average, idealAverage)
		} else {
			t.Logf("Class %d got an average of %v and the ideal was %v", i, results[i].average, idealAverage)
		}
	}
	clk.Run(nil)
}

// TestDummy should fail because the dummy QueueSet exercises no control
func TestDummy(t *testing.T) {
	exerciseQueueSetUniformScenario(t, NewDummyQueueSet(), []uniformClient{
		{1001001001, 5, 10, time.Second, time.Second},
		{2002002002, 2, 10, time.Second, time.Second / 2},
	}, 1, time.Second*10, false)
}

func ClockWait(clk *clock.FakeEventClock, wg *sync.WaitGroup, duration time.Duration) {
	dunch := make(chan struct{})
	clk.EventAfterDuration(func(time.Time) {
		wg.Add(1)
		close(dunch)
	}, duration)
	wg.Done()
	select {
	case <-dunch:
	}
}
