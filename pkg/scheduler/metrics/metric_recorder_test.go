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

package metrics

import (
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

var _ MetricRecorder = &fakePodsRecorder{}

type fakePodsRecorder struct {
	counter int64
}

func (r *fakePodsRecorder) Inc() {
	atomic.AddInt64(&r.counter, 1)
}

func (r *fakePodsRecorder) Dec() {
	atomic.AddInt64(&r.counter, -1)
}

func (r *fakePodsRecorder) Clear() {
	atomic.StoreInt64(&r.counter, 0)
}

func TestInc(t *testing.T) {
	fakeRecorder := fakePodsRecorder{}
	var wg sync.WaitGroup
	loops := 100
	wg.Add(loops)
	for i := 0; i < loops; i++ {
		go func() {
			fakeRecorder.Inc()
			wg.Done()
		}()
	}
	wg.Wait()
	if fakeRecorder.counter != int64(loops) {
		t.Errorf("Expected %v, got %v", loops, fakeRecorder.counter)
	}
}

func TestDec(t *testing.T) {
	fakeRecorder := fakePodsRecorder{counter: 100}
	var wg sync.WaitGroup
	loops := 100
	wg.Add(loops)
	for i := 0; i < loops; i++ {
		go func() {
			fakeRecorder.Dec()
			wg.Done()
		}()
	}
	wg.Wait()
	if fakeRecorder.counter != int64(0) {
		t.Errorf("Expected %v, got %v", loops, fakeRecorder.counter)
	}
}

func TestClear(t *testing.T) {
	fakeRecorder := fakePodsRecorder{}
	var wg sync.WaitGroup
	incLoops, decLoops := 100, 80
	wg.Add(incLoops + decLoops)
	for i := 0; i < incLoops; i++ {
		go func() {
			fakeRecorder.Inc()
			wg.Done()
		}()
	}
	for i := 0; i < decLoops; i++ {
		go func() {
			fakeRecorder.Dec()
			wg.Done()
		}()
	}
	wg.Wait()
	if fakeRecorder.counter != int64(incLoops-decLoops) {
		t.Errorf("Expected %v, got %v", incLoops-decLoops, fakeRecorder.counter)
	}
	// verify Clear() works
	fakeRecorder.Clear()
	if fakeRecorder.counter != int64(0) {
		t.Errorf("Expected %v, got %v", 0, fakeRecorder.counter)
	}
}

func TestInFlightEventAsync(t *testing.T) {
	r := &MetricAsyncRecorder{
		aggregatedInflightEventMetric:              map[gaugeVecMetricKey]int{},
		aggregatedInflightEventMetricLastFlushTime: time.Now(),
		aggregatedInflightEventMetricBufferCh:      make(chan *gaugeVecMetric, 100),
		interval:                                   time.Hour,
	}

	podAddLabel := "Pod/Add"
	r.ObserveInFlightEventsAsync(podAddLabel, 10, false)
	r.ObserveInFlightEventsAsync(podAddLabel, -1, false)
	r.ObserveInFlightEventsAsync(PodPoppedInFlightEvent, 1, false)

	if d := cmp.Diff(r.aggregatedInflightEventMetric, map[gaugeVecMetricKey]int{
		{metricName: InFlightEvents.Name, labelValue: podAddLabel}:            9,
		{metricName: InFlightEvents.Name, labelValue: PodPoppedInFlightEvent}: 1,
	}, cmp.AllowUnexported(gaugeVecMetric{})); d != "" {
		t.Errorf("unexpected aggregatedInflightEventMetric: %s", d)
	}

	r.aggregatedInflightEventMetricLastFlushTime = time.Now().Add(-time.Hour) // to test flush

	// It adds -4 and flushes the metric to the channel.
	r.ObserveInFlightEventsAsync(podAddLabel, -4, false)
	if len(r.aggregatedInflightEventMetric) != 0 {
		t.Errorf("aggregatedInflightEventMetric should be cleared, but got: %v", r.aggregatedInflightEventMetric)
	}

	got := []gaugeVecMetric{}
	for {
		select {
		case m := <-r.aggregatedInflightEventMetricBufferCh:
			got = append(got, *m)
			continue
		default:
		}
		// got all
		break
	}

	// sort got to avoid the flaky test
	sort.Slice(got, func(i, j int) bool {
		return got[i].labelValues[0] < got[j].labelValues[0]
	})

	if d := cmp.Diff(got, []gaugeVecMetric{
		{
			labelValues: []string{podAddLabel},
			valueToAdd:  5,
		},
		{
			labelValues: []string{PodPoppedInFlightEvent},
			valueToAdd:  1,
		},
	}, cmp.AllowUnexported(gaugeVecMetric{}), cmpopts.IgnoreFields(gaugeVecMetric{}, "metric")); d != "" {
		t.Errorf("unexpected metrics are sent to aggregatedInflightEventMetricBufferCh: %s", d)
	}

	// Test force flush
	r.ObserveInFlightEventsAsync(podAddLabel, 1, true)
	if len(r.aggregatedInflightEventMetric) != 0 {
		t.Errorf("aggregatedInflightEventMetric should be force-flushed, but got: %v", r.aggregatedInflightEventMetric)
	}
}
