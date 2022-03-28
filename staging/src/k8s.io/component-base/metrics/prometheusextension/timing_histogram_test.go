/*
Copyright 2022 The Kubernetes Authors.

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

package prometheusextension

import (
	"math"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"

	testclock "k8s.io/utils/clock/testing"
)

func TestTimingHistogramNonMonotonicBuckets(t *testing.T) {
	testCases := map[string][]float64{
		"not strictly monotonic":  {1, 2, 2, 3},
		"not monotonic at all":    {1, 2, 4, 3, 5},
		"have +Inf in the middle": {1, 2, math.Inf(+1), 3},
	}
	for name, buckets := range testCases {
		_, err := NewTimingHistogram(TimingHistogramOpts{
			Name:    "test_histogram",
			Help:    "helpless",
			Buckets: buckets,
		})
		if err == nil {
			t.Errorf("Buckets %v are %s but NewHTimingistogram did not complain.", buckets, name)
		}
	}
}

var testBuckets = []float64{0, 0.5, 1}
var value0 float64 = 0.25

func exerciseTimingHistogram(th TimingHistogram, t0 time.Time, clk *testclock.FakePassiveClock) func(t *testing.T) {
	return func(t *testing.T) {
		v0 := value0
		t1 := t0.Add(time.Nanosecond)
		var v1 float64 = 0.75
		clk.SetTime(t1)
		th.Set(v1)
		t2 := t1.Add(time.Microsecond)
		var d2 float64 = 0.5
		v2 := v1 + d2
		clk.SetTime(t2)
		th.Add(d2)
		t3 := t2
		for i := 0; i < 1000000; i++ {
			t3 = t3.Add(time.Nanosecond)
			clk.SetTime(t3)
			th.Set(v2)
		}
		var d3 float64 = -0.6
		v3 := v2 + d3
		th.Add(d3)
		t4 := t3.Add(time.Second)
		clk.SetTime(t4)

		metric := &dto.Metric{}
		err := th.Write(metric)
		if err != nil {
			t.Error(err)
		}
		wroteHist := metric.Histogram
		if want, got := uint64(t4.Sub(t0)), wroteHist.GetSampleCount(); want != got {
			t.Errorf("Wanted %v but got %v", want, got)
		}
		if want, got := float64(t1.Sub(t0))*v0+float64(t2.Sub(t1))*v1+float64(t3.Sub(t2))*v2+float64(t4.Sub(t3))*v3, wroteHist.GetSampleSum(); want != got {
			t.Errorf("Wanted %v but got %v", want, got)
		}
		wroteBuckets := wroteHist.GetBucket()
		if len(wroteBuckets) != len(testBuckets) {
			t.Errorf("Got buckets %#+v", wroteBuckets)
		}
		expectedCounts := []time.Duration{0, t1.Sub(t0), t2.Sub(t0) + t4.Sub(t3)}
		for idx, ub := range testBuckets {
			if want, got := wroteBuckets[idx].GetCumulativeCount(), uint64(expectedCounts[idx]); want != got {
				t.Errorf("In bucket %d, wanted %v but got %v", idx, want, got)
			}
			if want, got := ub, wroteBuckets[idx].GetUpperBound(); want != got {
				t.Errorf("In bucket %d, wanted %v but got %v", idx, want, got)
			}
		}
	}
}

func TestTimeIntegration(t *testing.T) {
	t0 := time.Now()
	clk := testclock.NewFakePassiveClock(t0)
	th, err := NewTestableTimingHistogram(clk, TimingHistogramOpts{
		Name:         "TestTimeIntegration",
		Help:         "helpless",
		Buckets:      testBuckets,
		InitialValue: value0,
	})
	if err != nil {
		t.Error(err)
		return
	}
	t.Run("non-vec", exerciseTimingHistogram(th, t0, clk))
}

func TestVecTimeIntegration(t *testing.T) {
	t0 := time.Now()
	clk := testclock.NewFakePassiveClock(t0)
	thv := NewTestableTimingHistogramVec(clk, TimingHistogramOpts{
		Name:         "TestTimeIntegration",
		Help:         "helpless",
		Buckets:      testBuckets,
		InitialValue: value0,
	},
		[]string{"l1", "l2"})
	th1, err := thv.GetMetricWithLabelValues("v1", "v2")
	if err != nil {
		t.Error(err)
		return
	}
	t.Run("th1", exerciseTimingHistogram(th1.(TimingHistogram), t0, clk))
	clk.SetTime(t0)
	th2, err := thv.GetMetricWithLabelValues("u1", "u2")
	if err != nil {
		t.Error(err)
		return
	}
	if th1 == th2 {
		t.Errorf("GetMetricWithLabelValues returned same metric for different labels")
	}
	t.Run("th2", exerciseTimingHistogram(th2.(TimingHistogram), t0, clk))
}
