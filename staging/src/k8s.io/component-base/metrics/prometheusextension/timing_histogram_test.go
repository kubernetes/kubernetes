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

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
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

func exerciseTimingHistogramAndCollector(th GaugeOps, t0 time.Time, clk *unsyncFakeClock, collect func(chan<- prometheus.Metric), expectCollection ...GaugeOps) func(t *testing.T) {
	return func(t *testing.T) {
		exerciseTimingHistogramData(t, th, t0, clk)
		exerciseTimingHistogramCollector(t, collect, expectCollection)
	}
}

func exerciseTimingHistogramCollector(t *testing.T, collect func(chan<- prometheus.Metric), expectCollection []GaugeOps) {
	remainingCollection := expectCollection
	metch := make(chan prometheus.Metric)
	go func() {
		collect(metch)
		close(metch)
	}()
	for collected := range metch {
		collectedGO := collected.(GaugeOps)
		newRem, found := findAndRemove(remainingCollection, collectedGO)
		if !found {
			t.Errorf("Collected unexpected value %#+v", collected)
		}
		remainingCollection = newRem
	}
	if len(remainingCollection) > 0 {
		t.Errorf("Collection omitted %#+v", remainingCollection)
	}
}

var thTestBuckets = []float64{0, 0.5, 1}
var thTestV0 float64 = 0.25

// exerciseTimingHistogramData takes the given histogram through the following points in (time,value) space.
// t0 is the clock time of the histogram's construction
// value=v0 for t0 <= t <= t1 where v0 = 0.25 and t1 = t0 + 1 ns
// value=v1 for t1 <= t <= t2 where v1 = 0.75 and t2 = t1 + 1 microsecond
// value=v2 for t2 <= t <= t3 where v2 = 1.25 and t3 = t2 + 1 millisecond
// value=v3 for t3 <= t <= t4 where v3 = 0.65 and t4 = t3 + 1 second
func exerciseTimingHistogramData(t *testing.T, th GaugeOps, t0 time.Time, clk *unsyncFakeClock) {
	t1 := t0.Add(time.Nanosecond)
	v0 := thTestV0
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
	writer := th.(prometheus.Metric)
	err := writer.Write(metric)
	if err != nil {
		t.Error(err)
	}
	wroteHist := metric.Histogram
	if want, got := uint64(t4.Sub(t0)), wroteHist.GetSampleCount(); want != got {
		t.Errorf("Wanted %v but got %v", want, got)
	}
	if want, got := tDiff(t1, t0)*v0+tDiff(t2, t1)*v1+tDiff(t3, t2)*v2+tDiff(t4, t3)*v3, wroteHist.GetSampleSum(); want != got {
		t.Errorf("Wanted %v but got %v", want, got)
	}
	wroteBuckets := wroteHist.GetBucket()
	if len(wroteBuckets) != len(thTestBuckets) {
		t.Errorf("Got buckets %#+v", wroteBuckets)
	}
	expectedCounts := []time.Duration{0, t1.Sub(t0), t2.Sub(t0) + t4.Sub(t3)}
	for idx, ub := range thTestBuckets {
		if want, got := uint64(expectedCounts[idx]), wroteBuckets[idx].GetCumulativeCount(); want != got {
			t.Errorf("In bucket %d, wanted %v but got %v", idx, want, got)
		}
		if want, got := ub, wroteBuckets[idx].GetUpperBound(); want != got {
			t.Errorf("In bucket %d, wanted %v but got %v", idx, want, got)
		}
	}
}

// tDiff returns a time difference as float
func tDiff(hi, lo time.Time) float64 { return float64(hi.Sub(lo)) }

func findAndRemove(metrics []GaugeOps, seek GaugeOps) ([]GaugeOps, bool) {
	for idx, metric := range metrics {
		if metric == seek {
			return append(append([]GaugeOps{}, metrics[:idx]...), metrics[idx+1:]...), true
		}
	}
	return metrics, false
}

func TestTimeIntegrationDirect(t *testing.T) {
	t0 := time.Now()
	clk := &unsyncFakeClock{t0}
	th, err := NewTestableTimingHistogram(clk.Now, TimingHistogramOpts{
		Name:         "TestTimeIntegration",
		Help:         "helpless",
		Buckets:      thTestBuckets,
		InitialValue: thTestV0,
	})
	if err != nil {
		t.Error(err)
		return
	}
	t.Run("non-vec", exerciseTimingHistogramAndCollector(th, t0, clk, th.Collect, th))
}

func TestTimingHistogramVec(t *testing.T) {
	t0 := time.Now()
	clk := &unsyncFakeClock{t0}
	vec := NewTestableTimingHistogramVec(clk.Now, TimingHistogramOpts{
		Name:         "TestTimeIntegration",
		Help:         "helpless",
		Buckets:      thTestBuckets,
		InitialValue: thTestV0,
	}, "k1", "k2")
	th1 := vec.With(prometheus.Labels{"k1": "a", "k2": "x"})
	th1b := vec.WithLabelValues("a", "x")
	if th1 != th1b {
		t.Errorf("Vector not functional")
	}
	t.Run("th1", exerciseTimingHistogramAndCollector(th1, t0, clk, vec.Collect, th1))
	t0 = clk.Now()
	th2 := vec.WithLabelValues("a", "y")
	if th1 == th2 {
		t.Errorf("Vector does not distinguish label values")
	}
	t.Run("th2", exerciseTimingHistogramAndCollector(th2, t0, clk, vec.Collect, th1, th2))
	t0 = clk.Now()
	th3 := vec.WithLabelValues("b", "y")
	if th1 == th3 || th2 == th3 {
		t.Errorf("Vector does not distinguish label values")
	}
	t.Run("th2", exerciseTimingHistogramAndCollector(th3, t0, clk, vec.Collect, th1, th2, th3))
}

type unsyncFakeClock struct {
	now time.Time
}

func (ufc *unsyncFakeClock) Now() time.Time {
	return ufc.now
}

func (ufc *unsyncFakeClock) SetTime(now time.Time) {
	ufc.now = now
}

func BenchmarkTimingHistogramDirect(b *testing.B) {
	b.StopTimer()
	now := time.Now()
	hist, err := NewTestableTimingHistogram(func() time.Time { return now }, TimingHistogramOpts{
		Namespace: "testns",
		Subsystem: "testsubsys",
		Name:      "testhist",
		Help:      "Me",
		Buckets:   []float64{1, 2, 4, 8, 16},
	})
	if err != nil {
		b.Error(err)
	}
	var x int
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		now = now.Add(time.Duration(31-x) * time.Microsecond)
		hist.Set(float64(x))
		x = (x + i) % 23
	}
}
func BenchmarkTimingHistogramVecEltCached(b *testing.B) {
	b.StopTimer()
	now := time.Now()
	vec := NewTestableTimingHistogramVec(func() time.Time { return now }, TimingHistogramOpts{
		Namespace: "testns",
		Subsystem: "testsubsys",
		Name:      "testhist",
		Help:      "Me",
		Buckets:   []float64{1, 2, 4, 8, 16},
	},
		"label1", "label2")
	hist := vec.WithLabelValues("val1", "val2")
	var x int
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		now = now.Add(time.Duration(31-x) * time.Microsecond)
		hist.Set(float64(x))
		x = (x + i) % 23
	}
}

func BenchmarkTimingHistogramVecEltFetched(b *testing.B) {
	b.StopTimer()
	now := time.Now()
	vec := NewTestableTimingHistogramVec(func() time.Time { return now }, TimingHistogramOpts{
		Namespace: "testns",
		Subsystem: "testsubsys",
		Name:      "testhist",
		Help:      "Me",
		Buckets:   []float64{1, 2, 4, 8, 16},
	},
		"label1", "label2")
	var x int
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		now = now.Add(time.Duration(31-x) * time.Microsecond)
		vec.WithLabelValues("val1", "val2").Set(float64(x))
		x = (x + i) % 23
	}
}
