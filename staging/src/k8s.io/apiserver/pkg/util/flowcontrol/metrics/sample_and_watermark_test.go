/*
Copyright 2021 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"math/rand"
	"testing"
	"time"

	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/klog/v2"
	testclock "k8s.io/utils/clock/testing"
)

const (
	samplesHistName       = "sawtestsamples"
	samplingPeriod        = time.Millisecond
	ddtRangeCentiPeriods  = 300
	ddtOffsetCentiPeriods = 50
	numIterations         = 100
)

var errMetricNotFound = errors.New("not found")

/* TestSampler does a rough behavioral test of the sampling in a
   SampleAndWatermarkHistograms.  The test creates one and exercises
   it, checking that the count in the sampling histogram is correct at
   each step.  The sampling histogram is expected to get one
   observation at the end of each sampling period.  A fake clock is
   used, and the exercise consists of repeatedly changing that fake
   clock by an amount of time chosen uniformly at random from a range
   that goes from a little negative to somewhat more than two sampling
   periods.  The negative changes are included because small negative
   changes have been observed in real monotonic clock readings (see
   issue #96459) and we want to test that they are properly tolerated.
   The designed toleration is to pretend that the clock did not
   change, until it resumes net forward progress.
*/
func TestSampler(t *testing.T) {
	t0 := time.Now()
	clk := testclock.NewFakePassiveClock(t0)
	buckets := []float64{0, 1}
	gen := NewSampleAndWaterMarkHistogramsGenerator(clk, samplingPeriod,
		&compbasemetrics.HistogramOpts{Name: samplesHistName, Buckets: buckets},
		&compbasemetrics.HistogramOpts{Name: "marks", Buckets: buckets},
		[]string{})
	saw := gen.Generate(0, 1, []string{})
	toRegister := gen.metrics()
	registry := compbasemetrics.NewKubeRegistry()
	for _, reg := range toRegister {
		registry.MustRegister(reg)
	}
	// `dt` is the admitted cumulative difference in fake time
	// since the start of the test.  "admitted" means this is
	// never allowed to decrease, which matches the designed
	// toleration for negative monotonic clock changes.
	var dt time.Duration
	// `t1` is the current fake time
	t1 := t0.Add(dt)
	klog.Infof("Expect about %v warnings about time going backwards; this is fake time deliberately misbehaving.", (numIterations*ddtOffsetCentiPeriods)/ddtRangeCentiPeriods)
	t.Logf("t0=%s", t0)
	for i := 0; i < numIterations; i++ {
		// `ddt` is the next step to take in fake time
		ddt := time.Duration(rand.Intn(ddtRangeCentiPeriods)-ddtOffsetCentiPeriods) * samplingPeriod / 100
		t1 = t1.Add(ddt)
		diff := t1.Sub(t0)
		if diff > dt {
			dt = diff
		}
		clk.SetTime(t1)
		saw.Observe(1)
		expectedCount := int64(dt / samplingPeriod)
		actualCount, err := getHistogramCount(registry, samplesHistName)
		if err != nil && !(err == errMetricNotFound && expectedCount == 0) {
			t.Fatalf("For t0=%s, t1=%s, failed to getHistogramCount: %#+v", t0, t1, err)
		}
		t.Logf("For i=%d, ddt=%s, t1=%s, diff=%s, dt=%s, count=%d", i, ddt, t1, diff, dt, actualCount)
		if expectedCount != actualCount {
			t.Errorf("For i=%d, t0=%s, ddt=%s, t1=%s, expectedCount=%d, actualCount=%d", i, t0, ddt, t1, expectedCount, actualCount)
		}
	}
}

/* getHistogramCount returns the count of the named histogram or an error (if any) */
func getHistogramCount(registry compbasemetrics.KubeRegistry, metricName string) (int64, error) {
	mfs, err := registry.Gather()
	if err != nil {
		return 0, fmt.Errorf("failed to gather metrics: %w", err)
	}
	for _, mf := range mfs {
		thisName := mf.GetName()
		if thisName != metricName {
			continue
		}
		metric := mf.GetMetric()[0]
		hist := metric.GetHistogram()
		if hist == nil {
			return 0, errors.New("dto.Metric has nil Histogram")
		}
		if hist.SampleCount == nil {
			return 0, errors.New("dto.Histogram has nil SampleCount")
		}
		return int64(*hist.SampleCount), nil
	}
	return 0, errMetricNotFound
}
