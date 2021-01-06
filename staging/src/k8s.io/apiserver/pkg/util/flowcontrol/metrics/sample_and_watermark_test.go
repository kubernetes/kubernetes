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
	"fmt"
	"math/rand"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
)

const (
	samplesHistName = "sawtestsamples"
	ddtRange        = 3000
	ddtOffset       = 500
	numIterations   = 100
)

func TestSampler(t *testing.T) {
	t0 := time.Now()
	clk := clock.NewFakePassiveClock(t0)
	buckets := []float64{0, 1}
	const samplingPeriod = time.Millisecond
	gen := NewSampleAndWaterMarkHistogramsGenerator(clk, samplingPeriod,
		&compbasemetrics.HistogramOpts{Name: samplesHistName, Buckets: buckets},
		&compbasemetrics.HistogramOpts{Name: "marks", Buckets: buckets},
		[]string{})
	saw := gen.Generate(0, 1, []string{})
	regs := gen.metrics()
	for _, reg := range regs {
		legacyregistry.MustRegister(reg)
	}
	dt := 2 * samplingPeriod
	t1 := t0.Add(dt)
	klog.Infof("Expect about %v warnings about time going backwards; this is fake time deliberately misbehaving.", (numIterations*ddtOffset)/ddtRange)
	t.Logf("t0=%s", t0)
	for i := 0; i < numIterations; i++ {
		ddt := time.Microsecond * time.Duration(rand.Intn(ddtRange)-ddtOffset)
		t1 = t1.Add(ddt)
		diff := t1.Sub(t0)
		if diff > dt {
			dt = diff
		}
		clk.SetTime(t1)
		saw.Set(1)
		expectedCount := int64(dt / time.Millisecond)
		actualCount, err := getHistogramCount(regs, samplesHistName)
		if err != nil {
			t.Fatalf("For t0=%s, t1=%s, failed to getHistogramCount: %#+v", t0, t1, err)
		}
		t.Logf("For i=%d, ddt=%s, t1=%s, diff=%s, dt=%s, count=%d", i, ddt, t1, diff, dt, actualCount)
		if expectedCount != actualCount {
			t.Errorf("For i=%d, t0=%s, ddt=%s, t1=%s, expectedCount=%d, actualCount=%d", i, t0, ddt, t1, expectedCount, actualCount)
		}
	}
}

func getHistogramCount(regs Registerables, metricName string) (int64, error) {
	considered := []string{}
	mfs, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		return 0, fmt.Errorf("failed to gather metrics: %s", err)
	}
	for _, mf := range mfs {
		thisName := mf.GetName()
		if thisName != metricName {
			considered = append(considered, thisName)
			continue
		}
		metric := mf.GetMetric()[0]
		hist := metric.GetHistogram()
		if hist == nil {
			return 0, fmt.Errorf("dto.Metric has nil Histogram")
		}
		if hist.SampleCount == nil {
			return 0, fmt.Errorf("dto.Histogram has nil SampleCount")
		}
		return int64(*hist.SampleCount), nil
	}
	return 0, fmt.Errorf("not found, considered=%#+v", considered)
}
