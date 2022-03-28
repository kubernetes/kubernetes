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

package metrics

import (
	"fmt"
	"sync"
	"time"

	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

const (
	labelNameLB = "lb"
)

// NewSampleAndCountHistogramsGenerator makes a new one
func NewSampleAndCountHistogramsPairGenerator(clock clock.PassiveClock, samplePeriod time.Duration, sampleOpts *compbasemetrics.HistogramOpts, countOpts *compbasemetrics.CounterOpts, countBuckets []float64, labelNames []string) DelegatingRatioedChangeObserverPairGenerator {
	return DelegatingRatioedChangeObserverPairGenerator{
		urGenerator: NewSampleAndCountHistogramsGenerator(clock, samplePeriod, sampleOpts, countOpts, countBuckets, append([]string{labelNamePhase}, labelNames...)),
	}
}

type SampleAndCountObserverGenerator struct {
	*sampleAndCountObserverGenerator
}

type sampleAndCountObserverGenerator struct {
	clock             clock.PassiveClock
	t0                time.Time
	samplePeriod      time.Duration
	samples           *compbasemetrics.HistogramVec
	countBuckets      []float64
	bucketLabelValues []string
	counts            *compbasemetrics.CounterVec
}

var _ RatioedChangeObserverGenerator = SampleAndCountObserverGenerator{}

// NewSampleAndCountHistogramsGenerator makes a new one
func NewSampleAndCountHistogramsGenerator(clock clock.PassiveClock, samplePeriod time.Duration, sampleOpts *compbasemetrics.HistogramOpts, countOpts *compbasemetrics.CounterOpts, countBuckets []float64, labelNames []string) SampleAndCountObserverGenerator {
	bucketLabelValues := make([]string, len(countBuckets))
	for idx, lb := range countBuckets {
		bucketLabelValues[idx] = fmt.Sprintf("%g", lb)
	}
	return SampleAndCountObserverGenerator{
		&sampleAndCountObserverGenerator{
			clock:             clock,
			t0:                clock.Now(),
			samplePeriod:      samplePeriod,
			samples:           compbasemetrics.NewHistogramVec(sampleOpts, labelNames),
			countBuckets:      countBuckets,
			bucketLabelValues: bucketLabelValues,
			counts:            compbasemetrics.NewCounterVec(countOpts, append([]string{labelNameLB}, labelNames...)),
		}}
}

func (swg *sampleAndCountObserverGenerator) quantize(when time.Time) int64 {
	return int64(when.Sub(swg.t0) / swg.samplePeriod)
}

// Generate makes a new RatioedChangeObserver
func (swg *sampleAndCountObserverGenerator) Generate(initialNumerator, initialDenominator float64, labelValues []string) RatioedChangeObserver {
	ratio := initialNumerator / initialDenominator
	when := swg.clock.Now()
	countLabelValues := [][]string{}
	for idx := range swg.countBuckets {
		countLabelValues = append(countLabelValues, append([]string{swg.bucketLabelValues[idx]}, labelValues...))
	}
	return &sampleAndCountHistograms{
		sampleAndCountObserverGenerator: swg,
		labelValues:                     labelValues,
		countLabelValues:                countLabelValues,
		denominator:                     initialDenominator,
		sampleAndCountAccumulator: sampleAndCountAccumulator{
			lastSet:    when,
			lastSetInt: swg.quantize(when),
			numerator:  initialNumerator,
			ratio:      ratio,
			lastBucket: findBucket(swg.countBuckets, ratio),
		}}
}

func (swg *sampleAndCountObserverGenerator) Metrics() Registerables {
	return Registerables{swg.samples, swg.counts}
}

type sampleAndCountHistograms struct {
	*sampleAndCountObserverGenerator
	labelValues      []string
	countLabelValues [][]string // one []string for each bucket

	sync.Mutex
	denominator float64
	sampleAndCountAccumulator
}

type sampleAndCountAccumulator struct {
	lastSet    time.Time
	lastSetInt int64 // lastSet / samplePeriod
	numerator  float64
	ratio      float64 // numerator/denominator
	lastBucket int     // negative when not in a bucket
}

var _ RatioedChangeObserver = (*sampleAndCountHistograms)(nil)

func (saw *sampleAndCountHistograms) Add(deltaNumerator float64) {
	saw.innerSet(func() {
		saw.numerator += deltaNumerator
	})
}

func (saw *sampleAndCountHistograms) Observe(numerator float64) {
	saw.innerSet(func() {
		saw.numerator = numerator
	})
}

func (saw *sampleAndCountHistograms) SetDenominator(denominator float64) {
	saw.innerSet(func() {
		saw.denominator = denominator
	})
}

func findBucket(countBuckets []float64, x float64) int {
	if x < countBuckets[0] {
		return -1
	}
	var idx int
	for idx = len(countBuckets) - 1; idx > 0 && x < countBuckets[idx]; idx-- {
	}
	return idx
}

func (saw *sampleAndCountHistograms) innerSet(updateNumeratorOrDenominator func()) {
	when, whenInt, acc, wellOrdered := func() (time.Time, int64, sampleAndCountAccumulator, bool) {
		saw.Lock()
		defer saw.Unlock()
		// Moved these variables here to tiptoe around https://github.com/golang/go/issues/43570 for #97685
		when := saw.clock.Now()
		whenInt := saw.quantize(when)
		acc := saw.sampleAndCountAccumulator
		dt := when.Sub(acc.lastSet)
		wellOrdered := dt >= 0
		updateNumeratorOrDenominator()
		saw.ratio = saw.numerator / saw.denominator
		if wellOrdered {
			bucket := findBucket(saw.countBuckets, saw.ratio)
			if saw.lastBucket >= 0 {
				saw.counts.WithLabelValues(saw.countLabelValues[saw.lastBucket]...).Add(dt.Seconds())
			}
			saw.lastSet = when
			saw.lastBucket = bucket
		}
		// `wellOrdered` should always be true because we are using
		// monotonic clock readings and they never go backwards.  Yet
		// very small backwards steps (under 1 microsecond) have been
		// observed
		// (https://github.com/kubernetes/kubernetes/issues/96459).
		// In the backwards case, treat the current reading as if it
		// had occurred at time `saw.lastSet` and log an error.  It
		// would be wrong to update `saw.lastSet` in this case because
		// that plants a time bomb for future updates to
		// `saw.lastSetInt`.
		return when, whenInt, acc, wellOrdered
	}()
	if !wellOrdered {
		lastSetS := acc.lastSet.String()
		whenS := when.String()
		klog.Errorf("Time went backwards from %s to %s for labelValues=%#+v", lastSetS, whenS, saw.labelValues)
	}
	for acc.lastSetInt < whenInt {
		saw.samples.WithLabelValues(saw.labelValues...).Observe(acc.ratio)
		acc.lastSetInt++
	}
}
