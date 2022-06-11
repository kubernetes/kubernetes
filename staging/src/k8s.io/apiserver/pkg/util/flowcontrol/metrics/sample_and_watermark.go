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
	"sync"
	"time"

	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

const (
	labelNameMark       = "mark"
	labelValueLo        = "low"
	labelValueHi        = "high"
	LabelNamePhase      = "phase"
	LabelValueWaiting   = "waiting"
	LabelValueExecuting = "executing"
)

// SampleAndWaterMarkObserverVec creates RatioedGauges that
// populate histograms of samples and low- and high-water-marks.  The
// generator has a samplePeriod, and the histograms get an observation
// every samplePeriod.  The sampling windows are quantized based on
// the monotonic rather than wall-clock times.  The `t0` field is
// there so to provide a baseline for monotonic clock differences.
type SampleAndWaterMarkObserverVec struct {
	*sampleAndWaterMarkObserverVec
}

type sampleAndWaterMarkObserverVec struct {
	clock        clock.PassiveClock
	t0           time.Time
	samplePeriod time.Duration
	samples      *compbasemetrics.HistogramVec
	waterMarks   *compbasemetrics.HistogramVec
}

var _ RatioedGaugeVec = SampleAndWaterMarkObserverVec{}

// NewSampleAndWaterMarkHistogramsVec makes a new one
func NewSampleAndWaterMarkHistogramsVec(clock clock.PassiveClock, samplePeriod time.Duration, sampleOpts, waterMarkOpts *compbasemetrics.HistogramOpts, labelNames []string) SampleAndWaterMarkObserverVec {
	return SampleAndWaterMarkObserverVec{
		&sampleAndWaterMarkObserverVec{
			clock:        clock,
			t0:           clock.Now(),
			samplePeriod: samplePeriod,
			samples:      compbasemetrics.NewHistogramVec(sampleOpts, labelNames),
			waterMarks:   compbasemetrics.NewHistogramVec(waterMarkOpts, append([]string{labelNameMark}, labelNames...)),
		}}
}

func (swg *sampleAndWaterMarkObserverVec) quantize(when time.Time) int64 {
	return int64(when.Sub(swg.t0) / swg.samplePeriod)
}

// NewForLabelValuesSafe makes a new RatioedGauge
func (swg *sampleAndWaterMarkObserverVec) NewForLabelValuesSafe(initialNumerator, initialDenominator float64, labelValues []string) RatioedGauge {
	ratio := initialNumerator / initialDenominator
	when := swg.clock.Now()
	return &sampleAndWaterMarkHistograms{
		sampleAndWaterMarkObserverVec: swg,
		labelValues:                   labelValues,
		loLabelValues:                 append([]string{labelValueLo}, labelValues...),
		hiLabelValues:                 append([]string{labelValueHi}, labelValues...),
		denominator:                   initialDenominator,
		sampleAndWaterMarkAccumulator: sampleAndWaterMarkAccumulator{
			lastSet:    when,
			lastSetInt: swg.quantize(when),
			numerator:  initialNumerator,
			ratio:      ratio,
			loRatio:    ratio,
			hiRatio:    ratio,
		}}
}

func (swg *sampleAndWaterMarkObserverVec) metrics() Registerables {
	return Registerables{swg.samples, swg.waterMarks}
}

type sampleAndWaterMarkHistograms struct {
	*sampleAndWaterMarkObserverVec
	labelValues                  []string
	loLabelValues, hiLabelValues []string

	sync.Mutex
	denominator float64
	sampleAndWaterMarkAccumulator
}

type sampleAndWaterMarkAccumulator struct {
	lastSet          time.Time
	lastSetInt       int64 // lastSet / samplePeriod
	numerator        float64
	ratio            float64 // numerator/denominator
	loRatio, hiRatio float64
}

var _ RatioedGauge = (*sampleAndWaterMarkHistograms)(nil)

func (saw *sampleAndWaterMarkHistograms) Set(numerator float64) {
	saw.innerSet(func() {
		saw.numerator = numerator
	})
}

func (saw *sampleAndWaterMarkHistograms) Add(deltaNumerator float64) {
	saw.innerSet(func() {
		saw.numerator += deltaNumerator
	})
}
func (saw *sampleAndWaterMarkHistograms) Sub(deltaNumerator float64) {
	saw.innerSet(func() {
		saw.numerator -= deltaNumerator
	})
}

func (saw *sampleAndWaterMarkHistograms) Inc() {
	saw.innerSet(func() {
		saw.numerator += 1
	})
}
func (saw *sampleAndWaterMarkHistograms) Dec() {
	saw.innerSet(func() {
		saw.numerator -= 1
	})
}

func (saw *sampleAndWaterMarkHistograms) SetToCurrentTime() {
	saw.innerSet(func() {
		saw.numerator = float64(saw.clock.Now().Sub(time.Unix(0, 0)))
	})
}

func (saw *sampleAndWaterMarkHistograms) SetDenominator(denominator float64) {
	saw.innerSet(func() {
		saw.denominator = denominator
	})
}

func (saw *sampleAndWaterMarkHistograms) innerSet(updateNumeratorOrDenominator func()) {
	when, whenInt, acc, wellOrdered := func() (time.Time, int64, sampleAndWaterMarkAccumulator, bool) {
		saw.Lock()
		defer saw.Unlock()
		// Moved these variables here to tiptoe around https://github.com/golang/go/issues/43570 for #97685
		when := saw.clock.Now()
		whenInt := saw.quantize(when)
		acc := saw.sampleAndWaterMarkAccumulator
		wellOrdered := !when.Before(acc.lastSet)
		updateNumeratorOrDenominator()
		saw.ratio = saw.numerator / saw.denominator
		if wellOrdered {
			if acc.lastSetInt < whenInt {
				saw.loRatio, saw.hiRatio = acc.ratio, acc.ratio
				saw.lastSetInt = whenInt
			}
			saw.lastSet = when
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
		if saw.ratio < saw.loRatio {
			saw.loRatio = saw.ratio
		} else if saw.ratio > saw.hiRatio {
			saw.hiRatio = saw.ratio
		}
		return when, whenInt, acc, wellOrdered
	}()
	if !wellOrdered {
		lastSetS := acc.lastSet.String()
		whenS := when.String()
		klog.Errorf("Time went backwards from %s to %s for labelValues=%#+v", lastSetS, whenS, saw.labelValues)
	}
	for acc.lastSetInt < whenInt {
		saw.samples.WithLabelValues(saw.labelValues...).Observe(acc.ratio)
		saw.waterMarks.WithLabelValues(saw.loLabelValues...).Observe(acc.loRatio)
		saw.waterMarks.WithLabelValues(saw.hiLabelValues...).Observe(acc.hiRatio)
		acc.lastSetInt++
		acc.loRatio, acc.hiRatio = acc.ratio, acc.ratio
	}
}
