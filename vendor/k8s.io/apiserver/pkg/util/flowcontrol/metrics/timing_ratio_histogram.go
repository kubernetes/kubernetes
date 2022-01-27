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
	"context"
	"sync"
	"time"

	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/klog/v2"
)

// TimingRatioHistogram is essentially a gauge for a ratio where the client
// independently controls the numerator and denominator.
// When scraped it produces a histogram of samples of the ratio
// taken at the end of every nanosecond.
// `*TimingRatioHistogram` implements both Registerable and RatioedGauge.
type TimingRatioHistogram struct {
	// The implementation is layered on TimingHistogram,
	// adding the division by an occasionally adjusted denominator.

	// Registerable is the registerable aspect.
	// That is the registerable aspect of the underlying TimingHistogram.
	compbasemetrics.Registerable

	// timingRatioHistogramInner implements the RatioedGauge aspect.
	timingRatioHistogramInner
}

// TimingRatioHistogramOpts is the constructor parameters of a TimingRatioHistogram.
// The `TimingHistogramOpts.InitialValue` is the initial numerator.
type TimingRatioHistogramOpts struct {
	compbasemetrics.TimingHistogramOpts
	InitialDenominator float64
}

// timingRatioHistogramInner implements the instrumentation aspect
type timingRatioHistogramInner struct {
	nowFunc         func() time.Time
	getGaugeOfRatio func() Gauge
	sync.Mutex
	// access only with mutex locked
	numerator, denominator float64
}

var _ RatioedGauge = &timingRatioHistogramInner{}
var _ RatioedGauge = &TimingRatioHistogram{}
var _ compbasemetrics.Registerable = &TimingRatioHistogram{}

// NewTimingHistogram returns an object which is TimingHistogram-like. However, nothing
// will be measured until the histogram is registered in at least one registry.
func NewTimingRatioHistogram(opts *TimingRatioHistogramOpts) *TimingRatioHistogram {
	return NewTestableTimingRatioHistogram(time.Now, opts)
}

// NewTestableTimingHistogram adds injection of the clock
func NewTestableTimingRatioHistogram(nowFunc func() time.Time, opts *TimingRatioHistogramOpts) *TimingRatioHistogram {
	ratioedOpts := opts.TimingHistogramOpts
	ratioedOpts.InitialValue /= opts.InitialDenominator
	th := compbasemetrics.NewTestableTimingHistogram(nowFunc, &ratioedOpts)
	return &TimingRatioHistogram{
		Registerable: th,
		timingRatioHistogramInner: timingRatioHistogramInner{
			nowFunc:         nowFunc,
			getGaugeOfRatio: func() Gauge { return th },
			numerator:       opts.InitialValue,
			denominator:     opts.InitialDenominator,
		}}
}

func (trh *timingRatioHistogramInner) Set(numerator float64) {
	trh.Lock()
	defer trh.Unlock()
	trh.numerator = numerator
	ratio := numerator / trh.denominator
	trh.getGaugeOfRatio().Set(ratio)
}

func (trh *timingRatioHistogramInner) Add(deltaNumerator float64) {
	trh.Lock()
	defer trh.Unlock()
	numerator := trh.numerator + deltaNumerator
	trh.numerator = numerator
	ratio := numerator / trh.denominator
	trh.getGaugeOfRatio().Set(ratio)
}

func (trh *timingRatioHistogramInner) Sub(deltaNumerator float64) {
	trh.Add(-deltaNumerator)
}

func (trh *timingRatioHistogramInner) Inc() {
	trh.Add(1)
}

func (trh *timingRatioHistogramInner) Dec() {
	trh.Add(-1)
}

func (trh *timingRatioHistogramInner) SetToCurrentTime() {
	trh.Set(float64(trh.nowFunc().Sub(time.Unix(0, 0))))
}

func (trh *timingRatioHistogramInner) SetDenominator(denominator float64) {
	trh.Lock()
	defer trh.Unlock()
	trh.denominator = denominator
	ratio := trh.numerator / denominator
	trh.getGaugeOfRatio().Set(ratio)
}

// WithContext allows the normal TimingHistogram metric to pass in context.
// The context is no-op at the current level of development.
func (trh *timingRatioHistogramInner) WithContext(ctx context.Context) RatioedGauge {
	return trh
}

// TimingRatioHistogramVec is a collection of TimingRatioHistograms that differ
// only in label values.
// `*TimingRatioHistogramVec` implements both Registerable and RatioedGaugeVec.
type TimingRatioHistogramVec struct {
	// promote only the Registerable methods
	compbasemetrics.Registerable
	// delegate is TimingHistograms of the ratio
	delegate compbasemetrics.GaugeVecMetric
}

var _ RatioedGaugeVec = &TimingRatioHistogramVec{}
var _ compbasemetrics.Registerable = &TimingRatioHistogramVec{}

// NewTimingHistogramVec constructs a new vector.
// `opts.InitialValue` is the initial ratio, but this applies
// only for the tiny period of time until NewForLabelValuesSafe sets
// the ratio based on the given initial numerator and denominator.
// Thus there is a tiny splinter of time during member construction when
// its underlying TimingHistogram is given the initial numerator rather than
// the initial ratio (which is obviously a non-issue when both are zero).
// Note the difficulties associated with extracting a member
// before registering the vector.
func NewTimingRatioHistogramVec(opts *compbasemetrics.TimingHistogramOpts, labelNames ...string) *TimingRatioHistogramVec {
	return NewTestableTimingRatioHistogramVec(time.Now, opts, labelNames...)
}

// NewTestableTimingHistogramVec adds injection of the clock.
func NewTestableTimingRatioHistogramVec(nowFunc func() time.Time, opts *compbasemetrics.TimingHistogramOpts, labelNames ...string) *TimingRatioHistogramVec {
	delegate := compbasemetrics.NewTestableTimingHistogramVec(nowFunc, opts, labelNames)
	return &TimingRatioHistogramVec{
		Registerable: delegate,
		delegate:     delegate,
	}
}

func (v *TimingRatioHistogramVec) metrics() Registerables {
	return Registerables{v}
}

// NewForLabelValuesChecked will return an error if this vec is not hidden and not yet registered
// or there is a syntactic problem with the labelValues.
func (v *TimingRatioHistogramVec) NewForLabelValuesChecked(initialNumerator, initialDenominator float64, labelValues []string) (RatioedGauge, error) {
	underMember, err := v.delegate.WithLabelValuesChecked(labelValues...)
	if err != nil {
		return noopRatioed{}, err
	}
	underMember.Set(initialNumerator / initialDenominator)
	return &timingRatioHistogramInner{
		getGaugeOfRatio: func() Gauge { return underMember },
		numerator:       initialNumerator,
		denominator:     initialDenominator,
	}, nil
}

// NewForLabelValuesSafe is the same as NewForLabelValuesChecked in cases where that does not
// return an error.  When the unsafe version returns an error due to the vector not being
// registered yet, the safe version returns an object that implements its methods
// by looking up the relevant vector member in each call (thus getting a non-noop after registration).
// In the other error cases the object returned here is a noop.
func (v *TimingRatioHistogramVec) NewForLabelValuesSafe(initialNumerator, initialDenominator float64, labelValues []string) RatioedGauge {
	tro, err := v.NewForLabelValuesChecked(initialNumerator, initialDenominator, labelValues)
	if err == nil {
		klog.V(3).InfoS("TimingRatioHistogramVec.NewForLabelValuesSafe hit the efficient case", "fqName", v.FQName(), "labelValues", labelValues)
		return tro
	}
	if !compbasemetrics.ErrIsNotRegistered(err) {
		klog.ErrorS(err, "Failed to extract TimingRatioHistogramVec member, using noop instead", "vectorname", v.FQName(), "labelValues", labelValues)
		return tro
	}
	klog.V(3).InfoS("TimingRatioHistogramVec.NewForLabelValuesSafe hit the inefficient case", "fqName", v.FQName(), "labelValues", labelValues)
	// At this point we know v.NewForLabelValuesChecked(..) returns a permanent noop,
	// which we precisely want to avoid using.  Instead, make our own gauge that
	// fetches the element on every Set.
	return &timingRatioHistogramInner{
		getGaugeOfRatio: func() Gauge { return v.delegate.WithLabelValues(labelValues...) },
		numerator:       initialNumerator,
		denominator:     initialDenominator,
	}
}

type noopRatioed struct{}

func (noopRatioed) Set(float64)            {}
func (noopRatioed) Add(float64)            {}
func (noopRatioed) Sub(float64)            {}
func (noopRatioed) Inc()                   {}
func (noopRatioed) Dec()                   {}
func (noopRatioed) SetToCurrentTime()      {}
func (noopRatioed) SetDenominator(float64) {}

func (v *TimingRatioHistogramVec) Reset() {
	v.delegate.Reset()
}
