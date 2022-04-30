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
// `*TimingRatioHistogram` implements both Registerable and RatioedObserver.
type TimingRatioHistogram struct {
	// The implementation is layered on TimingHistogram,
	// adding the division by an occasionally adjusted denominator.

	// Registerable is the registerable aspect.
	// That is the registerable aspect of the underlying TimingHistogram.
	compbasemetrics.Registerable

	// timingRatioHistogramInner implements the RatioedObsserver aspect.
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
	under interface{ Set(float64) }
	sync.Mutex
	numerator, denominator float64 // access only with mutex locked
}

var _ RatioedObserver = &timingRatioHistogramInner{}
var _ RatioedObserver = &TimingRatioHistogram{}
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
			under:       th,
			numerator:   opts.InitialValue,
			denominator: opts.InitialDenominator,
		}}
}

func (trh *timingRatioHistogramInner) Observe(numerator float64) {
	trh.Lock()
	defer trh.Unlock()
	trh.numerator = numerator
	ratio := numerator / trh.denominator
	trh.under.Set(ratio)
}

func (trh *timingRatioHistogramInner) SetDenominator(denominator float64) {
	trh.Lock()
	defer trh.Unlock()
	trh.denominator = denominator
	ratio := trh.numerator / denominator
	trh.under.Set(ratio)
}

// WithContext allows the normal TimingHistogram metric to pass in context.
// The context is no-op at the current level of development.
func (trh *timingRatioHistogramInner) WithContext(ctx context.Context) RatioedObserver {
	return trh
}

// TimingRatioHistogramVec is a collection of TimingRatioHistograms that differ
// only in label values.
// `*TimingRatioHistogramVec` implements both Registerable and RatioedObserverVec.
type TimingRatioHistogramVec struct {
	compbasemetrics.Registerable                                // promote only the Registerable methods
	under                        compbasemetrics.GaugeVecMetric // TimingHistograms of the ratio
	initialNumerator             float64
}

var _ RatioedObserverVec = &TimingRatioHistogramVec{}
var _ compbasemetrics.Registerable = &TimingRatioHistogramVec{}

// NewTimingHistogramVec constructs a new vector.
// `opts.InitialValue` is the initial numerator.
// The initial denominator can be different for each member of the vector
// and is supplied when extracting a member.
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
	under := compbasemetrics.NewTestableTimingHistogramVec(nowFunc, opts, labelNames)
	return &TimingRatioHistogramVec{
		Registerable:     under,
		under:            under,
		initialNumerator: opts.InitialValue,
	}
}

func (v *TimingRatioHistogramVec) metrics() Registerables {
	return Registerables{v}
}

// WithLabelValuesChecked will return an error if this vec is not hidden and not yet registered
// or there is a syntactic problem with the labelValues.
func (v *TimingRatioHistogramVec) WithLabelValuesChecked(initialDenominator float64, labelValues ...string) (RatioedObserver, error) {
	underMember, err := v.under.WithLabelValuesChecked(labelValues...)
	if err != nil {
		return noopRatioed{}, err
	}
	underMember.Set(v.initialNumerator / initialDenominator)
	return &timingRatioHistogramInner{
			under:       underMember,
			numerator:   v.initialNumerator,
			denominator: initialDenominator,
		},
		nil
}

// WithLabelValuesSafe is the same as WithLabelValuesChecked in cases where that does not
// return an error.  When the unsafe version returns an error due to the vector not being
// registered yet, the safe version returns an object that implements its methods
// by first calling WithLabelValuesChecked and then delegating to whatever observer that returns.
// In the other error cases the object returned here is a noop.
func (v *TimingRatioHistogramVec) WithLabelValuesSafe(initialDenominator float64, labelValues ...string) RatioedObserver {
	tro, err := v.WithLabelValuesChecked(initialDenominator, labelValues...)
	if err == nil {
		return tro
	}
	if !compbasemetrics.ErrIsNotRegistered(err) {
		klog.ErrorS(err, "Failed to extract TimingRatioHistogramVec member, using noop instead", "vectorname", v.FQName(), "labelValues", labelValues)
		return tro
	}
	return &timingRatioHistogramInner{
		under:       &timingHistogramVecElt{under: v.under, labelValues: labelValues},
		numerator:   v.initialNumerator,
		denominator: initialDenominator,
	}
}

type timingHistogramVecElt struct {
	under       compbasemetrics.GaugeVecMetric
	labelValues []string
}

func (tve *timingHistogramVecElt) Set(x float64) {
	tve.under.WithLabelValues(tve.labelValues...).Set(x)
}

// With, if called after this vector has been
// registered in at least one registry and this vector is not
// hidden, will return a RatioedObserver that is NOT a noop along
// with nil error.  If called on a hidden vector then it will
// return a noop and a nil error.  Otherwise it returns a noop
// and an error that passes compbasemetrics.ErrIsNotRegistered.
func (v *TimingRatioHistogramVec) With(initialDenominator float64, labels map[string]string) (RatioedObserver, error) {
	underMember, err := v.under.WithChecked(labels)
	if err != nil {
		return noopRatioed{}, err
	}
	underMember.Set(v.initialNumerator / initialDenominator)
	return &timingRatioHistogramInner{under: underMember,
			numerator:   v.initialNumerator,
			denominator: initialDenominator},
		nil
}

type noopRatioed struct{}

func (noopRatioed) Observe(float64)        {}
func (noopRatioed) SetDenominator(float64) {}

func (v *TimingRatioHistogramVec) Reset() {
	v.under.Reset()
}

// WithContext returns wrapped TimingHistogramVec with context
func (v *TimingRatioHistogramVec) WithContext(ctx context.Context) *TimingRatioHistogramVecWithContext {
	return &TimingRatioHistogramVecWithContext{
		ctx:                     ctx,
		TimingRatioHistogramVec: v,
	}
}

// TimingHistogramVecWithContext is the wrapper of TimingHistogramVec with context.
type TimingRatioHistogramVecWithContext struct {
	*TimingRatioHistogramVec
	ctx context.Context
}

type TimingRatioHistogramPairVec struct {
	urVec *TimingRatioHistogramVec
}

var _ RatioedObserverPairVec = TimingRatioHistogramPairVec{}

// NewTimedRatioObserverPairVec makes a new pair generator
func NewTimingRatioHistogramPairVec(opts *compbasemetrics.TimingHistogramOpts, labelNames ...string) TimingRatioHistogramPairVec {
	return NewTestableTimingRatioHistogramPairVec(time.Now, opts, labelNames...)
}

// NewTimedRatioObserverPairVec makes a new pair generator
func NewTestableTimingRatioHistogramPairVec(nowFunc func() time.Time, opts *compbasemetrics.TimingHistogramOpts, labelNames ...string) TimingRatioHistogramPairVec {
	return TimingRatioHistogramPairVec{
		urVec: NewTestableTimingRatioHistogramVec(nowFunc, opts, append([]string{LabelNamePhase}, labelNames...)...),
	}
}

// WithLabelValues extracts a member if it is not broken
func (pv TimingRatioHistogramPairVec) WithLabelValuesChecked(initialWaitingDenominator, initialExecutingDenominator float64, labelValues ...string) (RatioedObserverPair, error) {
	RequestsWaiting, err := pv.urVec.WithLabelValuesChecked(initialWaitingDenominator, append([]string{LabelValueWaiting}, labelValues...)...)
	if err != nil {
		return RatioedObserverPair{}, err
	}
	RequestsExecuting, err := pv.urVec.WithLabelValuesChecked(initialExecutingDenominator, append([]string{LabelValueExecuting}, labelValues...)...)
	return RatioedObserverPair{RequestsWaiting: RequestsWaiting, RequestsExecuting: RequestsExecuting}, err
}

// WithLabelValuesSafe extracts a member that will always work right and be more expensive
func (pv TimingRatioHistogramPairVec) WithLabelValuesSafe(initialWaitingDenominator, initialExecutingDenominator float64, labelValues ...string) RatioedObserverPair {
	return RatioedObserverPair{
		RequestsWaiting:   pv.urVec.WithLabelValuesSafe(initialWaitingDenominator, append([]string{LabelValueWaiting}, labelValues...)...),
		RequestsExecuting: pv.urVec.WithLabelValuesSafe(initialExecutingDenominator, append([]string{LabelValueExecuting}, labelValues...)...),
	}
}

func (pv TimingRatioHistogramPairVec) metrics() Registerables {
	return pv.urVec.metrics()
}
