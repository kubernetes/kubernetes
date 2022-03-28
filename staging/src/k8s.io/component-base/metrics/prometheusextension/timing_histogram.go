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
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"k8s.io/utils/clock"
)

// WritableVariable is a `float64`-valued variable that can be written to.
type WritableVariable interface {
	// Set the variable to the given value.
	Set(float64)

	// Add the given change to the variable
	Add(float64)
}

// A TimingHistogram tracks how long a `float64` variable spends in
// ranges defined by buckets.  Time is counted in nanoseconds.  The
// histogram's sum is the integral over time (in nanoseconds, from
// creation of the histogram) of the variable's value.
type TimingHistogram interface {
	prometheus.Metric
	prometheus.Collector
	WritableVariable
}

// TimingHistogramOpts is the parameters of the TimingHistogram constructor
type TimingHistogramOpts struct {
	Namespace   string
	Subsystem   string
	Name        string
	Help        string
	ConstLabels prometheus.Labels

	// Buckets defines the buckets into which observations are
	// accumulated. Each element in the slice is the upper
	// inclusive bound of a bucket. The values must be sorted in
	// strictly increasing order. There is no need to add a
	// highest bucket with +Inf bound. The default value is
	// prometheus.DefBuckets.
	Buckets []float64

	// The initial value of the variable.
	InitialValue float64
}

// NewTimingHistogram creates a new TimingHistogram
func NewTimingHistogram(opts TimingHistogramOpts) (TimingHistogram, error) {
	return NewTestableTimingHistogram(clock.RealClock{}, opts)
}

// NewTestableTimingHistogram creates a TimingHistogram that uses a mockable clock
func NewTestableTimingHistogram(clock clock.PassiveClock, opts TimingHistogramOpts) (TimingHistogram, error) {
	desc := prometheus.NewDesc(
		prometheus.BuildFQName(opts.Namespace, opts.Subsystem, opts.Name),
		opts.Help,
		nil,
		opts.ConstLabels,
	)
	return newTimingHistogram(clock, desc, opts)
}

func newTimingHistogram(clock clock.PassiveClock, desc *prometheus.Desc, opts TimingHistogramOpts, variableLabelValues ...string) (TimingHistogram, error) {
	if len(opts.Buckets) == 0 {
		opts.Buckets = prometheus.DefBuckets
	}

	for i, upperBound := range opts.Buckets {
		if i < len(opts.Buckets)-1 {
			if upperBound >= opts.Buckets[i+1] {
				return nil, fmt.Errorf(
					"histogram buckets must be in increasing order: %f >= %f",
					upperBound, opts.Buckets[i+1],
				)
			}
		} else {
			if math.IsInf(upperBound, +1) {
				// The +Inf bucket is implicit. Remove it here.
				opts.Buckets = opts.Buckets[:i]
			}
		}
	}
	upperBounds := make([]float64, len(opts.Buckets))
	copy(upperBounds, opts.Buckets)

	return &timingHistogram{
		desc:                desc,
		variableLabelValues: variableLabelValues,
		clock:               clock,
		upperBounds:         upperBounds,
		buckets:             make([]time.Duration, len(upperBounds)+1),
		lastSetTime:         clock.Now(),
		value:               opts.InitialValue,
		hotCount:            initialHotCount,
	}, nil
}

// initialHotCount is the negative of the number of terms
// that are summed into sumHot before it makes another term
// of sumCold.
const initialHotCount = -1000000

type timingHistogram struct {
	desc                *prometheus.Desc
	variableLabelValues []string
	clock               clock.PassiveClock
	upperBounds         []float64 // exclusive of +Inf

	lock sync.Mutex // applies to all the following

	// buckets is longer by one than upperBounds.
	// For 0 <= idx < len(upperBounds), buckets[idx] holds the
	// accumulated time.Duration that value has been <=
	// upperBounds[idx] but not <= upperBounds[idx-1].
	// buckets[len(upperBounds)] holds the accumulated
	// time.Duration when value fit in no other bucket.
	buckets []time.Duration

	// identifies when value was last set
	lastSetTime time.Time
	value       float64

	// sumHot + sumCold is the integral of value over time (in
	// nanoseconds).  Rather than risk loss of precision in one
	// float64, we do this sum hierarchically.  Many successive
	// increments are added into sumHot, and once in a great while
	// that is added into sumCold and reset to zero.
	sumHot  float64
	sumCold float64

	// hotCount is used to decide when to dump sumHot into sumCold.
	// hotCount counts upward from initialHotCount to zero.
	hotCount int
}

var _ TimingHistogram = &timingHistogram{}

func (sh *timingHistogram) Set(newValue float64) {
	sh.update(func(float64) float64 { return newValue })
}

func (sh *timingHistogram) Add(delta float64) {
	sh.update(func(oldValue float64) float64 { return oldValue + delta })
}

func (sh *timingHistogram) update(updateFn func(float64) float64) {
	sh.lock.Lock()
	defer sh.lock.Unlock()
	sh.updateLocked(updateFn)
}

func (sh *timingHistogram) updateLocked(updateFn func(float64) float64) {
	now := sh.clock.Now()
	delta := now.Sub(sh.lastSetTime)
	if delta > 0 {
		idx := sort.SearchFloat64s(sh.upperBounds, sh.value)
		sh.buckets[idx] += delta
		sh.lastSetTime = now
		sh.sumHot += float64(delta) * sh.value
		sh.hotCount++
		if sh.hotCount >= 0 {
			sh.sumCold += sh.sumHot
			sh.sumHot = 0
			sh.hotCount = initialHotCount
		}
	}
	sh.value = updateFn(sh.value)
}

func (sh *timingHistogram) Desc() *prometheus.Desc {
	return sh.desc
}

func (sh *timingHistogram) Write(dest *dto.Metric) error {
	sh.lock.Lock()
	defer sh.lock.Unlock()
	sh.updateLocked(func(x float64) float64 { return x })
	nBounds := len(sh.upperBounds)
	buckets := make(map[float64]uint64, nBounds)
	var cumCount uint64
	for idx, upperBound := range sh.upperBounds {
		cumCount += uint64(sh.buckets[idx])
		buckets[upperBound] = cumCount
	}
	cumCount += uint64(sh.buckets[nBounds])
	metric, err := prometheus.NewConstHistogram(sh.desc, cumCount, sh.sumHot+sh.sumCold, buckets, sh.variableLabelValues...)
	if err != nil {
		return err
	}
	return metric.Write(dest)
}

func (sh *timingHistogram) Describe(ch chan<- *prometheus.Desc) {
	ch <- sh.desc
}

func (sh *timingHistogram) Collect(ch chan<- prometheus.Metric) {
	sh.Add(0)
	ch <- sh
}
