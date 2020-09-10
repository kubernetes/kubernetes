/*
Copyright 2020 Mike Spreitzer.

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

package prometheus_extension

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"k8s.io/utils/clock"
)

// A SamplingHistogram samples the values of a float64 variable at a
// configured rate.  The samples contribute to a Histogram.
type SamplingHistogram interface {
	prometheus.Metric
	prometheus.Collector

	// Set the variable to the given value.
	Set(float64)

	// Add the given change to the variable
	Add(float64)
}

type SamplingHistogramOpts struct {
	prometheus.HistogramOpts

	// The initial value of the variable
	InitialValue float64

	// The variable is sampled once every this often
	SamplingPeriod time.Duration
}

func NewSamplingHistogram(opts SamplingHistogramOpts) SamplingHistogram {
	return NewTestableSamplingHistogram(clock.RealClock{}, opts)
}

func NewTestableSamplingHistogram(clock clock.Clock, opts SamplingHistogramOpts) SamplingHistogram {
	desc := prometheus.NewDesc(
		prometheus.BuildFQName(opts.Namespace, opts.Subsystem, opts.Name),
		opts.Help,
		nil,
		opts.ConstLabels,
	)
	return newSamplingHistogram(clock, desc, opts)
}

func newSamplingHistogram(clock clock.Clock, desc *prometheus.Desc, opts SamplingHistogramOpts, labelValues ...string) SamplingHistogram {
	return &samplingHistogram{
		samplingPeriod:  opts.SamplingPeriod,
		histogram:       prometheus.NewHistogram(opts.HistogramOpts),
		clock:           clock,
		lastSampleIndex: clock.Now().UnixNano() / int64(opts.SamplingPeriod),
		value:           opts.InitialValue,
	}
}

type samplingHistogram struct {
	samplingPeriod time.Duration
	histogram      prometheus.Histogram
	clock          clock.Clock
	lock           sync.Mutex

	// identifies the last sampling period completed
	lastSampleIndex int64
	value           float64
}

var _ SamplingHistogram = &samplingHistogram{}

func (sh *samplingHistogram) Set(newValue float64) {
	sh.Update(func(float64) float64 { return newValue })
}

func (sh *samplingHistogram) Add(delta float64) {
	sh.Update(func(oldValue float64) float64 { return oldValue + delta })
}

func (sh *samplingHistogram) Update(updateFn func(float64) float64) {
	oldValue, numSamples := func() (float64, int64) {
		sh.lock.Lock()
		defer sh.lock.Unlock()
		newSampleIndex := sh.clock.Now().UnixNano() / int64(sh.samplingPeriod)
		deltaIndex := newSampleIndex - sh.lastSampleIndex
		sh.lastSampleIndex = newSampleIndex
		oldValue := sh.value
		sh.value = updateFn(sh.value)
		return oldValue, deltaIndex
	}()
	for i := int64(0); i < numSamples; i++ {
		sh.histogram.Observe(oldValue)
	}
}

func (sh *samplingHistogram) Desc() *prometheus.Desc {
	return sh.histogram.Desc()
}

func (sh *samplingHistogram) Write(dest *dto.Metric) error {
	return sh.histogram.Write(dest)
}

func (sh *samplingHistogram) Describe(ch chan<- *prometheus.Desc) {
	sh.histogram.Describe(ch)
}

func (sh *samplingHistogram) Collect(ch chan<- prometheus.Metric) {
	sh.Add(0)
	sh.histogram.Collect(ch)
}
