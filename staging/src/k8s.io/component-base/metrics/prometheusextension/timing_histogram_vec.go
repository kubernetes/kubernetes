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
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// GaugeVecOps is a bunch of Gauge that have the same
// Desc and are distinguished by the values for their variable labels.
type GaugeVecOps interface {
	GetMetricWith(prometheus.Labels) (GaugeOps, error)
	GetMetricWithLabelValues(lvs ...string) (GaugeOps, error)
	With(prometheus.Labels) GaugeOps
	WithLabelValues(...string) GaugeOps
	CurryWith(prometheus.Labels) (GaugeVecOps, error)
	MustCurryWith(prometheus.Labels) GaugeVecOps
}

type TimingHistogramVec struct {
	*prometheus.MetricVec
}

var _ GaugeVecOps = &TimingHistogramVec{}
var _ prometheus.Collector = &TimingHistogramVec{}

func NewTimingHistogramVec(opts TimingHistogramOpts, labelNames ...string) *TimingHistogramVec {
	return NewTestableTimingHistogramVec(time.Now, opts, labelNames...)
}

func NewTestableTimingHistogramVec(nowFunc func() time.Time, opts TimingHistogramOpts, labelNames ...string) *TimingHistogramVec {
	desc := prometheus.NewDesc(
		prometheus.BuildFQName(opts.Namespace, opts.Subsystem, opts.Name),
		wrapTimingHelp(opts.Help),
		labelNames,
		opts.ConstLabels,
	)
	return &TimingHistogramVec{
		MetricVec: prometheus.NewMetricVec(desc, func(lvs ...string) prometheus.Metric {
			metric, err := newTimingHistogram(nowFunc, desc, opts, lvs...)
			if err != nil {
				panic(err) // like in prometheus.newHistogram
			}
			return metric
		}),
	}
}

func (hv *TimingHistogramVec) GetMetricWith(labels prometheus.Labels) (GaugeOps, error) {
	metric, err := hv.MetricVec.GetMetricWith(labels)
	if metric != nil {
		return metric.(GaugeOps), err
	}
	return nil, err
}

func (hv *TimingHistogramVec) GetMetricWithLabelValues(lvs ...string) (GaugeOps, error) {
	metric, err := hv.MetricVec.GetMetricWithLabelValues(lvs...)
	if metric != nil {
		return metric.(GaugeOps), err
	}
	return nil, err
}

func (hv *TimingHistogramVec) With(labels prometheus.Labels) GaugeOps {
	h, err := hv.GetMetricWith(labels)
	if err != nil {
		panic(err)
	}
	return h
}

func (hv *TimingHistogramVec) WithLabelValues(lvs ...string) GaugeOps {
	h, err := hv.GetMetricWithLabelValues(lvs...)
	if err != nil {
		panic(err)
	}
	return h
}

func (hv *TimingHistogramVec) CurryWith(labels prometheus.Labels) (GaugeVecOps, error) {
	vec, err := hv.MetricVec.CurryWith(labels)
	if vec != nil {
		return &TimingHistogramVec{MetricVec: vec}, err
	}
	return nil, err
}

func (hv *TimingHistogramVec) MustCurryWith(labels prometheus.Labels) GaugeVecOps {
	vec, err := hv.CurryWith(labels)
	if err != nil {
		panic(err)
	}
	return vec
}
