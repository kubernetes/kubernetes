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
	"github.com/prometheus/client_golang/prometheus"
)

// WeightedObserverVec is a bunch of WeightedObservers that have the same
// Desc and are distinguished by the values for their variable labels.
type WeightedObserverVec interface {
	GetMetricWith(prometheus.Labels) (WeightedObserver, error)
	GetMetricWithLabelValues(lvs ...string) (WeightedObserver, error)
	With(prometheus.Labels) WeightedObserver
	WithLabelValues(...string) WeightedObserver
	CurryWith(prometheus.Labels) (WeightedObserverVec, error)
	MustCurryWith(prometheus.Labels) WeightedObserverVec
}

// WeightedHistogramVec implements WeightedObserverVec
type WeightedHistogramVec struct {
	*prometheus.MetricVec
}

var _ WeightedObserverVec = &WeightedHistogramVec{}
var _ prometheus.Collector = &WeightedHistogramVec{}

func NewWeightedHistogramVec(opts WeightedHistogramOpts, labelNames ...string) *WeightedHistogramVec {
	desc := prometheus.NewDesc(
		prometheus.BuildFQName(opts.Namespace, opts.Subsystem, opts.Name),
		wrapWeightedHelp(opts.Help),
		labelNames,
		opts.ConstLabels,
	)
	return &WeightedHistogramVec{
		MetricVec: prometheus.NewMetricVec(desc, func(lvs ...string) prometheus.Metric {
			metric, err := newWeightedHistogram(desc, opts, lvs...)
			if err != nil {
				panic(err) // like in prometheus.newHistogram
			}
			return metric
		}),
	}
}

func (hv *WeightedHistogramVec) GetMetricWith(labels prometheus.Labels) (WeightedObserver, error) {
	metric, err := hv.MetricVec.GetMetricWith(labels)
	if metric != nil {
		return metric.(WeightedObserver), err
	}
	return nil, err
}

func (hv *WeightedHistogramVec) GetMetricWithLabelValues(lvs ...string) (WeightedObserver, error) {
	metric, err := hv.MetricVec.GetMetricWithLabelValues(lvs...)
	if metric != nil {
		return metric.(WeightedObserver), err
	}
	return nil, err
}

func (hv *WeightedHistogramVec) With(labels prometheus.Labels) WeightedObserver {
	h, err := hv.GetMetricWith(labels)
	if err != nil {
		panic(err)
	}
	return h
}

func (hv *WeightedHistogramVec) WithLabelValues(lvs ...string) WeightedObserver {
	h, err := hv.GetMetricWithLabelValues(lvs...)
	if err != nil {
		panic(err)
	}
	return h
}

func (hv *WeightedHistogramVec) CurryWith(labels prometheus.Labels) (WeightedObserverVec, error) {
	vec, err := hv.MetricVec.CurryWith(labels)
	if vec != nil {
		return &WeightedHistogramVec{MetricVec: vec}, err
	}
	return nil, err
}

func (hv *WeightedHistogramVec) MustCurryWith(labels prometheus.Labels) WeightedObserverVec {
	vec, err := hv.CurryWith(labels)
	if err != nil {
		panic(err)
	}
	return vec
}
