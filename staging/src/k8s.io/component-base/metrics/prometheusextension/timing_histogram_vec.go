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

	"k8s.io/utils/clock"
)

type TimingHistogramVec struct {
	*prometheus.MetricVec
}

func NewTimingHistogramVec(opts TimingHistogramOpts, variableLabelNames []string) *TimingHistogramVec {
	return NewTestableTimingHistogramVec(clock.RealClock{}, opts, variableLabelNames)
}

func NewTestableTimingHistogramVec(clk clock.PassiveClock, opts TimingHistogramOpts, variableLabelNames []string) *TimingHistogramVec {
	desc := prometheus.NewDesc(
		prometheus.BuildFQName(opts.Namespace, opts.Subsystem, opts.Name),
		opts.Help,
		variableLabelNames,
		opts.ConstLabels,
	)
	return &TimingHistogramVec{
		MetricVec: prometheus.NewMetricVec(desc, func(variableLabelValues ...string) prometheus.Metric {
			hist, err := newTimingHistogram(clk, desc, opts, variableLabelValues...)
			if err != nil {
				panic(err)
			}
			return hist
		}),
	}
}

func (v *TimingHistogramVec) GetMetricWithLabelValues(lvs ...string) (WritableVariable, error) {
	metric, err := v.MetricVec.GetMetricWithLabelValues(lvs...)
	if metric != nil {
		return metric.(WritableVariable), err
	}
	return nil, err
}

func (v *TimingHistogramVec) GetMetricWith(labels prometheus.Labels) (WritableVariable, error) {
	metric, err := v.MetricVec.GetMetricWith(labels)
	if metric != nil {
		return metric.(WritableVariable), err
	}
	return nil, err
}

func (v *TimingHistogramVec) WithLabelValues(lvs ...string) WritableVariable {
	h, err := v.GetMetricWithLabelValues(lvs...)
	if err != nil {
		panic(err)
	}
	return h
}

func (v *TimingHistogramVec) With(labels prometheus.Labels) WritableVariable {
	h, err := v.GetMetricWith(labels)
	if err != nil {
		panic(err)
	}
	return h
}
