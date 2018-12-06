/*
Copyright 2016 The Kubernetes Authors.

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

package prometheus

import (
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"

	"github.com/prometheus/client_golang/prometheus"
)

// Package prometheus sets the workqueue DefaultMetricsFactory to produce
// prometheus metrics. To use this package, you just have to import it.

func init() {
	workqueue.SetProvider(prometheusMetricsProvider{})
}

type prometheusMetricsProvider struct{}

func (prometheusMetricsProvider) NewDepthMetric(name string) workqueue.GaugeMetric {
	depth := prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem: name,
		Name:      "depth",
		Help:      "Current depth of workqueue: " + name,
	})
	if err := prometheus.Register(depth); err != nil {
		klog.Errorf("failed to register depth metric %v: %v", name, err)
	}
	return depth
}

func (prometheusMetricsProvider) NewAddsMetric(name string) workqueue.CounterMetric {
	adds := prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: name,
		Name:      "adds",
		Help:      "Total number of adds handled by workqueue: " + name,
	})
	if err := prometheus.Register(adds); err != nil {
		klog.Errorf("failed to register adds metric %v: %v", name, err)
	}
	return adds
}

func (prometheusMetricsProvider) NewLatencyMetric(name string) workqueue.SummaryMetric {
	latency := prometheus.NewSummary(prometheus.SummaryOpts{
		Subsystem: name,
		Name:      "queue_latency",
		Help:      "How long an item stays in workqueue" + name + " before being requested.",
	})
	if err := prometheus.Register(latency); err != nil {
		klog.Errorf("failed to register latency metric %v: %v", name, err)
	}
	return latency
}

func (prometheusMetricsProvider) NewWorkDurationMetric(name string) workqueue.SummaryMetric {
	workDuration := prometheus.NewSummary(prometheus.SummaryOpts{
		Subsystem: name,
		Name:      "work_duration",
		Help:      "How long processing an item from workqueue" + name + " takes.",
	})
	if err := prometheus.Register(workDuration); err != nil {
		klog.Errorf("failed to register work_duration metric %v: %v", name, err)
	}
	return workDuration
}

func (prometheusMetricsProvider) NewUnfinishedWorkSecondsMetric(name string) workqueue.SettableGaugeMetric {
	unfinished := prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem: name,
		Name:      "unfinished_work_seconds",
		Help: "How many seconds of work " + name + " has done that " +
			"is in progress and hasn't been observed by work_duration. Large " +
			"values indicate stuck threads. One can deduce the number of stuck " +
			"threads by observing the rate at which this increases.",
	})
	if err := prometheus.Register(unfinished); err != nil {
		klog.Errorf("failed to register unfinished_work_seconds metric %v: %v", name, err)
	}
	return unfinished
}

func (prometheusMetricsProvider) NewLongestRunningProcessorMicrosecondsMetric(name string) workqueue.SettableGaugeMetric {
	unfinished := prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem: name,
		Name:      "longest_running_processor_microseconds",
		Help: "How many microseconds has the longest running " +
			"processor for " + name + " been running.",
	})
	if err := prometheus.Register(unfinished); err != nil {
		klog.Errorf("failed to register longest_running_processor_microseconds metric %v: %v", name, err)
	}
	return unfinished
}

func (prometheusMetricsProvider) NewRetriesMetric(name string) workqueue.CounterMetric {
	retries := prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: name,
		Name:      "retries",
		Help:      "Total number of retries handled by workqueue: " + name,
	})
	if err := prometheus.Register(retries); err != nil {
		klog.Errorf("failed to register retries metric %v: %v", name, err)
	}
	return retries
}
