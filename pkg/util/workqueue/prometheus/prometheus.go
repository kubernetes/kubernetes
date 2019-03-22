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

// Metrics subsystem and keys used by the workqueue.
const (
	WorkQueueSubsystem         = "workqueue"
	DepthKey                   = "depth"
	AddsKey                    = "adds_total"
	QueueLatencyKey            = "queue_duration_seconds"
	WorkDurationKey            = "work_duration_seconds"
	UnfinishedWorkKey          = "unfinished_work_seconds"
	LongestRunningProcessorKey = "longest_running_processor_seconds"
	RetriesKey                 = "retries_total"
)

func init() {
	workqueue.SetProvider(prometheusMetricsProvider{})
}

type prometheusMetricsProvider struct{}

func (prometheusMetricsProvider) NewDepthMetric(name string) workqueue.GaugeMetric {
	depth := prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem:   WorkQueueSubsystem,
		Name:        DepthKey,
		Help:        "Current depth of workqueue",
		ConstLabels: prometheus.Labels{"name": name},
	})
	prometheus.Register(depth)
	return depth
}

func (prometheusMetricsProvider) NewAddsMetric(name string) workqueue.CounterMetric {
	adds := prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem:   WorkQueueSubsystem,
		Name:        AddsKey,
		Help:        "Total number of adds handled by workqueue",
		ConstLabels: prometheus.Labels{"name": name},
	})
	prometheus.Register(adds)
	return adds
}

func (prometheusMetricsProvider) NewLatencyMetric(name string) workqueue.HistogramMetric {
	latency := prometheus.NewHistogram(prometheus.HistogramOpts{
		Subsystem:   WorkQueueSubsystem,
		Name:        QueueLatencyKey,
		Help:        "How long in seconds an item stays in workqueue before being requested.",
		ConstLabels: prometheus.Labels{"name": name},
		Buckets:     prometheus.ExponentialBuckets(10e-9, 10, 10),
	})
	prometheus.Register(latency)
	return latency
}

func (prometheusMetricsProvider) NewWorkDurationMetric(name string) workqueue.HistogramMetric {
	workDuration := prometheus.NewHistogram(prometheus.HistogramOpts{
		Subsystem:   WorkQueueSubsystem,
		Name:        WorkDurationKey,
		Help:        "How long in seconds processing an item from workqueue takes.",
		ConstLabels: prometheus.Labels{"name": name},
		Buckets:     prometheus.ExponentialBuckets(10e-9, 10, 10),
	})
	prometheus.Register(workDuration)
	return workDuration
}

func (prometheusMetricsProvider) NewUnfinishedWorkSecondsMetric(name string) workqueue.SettableGaugeMetric {
	unfinished := prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem: WorkQueueSubsystem,
		Name:      UnfinishedWorkKey,
		Help: "How many seconds of work has done that " +
			"is in progress and hasn't been observed by work_duration. Large " +
			"values indicate stuck threads. One can deduce the number of stuck " +
			"threads by observing the rate at which this increases.",
		ConstLabels: prometheus.Labels{"name": name},
	})
	prometheus.Register(unfinished)
	return unfinished
}

func (prometheusMetricsProvider) NewLongestRunningProcessorSecondsMetric(name string) workqueue.SettableGaugeMetric {
	unfinished := prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem: WorkQueueSubsystem,
		Name:      LongestRunningProcessorKey,
		Help: "How many seconds has the longest running " +
			"processor for workqueue been running.",
		ConstLabels: prometheus.Labels{"name": name},
	})
	prometheus.Register(unfinished)
	return unfinished
}

func (prometheusMetricsProvider) NewRetriesMetric(name string) workqueue.CounterMetric {
	retries := prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem:   WorkQueueSubsystem,
		Name:        RetriesKey,
		Help:        "Total number of retries handled by workqueue",
		ConstLabels: prometheus.Labels{"name": name},
	})
	prometheus.Register(retries)
	return retries
}

// TODO(danielqsj): Remove the following metrics, they are deprecated
func (prometheusMetricsProvider) NewDeprecatedDepthMetric(name string) workqueue.GaugeMetric {
	depth := prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem: name,
		Name:      "depth",
		Help:      "(Deprecated) Current depth of workqueue: " + name,
	})
	if err := prometheus.Register(depth); err != nil {
		klog.Errorf("failed to register depth metric %v: %v", name, err)
	}
	return depth
}

func (prometheusMetricsProvider) NewDeprecatedAddsMetric(name string) workqueue.CounterMetric {
	adds := prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: name,
		Name:      "adds",
		Help:      "(Deprecated) Total number of adds handled by workqueue: " + name,
	})
	if err := prometheus.Register(adds); err != nil {
		klog.Errorf("failed to register adds metric %v: %v", name, err)
	}
	return adds
}

func (prometheusMetricsProvider) NewDeprecatedLatencyMetric(name string) workqueue.SummaryMetric {
	latency := prometheus.NewSummary(prometheus.SummaryOpts{
		Subsystem: name,
		Name:      "queue_latency",
		Help:      "(Deprecated) How long an item stays in workqueue" + name + " before being requested.",
	})
	if err := prometheus.Register(latency); err != nil {
		klog.Errorf("failed to register latency metric %v: %v", name, err)
	}
	return latency
}

func (prometheusMetricsProvider) NewDeprecatedWorkDurationMetric(name string) workqueue.SummaryMetric {
	workDuration := prometheus.NewSummary(prometheus.SummaryOpts{
		Subsystem: name,
		Name:      "work_duration",
		Help:      "(Deprecated) How long processing an item from workqueue" + name + " takes.",
	})
	if err := prometheus.Register(workDuration); err != nil {
		klog.Errorf("failed to register work_duration metric %v: %v", name, err)
	}
	return workDuration
}

func (prometheusMetricsProvider) NewDeprecatedUnfinishedWorkSecondsMetric(name string) workqueue.SettableGaugeMetric {
	unfinished := prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem: name,
		Name:      "unfinished_work_seconds",
		Help: "(Deprecated) How many seconds of work " + name + " has done that " +
			"is in progress and hasn't been observed by work_duration. Large " +
			"values indicate stuck threads. One can deduce the number of stuck " +
			"threads by observing the rate at which this increases.",
	})
	if err := prometheus.Register(unfinished); err != nil {
		klog.Errorf("failed to register unfinished_work_seconds metric %v: %v", name, err)
	}
	return unfinished
}

func (prometheusMetricsProvider) NewDeprecatedLongestRunningProcessorMicrosecondsMetric(name string) workqueue.SettableGaugeMetric {
	unfinished := prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem: name,
		Name:      "longest_running_processor_microseconds",
		Help: "(Deprecated) How many microseconds has the longest running " +
			"processor for " + name + " been running.",
	})
	if err := prometheus.Register(unfinished); err != nil {
		klog.Errorf("failed to register longest_running_processor_microseconds metric %v: %v", name, err)
	}
	return unfinished
}

func (prometheusMetricsProvider) NewDeprecatedRetriesMetric(name string) workqueue.CounterMetric {
	retries := prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: name,
		Name:      "retries",
		Help:      "(Deprecated) Total number of retries handled by workqueue: " + name,
	})
	if err := prometheus.Register(retries); err != nil {
		klog.Errorf("failed to register retries metric %v: %v", name, err)
	}
	return retries
}
