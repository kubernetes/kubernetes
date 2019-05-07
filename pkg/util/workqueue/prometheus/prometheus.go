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

var (
	depth = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: WorkQueueSubsystem,
			Name:      DepthKey,
			Help:      "Current depth of workqueue",
		},
		[]string{"name"},
	)

	adds = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: WorkQueueSubsystem,
			Name:      AddsKey,
			Help:      "Total number of adds handled by workqueue",
		},
		[]string{"name"},
	)

	latency = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: WorkQueueSubsystem,
			Name:      QueueLatencyKey,
			Help:      "How long in seconds an item stays in workqueue before being requested.",
			Buckets:   prometheus.ExponentialBuckets(10e-9, 10, 10),
		},
		[]string{"name"},
	)

	workDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: WorkQueueSubsystem,
			Name:      WorkDurationKey,
			Help:      "How long in seconds processing an item from workqueue takes.",
			Buckets:   prometheus.ExponentialBuckets(10e-9, 10, 10),
		},
		[]string{"name"},
	)

	unfinished = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: WorkQueueSubsystem,
			Name:      UnfinishedWorkKey,
			Help: "How many seconds of work has done that " +
				"is in progress and hasn't been observed by work_duration. Large " +
				"values indicate stuck threads. One can deduce the number of stuck " +
				"threads by observing the rate at which this increases.",
		},
		[]string{"name"},
	)

	longestRunningProcessor = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: WorkQueueSubsystem,
			Name:      LongestRunningProcessorKey,
			Help: "How many seconds has the longest running " +
				"processor for workqueue been running.",
		},
		[]string{"name"},
	)

	retries = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: WorkQueueSubsystem,
			Name:      RetriesKey,
			Help:      "Total number of retries handled by workqueue",
		},
		[]string{"name"},
	)
)

func registerMetrics() {
	prometheus.MustRegister(
		depth,
		adds,
		latency,
		workDuration,
		unfinished,
		longestRunningProcessor,
		retries,
	)
}

func init() {
	registerMetrics()
	workqueue.SetProvider(prometheusMetricsProvider{})
}

type prometheusMetricsProvider struct{}

func (prometheusMetricsProvider) NewDepthMetric(name string) workqueue.GaugeMetric {
	return depth.WithLabelValues(name)
}

func (prometheusMetricsProvider) NewAddsMetric(name string) workqueue.CounterMetric {
	return adds.WithLabelValues(name)
}

func (prometheusMetricsProvider) NewLatencyMetric(name string) workqueue.HistogramMetric {
	return latency.WithLabelValues(name)
}

func (prometheusMetricsProvider) NewWorkDurationMetric(name string) workqueue.HistogramMetric {
	return workDuration.WithLabelValues(name)
}

func (prometheusMetricsProvider) NewUnfinishedWorkSecondsMetric(name string) workqueue.SettableGaugeMetric {
	return unfinished.WithLabelValues(name)
}

func (prometheusMetricsProvider) NewLongestRunningProcessorSecondsMetric(name string) workqueue.SettableGaugeMetric {
	return longestRunningProcessor.WithLabelValues(name)
}

func (prometheusMetricsProvider) NewRetriesMetric(name string) workqueue.CounterMetric {
	return retries.WithLabelValues(name)
}

// TODO(danielqsj): Remove the following metrics, they are deprecated

// mustRegister attempts to register the given collector with the given metric, and name
// and returns the registered collector. The caller must use the returned collector
// as it might point to a different instance of an already registered collector.
func mustRegister(metric, name string, c prometheus.Collector) prometheus.Collector {
	err := prometheus.Register(c)
	if err == nil {
		return c
	}

	if aerr, ok := err.(prometheus.AlreadyRegisteredError); ok {
		klog.V(4).Infof("reusing already registered metric %v name %v", metric, name)
		return aerr.ExistingCollector
	}

	// this should fail hard as this indicates a programmatic error, i.e.
	// an invalid or duplicate metric descriptor,
	// a previously registered descriptor with the same fqdn but different labels,
	// or inconsistent label names or help strings for the same fqdn.
	klog.Fatalf("failed to register metric %v name %v: %v", metric, name, err)
	return nil
}

func (prometheusMetricsProvider) NewDeprecatedDepthMetric(name string) workqueue.GaugeMetric {
	depth := prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem: name,
		Name:      "depth",
		Help:      "(Deprecated) Current depth of workqueue: " + name,
	})

	return mustRegister("depth", name, depth).(prometheus.Gauge)
}

func (prometheusMetricsProvider) NewDeprecatedAddsMetric(name string) workqueue.CounterMetric {
	adds := prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: name,
		Name:      "adds",
		Help:      "(Deprecated) Total number of adds handled by workqueue: " + name,
	})

	return mustRegister("adds", name, adds).(prometheus.Counter)
}

func (prometheusMetricsProvider) NewDeprecatedLatencyMetric(name string) workqueue.SummaryMetric {
	latency := prometheus.NewSummary(prometheus.SummaryOpts{
		Subsystem: name,
		Name:      "queue_latency",
		Help:      "(Deprecated) How long an item stays in workqueue" + name + " before being requested.",
	})

	return mustRegister("queue_latency", name, latency).(prometheus.Summary)
}

func (prometheusMetricsProvider) NewDeprecatedWorkDurationMetric(name string) workqueue.SummaryMetric {
	workDuration := prometheus.NewSummary(prometheus.SummaryOpts{
		Subsystem: name,
		Name:      "work_duration",
		Help:      "(Deprecated) How long processing an item from workqueue" + name + " takes.",
	})

	return mustRegister("work_duration", name, workDuration).(prometheus.Summary)
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

	return mustRegister("unfinished_work_seconds", name, unfinished).(prometheus.Gauge)
}

func (prometheusMetricsProvider) NewDeprecatedLongestRunningProcessorMicrosecondsMetric(name string) workqueue.SettableGaugeMetric {
	unfinished := prometheus.NewGauge(prometheus.GaugeOpts{
		Subsystem: name,
		Name:      "longest_running_processor_microseconds",
		Help: "(Deprecated) How many microseconds has the longest running " +
			"processor for " + name + " been running.",
	})

	return mustRegister("longest_running_processor_microseconds", name, unfinished).(prometheus.Gauge)
}

func (prometheusMetricsProvider) NewDeprecatedRetriesMetric(name string) workqueue.CounterMetric {
	retries := prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: name,
		Name:      "retries",
		Help:      "(Deprecated) Total number of retries handled by workqueue: " + name,
	})

	return mustRegister("retries", name, retries).(prometheus.Counter)
}
