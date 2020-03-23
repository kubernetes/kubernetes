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
	"strings"
	"sync"
	"time"

	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	basemetricstestutil "k8s.io/component-base/metrics/testutil"
)

const (
	namespace = "apiserver"
	subsystem = "flowcontrol"
)

const (
	priorityLevel = "priorityLevel"
	flowSchema    = "flowSchema"
)

var (
	queueLengthBuckets            = []float64{0, 10, 25, 50, 100, 250, 500, 1000}
	requestDurationSecondsBuckets = []float64{0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30}
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	registerMetrics.Do(func() {
		for _, metric := range metrics {
			legacyregistry.MustRegister(metric)
		}
	})
}

type resettable interface {
	Reset()
}

// Reset all metrics to zero
func Reset() {
	for _, metric := range metrics {
		rm := metric.(resettable)
		rm.Reset()
	}
}

// GatherAndCompare the given metrics with the given Prometheus syntax expected value
func GatherAndCompare(expected string, metricNames ...string) error {
	return basemetricstestutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expected), metricNames...)
}

var (
	apiserverRejectedRequestsTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "rejected_requests_total",
			Help:      "Number of requests rejected by API Priority and Fairness system",
		},
		[]string{priorityLevel, flowSchema, "reason"},
	)
	apiserverDispatchedRequestsTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "dispatched_requests_total",
			Help:      "Number of requests released by API Priority and Fairness system for service",
		},
		[]string{priorityLevel, flowSchema},
	)
	apiserverCurrentInqueueRequests = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "current_inqueue_requests",
			Help:      "Number of requests currently pending in queues of the API Priority and Fairness system",
		},
		[]string{priorityLevel, flowSchema},
	)
	apiserverRequestQueueLength = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "request_queue_length_after_enqueue",
			Help:      "Length of queue in the API Priority and Fairness system, as seen by each request after it is enqueued",
			Buckets:   queueLengthBuckets,
		},
		[]string{priorityLevel, flowSchema},
	)
	apiserverRequestConcurrencyLimit = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "request_concurrency_limit",
			Help:      "Shared concurrency limit in the API Priority and Fairness system",
		},
		[]string{priorityLevel},
	)
	apiserverCurrentExecutingRequests = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "current_executing_requests",
			Help:      "Number of requests currently executing in the API Priority and Fairness system",
		},
		[]string{priorityLevel, flowSchema},
	)
	apiserverRequestWaitingSeconds = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "request_wait_duration_seconds",
			Help:      "Length of time a request spent waiting in its queue",
			Buckets:   requestDurationSecondsBuckets,
		},
		[]string{priorityLevel, flowSchema, "execute"},
	)
	apiserverRequestExecutionSeconds = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "request_execution_seconds",
			Help:      "Duration of request execution in the API Priority and Fairness system",
			Buckets:   requestDurationSecondsBuckets,
		},
		[]string{priorityLevel, flowSchema},
	)
	metrics = []compbasemetrics.Registerable{
		apiserverRejectedRequestsTotal,
		apiserverDispatchedRequestsTotal,
		apiserverCurrentInqueueRequests,
		apiserverRequestQueueLength,
		apiserverRequestConcurrencyLimit,
		apiserverCurrentExecutingRequests,
		apiserverRequestWaitingSeconds,
		apiserverRequestExecutionSeconds,
	}
)

// AddRequestsInQueues adds the given delta to the gauge of the # of requests in the queues of the specified flowSchema and priorityLevel
func AddRequestsInQueues(priorityLevel, flowSchema string, delta int) {
	apiserverCurrentInqueueRequests.WithLabelValues(priorityLevel, flowSchema).Add(float64(delta))
}

// AddRequestsExecuting adds the given delta to the gauge of executing requests of the given flowSchema and priorityLevel
func AddRequestsExecuting(priorityLevel, flowSchema string, delta int) {
	apiserverCurrentExecutingRequests.WithLabelValues(priorityLevel, flowSchema).Add(float64(delta))
}

// UpdateSharedConcurrencyLimit updates the value for the concurrency limit in flow control
func UpdateSharedConcurrencyLimit(priorityLevel string, limit int) {
	apiserverRequestConcurrencyLimit.WithLabelValues(priorityLevel).Set(float64(limit))
}

// AddReject increments the # of rejected requests for flow control
func AddReject(priorityLevel, flowSchema, reason string) {
	apiserverRejectedRequestsTotal.WithLabelValues(priorityLevel, flowSchema, reason).Add(1)
}

// AddDispatch increments the # of dispatched requests for flow control
func AddDispatch(priorityLevel, flowSchema string) {
	apiserverDispatchedRequestsTotal.WithLabelValues(priorityLevel, flowSchema).Add(1)
}

// ObserveQueueLength observes the queue length for flow control
func ObserveQueueLength(priorityLevel, flowSchema string, length int) {
	apiserverRequestQueueLength.WithLabelValues(priorityLevel, flowSchema).Observe(float64(length))
}

// ObserveWaitingDuration observes the queue length for flow control
func ObserveWaitingDuration(priorityLevel, flowSchema, execute string, waitTime time.Duration) {
	apiserverRequestWaitingSeconds.WithLabelValues(priorityLevel, flowSchema, execute).Observe(waitTime.Seconds())
}

// ObserveExecutionDuration observes the execution duration for flow control
func ObserveExecutionDuration(priorityLevel, flowSchema string, executionTime time.Duration) {
	apiserverRequestExecutionSeconds.WithLabelValues(priorityLevel, flowSchema).Observe(executionTime.Seconds())
}
