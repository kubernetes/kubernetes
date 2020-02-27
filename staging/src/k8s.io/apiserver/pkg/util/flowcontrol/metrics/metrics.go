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
	"sync"
	"time"

	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
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

var (
	apiserverRejectedRequestsTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "rejected_requests_total",
			Help:      "Number of rejected requests by api priority and fairness system",
		},
		[]string{priorityLevel, "reason"},
	)
	apiserverCurrentInqueueRequests = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "current_inqueue_requests",
			Help:      "Number of requests currently pending in the queue by the api priority and fairness system",
		},
		[]string{priorityLevel},
	)
	apiserverRequestQueueLength = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "request_queue_length",
			Help:      "Length of queue in the api priority and fairness system",
			Buckets:   queueLengthBuckets,
		},
		[]string{priorityLevel},
	)
	apiserverRequestConcurrencyLimit = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "request_concurrency_limit",
			Help:      "Shared concurrency limit in the api priority and fairness system",
		},
		[]string{priorityLevel},
	)
	apiserverCurrentExecutingRequests = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "current_executing_requests",
			Help:      "Number of requests currently executing in the api priority and fairness system",
		},
		[]string{priorityLevel},
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
			Help:      "Time of request executing in the api priority and fairness system",
			Buckets:   requestDurationSecondsBuckets,
		},
		[]string{priorityLevel, flowSchema},
	)
	metrics = []compbasemetrics.Registerable{
		apiserverRejectedRequestsTotal,
		apiserverCurrentInqueueRequests,
		apiserverRequestQueueLength,
		apiserverRequestConcurrencyLimit,
		apiserverCurrentExecutingRequests,
		apiserverRequestWaitingSeconds,
		apiserverRequestExecutionSeconds,
	}
)

// UpdateFlowControlRequestsInQueue updates the value for the # of requests in the specified queues in flow control
func UpdateFlowControlRequestsInQueue(priorityLevel string, inqueue int) {
	apiserverCurrentInqueueRequests.WithLabelValues(priorityLevel).Set(float64(inqueue))
}

// UpdateFlowControlRequestsExecuting updates the value for the # of requests executing in flow control
func UpdateFlowControlRequestsExecuting(priorityLevel string, executing int) {
	apiserverCurrentExecutingRequests.WithLabelValues(priorityLevel).Set(float64(executing))
}

// UpdateSharedConcurrencyLimit updates the value for the concurrency limit in flow control
func UpdateSharedConcurrencyLimit(priorityLevel string, limit int) {
	apiserverRequestConcurrencyLimit.WithLabelValues(priorityLevel).Set(float64(limit))
}

// AddReject increments the # of rejected requests for flow control
func AddReject(priorityLevel string, reason string) {
	apiserverRejectedRequestsTotal.WithLabelValues(priorityLevel, reason).Add(1)
}

// ObserveQueueLength observes the queue length for flow control
func ObserveQueueLength(priorityLevel string, length int) {
	apiserverRequestQueueLength.WithLabelValues(priorityLevel).Observe(float64(length))
}

// ObserveWaitingDuration observes the queue length for flow control
func ObserveWaitingDuration(priorityLevel, flowSchema, execute string, waitTime time.Duration) {
	apiserverRequestWaitingSeconds.WithLabelValues(priorityLevel, flowSchema, execute).Observe(waitTime.Seconds())
}

// ObserveExecutionDuration observes the execution duration for flow control
func ObserveExecutionDuration(priorityLevel, flowSchema string, executionTime time.Duration) {
	apiserverRequestExecutionSeconds.WithLabelValues(priorityLevel, flowSchema).Observe(executionTime.Seconds())
}
