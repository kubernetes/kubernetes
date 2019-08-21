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
	"time"

	"github.com/prometheus/client_golang/prometheus"
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

func init() {
	prometheus.MustRegister(apiserver_rejected_requests)
	prometheus.MustRegister(apiserver_current_inqueue_requests)
	prometheus.MustRegister(apiserver_request_queue_length)
	prometheus.MustRegister(apiserver_request_concurrency_limit)
	prometheus.MustRegister(apiserver_current_executing_requests)
	prometheus.MustRegister(apiserver_request_waiting_seconds)
	prometheus.MustRegister(apiserver_request_execution_seconds)
}

var (
	apiserver_rejected_requests = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "rejected_requests",
			Help:      "Number of rejected requests by api priority and fairness system",
		},
		[]string{priorityLevel, "reason"},
	)
	apiserver_current_inqueue_requests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "current_inqueue_requests",
			Help:      "Number of requests currently pending in the queue by the api priority and fairness system",
		},
		[]string{priorityLevel},
	)
	apiserver_request_queue_length = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "request_queue_length",
			Help:      "Length of queue in the api priority and fairness system",
			Buckets:   queueLengthBuckets,
		},
		[]string{priorityLevel},
	)
	apiserver_request_concurrency_limit = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "request_concurrency_limit",
			Help:      "Shared concurrency limit in the api priority and fairness system",
		},
		[]string{priorityLevel},
	)
	apiserver_current_executing_requests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "current_executing_requests",
			Help:      "Number of requests currently executing in the api priority and fairness system",
		},
		[]string{priorityLevel},
	)
	apiserver_request_waiting_seconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "request_wait_duration_seconds",
			Help:      "Length of time a request spent waiting in its queue",
			Buckets:   requestDurationSecondsBuckets,
		},
		[]string{priorityLevel, flowSchema, "execute"},
	)
	apiserver_request_execution_seconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "request_execution_seconds",
			Help:      "Time of request executing in the api priority and fairness system",
			Buckets:   requestDurationSecondsBuckets,
		},
		[]string{priorityLevel, flowSchema},
	)
)

func UpdateFlowControlRequestsInQueue(priorityLevel string, inqueue int) {
	apiserver_current_inqueue_requests.WithLabelValues(priorityLevel).Set(float64(inqueue))
}

func UpdateFlowControlRequestsExecuting(priorityLevel string, executing int) {
	apiserver_current_executing_requests.WithLabelValues(priorityLevel).Set(float64(executing))
}

func UpdateSharedConcurrencyLimit(priorityLevel string, limit int) {
	apiserver_request_concurrency_limit.WithLabelValues(priorityLevel).Set(float64(limit))
}

func AddReject(priorityLevel string, reason string) {
	apiserver_rejected_requests.WithLabelValues(priorityLevel, reason).Add(1)
}

func ObserveQueueLength(priorityLevel string, length int) {
	apiserver_request_queue_length.WithLabelValues(priorityLevel).Observe(float64(length))
}

func ObserveWaitingDuration(priorityLevel, flowSchema, execute string, waitTime time.Duration) {
	apiserver_request_waiting_seconds.WithLabelValues(priorityLevel, flowSchema, execute).Observe(waitTime.Seconds())
}

func ObserveExecutionDuration(priorityLevel, flowSchema string, executionTime time.Duration) {
	apiserver_request_execution_seconds.WithLabelValues(priorityLevel, flowSchema).Observe(executionTime.Seconds())
}
