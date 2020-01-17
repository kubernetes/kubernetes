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
	prometheus.MustRegister(apiserverRejectedRequests)
	prometheus.MustRegister(apiserverCurrentInqueueRequests)
	prometheus.MustRegister(apiserverRequestQueueLength)
	prometheus.MustRegister(apiserverRequestConcurrencyLimit)
	prometheus.MustRegister(apiserverCurrentExecutingRequests)
	prometheus.MustRegister(apiserverRequestWaitingSeconds)
	prometheus.MustRegister(apiserverRequestExecutionSeconds)
}

var (
	apiserverRejectedRequests = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "rejectedRequests",
			Help:      "Number of rejected requests by api priority and fairness system",
		},
		[]string{priorityLevel, "reason"},
	)
	apiserverCurrentInqueueRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "currentInqueueRequests",
			Help:      "Number of requests currently pending in the queue by the api priority and fairness system",
		},
		[]string{priorityLevel},
	)
	apiserverRequestQueueLength = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "requestQueueLength",
			Help:      "Length of queue in the api priority and fairness system",
			Buckets:   queueLengthBuckets,
		},
		[]string{priorityLevel},
	)
	apiserverRequestConcurrencyLimit = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "requestConcurrencyLimit",
			Help:      "Shared concurrency limit in the api priority and fairness system",
		},
		[]string{priorityLevel},
	)
	apiserverCurrentExecutingRequests = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "currentExecutingRequests",
			Help:      "Number of requests currently executing in the api priority and fairness system",
		},
		[]string{priorityLevel},
	)
	apiserverRequestWaitingSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "request_wait_durationSeconds",
			Help:      "Length of time a request spent waiting in its queue",
			Buckets:   requestDurationSecondsBuckets,
		},
		[]string{priorityLevel, flowSchema, "execute"},
	)
	apiserverRequestExecutionSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "requestExecutionSeconds",
			Help:      "Time of request executing in the api priority and fairness system",
			Buckets:   requestDurationSecondsBuckets,
		},
		[]string{priorityLevel, flowSchema},
	)
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
	apiserverRejectedRequests.WithLabelValues(priorityLevel, reason).Add(1)
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
