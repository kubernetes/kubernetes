// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package v3rpc

import "github.com/prometheus/client_golang/prometheus"

var (
	receivedCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "grpc",
			Name:      "requests_total",
			Help:      "Counter of received requests.",
		}, []string{"grpc_service", "grpc_method"})

	failedCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "grpc",
			Name:      "requests_failed_total",
			Help:      "Counter of failed requests.",
		}, []string{"grpc_service", "grpc_method", "grpc_code"})

	handlingDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "etcd",
			Subsystem: "grpc",
			Name:      "unary_requests_duration_seconds",
			Help:      "Bucketed histogram of processing time (s) of handled unary (non-stream) requests.",
			Buckets:   prometheus.ExponentialBuckets(0.0005, 2, 13),
		}, []string{"grpc_service", "grpc_method"})

	sentBytes = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "client_grpc_sent_bytes_total",
		Help:      "The total number of bytes sent to grpc clients.",
	})

	receivedBytes = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "client_grpc_received_bytes_total",
		Help:      "The total number of bytes received from grpc clients.",
	})
)

func init() {
	prometheus.MustRegister(receivedCounter)
	prometheus.MustRegister(failedCounter)
	prometheus.MustRegister(handlingDuration)

	prometheus.MustRegister(sentBytes)
	prometheus.MustRegister(receivedBytes)
}
