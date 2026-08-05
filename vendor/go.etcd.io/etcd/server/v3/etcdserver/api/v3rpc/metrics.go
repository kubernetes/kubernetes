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

import (
	"github.com/prometheus/client_golang/prometheus"
)

var (
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

	streamFailures = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "network",
			Name:      "server_stream_failures_total",
			Help:      "The total number of stream failures from the local server.",
		},
		[]string{"Type", "API"},
	)

	clientRequests = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "etcd",
			Subsystem: "server",
			Name:      "client_requests_total",
			Help:      "The total number of client requests per client version.",
		},
		[]string{"type", "client_api_version"},
	)

	watchSendLoopWatchStreamDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Namespace: "etcd_debugging",
			Subsystem: "server",
			Name:      "watch_send_loop_watch_stream_duration_seconds",
			Help:      "The total duration in seconds of running through the send loop watch stream response all events.",
			// lowest bucket start of upper bound 0.001 sec (1 ms) with factor 2
			// highest bucket start of 0.001 sec * 2^13 == 8.192 sec
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 14),
		},
	)

	watchSendLoopWatchStreamDurationPerEvent = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Namespace: "etcd_debugging",
			Subsystem: "server",
			Name:      "watch_send_loop_watch_stream_duration_per_event_seconds",
			Help:      "The average duration in seconds of running through the send loop watch stream response, per event.",
			// lowest bucket start of upper bound 0.001 sec (1 ms) with factor 2
			// highest bucket start of 0.001 sec * 2^13 == 8.192 sec
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 14),
		},
	)

	watchSendLoopControlStreamDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Namespace: "etcd_debugging",
			Subsystem: "server",
			Name:      "watch_send_loop_control_stream_duration_seconds",
			Help:      "The total duration in seconds of running through the send loop control stream response.",
			// lowest bucket start of upper bound 0.001 sec (1 ms) with factor 2
			// highest bucket start of 0.001 sec * 2^13 == 8.192 sec
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 14),
		},
	)

	watchSendLoopProgressDuration = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Namespace: "etcd_debugging",
			Subsystem: "server",
			Name:      "watch_send_loop_progress_duration_seconds",
			Help:      "The total duration in seconds of running through the progress loop control stream response.",
			// lowest bucket start of upper bound 0.001 sec (1 ms) with factor 2
			// highest bucket start of 0.001 sec * 2^13 == 8.192 sec
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 14),
		},
	)
)

func init() {
	prometheus.MustRegister(sentBytes)
	prometheus.MustRegister(receivedBytes)
	prometheus.MustRegister(streamFailures)
	prometheus.MustRegister(clientRequests)
	prometheus.MustRegister(watchSendLoopWatchStreamDuration)
	prometheus.MustRegister(watchSendLoopWatchStreamDurationPerEvent)
	prometheus.MustRegister(watchSendLoopControlStreamDuration)
	prometheus.MustRegister(watchSendLoopProgressDuration)
}
