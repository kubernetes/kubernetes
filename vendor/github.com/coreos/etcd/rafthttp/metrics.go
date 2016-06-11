// Copyright 2015 The etcd Authors
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

package rafthttp

import "github.com/prometheus/client_golang/prometheus"

// TODO: record write/recv failures.
var (
	sentBytes = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "sent_bytes_total",
		Help:      "The total number of bytes sent.",
	},
		[]string{"To"},
	)

	receivedBytes = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "received_bytes_total",
		Help:      "The total number of bytes received.",
	},
		[]string{"From"},
	)

	rtts = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "round_trip_time_seconds",
		Help:      "Round-Trip-Time histogram between members.",
		Buckets:   prometheus.ExponentialBuckets(0.0001, 2, 14),
	},
		[]string{"To"},
	)
)

func init() {
	prometheus.MustRegister(sentBytes)
	prometheus.MustRegister(receivedBytes)
	prometheus.MustRegister(rtts)
}
