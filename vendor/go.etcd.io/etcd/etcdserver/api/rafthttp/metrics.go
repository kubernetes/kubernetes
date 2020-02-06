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

var (
	activePeers = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "active_peers",
		Help:      "The current number of active peer connections.",
	},
		[]string{"Local", "Remote"},
	)

	disconnectedPeers = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "disconnected_peers_total",
		Help:      "The total number of disconnected peers.",
	},
		[]string{"Local", "Remote"},
	)

	sentBytes = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "peer_sent_bytes_total",
		Help:      "The total number of bytes sent to peers.",
	},
		[]string{"To"},
	)

	receivedBytes = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "peer_received_bytes_total",
		Help:      "The total number of bytes received from peers.",
	},
		[]string{"From"},
	)

	sentFailures = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "peer_sent_failures_total",
		Help:      "The total number of send failures from peers.",
	},
		[]string{"To"},
	)

	recvFailures = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "peer_received_failures_total",
		Help:      "The total number of receive failures from peers.",
	},
		[]string{"From"},
	)

	snapshotSend = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "snapshot_send_success",
		Help:      "Total number of successful snapshot sends",
	},
		[]string{"To"},
	)

	snapshotSendInflights = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "snapshot_send_inflights_total",
		Help:      "Total number of inflight snapshot sends",
	},
		[]string{"To"},
	)

	snapshotSendFailures = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "snapshot_send_failures",
		Help:      "Total number of snapshot send failures",
	},
		[]string{"To"},
	)

	snapshotSendSeconds = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "snapshot_send_total_duration_seconds",
		Help:      "Total latency distributions of v3 snapshot sends",

		// lowest bucket start of upper bound 0.1 sec (100 ms) with factor 2
		// highest bucket start of 0.1 sec * 2^9 == 51.2 sec
		Buckets: prometheus.ExponentialBuckets(0.1, 2, 10),
	},
		[]string{"To"},
	)

	snapshotReceive = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "snapshot_receive_success",
		Help:      "Total number of successful snapshot receives",
	},
		[]string{"From"},
	)

	snapshotReceiveInflights = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "snapshot_receive_inflights_total",
		Help:      "Total number of inflight snapshot receives",
	},
		[]string{"From"},
	)

	snapshotReceiveFailures = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "snapshot_receive_failures",
		Help:      "Total number of snapshot receive failures",
	},
		[]string{"From"},
	)

	snapshotReceiveSeconds = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "snapshot_receive_total_duration_seconds",
		Help:      "Total latency distributions of v3 snapshot receives",

		// lowest bucket start of upper bound 0.1 sec (100 ms) with factor 2
		// highest bucket start of 0.1 sec * 2^9 == 51.2 sec
		Buckets: prometheus.ExponentialBuckets(0.1, 2, 10),
	},
		[]string{"From"},
	)

	rttSec = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "etcd",
		Subsystem: "network",
		Name:      "peer_round_trip_time_seconds",
		Help:      "Round-Trip-Time histogram between peers",

		// lowest bucket start of upper bound 0.0001 sec (0.1 ms) with factor 2
		// highest bucket start of 0.0001 sec * 2^15 == 3.2768 sec
		Buckets: prometheus.ExponentialBuckets(0.0001, 2, 16),
	},
		[]string{"To"},
	)
)

func init() {
	prometheus.MustRegister(activePeers)
	prometheus.MustRegister(disconnectedPeers)
	prometheus.MustRegister(sentBytes)
	prometheus.MustRegister(receivedBytes)
	prometheus.MustRegister(sentFailures)
	prometheus.MustRegister(recvFailures)

	prometheus.MustRegister(snapshotSend)
	prometheus.MustRegister(snapshotSendInflights)
	prometheus.MustRegister(snapshotSendFailures)
	prometheus.MustRegister(snapshotSendSeconds)
	prometheus.MustRegister(snapshotReceive)
	prometheus.MustRegister(snapshotReceiveInflights)
	prometheus.MustRegister(snapshotReceiveFailures)
	prometheus.MustRegister(snapshotReceiveSeconds)

	prometheus.MustRegister(rttSec)
}
