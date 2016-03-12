// Copyright 2015 CoreOS, Inc.
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

import (
	"time"

	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft/raftpb"
	"github.com/prometheus/client_golang/prometheus"
)

var (
	// TODO: create a separate histogram for recording
	// snapshot sending metric. snapshot can be large and
	// take a long time to send. So it needs a different
	// time range than other type of messages.
	msgSentDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "etcd",
			Subsystem: "rafthttp",
			Name:      "message_sent_latency_seconds",
			Help:      "message sent latency distributions.",
			Buckets:   prometheus.ExponentialBuckets(0.0005, 2, 13),
		},
		[]string{"sendingType", "remoteID", "msgType"},
	)

	msgSentFailed = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "rafthttp",
		Name:      "message_sent_failed_total",
		Help:      "The total number of failed messages sent.",
	},
		[]string{"sendingType", "remoteID", "msgType"},
	)
)

func init() {
	prometheus.MustRegister(msgSentDuration)
	prometheus.MustRegister(msgSentFailed)
}

func reportSentDuration(sendingType string, m raftpb.Message, duration time.Duration) {
	typ := m.Type.String()
	if isLinkHeartbeatMessage(m) {
		typ = "MsgLinkHeartbeat"
	}
	msgSentDuration.WithLabelValues(sendingType, types.ID(m.To).String(), typ).Observe(float64(duration) / float64(time.Second))
}

func reportSentFailure(sendingType string, m raftpb.Message) {
	typ := m.Type.String()
	if isLinkHeartbeatMessage(m) {
		typ = "MsgLinkHeartbeat"
	}
	msgSentFailed.WithLabelValues(sendingType, types.ID(m.To).String(), typ).Inc()
}
