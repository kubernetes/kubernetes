/*
Copyright 2022 The Kubernetes Authors.

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

// Package metrics provides metric definitions and helpers used
// across konnectivity client, server, and agent.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"google.golang.org/grpc/status"

	"sigs.k8s.io/apiserver-network-proxy/konnectivity-client/proto/client"
)

// Segment identifies one of four tunnel segments (e.g. from server to agent).
type Segment string

const (
	// SegmentFromClient indicates a packet from client to server.
	SegmentFromClient Segment = "from_client"
	// SegmentToClient indicates a packet from server to client.
	SegmentToClient Segment = "to_client"
	// SegmentFromAgent indicates a packet from agent to server.
	SegmentFromAgent Segment = "from_agent"
	// SegmentToAgent indicates a packet from server to agent.
	SegmentToAgent Segment = "to_agent"
)

func MakeStreamPacketsTotalMetric(namespace, subsystem string) *prometheus.CounterVec {
	return prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "stream_packets_total",
			Help:      "Count of packets processed, by segment and packet type (example: from_client, DIAL_REQ)",
		},
		[]string{"segment", "packet_type"},
	)
}

func MakeStreamErrorsTotalMetric(namespace, subsystem string) *prometheus.CounterVec {
	return prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "stream_errors_total",
			Help:      "Count of gRPC stream errors, by segment, grpc Code, packet type. (example: from_agent, Code.Unavailable, DIAL_RSP)",
		},
		[]string{"segment", "code", "packet_type"},
	)
}

func ObservePacket(m *prometheus.CounterVec, segment Segment, packetType client.PacketType) {
	m.WithLabelValues(string(segment), packetType.String()).Inc()
}

func ObserveStreamErrorNoPacket(m *prometheus.CounterVec, segment Segment, err error) {
	code := status.Code(err)
	m.WithLabelValues(string(segment), code.String(), "Unknown").Inc()
}

func ObserveStreamError(m *prometheus.CounterVec, segment Segment, err error, packetType client.PacketType) {
	code := status.Code(err)
	m.WithLabelValues(string(segment), code.String(), packetType.String()).Inc()
}
