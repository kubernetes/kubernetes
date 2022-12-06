// Package metrics provides metric definitions and helpers used by
// multiple binaries (across client, server, and agent).

package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"google.golang.org/grpc/status"

	"sigs.k8s.io/apiserver-network-proxy/konnectivity-client/proto/client"
)

// Segment identifies one of four tunnel segments (e.g. from server to backend).
type Segment string

const (
	// SegmentFromFrontend indicates a packet from client to server.
	SegmentFromFrontend Segment = "from_frontend"
	// SegmentToFrontend indicates a packet from server to client.
	SegmentToFrontend Segment = "to_frontend"
	// SegmentFromFrontend indicates a packet from agent to server.
	SegmentFromBackend Segment = "from_backend"
	// SegmentFromFrontend indicates a packet from server to agent.
	SegmentToBackend Segment = "to_backend"
)

func MakeStreamEventsTotalMetric(namespace, subsystem string) *prometheus.CounterVec {
	return prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "stream_events_total",
			Help:      "Count of packets processed, by segment and packet type (example: from_frontend, DIAL_REQ)",
		},
		[]string{"segment", "packet_type"},
	)
}

func MakeStreamEventsErrorsMetric(namespace, subsystem string) *prometheus.CounterVec {
	return prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "stream_events_error_total",
			Help:      "Count of stream_errors, by segment, grpc Code, packet type. (example: from_backend, Code.Unavailable, DIAL_RSP)",
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
