/*
Copyright 2017 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "apiserver"
	subsystem = "egress_dialer"

	// ProtocolHTTPConnect means that the proxy protocol is http-connect.
	ProtocolHTTPConnect = "http_connect"
	// ProtocolGRPC means that the proxy protocol is the GRPC protocol.
	ProtocolGRPC = "grpc"
	// TransportTCP means that the transport is TCP.
	TransportTCP = "tcp"
	// TransportUDS means that the transport is UDS.
	TransportUDS = "uds"
	// StageConnect indicates that the dial failed at establishing connection to the proxy server.
	StageConnect = "connect"
	// StageProxy indicates that the dial failed at requesting the proxy server to proxy.
	StageProxy = "proxy"
)

var (
	// Use buckets ranging from 5 ms to 12.5 seconds.
	latencyBuckets = []float64{0.005, 0.025, 0.1, 0.5, 2.5, 12.5}

	// Metrics provides access to all dial metrics.
	Metrics = newDialMetrics()
)

// DialMetrics instruments dials to proxy server with prometheus metrics.
type DialMetrics struct {
	clock     clock.Clock
	latencies *metrics.HistogramVec
	failures  *metrics.CounterVec
}

// newDialMetrics create a new DialMetrics, configured with default metric names.
func newDialMetrics() *DialMetrics {
	latencies := metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "dial_duration_seconds",
			Help:           "Dial latency histogram in seconds, labeled by the protocol (http-connect or grpc), transport (tcp or uds)",
			Buckets:        latencyBuckets,
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"protocol", "transport"},
	)

	failures := metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "dial_failure_count",
			Help:           "Dial failure count, labeled by the protocol (http-connect or grpc), transport (tcp or uds), and stage (connect or proxy). The stage indicates at which stage the dial failed",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"protocol", "transport", "stage"},
	)

	legacyregistry.MustRegister(latencies)
	legacyregistry.MustRegister(failures)
	return &DialMetrics{latencies: latencies, failures: failures, clock: clock.RealClock{}}
}

// Clock returns the clock.
func (m *DialMetrics) Clock() clock.Clock {
	return m.clock
}

// SetClock sets the clock.
func (m *DialMetrics) SetClock(c clock.Clock) {
	m.clock = c
}

// Reset resets the metrics.
func (m *DialMetrics) Reset() {
	m.latencies.Reset()
	m.failures.Reset()
}

// ObserveDialLatency records the latency of a dial, labeled by protocol, transport.
func (m *DialMetrics) ObserveDialLatency(elapsed time.Duration, protocol, transport string) {
	m.latencies.WithLabelValues(protocol, transport).Observe(elapsed.Seconds())
}

// ObserveDialFailure records a failed dial, labeled by protocol, transport, and the stage the dial failed at.
func (m *DialMetrics) ObserveDialFailure(protocol, transport, stage string) {
	m.failures.WithLabelValues(protocol, transport, stage).Inc()
}
