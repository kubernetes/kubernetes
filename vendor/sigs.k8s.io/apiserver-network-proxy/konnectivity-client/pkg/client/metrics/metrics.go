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

package metrics

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"

	sharedmetrics "sigs.k8s.io/apiserver-network-proxy/konnectivity-client/pkg/shared/metrics"
	"sigs.k8s.io/apiserver-network-proxy/konnectivity-client/proto/client"
)

const (
	Namespace = "apiserver"
	Subsystem = "konnectivity_client"
)

var (
	// Metrics provides access to all client metrics. The client
	// application is responsible for registering (via Metrics.RegisterMetrics).
	Metrics = newMetrics()
)

// ClientMetrics includes all the metrics of the konnectivity-client.
type ClientMetrics struct {
	tConnections       *prometheus.GaugeVec
	streamEvents       *prometheus.CounterVec
	streamEventsErrors *prometheus.CounterVec
	registerOnce       sync.Once
}

func newMetrics() *ClientMetrics {
	tConnections := prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: Namespace,
			Subsystem: Subsystem,
			Name:      "tunnel_connections",
			Help:      "Current number of open tunnel connections.",
		},
		[]string{},
	)
	streamEvents := sharedmetrics.MakeStreamEventsTotalMetric(Namespace, Subsystem)
	streamEventsErrors := sharedmetrics.MakeStreamEventsErrorsMetric(Namespace, Subsystem)
	return &ClientMetrics{
		tConnections:       tConnections,
		streamEvents:       streamEvents,
		streamEventsErrors: streamEventsErrors,
	}
}

// RegisterMetrics registers all metrics with the client application.
func (c *ClientMetrics) RegisterMetrics(r prometheus.Registerer) {
	c.registerOnce.Do(func() {
		r.MustRegister(c.tConnections)
		r.MustRegister(c.streamEvents)
		r.MustRegister(c.streamEventsErrors)
	})
}

// LegacyRegisterMetrics registers all metrics via MustRegister func.
// TODO: remove this once https://github.com/kubernetes/kubernetes/pull/114293 is available.
func (c *ClientMetrics) LegacyRegisterMetrics(mustRegisterFn func(...prometheus.Collector)) {
	c.registerOnce.Do(func() {
		mustRegisterFn(c.tConnections)
		mustRegisterFn(c.streamEvents)
		mustRegisterFn(c.streamEventsErrors)
	})
}

// Reset resets the metrics.
func (c *ClientMetrics) Reset() {
	c.tConnections.Reset()
	c.streamEvents.Reset()
	c.streamEventsErrors.Reset()
}

func (c *ClientMetrics) TunnelConnectionsInc() {
	c.tConnections.WithLabelValues().Inc()
}

func (c *ClientMetrics) TunnelConnectionsDec() {
	c.tConnections.WithLabelValues().Dec()
}

func (c *ClientMetrics) ObservePacket(segment sharedmetrics.Segment, packetType client.PacketType) {
	sharedmetrics.ObservePacket(c.streamEvents, segment, packetType)
}

func (c *ClientMetrics) ObserveStreamErrorNoPacket(segment sharedmetrics.Segment, err error) {
	sharedmetrics.ObserveStreamErrorNoPacket(c.streamEventsErrors, segment, err)
}

func (c *ClientMetrics) ObserveStreamError(segment sharedmetrics.Segment, err error, packetType client.PacketType) {
	sharedmetrics.ObserveStreamError(c.streamEventsErrors, segment, err, packetType)
}
