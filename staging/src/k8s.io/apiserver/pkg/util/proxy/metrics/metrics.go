/*
Copyright 2024 The Kubernetes Authors.

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
	"context"
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	subsystem  = "apiserver"
	statuscode = "code"
)

var registerMetricsOnce sync.Once

var (
	// streamTranslatorRequestsTotal counts the number of requests that were handled by
	// the StreamTranslatorProxy (RemoteCommand subprotocol).
	streamTranslatorRequestsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "stream_translator_requests_total",
			Help:           "Total number of requests that were handled by the StreamTranslatorProxy, which processes streaming RemoteCommand/V5",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{statuscode},
	)
	// streamTunnelRequestsTotal counts the number of requests that were handled by
	// the StreamTunnelProxy (PortForward subprotocol).
	streamTunnelRequestsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "stream_tunnel_requests_total",
			Help:           "Total number of requests that were handled by the StreamTunnelProxy, which processes streaming PortForward/V2",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{statuscode},
	)
	// websocketStreamingRequestsTotal counts WebSocket streaming requests (exec/attach/portforward)
	// routed by the API server, labeled by subresource and proxy_type.
	websocketStreamingRequestsTotal = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem: subsystem,
			Name:      "websocket_streaming_requests_total",
			Help: "Total number of WebSocket streaming requests (exec/attach/portforward) routed by the API server, " +
				"labeled by subresource and proxy_type. proxy_type is proxied_to_kubelet when the kubelet " +
				"handles the request directly; otherwise translated_at_apiserver.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"subresource", "proxy_type"},
	)
)

func Register() {
	registerMetricsOnce.Do(func() {
		legacyregistry.MustRegister(streamTranslatorRequestsTotal)
		legacyregistry.MustRegister(streamTunnelRequestsTotal)
		legacyregistry.MustRegister(websocketStreamingRequestsTotal)
	})
}

func ResetForTest() {
	streamTranslatorRequestsTotal.Reset()
	streamTunnelRequestsTotal.Reset()
	websocketStreamingRequestsTotal.Reset()
}

// IncStreamTranslatorRequest increments the # of requests handled by the StreamTranslatorProxy.
func IncStreamTranslatorRequest(ctx context.Context, status string) {
	streamTranslatorRequestsTotal.WithContext(ctx).WithLabelValues(status).Add(1)
}

// IncStreamTunnelRequest increments the # of requests handled by the StreamTunnelProxy.
func IncStreamTunnelRequest(ctx context.Context, status string) {
	streamTunnelRequestsTotal.WithContext(ctx).WithLabelValues(status).Add(1)
}

// IncWebSocketStreamingRequest increments the count of WebSocket streaming requests
// routed by the API server with the given subresource (exec/attach/portforward) and
// proxy_type (proxied_to_kubelet or translated_at_apiserver).
func IncWebSocketStreamingRequest(ctx context.Context, subresource, proxyType string) {
	websocketStreamingRequestsTotal.WithContext(ctx).WithLabelValues(subresource, proxyType).Add(1)
}
