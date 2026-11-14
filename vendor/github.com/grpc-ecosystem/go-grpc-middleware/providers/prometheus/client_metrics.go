// Copyright (c) The go-grpc-middleware Authors.
// Licensed under the Apache License 2.0.

package prometheus

import (
	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors"
	"github.com/prometheus/client_golang/prometheus"
	"google.golang.org/grpc"
)

// ClientMetrics represents a collection of metrics to be registered on a
// Prometheus metrics registry for a gRPC client.
type ClientMetrics struct {
	clientStartedCounter    *prometheus.CounterVec
	clientHandledCounter    *prometheus.CounterVec
	clientStreamMsgReceived *prometheus.CounterVec
	clientStreamMsgSent     *prometheus.CounterVec

	// clientHandledHistogram can be nil
	clientHandledHistogram *prometheus.HistogramVec
	// clientStreamRecvHistogram can be nil
	clientStreamRecvHistogram *prometheus.HistogramVec
	// clientStreamSendHistogram can be nil
	clientStreamSendHistogram *prometheus.HistogramVec

	// contextLabelNames stores the names of context labels
	contextLabelNames []string
}

// NewClientMetrics returns a new ClientMetrics object.
// NOTE: Remember to register ClientMetrics object using prometheus registry
// e.g. prometheus.MustRegister(myClientMetrics).
func NewClientMetrics(opts ...ClientMetricsOption) *ClientMetrics {
	config := &clientMetricsConfig{
		clientHandledHistogramFn:    func() *prometheus.HistogramVec { return nil },
		clientStreamRecvHistogramFn: func() *prometheus.HistogramVec { return nil },
		clientStreamSendHistogramFn: func() *prometheus.HistogramVec { return nil },
	}
	config.apply(opts)

	// Build label names by combining default labels with context labels
	defaultLabels := []string{"grpc_type", "grpc_service", "grpc_method"}
	defaultLabelsWithCode := []string{"grpc_type", "grpc_service", "grpc_method", "grpc_code"}

	startedLabels := append(defaultLabels, config.contextLabels...)
	handledLabels := append(defaultLabelsWithCode, config.contextLabels...)
	streamLabels := append(defaultLabels, config.contextLabels...)

	return &ClientMetrics{
		clientStartedCounter: prometheus.NewCounterVec(
			config.counterOpts.apply(prometheus.CounterOpts{
				Name: "grpc_client_started_total",
				Help: "Total number of RPCs started on the client.",
			}), startedLabels,
		),

		clientHandledCounter: prometheus.NewCounterVec(
			config.counterOpts.apply(prometheus.CounterOpts{
				Name: "grpc_client_handled_total",
				Help: "Total number of RPCs completed by the client, regardless of success or failure.",
			}), handledLabels,
		),

		clientStreamMsgReceived: prometheus.NewCounterVec(
			config.counterOpts.apply(prometheus.CounterOpts{
				Name: "grpc_client_msg_received_total",
				Help: "Total number of RPC stream messages received by the client.",
			}), streamLabels,
		),

		clientStreamMsgSent: prometheus.NewCounterVec(
			config.counterOpts.apply(prometheus.CounterOpts{
				Name: "grpc_client_msg_sent_total",
				Help: "Total number of gRPC stream messages sent by the client.",
			}), streamLabels,
		),

		clientHandledHistogram:    config.clientHandledHistogramFn(),
		clientStreamRecvHistogram: config.clientStreamRecvHistogramFn(),
		clientStreamSendHistogram: config.clientStreamSendHistogramFn(),

		contextLabelNames: config.contextLabels,
	}
}

// Describe sends the super-set of all possible descriptors of metrics
// collected by this Collector to the provided channel and returns once
// the last descriptor has been sent.
func (m *ClientMetrics) Describe(ch chan<- *prometheus.Desc) {
	m.clientStartedCounter.Describe(ch)
	m.clientHandledCounter.Describe(ch)
	m.clientStreamMsgReceived.Describe(ch)
	m.clientStreamMsgSent.Describe(ch)
	if m.clientHandledHistogram != nil {
		m.clientHandledHistogram.Describe(ch)
	}
	if m.clientStreamRecvHistogram != nil {
		m.clientStreamRecvHistogram.Describe(ch)
	}
	if m.clientStreamSendHistogram != nil {
		m.clientStreamSendHistogram.Describe(ch)
	}
}

// Collect is called by the Prometheus registry when collecting
// metrics. The implementation sends each collected metric via the
// provided channel and returns once the last metric has been sent.
func (m *ClientMetrics) Collect(ch chan<- prometheus.Metric) {
	m.clientStartedCounter.Collect(ch)
	m.clientHandledCounter.Collect(ch)
	m.clientStreamMsgReceived.Collect(ch)
	m.clientStreamMsgSent.Collect(ch)
	if m.clientHandledHistogram != nil {
		m.clientHandledHistogram.Collect(ch)
	}
	if m.clientStreamRecvHistogram != nil {
		m.clientStreamRecvHistogram.Collect(ch)
	}
	if m.clientStreamSendHistogram != nil {
		m.clientStreamSendHistogram.Collect(ch)
	}
}

// UnaryClientInterceptor is a gRPC client-side interceptor that provides Prometheus monitoring for Unary RPCs.
func (m *ClientMetrics) UnaryClientInterceptor(opts ...Option) grpc.UnaryClientInterceptor {
	return interceptors.UnaryClientInterceptor(&reportable{
		opts:          opts,
		clientMetrics: m,
	})
}

// StreamClientInterceptor is a gRPC client-side interceptor that provides Prometheus monitoring for Streaming RPCs.
func (m *ClientMetrics) StreamClientInterceptor(opts ...Option) grpc.StreamClientInterceptor {
	return interceptors.StreamClientInterceptor(&reportable{
		opts:          opts,
		clientMetrics: m,
	})
}
