// Copyright (c) The go-grpc-middleware Authors.
// Licensed under the Apache License 2.0.

package prometheus

import (
	"github.com/prometheus/client_golang/prometheus"
)

type clientMetricsConfig struct {
	counterOpts counterOptions
	// clientHandledHistogram can be nil.
	clientHandledHistogram *prometheus.HistogramVec
	// clientStreamRecvHistogram can be nil.
	clientStreamRecvHistogram *prometheus.HistogramVec
	// clientStreamSendHistogram can be nil.
	clientStreamSendHistogram *prometheus.HistogramVec
}

type ClientMetricsOption func(*clientMetricsConfig)

func (c *clientMetricsConfig) apply(opts []ClientMetricsOption) {
	for _, o := range opts {
		o(c)
	}
}

func WithClientCounterOptions(opts ...CounterOption) ClientMetricsOption {
	return func(o *clientMetricsConfig) {
		o.counterOpts = opts
	}
}

// WithClientHandlingTimeHistogram turns on recording of handling time of RPCs.
// Histogram metrics can be very expensive for Prometheus to retain and query.
func WithClientHandlingTimeHistogram(opts ...HistogramOption) ClientMetricsOption {
	return func(o *clientMetricsConfig) {
		o.clientHandledHistogram = prometheus.NewHistogramVec(
			histogramOptions(opts).apply(prometheus.HistogramOpts{
				Name:    "grpc_client_handling_seconds",
				Help:    "Histogram of response latency (seconds) of the gRPC until it is finished by the application.",
				Buckets: prometheus.DefBuckets,
			}),
			[]string{"grpc_type", "grpc_service", "grpc_method"},
		)
	}
}

// WithClientStreamRecvHistogram turns on recording of single message receive time of streaming RPCs.
// Histogram metrics can be very expensive for Prometheus to retain and query.
func WithClientStreamRecvHistogram(opts ...HistogramOption) ClientMetricsOption {
	return func(o *clientMetricsConfig) {
		o.clientStreamRecvHistogram = prometheus.NewHistogramVec(
			histogramOptions(opts).apply(prometheus.HistogramOpts{
				Name:    "grpc_client_msg_recv_handling_seconds",
				Help:    "Histogram of response latency (seconds) of the gRPC single message receive.",
				Buckets: prometheus.DefBuckets,
			}),
			[]string{"grpc_type", "grpc_service", "grpc_method"},
		)
	}
}

// WithClientStreamSendHistogram turns on recording of single message send time of streaming RPCs.
// Histogram metrics can be very expensive for Prometheus to retain and query.
func WithClientStreamSendHistogram(opts ...HistogramOption) ClientMetricsOption {
	return func(o *clientMetricsConfig) {
		o.clientStreamSendHistogram = prometheus.NewHistogramVec(
			histogramOptions(opts).apply(prometheus.HistogramOpts{
				Name:    "grpc_client_msg_send_handling_seconds",
				Help:    "Histogram of response latency (seconds) of the gRPC single message send.",
				Buckets: prometheus.DefBuckets,
			}),
			[]string{"grpc_type", "grpc_service", "grpc_method"},
		)
	}
}
