// Copyright (c) The go-grpc-middleware Authors.
// Licensed under the Apache License 2.0.

package prometheus

import (
	"github.com/prometheus/client_golang/prometheus"
)

type clientMetricsConfig struct {
	counterOpts counterOptions
	// clientHandledHistogramFn can be nil.
	clientHandledHistogramFn func() *prometheus.HistogramVec
	// clientStreamRecvHistogramFn can be nil.
	clientStreamRecvHistogramFn func() *prometheus.HistogramVec
	// clientStreamSendHistogramFn can be nil.
	clientStreamSendHistogramFn func() *prometheus.HistogramVec
	// contextLabels defines the names of dynamic labels to be extracted from context
	contextLabels []string
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
		o.clientHandledHistogramFn = func() *prometheus.HistogramVec {
			defaultLabels := []string{"grpc_type", "grpc_service", "grpc_method"}
			allLabels := append(defaultLabels, o.contextLabels...)

			return prometheus.NewHistogramVec(
				histogramOptions(opts).apply(&prometheus.HistogramOpts{
					Name:    "grpc_client_handling_seconds",
					Help:    "Histogram of response latency (seconds) of the gRPC until it is finished by the application.",
					Buckets: prometheus.DefBuckets,
				}),
				allLabels,
			)
		}
	}
}

// WithClientStreamRecvHistogram turns on recording of single message receive time of streaming RPCs.
// Histogram metrics can be very expensive for Prometheus to retain and query.
func WithClientStreamRecvHistogram(opts ...HistogramOption) ClientMetricsOption {
	return func(o *clientMetricsConfig) {
		o.clientStreamRecvHistogramFn = func() *prometheus.HistogramVec {
			defaultLabels := []string{"grpc_type", "grpc_service", "grpc_method"}
			allLabels := append(defaultLabels, o.contextLabels...)

			return prometheus.NewHistogramVec(
				histogramOptions(opts).apply(&prometheus.HistogramOpts{
					Name:    "grpc_client_msg_recv_handling_seconds",
					Help:    "Histogram of response latency (seconds) of the gRPC single message receive.",
					Buckets: prometheus.DefBuckets,
				}),
				allLabels,
			)
		}
	}
}

// WithClientStreamSendHistogram turns on recording of single message send time of streaming RPCs.
// Histogram metrics can be very expensive for Prometheus to retain and query.
func WithClientStreamSendHistogram(opts ...HistogramOption) ClientMetricsOption {
	return func(o *clientMetricsConfig) {
		o.clientStreamSendHistogramFn = func() *prometheus.HistogramVec {
			defaultLabels := []string{"grpc_type", "grpc_service", "grpc_method"}
			allLabels := append(defaultLabels, o.contextLabels...)

			return prometheus.NewHistogramVec(
				histogramOptions(opts).apply(&prometheus.HistogramOpts{
					Name:    "grpc_client_msg_send_handling_seconds",
					Help:    "Histogram of response latency (seconds) of the gRPC single message send.",
					Buckets: prometheus.DefBuckets,
				}),
				allLabels,
			)
		}
	}
}

// WithClientContextLabels configures the server metrics to include dynamic labels extracted from context.
// The provided label names will be added to all server metrics as dynamic labels.
// Use WithLabelsFromContext in the interceptor options to specify how to extract these labels from context.
func WithClientContextLabels(labelNames ...string) ClientMetricsOption {
	return func(o *clientMetricsConfig) {
		o.contextLabels = labelNames
	}
}
