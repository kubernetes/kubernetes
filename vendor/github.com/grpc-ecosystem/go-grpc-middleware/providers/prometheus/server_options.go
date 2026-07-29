// Copyright (c) The go-grpc-middleware Authors.
// Licensed under the Apache License 2.0.

package prometheus

import (
	"context"

	"github.com/prometheus/client_golang/prometheus"
)

type exemplarFromCtxFn func(ctx context.Context) prometheus.Labels
type labelsFromCtxFn func(metadata context.Context) prometheus.Labels

type serverMetricsConfig struct {
	counterOpts counterOptions
	// histogramOpts stores the options for creating the histogram with dynamic labels
	histogramOpts histogramOptions
	// enableHistogram indicates whether histogram should be created
	enableHistogram bool
	// contextLabels defines the names of dynamic labels to be extracted from context
	contextLabels []string
}

type ServerMetricsOption func(*serverMetricsConfig)

func (c *serverMetricsConfig) apply(opts []ServerMetricsOption) {
	for _, o := range opts {
		o(c)
	}
}

// WithServerCounterOptions sets counter options.
func WithServerCounterOptions(opts ...CounterOption) ServerMetricsOption {
	return func(o *serverMetricsConfig) {
		o.counterOpts = opts
	}
}

// WithServerHandlingTimeHistogram turns on recording of handling time of RPCs.
// Histogram metrics can be very expensive for Prometheus to retain and query.
func WithServerHandlingTimeHistogram(opts ...HistogramOption) ServerMetricsOption {
	return func(o *serverMetricsConfig) {
		o.histogramOpts = opts
		o.enableHistogram = true
	}
}

// WithContextLabels configures the server metrics to include dynamic labels extracted from context.
// The provided label names will be added to all server metrics as dynamic labels.
// Use WithLabelsFromContext in the interceptor options to specify how to extract these labels from context.
func WithContextLabels(labelNames ...string) ServerMetricsOption {
	return func(o *serverMetricsConfig) {
		o.contextLabels = labelNames
	}
}
