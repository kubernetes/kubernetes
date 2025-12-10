// Copyright (c) The go-grpc-middleware Authors.
// Licensed under the Apache License 2.0.

package prometheus

import (
	"github.com/prometheus/client_golang/prometheus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/status"
)

// FromError returns a grpc status. If the error code is neither a valid grpc status nor a context error, codes.Unknown
// will be set.
func FromError(err error) *status.Status {
	s, ok := status.FromError(err)
	// Mirror what the grpc server itself does, i.e. also convert context errors to status
	if !ok {
		s = status.FromContextError(err)
	}
	return s
}

// A CounterOption lets you add options to Counter metrics using With* funcs.
type CounterOption func(*prometheus.CounterOpts)

type counterOptions []CounterOption

func (co counterOptions) apply(o prometheus.CounterOpts) prometheus.CounterOpts {
	for _, f := range co {
		f(&o)
	}
	return o
}

// WithConstLabels allows you to add ConstLabels to Counter metrics.
func WithConstLabels(labels prometheus.Labels) CounterOption {
	return func(o *prometheus.CounterOpts) {
		o.ConstLabels = labels
	}
}

// WithSubsystem allows you to add a Subsystem to Counter metrics.
func WithSubsystem(subsystem string) CounterOption {
	return func(o *prometheus.CounterOpts) {
		o.Subsystem = subsystem
	}
}

// A HistogramOption lets you add options to Histogram metrics using With*
// funcs.
type HistogramOption func(*prometheus.HistogramOpts)

type histogramOptions []HistogramOption

func (ho histogramOptions) apply(o prometheus.HistogramOpts) prometheus.HistogramOpts {
	for _, f := range ho {
		f(&o)
	}
	return o
}

// WithHistogramBuckets allows you to specify custom bucket ranges for histograms if EnableHandlingTimeHistogram is on.
func WithHistogramBuckets(buckets []float64) HistogramOption {
	return func(o *prometheus.HistogramOpts) { o.Buckets = buckets }
}

// WithHistogramOpts allows you to specify HistogramOpts but makes sure the correct name and label is used.
// This function is helpful when specifying more than just the buckets, like using NativeHistograms.
func WithHistogramOpts(opts *prometheus.HistogramOpts) HistogramOption {
	// TODO: This isn't ideal either if new fields are added to prometheus.HistogramOpts.
	// Maybe we can change the interface to accept abitrary HistogramOpts and
	// only make sure to overwrite the necessary fields (name, labels).
	return func(o *prometheus.HistogramOpts) {
		o.Buckets = opts.Buckets
		o.NativeHistogramBucketFactor = opts.NativeHistogramBucketFactor
		o.NativeHistogramZeroThreshold = opts.NativeHistogramZeroThreshold
		o.NativeHistogramMaxBucketNumber = opts.NativeHistogramMaxBucketNumber
		o.NativeHistogramMinResetDuration = opts.NativeHistogramMinResetDuration
		o.NativeHistogramMaxZeroThreshold = opts.NativeHistogramMaxZeroThreshold
	}
}

// WithHistogramConstLabels allows you to add custom ConstLabels to
// histograms metrics.
func WithHistogramConstLabels(labels prometheus.Labels) HistogramOption {
	return func(o *prometheus.HistogramOpts) {
		o.ConstLabels = labels
	}
}

// WithHistogramSubsystem allows you to add a Subsystem to histograms metrics.
func WithHistogramSubsystem(subsystem string) HistogramOption {
	return func(o *prometheus.HistogramOpts) {
		o.Subsystem = subsystem
	}
}

func typeFromMethodInfo(mInfo *grpc.MethodInfo) grpcType {
	if !mInfo.IsClientStream && !mInfo.IsServerStream {
		return Unary
	}
	if mInfo.IsClientStream && !mInfo.IsServerStream {
		return ClientStream
	}
	if !mInfo.IsClientStream && mInfo.IsServerStream {
		return ServerStream
	}
	return BidiStream
}

// An Option lets you add options to prometheus interceptors using With* funcs.
type Option func(*config)

type config struct {
	exemplarFn exemplarFromCtxFn
}

func (c *config) apply(opts []Option) {
	for _, o := range opts {
		o(c)
	}
}

// WithExemplarFromContext sets function that will be used to deduce exemplar for all counter and histogram metrics.
func WithExemplarFromContext(exemplarFn exemplarFromCtxFn) Option {
	return func(o *config) {
		o.exemplarFn = exemplarFn
	}
}
