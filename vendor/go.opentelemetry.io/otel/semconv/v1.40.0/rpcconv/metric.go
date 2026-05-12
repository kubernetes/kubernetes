// Code generated from semantic convention specification. DO NOT EDIT.

// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package rpcconv provides types and functionality for OpenTelemetry semantic
// conventions in the "rpc" namespace.
package rpcconv

import (
	"context"
	"sync"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/noop"
)

var (
	addOptPool = &sync.Pool{New: func() any { return &[]metric.AddOption{} }}
	recOptPool = &sync.Pool{New: func() any { return &[]metric.RecordOption{} }}
)

// ErrorTypeAttr is an attribute conforming to the error.type semantic
// conventions. It represents the describes a class of error the operation ended
// with.
type ErrorTypeAttr string

var (
	// ErrorTypeOther is a fallback error value to be used when the instrumentation
	// doesn't define a custom value.
	ErrorTypeOther ErrorTypeAttr = "_OTHER"
)

// SystemNameAttr is an attribute conforming to the rpc.system.name semantic
// conventions. It represents the Remote Procedure Call (RPC) system.
type SystemNameAttr string

var (
	// SystemNameGRPC is the [gRPC].
	//
	// [gRPC]: https://grpc.io/
	SystemNameGRPC SystemNameAttr = "grpc"
	// SystemNameDubbo is the [Apache Dubbo].
	//
	// [Apache Dubbo]: https://dubbo.apache.org/
	SystemNameDubbo SystemNameAttr = "dubbo"
	// SystemNameConnectrpc is the [Connect RPC].
	//
	// [Connect RPC]: https://connectrpc.com/
	SystemNameConnectrpc SystemNameAttr = "connectrpc"
	// SystemNameJSONRPC is the [JSON-RPC].
	//
	// [JSON-RPC]: https://www.jsonrpc.org/
	SystemNameJSONRPC SystemNameAttr = "jsonrpc"
)

// ClientCallDuration is an instrument used to record metric values conforming to
// the "rpc.client.call.duration" semantic conventions. It represents the
// measures the duration of an outgoing Remote Procedure Call (RPC).
type ClientCallDuration struct {
	metric.Float64Histogram
}

var newClientCallDurationOpts = []metric.Float64HistogramOption{
	metric.WithDescription("Measures the duration of an outgoing Remote Procedure Call (RPC)."),
	metric.WithUnit("s"),
}

// NewClientCallDuration returns a new ClientCallDuration instrument.
func NewClientCallDuration(
	m metric.Meter,
	opt ...metric.Float64HistogramOption,
) (ClientCallDuration, error) {
	// Check if the meter is nil.
	if m == nil {
		return ClientCallDuration{noop.Float64Histogram{}}, nil
	}

	if len(opt) == 0 {
		opt = newClientCallDurationOpts
	} else {
		opt = append(opt, newClientCallDurationOpts...)
	}

	i, err := m.Float64Histogram(
		"rpc.client.call.duration",
		opt...,
	)
	if err != nil {
		return ClientCallDuration{noop.Float64Histogram{}}, err
	}
	return ClientCallDuration{i}, nil
}

// Inst returns the underlying metric instrument.
func (m ClientCallDuration) Inst() metric.Float64Histogram {
	return m.Float64Histogram
}

// Name returns the semantic convention name of the instrument.
func (ClientCallDuration) Name() string {
	return "rpc.client.call.duration"
}

// Unit returns the semantic convention unit of the instrument
func (ClientCallDuration) Unit() string {
	return "s"
}

// Description returns the semantic convention description of the instrument
func (ClientCallDuration) Description() string {
	return "Measures the duration of an outgoing Remote Procedure Call (RPC)."
}

// Record records val to the current distribution for attrs.
//
// The systemName is the the Remote Procedure Call (RPC) system.
//
// All additional attrs passed are included in the recorded value.
//
// When this metric is reported alongside an RPC client span, the metric value
// SHOULD be the same as the RPC client span duration.
func (m ClientCallDuration) Record(
	ctx context.Context,
	val float64,
	systemName SystemNameAttr,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Float64Histogram.Record(ctx, val, metric.WithAttributes(
			attribute.String("rpc.system.name", string(systemName)),
		))
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			append(
				attrs[:len(attrs):len(attrs)],
				attribute.String("rpc.system.name", string(systemName)),
			)...,
		),
	)

	m.Float64Histogram.Record(ctx, val, *o...)
}

// RecordSet records val to the current distribution for set.
//
// When this metric is reported alongside an RPC client span, the metric value
// SHOULD be the same as the RPC client span duration.
func (m ClientCallDuration) RecordSet(ctx context.Context, val float64, set attribute.Set) {
	if set.Len() == 0 {
		m.Float64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Float64Histogram.Record(ctx, val, *o...)
}

// AttrErrorType returns an optional attribute for the "error.type" semantic
// convention. It represents the describes a class of error the operation ended
// with.
func (ClientCallDuration) AttrErrorType(val ErrorTypeAttr) attribute.KeyValue {
	return attribute.String("error.type", string(val))
}

// AttrMethod returns an optional attribute for the "rpc.method" semantic
// convention. It represents the fully-qualified logical name of the method from
// the RPC interface perspective.
func (ClientCallDuration) AttrMethod(val string) attribute.KeyValue {
	return attribute.String("rpc.method", val)
}

// AttrResponseStatusCode returns an optional attribute for the
// "rpc.response.status_code" semantic convention. It represents the status code
// of the RPC returned by the RPC server or generated by the client.
func (ClientCallDuration) AttrResponseStatusCode(val string) attribute.KeyValue {
	return attribute.String("rpc.response.status_code", val)
}

// AttrServerAddress returns an optional attribute for the "server.address"
// semantic convention. It represents a string identifying a group of RPC server
// instances request is sent to.
func (ClientCallDuration) AttrServerAddress(val string) attribute.KeyValue {
	return attribute.String("server.address", val)
}

// AttrServerPort returns an optional attribute for the "server.port" semantic
// convention. It represents the server port number.
func (ClientCallDuration) AttrServerPort(val int) attribute.KeyValue {
	return attribute.Int("server.port", val)
}

// ServerCallDuration is an instrument used to record metric values conforming to
// the "rpc.server.call.duration" semantic conventions. It represents the
// measures the duration of an incoming Remote Procedure Call (RPC).
type ServerCallDuration struct {
	metric.Float64Histogram
}

var newServerCallDurationOpts = []metric.Float64HistogramOption{
	metric.WithDescription("Measures the duration of an incoming Remote Procedure Call (RPC)."),
	metric.WithUnit("s"),
}

// NewServerCallDuration returns a new ServerCallDuration instrument.
func NewServerCallDuration(
	m metric.Meter,
	opt ...metric.Float64HistogramOption,
) (ServerCallDuration, error) {
	// Check if the meter is nil.
	if m == nil {
		return ServerCallDuration{noop.Float64Histogram{}}, nil
	}

	if len(opt) == 0 {
		opt = newServerCallDurationOpts
	} else {
		opt = append(opt, newServerCallDurationOpts...)
	}

	i, err := m.Float64Histogram(
		"rpc.server.call.duration",
		opt...,
	)
	if err != nil {
		return ServerCallDuration{noop.Float64Histogram{}}, err
	}
	return ServerCallDuration{i}, nil
}

// Inst returns the underlying metric instrument.
func (m ServerCallDuration) Inst() metric.Float64Histogram {
	return m.Float64Histogram
}

// Name returns the semantic convention name of the instrument.
func (ServerCallDuration) Name() string {
	return "rpc.server.call.duration"
}

// Unit returns the semantic convention unit of the instrument
func (ServerCallDuration) Unit() string {
	return "s"
}

// Description returns the semantic convention description of the instrument
func (ServerCallDuration) Description() string {
	return "Measures the duration of an incoming Remote Procedure Call (RPC)."
}

// Record records val to the current distribution for attrs.
//
// The systemName is the the Remote Procedure Call (RPC) system.
//
// All additional attrs passed are included in the recorded value.
//
// When this metric is reported alongside an RPC server span, the metric value
// SHOULD be the same as the RPC server span duration.
func (m ServerCallDuration) Record(
	ctx context.Context,
	val float64,
	systemName SystemNameAttr,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Float64Histogram.Record(ctx, val, metric.WithAttributes(
			attribute.String("rpc.system.name", string(systemName)),
		))
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			append(
				attrs[:len(attrs):len(attrs)],
				attribute.String("rpc.system.name", string(systemName)),
			)...,
		),
	)

	m.Float64Histogram.Record(ctx, val, *o...)
}

// RecordSet records val to the current distribution for set.
//
// When this metric is reported alongside an RPC server span, the metric value
// SHOULD be the same as the RPC server span duration.
func (m ServerCallDuration) RecordSet(ctx context.Context, val float64, set attribute.Set) {
	if set.Len() == 0 {
		m.Float64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Float64Histogram.Record(ctx, val, *o...)
}

// AttrErrorType returns an optional attribute for the "error.type" semantic
// convention. It represents the describes a class of error the operation ended
// with.
func (ServerCallDuration) AttrErrorType(val ErrorTypeAttr) attribute.KeyValue {
	return attribute.String("error.type", string(val))
}

// AttrMethod returns an optional attribute for the "rpc.method" semantic
// convention. It represents the fully-qualified logical name of the method from
// the RPC interface perspective.
func (ServerCallDuration) AttrMethod(val string) attribute.KeyValue {
	return attribute.String("rpc.method", val)
}

// AttrResponseStatusCode returns an optional attribute for the
// "rpc.response.status_code" semantic convention. It represents the status code
// of the RPC returned by the RPC server or generated by the client.
func (ServerCallDuration) AttrResponseStatusCode(val string) attribute.KeyValue {
	return attribute.String("rpc.response.status_code", val)
}

// AttrServerAddress returns an optional attribute for the "server.address"
// semantic convention. It represents a string identifying a group of RPC server
// instances request is sent to.
func (ServerCallDuration) AttrServerAddress(val string) attribute.KeyValue {
	return attribute.String("server.address", val)
}

// AttrServerPort returns an optional attribute for the "server.port" semantic
// convention. It represents the server port number.
func (ServerCallDuration) AttrServerPort(val int) attribute.KeyValue {
	return attribute.Int("server.port", val)
}
