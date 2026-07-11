// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package internal provides internal functionality for the otelgrpc package.
package internal // import "go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc/internal"

import (
	"strings"

	"go.opentelemetry.io/otel/attribute"
	oldsemconv "go.opentelemetry.io/otel/semconv/v1.37.0" //nolint:depguard // Use of v1.37.0 is required for backward compatibility stability opt-in.
	semconv "go.opentelemetry.io/otel/semconv/v1.40.0"
)

// ParseFullMethod returns a span name following the OpenTelemetry semantic
// conventions as well as all applicable span attribute.KeyValue attributes based
// on a gRPC's FullMethod.
//
// Parsing is consistent with grpc-go implementation:
// https://github.com/grpc/grpc-go/blob/v1.57.0/internal/grpcutil/method.go#L26-L39
func ParseFullMethod(fullMethod string) (string, []attribute.KeyValue) {
	if !strings.HasPrefix(fullMethod, "/") {
		// Invalid format, does not follow `/package.service/method`.
		return fullMethod, nil
	}
	name := fullMethod[1:]
	return name, []attribute.KeyValue{semconv.RPCMethod(name)}
}

// ParseFullMethodOld returns a span name following the old OpenTelemetry semantic
// conventions as well as all applicable span attribute.KeyValue attributes based
// on a gRPC's FullMethod.
// Based on the implementation in:
// https://github.com/open-telemetry/opentelemetry-go-contrib/blob/072dcf8ad7e5e48b506e05720b29d8b078759606/instrumentation/google.golang.org/grpc/otelgrpc/internal/parse.go#L20
func ParseFullMethodOld(fullMethod string) (string, []attribute.KeyValue) {
	if !strings.HasPrefix(fullMethod, "/") {
		return fullMethod, nil
	}
	name := fullMethod[1:]
	parts := strings.Split(name, "/")
	if len(parts) < 2 {
		return name, []attribute.KeyValue{
			attribute.String("rpc.system", "grpc"),
		}
	}
	service := parts[0]
	method := parts[1]
	return name, []attribute.KeyValue{
		oldsemconv.RPCSystemKey.String("grpc"),
		oldsemconv.RPCServiceKey.String(service),
		oldsemconv.RPCMethodKey.String(method),
	}
}
