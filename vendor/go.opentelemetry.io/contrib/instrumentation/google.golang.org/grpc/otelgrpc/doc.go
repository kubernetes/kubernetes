// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

/*
Package otelgrpc is the instrumentation library for [google.golang.org/grpc].

Use [NewClientHandler] with [grpc.WithStatsHandler] to instrument a gRPC client.

Use [NewServerHandler] with [grpc.StatsHandler] to instrument a gRPC server.
*/
package otelgrpc // import "go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
