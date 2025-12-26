// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otelgrpc // import "go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"

// Version is the current release version of the gRPC instrumentation.
func Version() string {
	return "0.63.0"
	// This string is updated by the pre_release.sh script during release
}
