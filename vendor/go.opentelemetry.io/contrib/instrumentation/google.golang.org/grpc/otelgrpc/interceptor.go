// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otelgrpc

// gRPC tracing middleware
// https://opentelemetry.io/docs/specs/semconv/rpc/
import (
	"net"
	"net/url"
	"strconv"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	semconv "go.opentelemetry.io/otel/semconv/v1.43.0"
	grpc_codes "google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// serverAddrAttrsFromCanonicalTarget extracts server address attributes from a
// canonical gRPC target as returned by [google.golang.org/grpc.ClientConn.CanonicalTarget].
// Canonical targets always have the form "scheme://[authority]/endpoint", e.g.:
//
//	passthrough:///127.0.0.1:7777   → ServerAddress("127.0.0.1"), ServerPort(7777)
//	dns:///example.com:443          → ServerAddress("example.com"), ServerPort(443)
//	dns://authority/example.com:443 → ServerAddress("example.com"), ServerPort(443)
//	unix:///tmp/grpc.sock           → ServerAddress("/tmp/grpc.sock")
func serverAddrAttrsFromCanonicalTarget(target string) []attribute.KeyValue {
	// Fast path: confirmed host:port with numeric port, no scheme involved.
	if h, pStr, err := net.SplitHostPort(target); err == nil {
		if p, err := strconv.Atoi(pStr); err == nil {
			return []attribute.KeyValue{
				semconv.ServerAddress(h),
				semconv.ServerPort(p),
			}
		}
	}
	// gRPC URI: scheme://authority/endpoint  → endpoint in Path
	//           scheme:endpoint              → endpoint in Opaque
	u, err := url.Parse(target)
	if err != nil {
		return []attribute.KeyValue{semconv.ServerAddress(target)}
	}
	ep := u.Path
	if u.Scheme != "unix" && u.Scheme != "unix-abstract" {
		// Strip the leading "/" added by url.Parse for hierarchical URIs;
		// preserve it for unix socket paths where the slash is meaningful.
		ep = strings.TrimPrefix(ep, "/")
	}
	if ep == "" {
		ep = u.Opaque
	}
	if ep != "" {
		return serverAddrAttrs(ep)
	}
	return []attribute.KeyValue{semconv.ServerAddress(target)}
}

// serverAddrAttrs returns the server address attributes for the hostport.
func serverAddrAttrs(hostport string) []attribute.KeyValue {
	h, pStr, err := net.SplitHostPort(hostport)
	if err != nil {
		// The server.address attribute is required.
		return []attribute.KeyValue{semconv.ServerAddress(hostport)}
	}
	p, err := strconv.Atoi(pStr)
	if err != nil {
		return []attribute.KeyValue{semconv.ServerAddress(h)}
	}
	return []attribute.KeyValue{
		semconv.ServerAddress(h),
		semconv.ServerPort(p),
	}
}

// serverStatus returns a span status code and message for a given gRPC
// status code. It maps specific gRPC status codes to a corresponding span
// status code and message. This function is intended for use on the server
// side of a gRPC connection.
//
// If the gRPC status code is Unknown, DeadlineExceeded, Unimplemented,
// Internal, Unavailable, or DataLoss, it returns a span status code of Error
// and the message from the gRPC status. Otherwise, it returns a span status
// code of Unset and an empty message.
func serverStatus(grpcStatus *status.Status) (codes.Code, string) {
	switch grpcStatus.Code() {
	case grpc_codes.Unknown,
		grpc_codes.DeadlineExceeded,
		grpc_codes.Unimplemented,
		grpc_codes.Internal,
		grpc_codes.Unavailable,
		grpc_codes.DataLoss:
		return codes.Error, grpcStatus.Message()
	default:
		return codes.Unset, ""
	}
}
