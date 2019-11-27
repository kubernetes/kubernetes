// Copyright 2016 Michal Witkowski. All Rights Reserved.
// See LICENSE for licensing terms.

// gRPC Prometheus monitoring interceptors for server-side gRPC.

package grpc_prometheus

import (
	prom "github.com/prometheus/client_golang/prometheus"
	"google.golang.org/grpc"
)

var (
	// DefaultServerMetrics is the default instance of ServerMetrics. It is
	// intended to be used in conjunction the default Prometheus metrics
	// registry.
	DefaultServerMetrics = NewServerMetrics()

	// UnaryServerInterceptor is a gRPC server-side interceptor that provides Prometheus monitoring for Unary RPCs.
	UnaryServerInterceptor = DefaultServerMetrics.UnaryServerInterceptor()

	// StreamServerInterceptor is a gRPC server-side interceptor that provides Prometheus monitoring for Streaming RPCs.
	StreamServerInterceptor = DefaultServerMetrics.StreamServerInterceptor()
)

func init() {
	prom.MustRegister(DefaultServerMetrics.serverStartedCounter)
	prom.MustRegister(DefaultServerMetrics.serverHandledCounter)
	prom.MustRegister(DefaultServerMetrics.serverStreamMsgReceived)
	prom.MustRegister(DefaultServerMetrics.serverStreamMsgSent)
}

// Register takes a gRPC server and pre-initializes all counters to 0. This
// allows for easier monitoring in Prometheus (no missing metrics), and should
// be called *after* all services have been registered with the server. This
// function acts on the DefaultServerMetrics variable.
func Register(server *grpc.Server) {
	DefaultServerMetrics.InitializeMetrics(server)
}

// EnableHandlingTimeHistogram turns on recording of handling time
// of RPCs. Histogram metrics can be very expensive for Prometheus
// to retain and query. This function acts on the DefaultServerMetrics
// variable and the default Prometheus metrics registry.
func EnableHandlingTimeHistogram(opts ...HistogramOption) {
	DefaultServerMetrics.EnableHandlingTimeHistogram(opts...)
	prom.Register(DefaultServerMetrics.serverHandledHistogram)
}
