// Copyright 2018, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ocgrpc

import (
	"context"
	"go.opencensus.io/trace"

	"google.golang.org/grpc/stats"
)

// ServerHandler implements gRPC stats.Handler recording OpenCensus stats and
// traces. Use with gRPC servers.
//
// When installed (see Example), tracing metadata is read from inbound RPCs
// by default. If no tracing metadata is present, or if the tracing metadata is
// present but the SpanContext isn't sampled, then a new trace may be started
// (as determined by Sampler).
type ServerHandler struct {
	// IsPublicEndpoint may be set to true to always start a new trace around
	// each RPC. Any SpanContext in the RPC metadata will be added as a linked
	// span instead of making it the parent of the span created around the
	// server RPC.
	//
	// Be aware that if you leave this false (the default) on a public-facing
	// server, callers will be able to send tracing metadata in gRPC headers
	// and trigger traces in your backend.
	IsPublicEndpoint bool

	// StartOptions to use for to spans started around RPCs handled by this server.
	//
	// These will apply even if there is tracing metadata already
	// present on the inbound RPC but the SpanContext is not sampled. This
	// ensures that each service has some opportunity to be traced. If you would
	// like to not add any additional traces for this gRPC service, set:
	//
	//   StartOptions.Sampler = trace.ProbabilitySampler(0.0)
	//
	// StartOptions.SpanKind will always be set to trace.SpanKindServer
	// for spans started by this handler.
	StartOptions trace.StartOptions
}

var _ stats.Handler = (*ServerHandler)(nil)

// HandleConn exists to satisfy gRPC stats.Handler.
func (s *ServerHandler) HandleConn(ctx context.Context, cs stats.ConnStats) {
	// no-op
}

// TagConn exists to satisfy gRPC stats.Handler.
func (s *ServerHandler) TagConn(ctx context.Context, cti *stats.ConnTagInfo) context.Context {
	// no-op
	return ctx
}

// HandleRPC implements per-RPC tracing and stats instrumentation.
func (s *ServerHandler) HandleRPC(ctx context.Context, rs stats.RPCStats) {
	traceHandleRPC(ctx, rs)
	statsHandleRPC(ctx, rs)
}

// TagRPC implements per-RPC context management.
func (s *ServerHandler) TagRPC(ctx context.Context, rti *stats.RPCTagInfo) context.Context {
	ctx = s.traceTagRPC(ctx, rti)
	ctx = s.statsTagRPC(ctx, rti)
	return ctx
}
