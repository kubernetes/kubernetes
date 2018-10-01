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
	"go.opencensus.io/trace"
	"golang.org/x/net/context"

	"google.golang.org/grpc/stats"
)

// ClientHandler implements a gRPC stats.Handler for recording OpenCensus stats and
// traces. Use with gRPC clients only.
type ClientHandler struct {
	// StartOptions allows configuring the StartOptions used to create new spans.
	//
	// StartOptions.SpanKind will always be set to trace.SpanKindClient
	// for spans started by this handler.
	StartOptions trace.StartOptions
}

// HandleConn exists to satisfy gRPC stats.Handler.
func (c *ClientHandler) HandleConn(ctx context.Context, cs stats.ConnStats) {
	// no-op
}

// TagConn exists to satisfy gRPC stats.Handler.
func (c *ClientHandler) TagConn(ctx context.Context, cti *stats.ConnTagInfo) context.Context {
	// no-op
	return ctx
}

// HandleRPC implements per-RPC tracing and stats instrumentation.
func (c *ClientHandler) HandleRPC(ctx context.Context, rs stats.RPCStats) {
	traceHandleRPC(ctx, rs)
	statsHandleRPC(ctx, rs)
}

// TagRPC implements per-RPC context management.
func (c *ClientHandler) TagRPC(ctx context.Context, rti *stats.RPCTagInfo) context.Context {
	ctx = c.traceTagRPC(ctx, rti)
	ctx = c.statsTagRPC(ctx, rti)
	return ctx
}
