// Copyright 2017, OpenCensus Authors
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
	"strings"

	"google.golang.org/grpc/codes"

	"context"
	"go.opencensus.io/trace"
	"go.opencensus.io/trace/propagation"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/stats"
	"google.golang.org/grpc/status"
)

const traceContextKey = "grpc-trace-bin"

// TagRPC creates a new trace span for the client side of the RPC.
//
// It returns ctx with the new trace span added and a serialization of the
// SpanContext added to the outgoing gRPC metadata.
func (c *ClientHandler) traceTagRPC(ctx context.Context, rti *stats.RPCTagInfo) context.Context {
	name := strings.TrimPrefix(rti.FullMethodName, "/")
	name = strings.Replace(name, "/", ".", -1)
	ctx, span := trace.StartSpan(ctx, name,
		trace.WithSampler(c.StartOptions.Sampler),
		trace.WithSpanKind(trace.SpanKindClient)) // span is ended by traceHandleRPC
	traceContextBinary := propagation.Binary(span.SpanContext())
	return metadata.AppendToOutgoingContext(ctx, traceContextKey, string(traceContextBinary))
}

// TagRPC creates a new trace span for the server side of the RPC.
//
// It checks the incoming gRPC metadata in ctx for a SpanContext, and if
// it finds one, uses that SpanContext as the parent context of the new span.
//
// It returns ctx, with the new trace span added.
func (s *ServerHandler) traceTagRPC(ctx context.Context, rti *stats.RPCTagInfo) context.Context {
	md, _ := metadata.FromIncomingContext(ctx)
	name := strings.TrimPrefix(rti.FullMethodName, "/")
	name = strings.Replace(name, "/", ".", -1)
	traceContext := md[traceContextKey]
	var (
		parent     trace.SpanContext
		haveParent bool
	)
	if len(traceContext) > 0 {
		// Metadata with keys ending in -bin are actually binary. They are base64
		// encoded before being put on the wire, see:
		// https://github.com/grpc/grpc-go/blob/08d6261/Documentation/grpc-metadata.md#storing-binary-data-in-metadata
		traceContextBinary := []byte(traceContext[0])
		parent, haveParent = propagation.FromBinary(traceContextBinary)
		if haveParent && !s.IsPublicEndpoint {
			ctx, _ := trace.StartSpanWithRemoteParent(ctx, name, parent,
				trace.WithSpanKind(trace.SpanKindServer),
				trace.WithSampler(s.StartOptions.Sampler),
			)
			return ctx
		}
	}
	ctx, span := trace.StartSpan(ctx, name,
		trace.WithSpanKind(trace.SpanKindServer),
		trace.WithSampler(s.StartOptions.Sampler))
	if haveParent {
		span.AddLink(trace.Link{TraceID: parent.TraceID, SpanID: parent.SpanID, Type: trace.LinkTypeChild})
	}
	return ctx
}

func traceHandleRPC(ctx context.Context, rs stats.RPCStats) {
	span := trace.FromContext(ctx)
	// TODO: compressed and uncompressed sizes are not populated in every message.
	switch rs := rs.(type) {
	case *stats.Begin:
		span.AddAttributes(
			trace.BoolAttribute("Client", rs.Client),
			trace.BoolAttribute("FailFast", rs.FailFast))
	case *stats.InPayload:
		span.AddMessageReceiveEvent(0 /* TODO: messageID */, int64(rs.Length), int64(rs.WireLength))
	case *stats.OutPayload:
		span.AddMessageSendEvent(0, int64(rs.Length), int64(rs.WireLength))
	case *stats.End:
		if rs.Error != nil {
			s, ok := status.FromError(rs.Error)
			if ok {
				span.SetStatus(trace.Status{Code: int32(s.Code()), Message: s.Message()})
			} else {
				span.SetStatus(trace.Status{Code: int32(codes.Internal), Message: rs.Error.Error()})
			}
		}
		span.End()
	}
}
