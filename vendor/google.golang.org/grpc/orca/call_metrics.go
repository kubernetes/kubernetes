/*
 *
 * Copyright 2022 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package orca

import (
	"context"
	"sync"

	"google.golang.org/grpc"
	grpcinternal "google.golang.org/grpc/internal"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/orca/internal"
	"google.golang.org/protobuf/proto"
)

// CallMetricsRecorder allows a service method handler to record per-RPC
// metrics.  It contains all utilization-based metrics from
// ServerMetricsRecorder as well as additional request cost metrics.
type CallMetricsRecorder interface {
	ServerMetricsRecorder

	// SetRequestCost sets the relevant server metric.
	SetRequestCost(name string, val float64)
	// DeleteRequestCost deletes the relevant server metric to prevent it
	// from being sent.
	DeleteRequestCost(name string)

	// SetNamedMetric sets the relevant server metric.
	SetNamedMetric(name string, val float64)
	// DeleteNamedMetric deletes the relevant server metric to prevent it
	// from being sent.
	DeleteNamedMetric(name string)
}

type callMetricsRecorderCtxKey struct{}

// CallMetricsRecorderFromContext returns the RPC-specific custom metrics
// recorder embedded in the provided RPC context.
//
// Returns nil if no custom metrics recorder is found in the provided context,
// which will be the case when custom metrics reporting is not enabled.
func CallMetricsRecorderFromContext(ctx context.Context) CallMetricsRecorder {
	rw, ok := ctx.Value(callMetricsRecorderCtxKey{}).(*recorderWrapper)
	if !ok {
		return nil
	}
	return rw.recorder()
}

// recorderWrapper is a wrapper around a CallMetricsRecorder to ensure that
// concurrent calls to CallMetricsRecorderFromContext() results in only one
// allocation of the underlying metrics recorder, while also allowing for lazy
// initialization of the recorder itself.
type recorderWrapper struct {
	once sync.Once
	r    CallMetricsRecorder
	smp  ServerMetricsProvider
}

func (rw *recorderWrapper) recorder() CallMetricsRecorder {
	rw.once.Do(func() {
		rw.r = newServerMetricsRecorder()
	})
	return rw.r
}

// setTrailerMetadata adds a trailer metadata entry with key being set to
// `internal.TrailerMetadataKey` and value being set to the binary-encoded
// orca.OrcaLoadReport protobuf message.
//
// This function is called from the unary and streaming interceptors defined
// above. Any errors encountered here are not propagated to the caller because
// they are ignored there. Hence we simply log any errors encountered here at
// warning level, and return nothing.
func (rw *recorderWrapper) setTrailerMetadata(ctx context.Context) {
	var sm *ServerMetrics
	if rw.smp != nil {
		sm = rw.smp.ServerMetrics()
		sm.merge(rw.r.ServerMetrics())
	} else {
		sm = rw.r.ServerMetrics()
	}

	b, err := proto.Marshal(sm.toLoadReportProto())
	if err != nil {
		logger.Warningf("Failed to marshal load report: %v", err)
		return
	}
	if err := grpc.SetTrailer(ctx, metadata.Pairs(internal.TrailerMetadataKey, string(b))); err != nil {
		logger.Warningf("Failed to set trailer metadata: %v", err)
	}
}

var joinServerOptions = grpcinternal.JoinServerOptions.(func(...grpc.ServerOption) grpc.ServerOption)

// CallMetricsServerOption returns a server option which enables the reporting
// of per-RPC custom backend metrics for unary and streaming RPCs.
//
// Server applications interested in injecting custom backend metrics should
// pass the server option returned from this function as the first argument to
// grpc.NewServer().
//
// Subsequently, server RPC handlers can retrieve a reference to the RPC
// specific custom metrics recorder [CallMetricsRecorder] to be used, via a call
// to CallMetricsRecorderFromContext(), and inject custom metrics at any time
// during the RPC lifecycle.
//
// The injected custom metrics will be sent as part of trailer metadata, as a
// binary-encoded [ORCA LoadReport] protobuf message, with the metadata key
// being set be "endpoint-load-metrics-bin".
//
// If a non-nil ServerMetricsProvider is provided, the gRPC server will
// transmit the metrics it provides, overwritten by any per-RPC metrics given
// to the CallMetricsRecorder.  A ServerMetricsProvider is typically obtained
// by calling NewServerMetricsRecorder.
//
// [ORCA LoadReport]: https://github.com/cncf/xds/blob/main/xds/data/orca/v3/orca_load_report.proto#L15
func CallMetricsServerOption(smp ServerMetricsProvider) grpc.ServerOption {
	return joinServerOptions(grpc.ChainUnaryInterceptor(unaryInt(smp)), grpc.ChainStreamInterceptor(streamInt(smp)))
}

func unaryInt(smp ServerMetricsProvider) func(ctx context.Context, req any, _ *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
	return func(ctx context.Context, req any, _ *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		// We don't allocate the metric recorder here. It will be allocated the
		// first time the user calls CallMetricsRecorderFromContext().
		rw := &recorderWrapper{smp: smp}
		ctxWithRecorder := newContextWithRecorderWrapper(ctx, rw)

		resp, err := handler(ctxWithRecorder, req)

		// It is safe to access the underlying metric recorder inside the wrapper at
		// this point, as the user's RPC handler is done executing, and therefore
		// there will be no more calls to CallMetricsRecorderFromContext(), which is
		// where the metric recorder is lazy allocated.
		if rw.r != nil {
			rw.setTrailerMetadata(ctx)
		}
		return resp, err
	}
}

func streamInt(smp ServerMetricsProvider) func(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	return func(srv any, ss grpc.ServerStream, _ *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		// We don't allocate the metric recorder here. It will be allocated the
		// first time the user calls CallMetricsRecorderFromContext().
		rw := &recorderWrapper{smp: smp}
		ws := &wrappedStream{
			ServerStream: ss,
			ctx:          newContextWithRecorderWrapper(ss.Context(), rw),
		}

		err := handler(srv, ws)

		// It is safe to access the underlying metric recorder inside the wrapper at
		// this point, as the user's RPC handler is done executing, and therefore
		// there will be no more calls to CallMetricsRecorderFromContext(), which is
		// where the metric recorder is lazy allocated.
		if rw.r != nil {
			rw.setTrailerMetadata(ss.Context())
		}
		return err
	}
}

func newContextWithRecorderWrapper(ctx context.Context, r *recorderWrapper) context.Context {
	return context.WithValue(ctx, callMetricsRecorderCtxKey{}, r)
}

// wrappedStream wraps the grpc.ServerStream received by the streaming
// interceptor. Overrides only the Context() method to return a context which
// contains a reference to the CallMetricsRecorder corresponding to this
// stream.
type wrappedStream struct {
	grpc.ServerStream
	ctx context.Context
}

func (w *wrappedStream) Context() context.Context {
	return w.ctx
}
