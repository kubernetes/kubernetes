/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kubeletplugin

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"google.golang.org/grpc"
	"k8s.io/klog/v2"
)

var requestID int64

type grpcServer struct {
	grpcVerbosity int
	wg            sync.WaitGroup
	endpoint      endpoint
	server        *grpc.Server
}

type registerService func(s *grpc.Server)

// startGRPCServer sets up the GRPC server on a Unix domain socket and spawns a goroutine
// which handles requests for arbitrary services.
//
// errHandler gets invoked in the background when errors are encountered there.
// They are fatal and should cause the process to exit.
func startGRPCServer(logger klog.Logger, grpcVerbosity int, unaryInterceptors []grpc.UnaryServerInterceptor, streamInterceptors []grpc.StreamServerInterceptor, endpoint endpoint, errHandler func(ctx context.Context, err error), services ...registerService) (*grpcServer, error) {
	ctx := klog.NewContext(context.Background(), logger)

	s := &grpcServer{
		endpoint:      endpoint,
		grpcVerbosity: grpcVerbosity,
	}

	listener, err := endpoint.listen(ctx)
	if err != nil {
		return nil, fmt.Errorf("listen on %q: %w", s.endpoint.path(), err)
	}

	// Run a gRPC server. It will close the listening socket when
	// shutting down, so we don't need to do that.
	// The per-context logger always gets initialized because
	// there might be log output inside the method implementations.
	var opts []grpc.ServerOption
	finalUnaryInterceptors := []grpc.UnaryServerInterceptor{
		unaryContextInterceptor(ctx),
		s.interceptor,
	}
	finalUnaryInterceptors = append(finalUnaryInterceptors, unaryInterceptors...)
	finalStreamInterceptors := []grpc.StreamServerInterceptor{
		streamContextInterceptor(ctx),
		s.streamInterceptor,
	}
	finalStreamInterceptors = append(finalStreamInterceptors, streamInterceptors...)
	opts = append(opts, grpc.ChainUnaryInterceptor(finalUnaryInterceptors...))
	opts = append(opts, grpc.ChainStreamInterceptor(finalStreamInterceptors...))
	s.server = grpc.NewServer(opts...)
	for _, service := range services {
		service(s.server)
	}
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		err := s.server.Serve(listener)
		if err != nil {
			errHandler(ctx, err)
		} else {
			logger.V(3).Info("GRPC server terminated gracefully", "endpoint", endpoint.path())
		}
	}()

	logger.V(3).Info("GRPC server started", "endpoint", endpoint.path())
	return s, nil
}

// unaryContextInterceptor injects values from the context into the context
// used by the call chain.
func unaryContextInterceptor(valueCtx context.Context) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		ctx = mergeContexts(ctx, valueCtx)
		return handler(ctx, req)
	}
}

// streamContextInterceptor does the same as UnaryContextInterceptor for streams.
func streamContextInterceptor(valueCtx context.Context) grpc.StreamServerInterceptor {
	return func(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		ctx := mergeContexts(ss.Context(), valueCtx)
		return handler(srv, mergeServerStream{ServerStream: ss, ctx: ctx})
	}
}

type mergeServerStream struct {
	grpc.ServerStream
	ctx context.Context
}

func (m mergeServerStream) Context() context.Context {
	return m.ctx
}

// mergeContexts creates a new context where cancellation is handled by the
// root context. The values stored by the value context are used as fallback if
// the root context doesn't have a certain value.
func mergeContexts(rootCtx, valueCtx context.Context) context.Context {
	return mergeCtx{
		Context:  rootCtx,
		valueCtx: valueCtx,
	}
}

type mergeCtx struct {
	context.Context
	valueCtx context.Context
}

func (m mergeCtx) Value(i interface{}) interface{} {
	if v := m.Context.Value(i); v != nil {
		return v
	}
	return m.valueCtx.Value(i)
}

// interceptor is called for each request. It creates a logger with a unique,
// sequentially increasing request ID and adds that logger to the context. It
// also logs request and response.
func (s *grpcServer) interceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
	requestID := atomic.AddInt64(&requestID, 1)
	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "requestID", requestID, "method", info.FullMethod)
	ctx = klog.NewContext(ctx, logger)
	logger.V(s.grpcVerbosity).Info("handling request", "request", req)
	defer func() {
		if r := recover(); r != nil {
			logger.Error(nil, "handling request panicked", "panic", r, "request", req)
			panic(r)
		}
	}()
	resp, err = handler(ctx, req)
	if err != nil {
		logger.Error(err, "handling request failed")
	} else {
		logger.V(s.grpcVerbosity).Info("handling request succeeded", "response", resp)
	}
	return
}

func (s *grpcServer) streamInterceptor(server interface{}, stream grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	requestID := atomic.AddInt64(&requestID, 1)
	ctx := stream.Context()
	logger := klog.FromContext(ctx)
	logger = klog.LoggerWithValues(logger, "requestID", requestID, "method", info.FullMethod)
	ctx = klog.NewContext(ctx, logger)
	stream = logStream{
		ServerStream:  stream,
		ctx:           ctx,
		grpcVerbosity: s.grpcVerbosity,
	}
	logger.V(s.grpcVerbosity).Info("handling stream")
	err := handler(server, stream)
	if err != nil {
		logger.Error(err, "handling stream failed")
	} else {
		logger.V(s.grpcVerbosity).Info("handling stream succeeded")
	}
	return err

}

type logStream struct {
	grpc.ServerStream
	ctx           context.Context
	grpcVerbosity int
}

func (l logStream) Context() context.Context {
	return l.ctx
}

func (l logStream) SendMsg(msg interface{}) error {
	logger := klog.FromContext(l.ctx)
	logger.V(l.grpcVerbosity).Info("sending stream message", "message", msg)
	err := l.ServerStream.SendMsg(msg)
	if err != nil {
		logger.Error(err, "sending stream message failed")
	} else {
		logger.V(l.grpcVerbosity).Info("sending stream message succeeded")
	}
	return err
}

// stop ensures that the server is not running anymore and cleans up all resources.
// It is idempotent and may be called with a nil pointer.
func (s *grpcServer) stop() {
	if s == nil {
		return
	}
	if s.server != nil {
		s.server.Stop()
	}
	s.wg.Wait()
	s.server = nil
}
