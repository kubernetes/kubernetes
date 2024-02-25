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
	"net"
	"os"
	"sync"
	"sync/atomic"

	"google.golang.org/grpc"
	"k8s.io/klog/v2"
)

type grpcServer struct {
	logger        klog.Logger
	grpcVerbosity int
	wg            sync.WaitGroup
	endpoint      endpoint
	server        *grpc.Server
	requestID     int64
}

type registerService func(s *grpc.Server)

// endpoint defines where to listen for incoming connections.
// The listener always gets closed when shutting down.
//
// If the listener is not set, a new listener for a Unix domain socket gets
// created at the path.
//
// If the path is non-empty, then the socket will get removed when shutting
// down, regardless of who created the listener.
type endpoint struct {
	path     string
	listener net.Listener
}

// startGRPCServer sets up the GRPC server on a Unix domain socket and spawns a goroutine
// which handles requests for arbitrary services.
func startGRPCServer(logger klog.Logger, grpcVerbosity int, interceptors []grpc.UnaryServerInterceptor, endpoint endpoint, services ...registerService) (*grpcServer, error) {
	s := &grpcServer{
		logger:        logger,
		endpoint:      endpoint,
		grpcVerbosity: grpcVerbosity,
	}

	listener := endpoint.listener
	if listener == nil {
		// Remove any (probably stale) existing socket.
		if err := os.Remove(endpoint.path); err != nil && !os.IsNotExist(err) {
			return nil, fmt.Errorf("remove Unix domain socket: %v", err)
		}

		// Now we can use the endpoint for listening.
		l, err := net.Listen("unix", endpoint.path)
		if err != nil {
			return nil, fmt.Errorf("listen on %q: %v", endpoint.path, err)
		}
		listener = l
	}

	// Run a gRPC server. It will close the listening socket when
	// shutting down, so we don't need to do that.
	var opts []grpc.ServerOption
	var finalInterceptors []grpc.UnaryServerInterceptor
	if grpcVerbosity >= 0 {
		finalInterceptors = append(finalInterceptors, s.interceptor)
	}
	finalInterceptors = append(finalInterceptors, interceptors...)
	opts = append(opts, grpc.ChainUnaryInterceptor(finalInterceptors...))
	s.server = grpc.NewServer(opts...)
	for _, service := range services {
		service(s.server)
	}
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		err := s.server.Serve(listener)
		if err != nil {
			logger.Error(err, "GRPC server failed")
		} else {
			logger.V(3).Info("GRPC server terminated gracefully")
		}
	}()

	logger.Info("GRPC server started")
	return s, nil
}

// interceptor is called for each request. It creates a logger with a unique,
// sequentially increasing request ID and adds that logger to the context. It
// also logs request and response.
func (s *grpcServer) interceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
	requestID := atomic.AddInt64(&s.requestID, 1)
	logger := klog.LoggerWithValues(s.logger, "requestID", requestID)
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
		logger.Error(err, "handling request failed", "request", req)
	} else {
		logger.V(s.grpcVerbosity).Info("handling request succeeded", "response", resp)
	}
	return
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
	if s.endpoint.path != "" {
		if err := os.Remove(s.endpoint.path); err != nil && !os.IsNotExist(err) {
			s.logger.Error(err, "remove Unix socket")
		}
	}
	s.logger.V(3).Info("GRPC server stopped")
}
