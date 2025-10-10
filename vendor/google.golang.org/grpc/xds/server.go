/*
 *
 * Copyright 2020 gRPC authors.
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

package xds

import (
	"context"
	"errors"
	"fmt"
	"net"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	estats "google.golang.org/grpc/experimental/stats"
	"google.golang.org/grpc/internal"
	internalgrpclog "google.golang.org/grpc/internal/grpclog"
	"google.golang.org/grpc/internal/grpcsync"
	iresolver "google.golang.org/grpc/internal/resolver"
	istats "google.golang.org/grpc/internal/stats"
	"google.golang.org/grpc/internal/transport"
	"google.golang.org/grpc/internal/xds/bootstrap"
	"google.golang.org/grpc/internal/xds/server"
	"google.golang.org/grpc/internal/xds/xdsclient"
	"google.golang.org/grpc/internal/xds/xdsclient/xdsresource"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

const serverPrefix = "[xds-server %p] "

var (
	// These will be overridden in unit tests.
	xdsClientPool = xdsclient.DefaultPool
	newGRPCServer = func(opts ...grpc.ServerOption) grpcServer {
		return grpc.NewServer(opts...)
	}
)

// grpcServer contains methods from grpc.Server which are used by the
// GRPCServer type here. This is useful for overriding in unit tests.
type grpcServer interface {
	RegisterService(*grpc.ServiceDesc, any)
	Serve(net.Listener) error
	Stop()
	GracefulStop()
	GetServiceInfo() map[string]grpc.ServiceInfo
}

// GRPCServer wraps a gRPC server and provides server-side xDS functionality, by
// communication with a management server using xDS APIs. It implements the
// grpc.ServiceRegistrar interface and can be passed to service registration
// functions in IDL generated code.
type GRPCServer struct {
	gs             grpcServer
	quit           *grpcsync.Event
	logger         *internalgrpclog.PrefixLogger
	opts           *serverOptions
	xdsC           xdsclient.XDSClient
	xdsClientClose func()
}

// NewGRPCServer creates an xDS-enabled gRPC server using the passed in opts.
// The underlying gRPC server has no service registered and has not started to
// accept requests yet.
func NewGRPCServer(opts ...grpc.ServerOption) (*GRPCServer, error) {
	newOpts := []grpc.ServerOption{
		grpc.ChainUnaryInterceptor(xdsUnaryInterceptor),
		grpc.ChainStreamInterceptor(xdsStreamInterceptor),
	}
	newOpts = append(newOpts, opts...)
	s := &GRPCServer{
		gs:   newGRPCServer(newOpts...),
		quit: grpcsync.NewEvent(),
	}
	s.handleServerOptions(opts)

	var mrl estats.MetricsRecorder
	mrl = istats.NewMetricsRecorderList(nil)
	if srv, ok := s.gs.(*grpc.Server); ok { // Will hit in prod but not for testing.
		mrl = internal.MetricsRecorderForServer.(func(*grpc.Server) estats.MetricsRecorder)(srv)
	}

	// Initializing the xDS client upfront (instead of at serving time)
	// simplifies the code by eliminating the need for a mutex to protect the
	// xdsC and xdsClientClose fields.
	pool := xdsClientPool
	if s.opts.clientPoolForTesting != nil {
		pool = s.opts.clientPoolForTesting
	}
	xdsClient, xdsClientClose, err := pool.NewClient(xdsclient.NameForServer, mrl)
	if err != nil {
		return nil, fmt.Errorf("xDS client creation failed: %v", err)
	}

	// Validate the bootstrap configuration for server specific fields.

	// Listener resource name template is mandatory on the server side.
	cfg := xdsClient.BootstrapConfig()
	if cfg.ServerListenerResourceNameTemplate() == "" {
		xdsClientClose()
		return nil, errors.New("missing server_listener_resource_name_template in the bootstrap configuration")
	}

	s.xdsC = xdsClient
	s.xdsClientClose = xdsClientClose

	s.logger = internalgrpclog.NewPrefixLogger(logger, fmt.Sprintf(serverPrefix, s))
	s.logger.Infof("Created xds.GRPCServer")

	return s, nil
}

// handleServerOptions iterates through the list of server options passed in by
// the user, and handles the xDS server specific options.
func (s *GRPCServer) handleServerOptions(opts []grpc.ServerOption) {
	so := s.defaultServerOptions()
	for _, opt := range opts {
		if o, ok := opt.(*serverOption); ok {
			o.apply(so)
		}
	}
	s.opts = so
}

func (s *GRPCServer) defaultServerOptions() *serverOptions {
	return &serverOptions{
		// A default serving mode change callback which simply logs at the
		// default-visible log level. This will be used if the application does not
		// register a mode change callback.
		//
		// Note that this means that `s.opts.modeCallback` will never be nil and can
		// safely be invoked directly from `handleServingModeChanges`.
		modeCallback: s.loggingServerModeChangeCallback,
	}
}

func (s *GRPCServer) loggingServerModeChangeCallback(addr net.Addr, args ServingModeChangeArgs) {
	switch args.Mode {
	case connectivity.ServingModeServing:
		s.logger.Errorf("Listener %q entering mode: %q", addr.String(), args.Mode)
	case connectivity.ServingModeNotServing:
		s.logger.Errorf("Listener %q entering mode: %q due to error: %v", addr.String(), args.Mode, args.Err)
	}
}

// RegisterService registers a service and its implementation to the underlying
// gRPC server. It is called from the IDL generated code. This must be called
// before invoking Serve.
func (s *GRPCServer) RegisterService(sd *grpc.ServiceDesc, ss any) {
	s.gs.RegisterService(sd, ss)
}

// GetServiceInfo returns a map from service names to ServiceInfo.
// Service names include the package names, in the form of <package>.<service>.
func (s *GRPCServer) GetServiceInfo() map[string]grpc.ServiceInfo {
	return s.gs.GetServiceInfo()
}

// Serve gets the underlying gRPC server to accept incoming connections on the
// listener lis, which is expected to be listening on a TCP port.
//
// A connection to the management server, to receive xDS configuration, is
// initiated here.
//
// Serve will return a non-nil error unless Stop or GracefulStop is called.
func (s *GRPCServer) Serve(lis net.Listener) error {
	s.logger.Infof("Serve() passed a net.Listener on %s", lis.Addr().String())
	if _, ok := lis.Addr().(*net.TCPAddr); !ok {
		return fmt.Errorf("xds: GRPCServer expects listener to return a net.TCPAddr. Got %T", lis.Addr())
	}

	if s.quit.HasFired() {
		return grpc.ErrServerStopped
	}

	// The server listener resource name template from the bootstrap
	// configuration contains a template for the name of the Listener resource
	// to subscribe to for a gRPC server. If the token `%s` is present in the
	// string, it will be replaced with the server's listening "IP:port" (e.g.,
	// "0.0.0.0:8080", "[::]:8080").
	cfg := s.xdsC.BootstrapConfig()
	name := bootstrap.PopulateResourceTemplate(cfg.ServerListenerResourceNameTemplate(), lis.Addr().String())

	// Create a listenerWrapper which handles all functionality required by
	// this particular instance of Serve().
	lw := server.NewListenerWrapper(server.ListenerWrapperParams{
		Listener:             lis,
		ListenerResourceName: name,
		XDSClient:            s.xdsC,
		ModeCallback: func(addr net.Addr, mode connectivity.ServingMode, err error) {
			s.opts.modeCallback(addr, ServingModeChangeArgs{
				Mode: mode,
				Err:  err,
			})
		},
	})
	return s.gs.Serve(lw)
}

// Stop stops the underlying gRPC server. It immediately closes all open
// connections. It cancels all active RPCs on the server side and the
// corresponding pending RPCs on the client side will get notified by connection
// errors.
func (s *GRPCServer) Stop() {
	s.quit.Fire()
	s.gs.Stop()
	if s.xdsC != nil {
		s.xdsClientClose()
	}
}

// GracefulStop stops the underlying gRPC server gracefully. It stops the server
// from accepting new connections and RPCs and blocks until all the pending RPCs
// are finished.
func (s *GRPCServer) GracefulStop() {
	s.quit.Fire()
	s.gs.GracefulStop()
	if s.xdsC != nil {
		s.xdsClientClose()
	}
}

// routeAndProcess routes the incoming RPC to a configured route in the route
// table and also processes the RPC by running the incoming RPC through any HTTP
// Filters configured.
func routeAndProcess(ctx context.Context) error {
	conn := transport.GetConnection(ctx)
	cw, ok := conn.(interface {
		UsableRouteConfiguration() xdsresource.UsableRouteConfiguration
	})
	if !ok {
		return errors.New("missing virtual hosts in incoming context")
	}

	rc := cw.UsableRouteConfiguration()
	// Error out at routing l7 level with a status code UNAVAILABLE, represents
	// an nack before usable route configuration or resource not found for RDS
	// or error combining LDS + RDS (Shouldn't happen).
	if rc.Err != nil {
		if logger.V(2) {
			logger.Infof("RPC on connection with xDS Configuration error: %v", rc.Err)
		}
		return status.Error(codes.Unavailable, fmt.Sprintf("error from xDS configuration for matched route configuration: %v", rc.Err))
	}

	mn, ok := grpc.Method(ctx)
	if !ok {
		return errors.New("missing method name in incoming context")
	}
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return errors.New("missing metadata in incoming context")
	}
	// A41 added logic to the core grpc implementation to guarantee that once
	// the RPC gets to this point, there will be a single, unambiguous authority
	// present in the header map.
	authority := md.Get(":authority")
	vh := xdsresource.FindBestMatchingVirtualHostServer(authority[0], rc.VHS)
	if vh == nil {
		return rc.StatusErrWithNodeID(codes.Unavailable, "the incoming RPC did not match a configured Virtual Host")
	}

	var rwi *xdsresource.RouteWithInterceptors
	rpcInfo := iresolver.RPCInfo{
		Context: ctx,
		Method:  mn,
	}
	for _, r := range vh.Routes {
		if r.M.Match(rpcInfo) {
			// "NonForwardingAction is expected for all Routes used on
			// server-side; a route with an inappropriate action causes RPCs
			// matching that route to fail with UNAVAILABLE." - A36
			if r.ActionType != xdsresource.RouteActionNonForwardingAction {
				return rc.StatusErrWithNodeID(codes.Unavailable, "the incoming RPC matched to a route that was not of action type non forwarding")
			}
			rwi = &r
			break
		}
	}
	if rwi == nil {
		return rc.StatusErrWithNodeID(codes.Unavailable, "the incoming RPC did not match a configured Route")
	}
	for _, interceptor := range rwi.Interceptors {
		if err := interceptor.AllowRPC(ctx); err != nil {
			return rc.StatusErrWithNodeID(codes.PermissionDenied, "Incoming RPC is not allowed: %v", err)
		}
	}
	return nil
}

// xdsUnaryInterceptor is the unary interceptor added to the gRPC server to
// perform any xDS specific functionality on unary RPCs.
func xdsUnaryInterceptor(ctx context.Context, req any, _ *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp any, err error) {
	if err := routeAndProcess(ctx); err != nil {
		return nil, err
	}
	return handler(ctx, req)
}

// xdsStreamInterceptor is the stream interceptor added to the gRPC server to
// perform any xDS specific functionality on streaming RPCs.
func xdsStreamInterceptor(srv any, ss grpc.ServerStream, _ *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	if err := routeAndProcess(ss.Context()); err != nil {
		return err
	}
	return handler(srv, ss)
}
