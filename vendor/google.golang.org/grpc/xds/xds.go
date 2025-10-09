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

// Package xds contains an implementation of the xDS suite of protocols, to be
// used by gRPC client and server applications.
//
// On the client-side, users simply need to import this package to get all xDS
// functionality. On the server-side, users need to use the GRPCServer type
// exported by this package instead of the regular grpc.Server.
//
// See https://github.com/grpc/grpc-go/tree/master/examples/features/xds for
// example.
package xds

import (
	"fmt"

	"google.golang.org/grpc"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal"
	internaladmin "google.golang.org/grpc/internal/admin"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/xds/csds"

	_ "google.golang.org/grpc/credentials/tls/certprovider/pemfile"           // Register the file watcher certificate provider plugin.
	_ "google.golang.org/grpc/internal/xds/balancer"                          // Register the balancers.
	_ "google.golang.org/grpc/internal/xds/clusterspecifier/rls"              // Register the RLS cluster specifier plugin. Note that this does not register the RLS LB policy.
	_ "google.golang.org/grpc/internal/xds/httpfilter/fault"                  // Register the fault injection filter.
	_ "google.golang.org/grpc/internal/xds/httpfilter/rbac"                   // Register the RBAC filter.
	_ "google.golang.org/grpc/internal/xds/httpfilter/router"                 // Register the router filter.
	_ "google.golang.org/grpc/internal/xds/resolver"                          // Register the xds_resolver.
	_ "google.golang.org/grpc/internal/xds/xdsclient/xdslbregistry/converter" // Register the xDS LB Registry Converters.

	v3statusgrpc "github.com/envoyproxy/go-control-plane/envoy/service/status/v3"
)

var logger = grpclog.Component("xds")

func init() {
	internaladmin.AddService(func(registrar grpc.ServiceRegistrar) (func(), error) {
		var grpcServer *grpc.Server
		switch ss := registrar.(type) {
		case *grpc.Server:
			grpcServer = ss
		case *GRPCServer:
			sss, ok := ss.gs.(*grpc.Server)
			if !ok {
				logger.Warning("grpc server within xds.GRPCServer is not *grpc.Server, CSDS will not be registered")
				return nil, nil
			}
			grpcServer = sss
		default:
			// Returning an error would cause the top level admin.Register() to
			// fail. Log a warning instead.
			logger.Error("Server to register service on is neither a *grpc.Server or a *xds.GRPCServer, CSDS will not be registered")
			return nil, nil
		}

		csdss, err := csds.NewClientStatusDiscoveryServer()
		if err != nil {
			return nil, fmt.Errorf("failed to create csds server: %v", err)
		}
		v3statusgrpc.RegisterClientStatusDiscoveryServiceServer(grpcServer, csdss)
		return csdss.Close, nil
	})
}

// NewXDSResolverWithConfigForTesting creates a new xDS resolver builder using
// the provided xDS bootstrap config instead of the global configuration from
// the supported environment variables.  The resolver.Builder is meant to be
// used in conjunction with the grpc.WithResolvers DialOption.
//
// # Testing Only
//
// This function should ONLY be used for testing and may not work with some
// other features, including the CSDS service.
//
// # Experimental
//
// Notice: This API is EXPERIMENTAL and may be changed or removed in a
// later release.
func NewXDSResolverWithConfigForTesting(bootstrapConfig []byte) (resolver.Builder, error) {
	return internal.NewXDSResolverWithConfigForTesting.(func([]byte) (resolver.Builder, error))(bootstrapConfig)
}
