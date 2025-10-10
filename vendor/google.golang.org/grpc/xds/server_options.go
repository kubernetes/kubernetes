/*
 *
 * Copyright 2021 gRPC authors.
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
	"net"

	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/internal/xds/bootstrap"
	"google.golang.org/grpc/internal/xds/xdsclient"
)

type serverOptions struct {
	modeCallback         ServingModeCallbackFunc
	clientPoolForTesting *xdsclient.Pool
}

type serverOption struct {
	grpc.EmptyServerOption
	apply func(*serverOptions)
}

// ServingModeCallback returns a grpc.ServerOption which allows users to
// register a callback to get notified about serving mode changes.
func ServingModeCallback(cb ServingModeCallbackFunc) grpc.ServerOption {
	return &serverOption{apply: func(o *serverOptions) { o.modeCallback = cb }}
}

// ServingModeCallbackFunc is the callback that users can register to get
// notified about the server's serving mode changes. The callback is invoked
// with the address of the listener and its new mode.
//
// Users must not perform any blocking operations in this callback.
type ServingModeCallbackFunc func(addr net.Addr, args ServingModeChangeArgs)

// ServingModeChangeArgs wraps the arguments passed to the serving mode callback
// function.
type ServingModeChangeArgs struct {
	// Mode is the new serving mode of the server listener.
	Mode connectivity.ServingMode
	// Err is set to a non-nil error if the server has transitioned into
	// not-serving mode.
	Err error
}

// BootstrapContentsForTesting returns a grpc.ServerOption which allows users
// to inject a bootstrap configuration used by only this server, instead of the
// global configuration from the environment variables.
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
func BootstrapContentsForTesting(bootstrapContents []byte) grpc.ServerOption {
	config, err := bootstrap.NewConfigFromContents(bootstrapContents)
	if err != nil {
		logger.Warningf("Failed to parse bootstrap contents %s for server options: %v", string(bootstrapContents), err)
		return &serverOption{apply: func(o *serverOptions) { o.clientPoolForTesting = nil }}
	}
	return ClientPoolForTesting(xdsclient.NewPool(config))
}

// ClientPoolForTesting returns a grpc.ServerOption with the pool for xds
// clients. It allows users to set a pool for xDS clients sharing the bootstrap
// contents for this server.
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
func ClientPoolForTesting(pool *xdsclient.Pool) grpc.ServerOption {
	return &serverOption{apply: func(o *serverOptions) { o.clientPoolForTesting = pool }}
}
