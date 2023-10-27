/*
 *
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package service is a utility for calling the S2A handshaker service.
package service

import (
	"context"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	"google.golang.org/appengine"
	"google.golang.org/appengine/socket"
	grpc "google.golang.org/grpc"
	"google.golang.org/grpc/grpclog"
)

// An environment variable, if true, opportunistically use AppEngine-specific dialer to call S2A.
const enableAppEngineDialerEnv = "S2A_ENABLE_APP_ENGINE_DIALER"

var (
	// appEngineDialerHook is an AppEngine-specific dial option that is set
	// during init time. If nil, then the application is not running on Google
	// AppEngine.
	appEngineDialerHook func(context.Context) grpc.DialOption
	// mu guards hsConnMap and hsDialer.
	mu sync.Mutex
	// hsConnMap represents a mapping from an S2A handshaker service address
	// to a corresponding connection to an S2A handshaker service instance.
	hsConnMap = make(map[string]*grpc.ClientConn)
	// hsDialer will be reassigned in tests.
	hsDialer = grpc.Dial
)

func init() {
	if !appengine.IsAppEngine() && !appengine.IsDevAppServer() {
		return
	}
	appEngineDialerHook = func(ctx context.Context) grpc.DialOption {
		return grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return socket.DialTimeout(ctx, "tcp", addr, timeout)
		})
	}
}

// Dial dials the S2A handshaker service. If a connection has already been
// established, this function returns it. Otherwise, a new connection is
// created.
func Dial(handshakerServiceAddress string) (*grpc.ClientConn, error) {
	mu.Lock()
	defer mu.Unlock()

	hsConn, ok := hsConnMap[handshakerServiceAddress]
	if !ok {
		// Create a new connection to the S2A handshaker service. Note that
		// this connection stays open until the application is closed.
		grpcOpts := []grpc.DialOption{
			grpc.WithInsecure(),
		}
		if enableAppEngineDialer() && appEngineDialerHook != nil {
			if grpclog.V(1) {
				grpclog.Info("Using AppEngine-specific dialer to talk to S2A.")
			}
			grpcOpts = append(grpcOpts, appEngineDialerHook(context.Background()))
		}
		var err error
		hsConn, err = hsDialer(handshakerServiceAddress, grpcOpts...)
		if err != nil {
			return nil, err
		}
		hsConnMap[handshakerServiceAddress] = hsConn
	}
	return hsConn, nil
}

func enableAppEngineDialer() bool {
	if strings.ToLower(os.Getenv(enableAppEngineDialerEnv)) == "true" {
		return true
	}
	return false
}
