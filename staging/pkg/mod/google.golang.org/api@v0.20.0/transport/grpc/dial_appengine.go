// Copyright 2016 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package grpc

import (
	"context"
	"net"
	"time"

	"google.golang.org/appengine"
	"google.golang.org/appengine/socket"
	"google.golang.org/grpc"
)

func init() {
	// NOTE: dev_appserver doesn't currently support SSL.
	// When it does, this code can be removed.
	if appengine.IsDevAppServer() {
		return
	}

	appengineDialerHook = func(ctx context.Context) grpc.DialOption {
		return grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return socket.DialTimeout(ctx, "tcp", addr, timeout)
		})
	}
}
