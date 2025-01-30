// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proxy

import (
	"context"
	"net"
)

// A ContextDialer dials using a context.
type ContextDialer interface {
	DialContext(ctx context.Context, network, address string) (net.Conn, error)
}

// Dial works like DialContext on net.Dialer but using a dialer returned by FromEnvironment.
//
// The passed ctx is only used for returning the Conn, not the lifetime of the Conn.
//
// Custom dialers (registered via RegisterDialerType) that do not implement ContextDialer
// can leak a goroutine for as long as it takes the underlying Dialer implementation to timeout.
//
// A Conn returned from a successful Dial after the context has been cancelled will be immediately closed.
func Dial(ctx context.Context, network, address string) (net.Conn, error) {
	d := FromEnvironment()
	if xd, ok := d.(ContextDialer); ok {
		return xd.DialContext(ctx, network, address)
	}
	return dialContext(ctx, d, network, address)
}

// WARNING: this can leak a goroutine for as long as the underlying Dialer implementation takes to timeout
// A Conn returned from a successful Dial after the context has been cancelled will be immediately closed.
func dialContext(ctx context.Context, d Dialer, network, address string) (net.Conn, error) {
	var (
		conn net.Conn
		done = make(chan struct{}, 1)
		err  error
	)
	go func() {
		conn, err = d.Dial(network, address)
		close(done)
		if conn != nil && ctx.Err() != nil {
			conn.Close()
		}
	}()
	select {
	case <-ctx.Done():
		err = ctx.Err()
	case <-done:
	}
	return conn, err
}
