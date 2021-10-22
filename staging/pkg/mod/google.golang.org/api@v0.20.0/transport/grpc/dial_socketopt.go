// Copyright 2019 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.11,linux

package grpc

import (
	"context"
	"net"
	"syscall"

	"golang.org/x/sys/unix"
	"google.golang.org/grpc"
)

const (
	// defaultTCPUserTimeout is the default TCP_USER_TIMEOUT socket option. By
	// default is 20 seconds.
	tcpUserTimeoutMilliseconds = 20000
)

func init() {
	// timeoutDialerOption is a grpc.DialOption that contains dialer with
	// socket option TCP_USER_TIMEOUT. This dialer requires go versions 1.11+.
	timeoutDialerOption = grpc.WithContextDialer(dialTCPUserTimeout)
}

func dialTCPUserTimeout(ctx context.Context, addr string) (net.Conn, error) {
	control := func(network, address string, c syscall.RawConn) error {
		var syscallErr error
		controlErr := c.Control(func(fd uintptr) {
			syscallErr = syscall.SetsockoptInt(
				int(fd), syscall.IPPROTO_TCP, unix.TCP_USER_TIMEOUT, tcpUserTimeoutMilliseconds)
		})
		if syscallErr != nil {
			return syscallErr
		}
		if controlErr != nil {
			return controlErr
		}
		return nil
	}
	d := &net.Dialer{
		Control: control,
	}
	return d.DialContext(ctx, "tcp", addr)
}
