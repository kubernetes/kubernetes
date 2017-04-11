// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build !appengine

package socket

import (
	"net"
	"time"

	"golang.org/x/net/context"
)

// Dial connects to the address addr on the network protocol.
// The address format is host:port, where host may be a hostname or an IP address.
// Known protocols are "tcp" and "udp".
// The returned connection satisfies net.Conn, and is valid while ctx is valid;
// if the connection is to be used after ctx becomes invalid, invoke SetContext
// with the new context.
func Dial(ctx context.Context, protocol, addr string) (*Conn, error) {
	conn, err := net.Dial(protocol, addr)
	if err != nil {
		return nil, err
	}
	return &Conn{conn}, nil
}

// DialTimeout is like Dial but takes a timeout.
// The timeout includes name resolution, if required.
func DialTimeout(ctx context.Context, protocol, addr string, timeout time.Duration) (*Conn, error) {
	conn, err := net.DialTimeout(protocol, addr, timeout)
	if err != nil {
		return nil, err
	}
	return &Conn{conn}, nil
}

// LookupIP returns the given host's IP addresses.
func LookupIP(ctx context.Context, host string) (addrs []net.IP, err error) {
	return net.LookupIP(host)
}

// Conn represents a socket connection.
// It implements net.Conn.
type Conn struct {
	net.Conn
}

// SetContext sets the context that is used by this Conn.
// It is usually used only when using a Conn that was created in a different context,
// such as when a connection is created during a warmup request but used while
// servicing a user request.
func (cn *Conn) SetContext(ctx context.Context) {
	// This function is not required in App Engine "flexible environment".
}

// KeepAlive signals that the connection is still in use.
// It may be called to prevent the socket being closed due to inactivity.
func (cn *Conn) KeepAlive() error {
	// This function is not required in App Engine "flexible environment".
	return nil
}
