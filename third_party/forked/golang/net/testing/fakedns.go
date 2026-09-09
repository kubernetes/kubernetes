// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package is copied from Go library net.
// https://golang.org/src/net/dnsclient_unix_test.go
// The original private type FakeDNSServer
// is exported as public type for testing.

package testing

import (
	"context"
	"errors"
	"fmt"
	"net"
	"time"

	_ "testing"

	"golang.org/x/net/dns/dnsmessage"
)

type FakeDNSServer struct {
	ResponseHandler func(n, s string, q dnsmessage.Message, t time.Time) (dnsmessage.Message, error)
	alwaysTCP       bool
}

func (server *FakeDNSServer) DialContext(_ context.Context, n, s string) (net.Conn, error) {
	if server.alwaysTCP || n == "tcp" || n == "tcp4" || n == "tcp6" {
		return &fakeDNSConn{tcp: true, server: server, n: n, s: s}, nil
	}
	return &fakeDNSPacketConn{fakeDNSConn: fakeDNSConn{tcp: false, server: server, n: n, s: s}}, nil
}

type fakeDNSConn struct {
	net.Conn
	tcp    bool
	server *FakeDNSServer
	n      string
	s      string
	q      dnsmessage.Message
	t      time.Time
	buf    []byte
}

func (f *fakeDNSConn) Close() error {
	return nil
}

func (f *fakeDNSConn) Read(b []byte) (int, error) {
	if len(f.buf) > 0 {
		n := copy(b, f.buf)
		f.buf = f.buf[n:]
		return n, nil
	}

	resp, err := f.server.ResponseHandler(f.n, f.s, f.q, f.t)
	if err != nil {
		return 0, err
	}

	bb := make([]byte, 2, 514)
	bb, err = resp.AppendPack(bb)
	if err != nil {
		return 0, fmt.Errorf("cannot marshal DNS message: %v", err)
	}

	if f.tcp {
		l := len(bb) - 2
		bb[0] = byte(l >> 8)
		bb[1] = byte(l)
		f.buf = bb
		return f.Read(b)
	}

	bb = bb[2:]
	if len(b) < len(bb) {
		return 0, errors.New("read would fragment DNS message")
	}

	copy(b, bb)
	return len(bb), nil
}

func (f *fakeDNSConn) Write(b []byte) (int, error) {
	if f.tcp && len(b) >= 2 {
		b = b[2:]
	}
	if f.q.Unpack(b) != nil {
		return 0, fmt.Errorf("cannot unmarshal DNS message fake %s (%d)", f.n, len(b))
	}
	return len(b), nil
}

func (f *fakeDNSConn) SetDeadline(t time.Time) error {
	f.t = t
	return nil
}

type fakeDNSPacketConn struct {
	net.PacketConn
	fakeDNSConn
}

func (f *fakeDNSPacketConn) SetDeadline(t time.Time) error {
	return f.fakeDNSConn.SetDeadline(t)
}

func (f *fakeDNSPacketConn) Close() error {
	return f.fakeDNSConn.Close()
}
