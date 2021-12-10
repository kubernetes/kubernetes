// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package socket

import (
	"errors"
	"net"
	"os"
	"syscall"
)

// A Conn represents a raw connection.
type Conn struct {
	network string
	c       syscall.RawConn
}

// tcpConn is an interface implemented by net.TCPConn.
// It can be used for interface assertions to check if a net.Conn is a TCP connection.
type tcpConn interface {
	SyscallConn() (syscall.RawConn, error)
	SetLinger(int) error
}

var _ tcpConn = (*net.TCPConn)(nil)

// udpConn is an interface implemented by net.UDPConn.
// It can be used for interface assertions to check if a net.Conn is a UDP connection.
type udpConn interface {
	SyscallConn() (syscall.RawConn, error)
	ReadMsgUDP(b, oob []byte) (n, oobn, flags int, addr *net.UDPAddr, err error)
}

var _ udpConn = (*net.UDPConn)(nil)

// ipConn is an interface implemented by net.IPConn.
// It can be used for interface assertions to check if a net.Conn is an IP connection.
type ipConn interface {
	SyscallConn() (syscall.RawConn, error)
	ReadMsgIP(b, oob []byte) (n, oobn, flags int, addr *net.IPAddr, err error)
}

var _ ipConn = (*net.IPConn)(nil)

// NewConn returns a new raw connection.
func NewConn(c net.Conn) (*Conn, error) {
	var err error
	var cc Conn
	switch c := c.(type) {
	case tcpConn:
		cc.network = "tcp"
		cc.c, err = c.SyscallConn()
	case udpConn:
		cc.network = "udp"
		cc.c, err = c.SyscallConn()
	case ipConn:
		cc.network = "ip"
		cc.c, err = c.SyscallConn()
	default:
		return nil, errors.New("unknown connection type")
	}
	if err != nil {
		return nil, err
	}
	return &cc, nil
}

func (o *Option) get(c *Conn, b []byte) (int, error) {
	var operr error
	var n int
	fn := func(s uintptr) {
		n, operr = getsockopt(s, o.Level, o.Name, b)
	}
	if err := c.c.Control(fn); err != nil {
		return 0, err
	}
	return n, os.NewSyscallError("getsockopt", operr)
}

func (o *Option) set(c *Conn, b []byte) error {
	var operr error
	fn := func(s uintptr) {
		operr = setsockopt(s, o.Level, o.Name, b)
	}
	if err := c.c.Control(fn); err != nil {
		return err
	}
	return os.NewSyscallError("setsockopt", operr)
}
