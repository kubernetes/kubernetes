// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris windows

package socket

import (
	"os"
	"syscall"
)

func (c *Conn) recvMsg(m *Message, flags int) error {
	var h msghdr
	vs := make([]iovec, len(m.Buffers))
	var sa []byte
	if c.network != "tcp" {
		sa = make([]byte, sizeofSockaddrInet6)
	}
	h.pack(vs, m.Buffers, m.OOB, sa)
	var operr error
	var n int
	fn := func(s uintptr) bool {
		n, operr = recvmsg(s, &h, flags)
		if operr == syscall.EAGAIN {
			return false
		}
		return true
	}
	if err := c.c.Read(fn); err != nil {
		return err
	}
	if operr != nil {
		return os.NewSyscallError("recvmsg", operr)
	}
	if c.network != "tcp" {
		var err error
		m.Addr, err = parseInetAddr(sa[:], c.network)
		if err != nil {
			return err
		}
	}
	m.N = n
	m.NN = h.controllen()
	m.Flags = h.flags()
	return nil
}

func (c *Conn) sendMsg(m *Message, flags int) error {
	var h msghdr
	vs := make([]iovec, len(m.Buffers))
	var sa []byte
	if m.Addr != nil {
		sa = marshalInetAddr(m.Addr)
	}
	h.pack(vs, m.Buffers, m.OOB, sa)
	var operr error
	var n int
	fn := func(s uintptr) bool {
		n, operr = sendmsg(s, &h, flags)
		if operr == syscall.EAGAIN {
			return false
		}
		return true
	}
	if err := c.c.Write(fn); err != nil {
		return err
	}
	if operr != nil {
		return os.NewSyscallError("sendmsg", operr)
	}
	m.N = n
	m.NN = len(m.OOB)
	return nil
}
