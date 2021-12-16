// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris || windows || zos
// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris windows zos

package socket

import (
	"os"
)

func (c *Conn) recvMsg(m *Message, flags int) error {
	m.raceWrite()
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
		return ioComplete(flags, operr)
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
	m.raceRead()
	var h msghdr
	vs := make([]iovec, len(m.Buffers))
	var sa []byte
	if m.Addr != nil {
		var a [sizeofSockaddrInet6]byte
		n := marshalInetAddr(m.Addr, a[:])
		sa = a[:n]
	}
	h.pack(vs, m.Buffers, m.OOB, sa)
	var operr error
	var n int
	fn := func(s uintptr) bool {
		n, operr = sendmsg(s, &h, flags)
		return ioComplete(flags, operr)
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
