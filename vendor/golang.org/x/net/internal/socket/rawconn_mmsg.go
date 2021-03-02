// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux
// +build linux

package socket

import (
	"net"
	"os"
	"syscall"
)

func (c *Conn) recvMsgs(ms []Message, flags int) (int, error) {
	for i := range ms {
		ms[i].raceWrite()
	}
	hs := make(mmsghdrs, len(ms))
	var parseFn func([]byte, string) (net.Addr, error)
	if c.network != "tcp" {
		parseFn = parseInetAddr
	}
	if err := hs.pack(ms, parseFn, nil); err != nil {
		return 0, err
	}
	var operr error
	var n int
	fn := func(s uintptr) bool {
		n, operr = recvmmsg(s, hs, flags)
		if operr == syscall.EAGAIN {
			return false
		}
		return true
	}
	if err := c.c.Read(fn); err != nil {
		return n, err
	}
	if operr != nil {
		return n, os.NewSyscallError("recvmmsg", operr)
	}
	if err := hs[:n].unpack(ms[:n], parseFn, c.network); err != nil {
		return n, err
	}
	return n, nil
}

func (c *Conn) sendMsgs(ms []Message, flags int) (int, error) {
	for i := range ms {
		ms[i].raceRead()
	}
	hs := make(mmsghdrs, len(ms))
	var marshalFn func(net.Addr) []byte
	if c.network != "tcp" {
		marshalFn = marshalInetAddr
	}
	if err := hs.pack(ms, nil, marshalFn); err != nil {
		return 0, err
	}
	var operr error
	var n int
	fn := func(s uintptr) bool {
		n, operr = sendmmsg(s, hs, flags)
		if operr == syscall.EAGAIN {
			return false
		}
		return true
	}
	if err := c.c.Write(fn); err != nil {
		return n, err
	}
	if operr != nil {
		return n, os.NewSyscallError("sendmmsg", operr)
	}
	if err := hs[:n].unpack(ms[:n], nil, ""); err != nil {
		return n, err
	}
	return n, nil
}
