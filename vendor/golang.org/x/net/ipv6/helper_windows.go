// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6

import (
	"net"
	"reflect"
	"syscall"
)

func (c *genericOpt) sysfd() (syscall.Handle, error) {
	switch p := c.Conn.(type) {
	case *net.TCPConn, *net.UDPConn, *net.IPConn:
		return sysfd(p)
	}
	return syscall.InvalidHandle, errInvalidConnType
}

func (c *dgramOpt) sysfd() (syscall.Handle, error) {
	switch p := c.PacketConn.(type) {
	case *net.UDPConn, *net.IPConn:
		return sysfd(p.(net.Conn))
	}
	return syscall.InvalidHandle, errInvalidConnType
}

func (c *payloadHandler) sysfd() (syscall.Handle, error) {
	return sysfd(c.PacketConn.(net.Conn))
}

func sysfd(c net.Conn) (syscall.Handle, error) {
	cv := reflect.ValueOf(c)
	switch ce := cv.Elem(); ce.Kind() {
	case reflect.Struct:
		netfd := ce.FieldByName("conn").FieldByName("fd")
		switch fe := netfd.Elem(); fe.Kind() {
		case reflect.Struct:
			fd := fe.FieldByName("sysfd")
			return syscall.Handle(fd.Uint()), nil
		}
	}
	return syscall.InvalidHandle, errInvalidConnType
}
