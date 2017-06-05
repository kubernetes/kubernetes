// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd

package ipv4

import (
	"net"
	"reflect"
)

func (c *genericOpt) sysfd() (int, error) {
	switch p := c.Conn.(type) {
	case *net.TCPConn, *net.UDPConn, *net.IPConn:
		return sysfd(p)
	}
	return 0, errInvalidConnType
}

func (c *dgramOpt) sysfd() (int, error) {
	switch p := c.PacketConn.(type) {
	case *net.UDPConn, *net.IPConn:
		return sysfd(p.(net.Conn))
	}
	return 0, errInvalidConnType
}

func (c *payloadHandler) sysfd() (int, error) {
	return sysfd(c.PacketConn.(net.Conn))
}

func (c *packetHandler) sysfd() (int, error) {
	return sysfd(c.c)
}

func sysfd(c net.Conn) (int, error) {
	cv := reflect.ValueOf(c)
	switch ce := cv.Elem(); ce.Kind() {
	case reflect.Struct:
		netfd := ce.FieldByName("conn").FieldByName("fd")
		switch fe := netfd.Elem(); fe.Kind() {
		case reflect.Struct:
			fd := fe.FieldByName("sysfd")
			return int(fd.Int()), nil
		}
	}
	return 0, errInvalidConnType
}
