// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !aix && !darwin && !dragonfly && !freebsd && !netbsd && !openbsd && !solaris && !windows
// +build !aix,!darwin,!dragonfly,!freebsd,!netbsd,!openbsd,!solaris,!windows

package ipv4

import (
	"net"

	"golang.org/x/net/internal/socket"
)

func (so *sockOpt) setIPMreq(c *socket.Conn, ifi *net.Interface, grp net.IP) error {
	return errNotImplemented
}

func (so *sockOpt) getMulticastIf(c *socket.Conn) (*net.Interface, error) {
	return nil, errNotImplemented
}

func (so *sockOpt) setMulticastIf(c *socket.Conn, ifi *net.Interface) error {
	return errNotImplemented
}
