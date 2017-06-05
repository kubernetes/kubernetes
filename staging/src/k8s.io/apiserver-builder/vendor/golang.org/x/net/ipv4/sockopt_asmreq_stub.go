// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !darwin,!dragonfly,!freebsd,!netbsd,!openbsd,!windows

package ipv4

import "net"

func setsockoptIPMreq(fd, name int, ifi *net.Interface, grp net.IP) error {
	return errOpNoSupport
}

func getsockoptInterface(fd, name int) (*net.Interface, error) {
	return nil, errOpNoSupport
}

func setsockoptInterface(fd, name int, ifi *net.Interface) error {
	return errOpNoSupport
}
