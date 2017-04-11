// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6

import (
	"net"
	"os"
	"syscall"
	"unsafe"
)

func setsockoptIPMreq(fd syscall.Handle, opt *sockOpt, ifi *net.Interface, grp net.IP) error {
	var mreq sysIPv6Mreq
	copy(mreq.Multiaddr[:], grp)
	if ifi != nil {
		mreq.setIfindex(ifi.Index)
	}
	return os.NewSyscallError("setsockopt", syscall.Setsockopt(fd, int32(opt.level), int32(opt.name), (*byte)(unsafe.Pointer(&mreq)), sysSizeofIPv6Mreq))
}
