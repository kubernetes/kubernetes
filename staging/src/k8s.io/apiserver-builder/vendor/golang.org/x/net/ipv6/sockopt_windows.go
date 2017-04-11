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

func getInt(fd syscall.Handle, opt *sockOpt) (int, error) {
	if opt.name < 1 || opt.typ != ssoTypeInt {
		return 0, errOpNoSupport
	}
	var i int32
	l := int32(4)
	if err := syscall.Getsockopt(fd, int32(opt.level), int32(opt.name), (*byte)(unsafe.Pointer(&i)), &l); err != nil {
		return 0, os.NewSyscallError("getsockopt", err)
	}
	return int(i), nil
}

func setInt(fd syscall.Handle, opt *sockOpt, v int) error {
	if opt.name < 1 || opt.typ != ssoTypeInt {
		return errOpNoSupport
	}
	i := int32(v)
	return os.NewSyscallError("setsockopt", syscall.Setsockopt(fd, int32(opt.level), int32(opt.name), (*byte)(unsafe.Pointer(&i)), 4))
}

func getInterface(fd syscall.Handle, opt *sockOpt) (*net.Interface, error) {
	if opt.name < 1 || opt.typ != ssoTypeInterface {
		return nil, errOpNoSupport
	}
	var i int32
	l := int32(4)
	if err := syscall.Getsockopt(fd, int32(opt.level), int32(opt.name), (*byte)(unsafe.Pointer(&i)), &l); err != nil {
		return nil, os.NewSyscallError("getsockopt", err)
	}
	if i == 0 {
		return nil, nil
	}
	ifi, err := net.InterfaceByIndex(int(i))
	if err != nil {
		return nil, err
	}
	return ifi, nil
}

func setInterface(fd syscall.Handle, opt *sockOpt, ifi *net.Interface) error {
	if opt.name < 1 || opt.typ != ssoTypeInterface {
		return errOpNoSupport
	}
	var i int32
	if ifi != nil {
		i = int32(ifi.Index)
	}
	return os.NewSyscallError("setsockopt", syscall.Setsockopt(fd, int32(opt.level), int32(opt.name), (*byte)(unsafe.Pointer(&i)), 4))
}

func getICMPFilter(fd syscall.Handle, opt *sockOpt) (*ICMPFilter, error) {
	return nil, errOpNoSupport
}

func setICMPFilter(fd syscall.Handle, opt *sockOpt, f *ICMPFilter) error {
	return errOpNoSupport
}

func getMTUInfo(fd syscall.Handle, opt *sockOpt) (*net.Interface, int, error) {
	return nil, 0, errOpNoSupport
}

func setGroup(fd syscall.Handle, opt *sockOpt, ifi *net.Interface, grp net.IP) error {
	if opt.name < 1 || opt.typ != ssoTypeIPMreq {
		return errOpNoSupport
	}
	return setsockoptIPMreq(fd, opt, ifi, grp)
}

func setSourceGroup(fd syscall.Handle, opt *sockOpt, ifi *net.Interface, grp, src net.IP) error {
	// TODO(mikio): implement this
	return errOpNoSupport
}
