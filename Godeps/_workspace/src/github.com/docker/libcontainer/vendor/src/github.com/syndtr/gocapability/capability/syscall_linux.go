// Copyright (c) 2013, Suryandaru Triandana <syndtr@gmail.com>
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package capability

import (
	"syscall"
	"unsafe"
)

type capHeader struct {
	version uint32
	pid     int
}

type capData struct {
	effective   uint32
	permitted   uint32
	inheritable uint32
}

func capget(hdr *capHeader, data *capData) (err error) {
	_, _, e1 := syscall.Syscall(syscall.SYS_CAPGET, uintptr(unsafe.Pointer(hdr)), uintptr(unsafe.Pointer(data)), 0)
	if e1 != 0 {
		err = e1
	}
	return
}

func capset(hdr *capHeader, data *capData) (err error) {
	_, _, e1 := syscall.Syscall(syscall.SYS_CAPSET, uintptr(unsafe.Pointer(hdr)), uintptr(unsafe.Pointer(data)), 0)
	if e1 != 0 {
		err = e1
	}
	return
}

func prctl(option int, arg2, arg3, arg4, arg5 uintptr) (err error) {
	_, _, e1 := syscall.Syscall6(syscall.SYS_PRCTL, uintptr(option), arg2, arg3, arg4, arg5, 0)
	if e1 != 0 {
		err = e1
	}
	return
}

const (
	vfsXattrName = "security.capability"

	vfsCapVerMask = 0xff000000
	vfsCapVer1    = 0x01000000
	vfsCapVer2    = 0x02000000

	vfsCapFlagMask      = ^vfsCapVerMask
	vfsCapFlageffective = 0x000001

	vfscapDataSizeV1 = 4 * (1 + 2*1)
	vfscapDataSizeV2 = 4 * (1 + 2*2)
)

type vfscapData struct {
	magic uint32
	data  [2]struct {
		permitted   uint32
		inheritable uint32
	}
	effective [2]uint32
	version   int8
}

var (
	_vfsXattrName *byte
)

func init() {
	_vfsXattrName, _ = syscall.BytePtrFromString(vfsXattrName)
}

func getVfsCap(path string, dest *vfscapData) (err error) {
	var _p0 *byte
	_p0, err = syscall.BytePtrFromString(path)
	if err != nil {
		return
	}
	r0, _, e1 := syscall.Syscall6(syscall.SYS_GETXATTR, uintptr(unsafe.Pointer(_p0)), uintptr(unsafe.Pointer(_vfsXattrName)), uintptr(unsafe.Pointer(dest)), vfscapDataSizeV2, 0, 0)
	if e1 != 0 {
		err = e1
	}
	switch dest.magic & vfsCapVerMask {
	case vfsCapVer1:
		dest.version = 1
		if r0 != vfscapDataSizeV1 {
			return syscall.EINVAL
		}
		dest.data[1].permitted = 0
		dest.data[1].inheritable = 0
	case vfsCapVer2:
		dest.version = 2
		if r0 != vfscapDataSizeV2 {
			return syscall.EINVAL
		}
	default:
		return syscall.EINVAL
	}
	if dest.magic&vfsCapFlageffective != 0 {
		dest.effective[0] = dest.data[0].permitted | dest.data[0].inheritable
		dest.effective[1] = dest.data[1].permitted | dest.data[1].inheritable
	} else {
		dest.effective[0] = 0
		dest.effective[1] = 0
	}
	return
}

func setVfsCap(path string, data *vfscapData) (err error) {
	var _p0 *byte
	_p0, err = syscall.BytePtrFromString(path)
	if err != nil {
		return
	}
	var size uintptr
	if data.version == 1 {
		data.magic = vfsCapVer1
		size = vfscapDataSizeV1
	} else if data.version == 2 {
		data.magic = vfsCapVer2
		if data.effective[0] != 0 || data.effective[1] != 0 {
			data.magic |= vfsCapFlageffective
			data.data[0].permitted |= data.effective[0]
			data.data[1].permitted |= data.effective[1]
		}
		size = vfscapDataSizeV2
	} else {
		return syscall.EINVAL
	}
	_, _, e1 := syscall.Syscall6(syscall.SYS_SETXATTR, uintptr(unsafe.Pointer(_p0)), uintptr(unsafe.Pointer(_vfsXattrName)), uintptr(unsafe.Pointer(data)), size, 0, 0)
	if e1 != 0 {
		err = e1
	}
	return
}
