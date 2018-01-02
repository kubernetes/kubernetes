// Copyright 2014 Red Hat, Inc
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build linux

package fileutil

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

func hasHardLinks(fi os.FileInfo) bool {
	// On directories, Nlink doesn't make sense when checking for hard links
	return !fi.IsDir() && fi.Sys().(*syscall.Stat_t).Nlink > 1
}

func getInode(fi os.FileInfo) uint64 {
	return fi.Sys().(*syscall.Stat_t).Ino
}

// These functions are from github.com/docker/docker/pkg/system

// TODO(sgotti) waiting for a utimensat functions accepting flags and a
// LUtimesNano using it in https://github.com/golang/sys/
func LUtimesNano(path string, ts []syscall.Timespec) error {
	// These are not currently available in syscall
	AT_FDCWD := -100
	AT_SYMLINK_NOFOLLOW := 0x100

	var _path *byte
	_path, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}

	if _, _, err := syscall.Syscall6(syscall.SYS_UTIMENSAT, uintptr(AT_FDCWD), uintptr(unsafe.Pointer(_path)), uintptr(unsafe.Pointer(&ts[0])), uintptr(AT_SYMLINK_NOFOLLOW), 0, 0); err != 0 && err != syscall.ENOSYS {
		return err
	}

	return nil
}

// Returns a nil slice and nil error if the xattr is not set
func Lgetxattr(path string, attr string) ([]byte, error) {
	pathBytes, err := syscall.BytePtrFromString(path)
	if err != nil {
		return nil, err
	}
	attrBytes, err := syscall.BytePtrFromString(attr)
	if err != nil {
		return nil, err
	}

	dest := make([]byte, 128)
	destBytes := unsafe.Pointer(&dest[0])
	sz, _, errno := syscall.Syscall6(syscall.SYS_LGETXATTR, uintptr(unsafe.Pointer(pathBytes)), uintptr(unsafe.Pointer(attrBytes)), uintptr(destBytes), uintptr(len(dest)), 0, 0)
	if errno == syscall.ENODATA {
		return nil, nil
	}
	if errno == syscall.ERANGE {
		dest = make([]byte, sz)
		destBytes := unsafe.Pointer(&dest[0])
		sz, _, errno = syscall.Syscall6(syscall.SYS_LGETXATTR, uintptr(unsafe.Pointer(pathBytes)), uintptr(unsafe.Pointer(attrBytes)), uintptr(destBytes), uintptr(len(dest)), 0, 0)
	}
	if errno != 0 {
		return nil, errno
	}

	return dest[:sz], nil
}

var _zero uintptr

func Lsetxattr(path string, attr string, data []byte, flags int) error {
	pathBytes, err := syscall.BytePtrFromString(path)
	if err != nil {
		return err
	}
	attrBytes, err := syscall.BytePtrFromString(attr)
	if err != nil {
		return err
	}
	var dataBytes unsafe.Pointer
	if len(data) > 0 {
		dataBytes = unsafe.Pointer(&data[0])
	} else {
		dataBytes = unsafe.Pointer(&_zero)
	}
	_, _, errno := syscall.Syscall6(syscall.SYS_LSETXATTR, uintptr(unsafe.Pointer(pathBytes)), uintptr(unsafe.Pointer(attrBytes)), uintptr(dataBytes), uintptr(len(data)), uintptr(flags), 0)
	if errno != 0 {
		return errno
	}
	return nil
}

// GetDeviceInfo returns the type, major, and minor numbers of a device.
// Kind is 'b' or 'c' for block and character devices, respectively.
// This does not follow symlinks.
func GetDeviceInfo(path string) (kind rune, major uint64, minor uint64, err error) {
	d, err := os.Lstat(path)
	if err != nil {
		return
	}
	mode := d.Mode()

	if mode&os.ModeDevice == 0 {
		err = fmt.Errorf("not a device: %s", path)
		return
	}
	stat_t, ok := d.Sys().(*syscall.Stat_t)
	if !ok {
		err = fmt.Errorf("cannot determine device number")
		return
	}
	return getDeviceInfo(mode, stat_t.Rdev)
}

// Parse the device info out of the mode bits. Separate for testability.
func getDeviceInfo(mode os.FileMode, rdev uint64) (kind rune, major uint64, minor uint64, err error) {
	kind = 'b'

	if mode&os.ModeCharDevice != 0 {
		kind = 'c'
	}

	major = (rdev >> 8) & 0xfff
	minor = (rdev & 0xff) | ((rdev >> 12) & 0xfff00)

	return
}
