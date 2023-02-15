//go:build freebsd
// +build freebsd

/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package subpath

import (
	"errors"
	"fmt"
	"unsafe"
	"k8s.io/mount-utils"
	"k8s.io/utils/nsenter"
)

/*
#include <fcntl.h>
#include <sys/stat.h>
#include <stdlib.h>

int openat_no_vargs(int fd, const char *path, int flags, uint32_t mode) { return openat(fd, path, flags, mode); }
*/
import "C"

const (
	O_PATH_PORTABLE = C.O_PATH // TODO: rebind with libc directly
)

var errUnsupported = errors.New("util/subpath on this platform is not fully supported")

// New returns a subpath.Interface for the current system.
func New(mount.Interface) Interface {
	return &subpath{}
}

// NewNSEnter is to satisfy the compiler for having NewSubpathNSEnter exist for all
// OS choices. however, NSEnter is only valid on Linux
func NewNSEnter(mounter mount.Interface, ne *nsenter.Nsenter, rootDir string) Interface {
	return nil
}

func (sp *subpath) PrepareSafeSubpath(subPath Subpath) (newHostPath string, cleanupAction func(), err error) {
	return subPath.Path, nil, errUnsupported
}

// This call is not implemented in golang unix/syscall, we need to bind
// from FreeBSD libc
func doOpenat(fd int, path string, flags int, mode uint32) (int, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	res := C.openat_no_vargs(C.int(fd), cPath, C.int(flags), C.uint(mode))
	if res < 0 {
		return 0, fmt.Errorf("openat failed for path %s", path)
	}
	return int(res), nil
}

// This call is not implemented in golang unix/syscall, we need to bind
// from FreeBSD libc
func doMkdirat(dirfd int, path string, mode uint32) (err error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	res := C.mkdirat(C.int(dirfd), cPath, C.ushort(mode))
	if res < 0 {
		return fmt.Errorf("mkdirat failed for path %s", path)
	}
	return nil

}
