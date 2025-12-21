//go:build linux

// SPDX-License-Identifier: MPL-2.0
/*
 * libpathrs: safe path resolution on Linux
 * Copyright (C) 2019-2025 Aleksa Sarai <cyphar@cyphar.com>
 * Copyright (C) 2019-2025 SUSE LLC
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// Package libpathrs is an internal thin wrapper around the libpathrs C API.
package libpathrs

import (
	"fmt"
	"syscall"
	"unsafe"
)

/*
// TODO: Figure out if we need to add support for linking against libpathrs
//       statically even if in dynamically linked builds in order to make
//       packaging a bit easier (using "-Wl,-Bstatic -lpathrs -Wl,-Bdynamic" or
//       "-l:pathrs.a").
#cgo pkg-config: pathrs
#include <pathrs.h>

// This is a workaround for unsafe.Pointer() not working for non-void pointers.
char *cast_ptr(void *ptr) { return ptr; }
*/
import "C"

func fetchError(errID C.int) error {
	if errID >= C.__PATHRS_MAX_ERR_VALUE {
		return nil
	}
	cErr := C.pathrs_errorinfo(errID)
	defer C.pathrs_errorinfo_free(cErr)

	var err error
	if cErr != nil {
		err = &Error{
			errno:       syscall.Errno(cErr.saved_errno),
			description: C.GoString(cErr.description),
		}
	}
	return err
}

// OpenRoot wraps pathrs_open_root.
func OpenRoot(path string) (uintptr, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	fd := C.pathrs_open_root(cPath)
	return uintptr(fd), fetchError(fd)
}

// Reopen wraps pathrs_reopen.
func Reopen(fd uintptr, flags int) (uintptr, error) {
	newFd := C.pathrs_reopen(C.int(fd), C.int(flags))
	return uintptr(newFd), fetchError(newFd)
}

// InRootResolve wraps pathrs_inroot_resolve.
func InRootResolve(rootFd uintptr, path string) (uintptr, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	fd := C.pathrs_inroot_resolve(C.int(rootFd), cPath)
	return uintptr(fd), fetchError(fd)
}

// InRootResolveNoFollow wraps pathrs_inroot_resolve_nofollow.
func InRootResolveNoFollow(rootFd uintptr, path string) (uintptr, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	fd := C.pathrs_inroot_resolve_nofollow(C.int(rootFd), cPath)
	return uintptr(fd), fetchError(fd)
}

// InRootOpen wraps pathrs_inroot_open.
func InRootOpen(rootFd uintptr, path string, flags int) (uintptr, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	fd := C.pathrs_inroot_open(C.int(rootFd), cPath, C.int(flags))
	return uintptr(fd), fetchError(fd)
}

// InRootReadlink wraps pathrs_inroot_readlink.
func InRootReadlink(rootFd uintptr, path string) (string, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	size := 128
	for {
		linkBuf := make([]byte, size)
		n := C.pathrs_inroot_readlink(C.int(rootFd), cPath, C.cast_ptr(unsafe.Pointer(&linkBuf[0])), C.ulong(len(linkBuf)))
		switch {
		case int(n) < C.__PATHRS_MAX_ERR_VALUE:
			return "", fetchError(n)
		case int(n) <= len(linkBuf):
			return string(linkBuf[:int(n)]), nil
		default:
			// The contents were truncated. Unlike readlinkat, pathrs returns
			// the size of the link when it checked. So use the returned size
			// as a basis for the reallocated size (but in order to avoid a DoS
			// where a magic-link is growing by a single byte each iteration,
			// make sure we are a fair bit larger).
			size += int(n)
		}
	}
}

// InRootRmdir wraps pathrs_inroot_rmdir.
func InRootRmdir(rootFd uintptr, path string) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	err := C.pathrs_inroot_rmdir(C.int(rootFd), cPath)
	return fetchError(err)
}

// InRootUnlink wraps pathrs_inroot_unlink.
func InRootUnlink(rootFd uintptr, path string) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	err := C.pathrs_inroot_unlink(C.int(rootFd), cPath)
	return fetchError(err)
}

// InRootRemoveAll wraps pathrs_inroot_remove_all.
func InRootRemoveAll(rootFd uintptr, path string) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	err := C.pathrs_inroot_remove_all(C.int(rootFd), cPath)
	return fetchError(err)
}

// InRootCreat wraps pathrs_inroot_creat.
func InRootCreat(rootFd uintptr, path string, flags int, mode uint32) (uintptr, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	fd := C.pathrs_inroot_creat(C.int(rootFd), cPath, C.int(flags), C.uint(mode))
	return uintptr(fd), fetchError(fd)
}

// InRootRename wraps pathrs_inroot_rename.
func InRootRename(rootFd uintptr, src, dst string, flags uint) error {
	cSrc := C.CString(src)
	defer C.free(unsafe.Pointer(cSrc))

	cDst := C.CString(dst)
	defer C.free(unsafe.Pointer(cDst))

	err := C.pathrs_inroot_rename(C.int(rootFd), cSrc, cDst, C.uint(flags))
	return fetchError(err)
}

// InRootMkdir wraps pathrs_inroot_mkdir.
func InRootMkdir(rootFd uintptr, path string, mode uint32) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	err := C.pathrs_inroot_mkdir(C.int(rootFd), cPath, C.uint(mode))
	return fetchError(err)
}

// InRootMkdirAll wraps pathrs_inroot_mkdir_all.
func InRootMkdirAll(rootFd uintptr, path string, mode uint32) (uintptr, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	fd := C.pathrs_inroot_mkdir_all(C.int(rootFd), cPath, C.uint(mode))
	return uintptr(fd), fetchError(fd)
}

// InRootMknod wraps pathrs_inroot_mknod.
func InRootMknod(rootFd uintptr, path string, mode uint32, dev uint64) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	err := C.pathrs_inroot_mknod(C.int(rootFd), cPath, C.uint(mode), C.dev_t(dev))
	return fetchError(err)
}

// InRootSymlink wraps pathrs_inroot_symlink.
func InRootSymlink(rootFd uintptr, path, target string) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cTarget := C.CString(target)
	defer C.free(unsafe.Pointer(cTarget))

	err := C.pathrs_inroot_symlink(C.int(rootFd), cPath, cTarget)
	return fetchError(err)
}

// InRootHardlink wraps pathrs_inroot_hardlink.
func InRootHardlink(rootFd uintptr, path, target string) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cTarget := C.CString(target)
	defer C.free(unsafe.Pointer(cTarget))

	err := C.pathrs_inroot_hardlink(C.int(rootFd), cPath, cTarget)
	return fetchError(err)
}

// ProcBase is pathrs_proc_base_t (uint64_t).
type ProcBase C.pathrs_proc_base_t

// FIXME: We need to open-code the constants because CGo unfortunately will
// implicitly convert any non-literal constants (i.e. those resolved using gcc)
// to signed integers. See <https://github.com/golang/go/issues/39136> for some
// more information on the underlying issue (though.
const (
	// ProcRoot is PATHRS_PROC_ROOT.
	ProcRoot ProcBase = 0xFFFF_FFFE_7072_6F63 // C.PATHRS_PROC_ROOT
	// ProcSelf is PATHRS_PROC_SELF.
	ProcSelf ProcBase = 0xFFFF_FFFE_091D_5E1F // C.PATHRS_PROC_SELF
	// ProcThreadSelf is PATHRS_PROC_THREAD_SELF.
	ProcThreadSelf ProcBase = 0xFFFF_FFFE_3EAD_5E1F // C.PATHRS_PROC_THREAD_SELF

	// ProcBaseTypeMask is __PATHRS_PROC_TYPE_MASK.
	ProcBaseTypeMask ProcBase = 0xFFFF_FFFF_0000_0000 // C.__PATHRS_PROC_TYPE_MASK
	// ProcBaseTypePid is __PATHRS_PROC_TYPE_PID.
	ProcBaseTypePid ProcBase = 0x8000_0000_0000_0000 // C.__PATHRS_PROC_TYPE_PID

	// ProcDefaultRootFd is PATHRS_PROC_DEFAULT_ROOTFD.
	ProcDefaultRootFd = -int(syscall.EBADF) // C.PATHRS_PROC_DEFAULT_ROOTFD
)

func assertEqual[T comparable](a, b T, msg string) {
	if a != b {
		panic(fmt.Sprintf("%s ((%T) %#v != (%T) %#v)", msg, a, a, b, b))
	}
}

// Verify that the values above match the actual C values. Unfortunately, Go
// only allows us to forcefully cast int64 to uint64 if you use a temporary
// variable, which means we cannot do it in a const context and thus need to do
// it at runtime (even though it is a check that fundamentally could be done at
// compile-time)...
func init() {
	var (
		actualProcRoot       int64 = C.PATHRS_PROC_ROOT
		actualProcSelf       int64 = C.PATHRS_PROC_SELF
		actualProcThreadSelf int64 = C.PATHRS_PROC_THREAD_SELF
	)

	assertEqual(ProcRoot, ProcBase(actualProcRoot), "PATHRS_PROC_ROOT")
	assertEqual(ProcSelf, ProcBase(actualProcSelf), "PATHRS_PROC_SELF")
	assertEqual(ProcThreadSelf, ProcBase(actualProcThreadSelf), "PATHRS_PROC_THREAD_SELF")

	var (
		actualProcBaseTypeMask uint64 = C.__PATHRS_PROC_TYPE_MASK
		actualProcBaseTypePid  uint64 = C.__PATHRS_PROC_TYPE_PID
	)

	assertEqual(ProcBaseTypeMask, ProcBase(actualProcBaseTypeMask), "__PATHRS_PROC_TYPE_MASK")
	assertEqual(ProcBaseTypePid, ProcBase(actualProcBaseTypePid), "__PATHRS_PROC_TYPE_PID")

	assertEqual(ProcDefaultRootFd, int(C.PATHRS_PROC_DEFAULT_ROOTFD), "PATHRS_PROC_DEFAULT_ROOTFD")
}

// ProcPid reimplements the PROC_PID(x) conversion.
func ProcPid(pid uint32) ProcBase { return ProcBaseTypePid | ProcBase(pid) }

// ProcOpenat wraps pathrs_proc_openat.
func ProcOpenat(procRootFd int, base ProcBase, path string, flags int) (uintptr, error) {
	cBase := C.pathrs_proc_base_t(base)

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	fd := C.pathrs_proc_openat(C.int(procRootFd), cBase, cPath, C.int(flags))
	return uintptr(fd), fetchError(fd)
}

// ProcReadlinkat wraps pathrs_proc_readlinkat.
func ProcReadlinkat(procRootFd int, base ProcBase, path string) (string, error) {
	// TODO: See if we can unify this code with InRootReadlink.

	cBase := C.pathrs_proc_base_t(base)

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	size := 128
	for {
		linkBuf := make([]byte, size)
		n := C.pathrs_proc_readlinkat(
			C.int(procRootFd), cBase, cPath,
			C.cast_ptr(unsafe.Pointer(&linkBuf[0])), C.ulong(len(linkBuf)))
		switch {
		case int(n) < C.__PATHRS_MAX_ERR_VALUE:
			return "", fetchError(n)
		case int(n) <= len(linkBuf):
			return string(linkBuf[:int(n)]), nil
		default:
			// The contents were truncated. Unlike readlinkat, pathrs returns
			// the size of the link when it checked. So use the returned size
			// as a basis for the reallocated size (but in order to avoid a DoS
			// where a magic-link is growing by a single byte each iteration,
			// make sure we are a fair bit larger).
			size += int(n)
		}
	}
}

// ProcfsOpenHow is pathrs_procfs_open_how (struct).
type ProcfsOpenHow C.pathrs_procfs_open_how

const (
	// ProcfsNewUnmasked is PATHRS_PROCFS_NEW_UNMASKED.
	ProcfsNewUnmasked = C.PATHRS_PROCFS_NEW_UNMASKED
)

// Flags returns a pointer to the internal flags field to allow other packages
// to modify structure fields that are internal due to Go's visibility model.
func (how *ProcfsOpenHow) Flags() *C.uint64_t { return &how.flags }

// ProcfsOpen is pathrs_procfs_open (sizeof(*how) is passed automatically).
func ProcfsOpen(how *ProcfsOpenHow) (uintptr, error) {
	fd := C.pathrs_procfs_open((*C.pathrs_procfs_open_how)(how), C.size_t(unsafe.Sizeof(*how)))
	return uintptr(fd), fetchError(fd)
}
