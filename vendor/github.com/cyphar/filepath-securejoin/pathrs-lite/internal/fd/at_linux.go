// SPDX-License-Identifier: MPL-2.0

//go:build linux

// Copyright (C) 2024-2025 Aleksa Sarai <cyphar@cyphar.com>
// Copyright (C) 2024-2025 SUSE LLC
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

package fd

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"golang.org/x/sys/unix"

	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/gocompat"
)

// prepareAtWith returns -EBADF (an invalid fd) if dir is nil, otherwise using
// the dir.Fd(). We use -EBADF because in filepath-securejoin we generally
// don't want to allow relative-to-cwd paths. The returned path is an
// *informational* string that describes a reasonable pathname for the given
// *at(2) arguments. You must not use the full path for any actual filesystem
// operations.
func prepareAt(dir Fd, path string) (dirFd int, unsafeUnmaskedPath string) {
	dirFd, dirPath := -int(unix.EBADF), "."
	if dir != nil {
		dirFd, dirPath = int(dir.Fd()), dir.Name()
	}
	if !filepath.IsAbs(path) {
		// only prepend the dirfd path for relative paths
		path = dirPath + "/" + path
	}
	// NOTE: If path is "." or "", the returned path won't be filepath.Clean,
	// but that's okay since this path is either used for errors (in which case
	// a trailing "/" or "/." is important information) or will be
	// filepath.Clean'd later (in the case of fd.Openat).
	return dirFd, path
}

// Openat is an [Fd]-based wrapper around unix.Openat.
func Openat(dir Fd, path string, flags int, mode int) (*os.File, error) { //nolint:unparam // wrapper func
	dirFd, fullPath := prepareAt(dir, path)
	// Make sure we always set O_CLOEXEC.
	flags |= unix.O_CLOEXEC
	fd, err := unix.Openat(dirFd, path, flags, uint32(mode))
	if err != nil {
		return nil, &os.PathError{Op: "openat", Path: fullPath, Err: err}
	}
	runtime.KeepAlive(dir)
	// openat is only used with lexically-safe paths so we can use
	// filepath.Clean here, and also the path itself is not going to be used
	// for actual path operations.
	fullPath = filepath.Clean(fullPath)
	return os.NewFile(uintptr(fd), fullPath), nil
}

// Fstatat is an [Fd]-based wrapper around unix.Fstatat.
func Fstatat(dir Fd, path string, flags int) (unix.Stat_t, error) {
	dirFd, fullPath := prepareAt(dir, path)
	var stat unix.Stat_t
	if err := unix.Fstatat(dirFd, path, &stat, flags); err != nil {
		return stat, &os.PathError{Op: "fstatat", Path: fullPath, Err: err}
	}
	runtime.KeepAlive(dir)
	return stat, nil
}

// Faccessat is an [Fd]-based wrapper around unix.Faccessat.
func Faccessat(dir Fd, path string, mode uint32, flags int) error {
	dirFd, fullPath := prepareAt(dir, path)
	err := unix.Faccessat(dirFd, path, mode, flags)
	if err != nil {
		err = &os.PathError{Op: "faccessat", Path: fullPath, Err: err}
	}
	runtime.KeepAlive(dir)
	return err
}

// Readlinkat is an [Fd]-based wrapper around unix.Readlinkat.
func Readlinkat(dir Fd, path string) (string, error) {
	dirFd, fullPath := prepareAt(dir, path)
	size := 4096
	for {
		linkBuf := make([]byte, size)
		n, err := unix.Readlinkat(dirFd, path, linkBuf)
		if err != nil {
			return "", &os.PathError{Op: "readlinkat", Path: fullPath, Err: err}
		}
		runtime.KeepAlive(dir)
		if n != size {
			return string(linkBuf[:n]), nil
		}
		// Possible truncation, resize the buffer.
		size *= 2
	}
}

const (
	// STATX_MNT_ID_UNIQUE is provided in golang.org/x/sys@v0.20.0, but in order to
	// avoid bumping the requirement for a single constant we can just define it
	// ourselves.
	_STATX_MNT_ID_UNIQUE = 0x4000 //nolint:revive // unix.* name

	// We don't care which mount ID we get. The kernel will give us the unique
	// one if it is supported. If the kernel doesn't support
	// STATX_MNT_ID_UNIQUE, the bit is ignored and the returned request mask
	// will only contain STATX_MNT_ID (if supported).
	wantStatxMntMask = _STATX_MNT_ID_UNIQUE | unix.STATX_MNT_ID
)

var hasStatxMountID = gocompat.SyncOnceValue(func() bool {
	var stx unix.Statx_t
	err := unix.Statx(-int(unix.EBADF), "/", 0, wantStatxMntMask, &stx)
	return err == nil && stx.Mask&wantStatxMntMask != 0
})

// GetMountID gets the mount identifier associated with the fd and path
// combination. It is effectively a wrapper around fetching
// STATX_MNT_ID{,_UNIQUE} with unix.Statx, but with a fallback to 0 if the
// kernel doesn't support the feature.
func GetMountID(dir Fd, path string) (uint64, error) {
	// If we don't have statx(STATX_MNT_ID*) support, we can't do anything.
	if !hasStatxMountID() {
		return 0, nil
	}

	dirFd, fullPath := prepareAt(dir, path)

	var stx unix.Statx_t
	err := unix.Statx(dirFd, path, unix.AT_EMPTY_PATH|unix.AT_SYMLINK_NOFOLLOW, wantStatxMntMask, &stx)
	if stx.Mask&wantStatxMntMask == 0 {
		// It's not a kernel limitation, for some reason we couldn't get a
		// mount ID. Assume it's some kind of attack.
		err = fmt.Errorf("could not get mount id: %w", err)
	}
	if err != nil {
		return 0, &os.PathError{Op: "statx(STATX_MNT_ID_...)", Path: fullPath, Err: err}
	}
	runtime.KeepAlive(dir)
	return stx.Mnt_id, nil
}
