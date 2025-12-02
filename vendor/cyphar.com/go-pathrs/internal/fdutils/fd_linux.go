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

// Package fdutils contains a few helper methods when dealing with *os.File and
// file descriptors.
package fdutils

import (
	"fmt"
	"os"

	"golang.org/x/sys/unix"

	"cyphar.com/go-pathrs/internal/libpathrs"
)

// DupFd makes a duplicate of the given fd.
func DupFd(fd uintptr, name string) (*os.File, error) {
	newFd, err := unix.FcntlInt(fd, unix.F_DUPFD_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("fcntl(F_DUPFD_CLOEXEC): %w", err)
	}
	return os.NewFile(uintptr(newFd), name), nil
}

// WithFileFd is a more ergonomic wrapper around file.SyscallConn().Control().
func WithFileFd[T any](file *os.File, fn func(fd uintptr) (T, error)) (T, error) {
	conn, err := file.SyscallConn()
	if err != nil {
		return *new(T), err
	}
	var (
		ret      T
		innerErr error
	)
	if err := conn.Control(func(fd uintptr) {
		ret, innerErr = fn(fd)
	}); err != nil {
		return *new(T), err
	}
	return ret, innerErr
}

// DupFile makes a duplicate of the given file.
func DupFile(file *os.File) (*os.File, error) {
	return WithFileFd(file, func(fd uintptr) (*os.File, error) {
		return DupFd(fd, file.Name())
	})
}

// MkFile creates a new *os.File from the provided file descriptor. However,
// unlike os.NewFile, the file's Name is based on the real path (provided by
// /proc/self/fd/$n).
func MkFile(fd uintptr) (*os.File, error) {
	fdPath := fmt.Sprintf("fd/%d", fd)
	fdName, err := libpathrs.ProcReadlinkat(libpathrs.ProcDefaultRootFd, libpathrs.ProcThreadSelf, fdPath)
	if err != nil {
		_ = unix.Close(int(fd))
		return nil, fmt.Errorf("failed to fetch real name of fd %d: %w", fd, err)
	}
	// TODO: Maybe we should prefix this name with something to indicate to
	// users that they must not use this path as a "safe" path. Something like
	// "//pathrs-handle:/foo/bar"?
	return os.NewFile(fd, fdName), nil
}
