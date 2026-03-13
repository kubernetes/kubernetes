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
	"os"
	"runtime"

	"golang.org/x/sys/unix"
)

// Fsopen is an [Fd]-based wrapper around unix.Fsopen.
func Fsopen(fsName string, flags int) (*os.File, error) {
	// Make sure we always set O_CLOEXEC.
	flags |= unix.FSOPEN_CLOEXEC
	fd, err := unix.Fsopen(fsName, flags)
	if err != nil {
		return nil, os.NewSyscallError("fsopen "+fsName, err)
	}
	return os.NewFile(uintptr(fd), "fscontext:"+fsName), nil
}

// Fsmount is an [Fd]-based wrapper around unix.Fsmount.
func Fsmount(ctx Fd, flags, mountAttrs int) (*os.File, error) {
	// Make sure we always set O_CLOEXEC.
	flags |= unix.FSMOUNT_CLOEXEC
	fd, err := unix.Fsmount(int(ctx.Fd()), flags, mountAttrs)
	if err != nil {
		return nil, os.NewSyscallError("fsmount "+ctx.Name(), err)
	}
	return os.NewFile(uintptr(fd), "fsmount:"+ctx.Name()), nil
}

// OpenTree is an [Fd]-based wrapper around unix.OpenTree.
func OpenTree(dir Fd, path string, flags uint) (*os.File, error) {
	dirFd, fullPath := prepareAt(dir, path)
	// Make sure we always set O_CLOEXEC.
	flags |= unix.OPEN_TREE_CLOEXEC
	fd, err := unix.OpenTree(dirFd, path, flags)
	if err != nil {
		return nil, &os.PathError{Op: "open_tree", Path: fullPath, Err: err}
	}
	runtime.KeepAlive(dir)
	return os.NewFile(uintptr(fd), fullPath), nil
}
