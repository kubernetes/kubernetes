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
	"runtime"

	"golang.org/x/sys/unix"

	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal"
)

// DupWithName creates a new file descriptor referencing the same underlying
// file, but with the provided name instead of fd.Name().
func DupWithName(fd Fd, name string) (*os.File, error) {
	fd2, err := unix.FcntlInt(fd.Fd(), unix.F_DUPFD_CLOEXEC, 0)
	if err != nil {
		return nil, os.NewSyscallError("fcntl(F_DUPFD_CLOEXEC)", err)
	}
	runtime.KeepAlive(fd)
	return os.NewFile(uintptr(fd2), name), nil
}

// Dup creates a new file description referencing the same underlying file.
func Dup(fd Fd) (*os.File, error) {
	return DupWithName(fd, fd.Name())
}

// Fstat is an [Fd]-based wrapper around unix.Fstat.
func Fstat(fd Fd) (unix.Stat_t, error) {
	var stat unix.Stat_t
	if err := unix.Fstat(int(fd.Fd()), &stat); err != nil {
		return stat, &os.PathError{Op: "fstat", Path: fd.Name(), Err: err}
	}
	runtime.KeepAlive(fd)
	return stat, nil
}

// Fstatfs is an [Fd]-based wrapper around unix.Fstatfs.
func Fstatfs(fd Fd) (unix.Statfs_t, error) {
	var statfs unix.Statfs_t
	if err := unix.Fstatfs(int(fd.Fd()), &statfs); err != nil {
		return statfs, &os.PathError{Op: "fstatfs", Path: fd.Name(), Err: err}
	}
	runtime.KeepAlive(fd)
	return statfs, nil
}

// IsDeadInode detects whether the file has been unlinked from a filesystem and
// is thus a "dead inode" from the kernel's perspective.
func IsDeadInode(file Fd) error {
	// If the nlink of a file drops to 0, there is an attacker deleting
	// directories during our walk, which could result in weird /proc values.
	// It's better to error out in this case.
	stat, err := Fstat(file)
	if err != nil {
		return fmt.Errorf("check for dead inode: %w", err)
	}
	if stat.Nlink == 0 {
		err := internal.ErrDeletedInode
		if stat.Mode&unix.S_IFMT == unix.S_IFDIR {
			err = internal.ErrInvalidDirectory
		}
		return fmt.Errorf("%w %q", err, file.Name())
	}
	return nil
}
