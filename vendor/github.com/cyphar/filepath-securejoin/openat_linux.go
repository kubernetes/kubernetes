//go:build linux

// Copyright (C) 2024 SUSE LLC. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package securejoin

import (
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

func dupFile(f *os.File) (*os.File, error) {
	fd, err := unix.FcntlInt(f.Fd(), unix.F_DUPFD_CLOEXEC, 0)
	if err != nil {
		return nil, os.NewSyscallError("fcntl(F_DUPFD_CLOEXEC)", err)
	}
	return os.NewFile(uintptr(fd), f.Name()), nil
}

func openatFile(dir *os.File, path string, flags int, mode int) (*os.File, error) {
	// Make sure we always set O_CLOEXEC.
	flags |= unix.O_CLOEXEC
	fd, err := unix.Openat(int(dir.Fd()), path, flags, uint32(mode))
	if err != nil {
		return nil, &os.PathError{Op: "openat", Path: dir.Name() + "/" + path, Err: err}
	}
	// All of the paths we use with openatFile(2) are guaranteed to be
	// lexically safe, so we can use path.Join here.
	fullPath := filepath.Join(dir.Name(), path)
	return os.NewFile(uintptr(fd), fullPath), nil
}

func fstatatFile(dir *os.File, path string, flags int) (unix.Stat_t, error) {
	var stat unix.Stat_t
	if err := unix.Fstatat(int(dir.Fd()), path, &stat, flags); err != nil {
		return stat, &os.PathError{Op: "fstatat", Path: dir.Name() + "/" + path, Err: err}
	}
	return stat, nil
}

func readlinkatFile(dir *os.File, path string) (string, error) {
	size := 4096
	for {
		linkBuf := make([]byte, size)
		n, err := unix.Readlinkat(int(dir.Fd()), path, linkBuf)
		if err != nil {
			return "", &os.PathError{Op: "readlinkat", Path: dir.Name() + "/" + path, Err: err}
		}
		if n != size {
			return string(linkBuf[:n]), nil
		}
		// Possible truncation, resize the buffer.
		size *= 2
	}
}
