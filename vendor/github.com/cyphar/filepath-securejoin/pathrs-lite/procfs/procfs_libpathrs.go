// SPDX-License-Identifier: MPL-2.0

//go:build libpathrs

// Copyright (C) 2024-2025 Aleksa Sarai <cyphar@cyphar.com>
// Copyright (C) 2024-2025 SUSE LLC
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Package procfs provides a safe API for operating on /proc on Linux.
package procfs

import (
	"os"
	"strconv"

	"cyphar.com/go-pathrs/procfs"
	"golang.org/x/sys/unix"
)

// ProcThreadSelfCloser is a callback that needs to be called when you are done
// operating on an [os.File] fetched using [Handle.OpenThreadSelf].
//
// [os.File]: https://pkg.go.dev/os#File
type ProcThreadSelfCloser = procfs.ThreadCloser

// Handle is a wrapper around an *os.File handle to "/proc", which can be used
// to do further procfs-related operations in a safe way.
type Handle struct {
	inner *procfs.Handle
}

// Close close the resources associated with this [Handle]. Note that if this
// [Handle] was created with [OpenProcRoot], on some kernels the underlying
// procfs handle is cached and so this Close operation may be a no-op. However,
// you should always call Close on [Handle]s once you are done with them.
func (proc *Handle) Close() error { return proc.inner.Close() }

// OpenProcRoot tries to open a "safer" handle to "/proc" (i.e., one with the
// "subset=pid" mount option applied, available from Linux 5.8). Unless you
// plan to do many [Handle.OpenRoot] operations, users should prefer to use
// this over [OpenUnsafeProcRoot] which is far more dangerous to keep open.
//
// If a safe handle cannot be opened, OpenProcRoot will fall back to opening a
// regular "/proc" handle.
//
// Note that using [Handle.OpenRoot] will still work with handles returned by
// this function. If a subpath cannot be operated on with a safe "/proc"
// handle, then [OpenUnsafeProcRoot] will be called internally and a temporary
// unsafe handle will be used.
func OpenProcRoot() (*Handle, error) {
	proc, err := procfs.Open()
	if err != nil {
		return nil, err
	}
	return &Handle{inner: proc}, nil
}

// OpenUnsafeProcRoot opens a handle to "/proc" without any overmounts or
// masked paths. You must be extremely careful to make sure this handle is
// never leaked to a container and that you program cannot be tricked into
// writing to arbitrary paths within it.
//
// This is not necessary if you just wish to use [Handle.OpenRoot], as handles
// returned by [OpenProcRoot] will fall back to using a *temporary* unsafe
// handle in that case. You should only really use this if you need to do many
// operations with [Handle.OpenRoot] and the performance overhead of making
// many procfs handles is an issue. If you do use OpenUnsafeProcRoot, you
// should make sure to close the handle as soon as possible to avoid
// known-fd-number attacks.
func OpenUnsafeProcRoot() (*Handle, error) {
	proc, err := procfs.Open(procfs.UnmaskedProcRoot)
	if err != nil {
		return nil, err
	}
	return &Handle{inner: proc}, nil
}

// OpenThreadSelf returns a handle to "/proc/thread-self/<subpath>" (or an
// equivalent handle on older kernels where "/proc/thread-self" doesn't exist).
// Once finished with the handle, you must call the returned closer function
// ([runtime.UnlockOSThread]). You must not pass the returned *os.File to other
// Go threads or use the handle after calling the closer.
//
// [runtime.UnlockOSThread]: https://pkg.go.dev/runtime#UnlockOSThread
func (proc *Handle) OpenThreadSelf(subpath string) (*os.File, ProcThreadSelfCloser, error) {
	return proc.inner.OpenThreadSelf(subpath, unix.O_PATH|unix.O_NOFOLLOW)
}

// OpenSelf returns a handle to /proc/self/<subpath>.
//
// Note that in Go programs with non-homogenous threads, this may result in
// spurious errors. If you are monkeying around with APIs that are
// thread-specific, you probably want to use [Handle.OpenThreadSelf] instead
// which will guarantee that the handle refers to the same thread as the caller
// is executing on.
func (proc *Handle) OpenSelf(subpath string) (*os.File, error) {
	return proc.inner.OpenSelf(subpath, unix.O_PATH|unix.O_NOFOLLOW)
}

// OpenRoot returns a handle to /proc/<subpath>.
//
// You should only use this when you need to operate on global procfs files
// (such as sysctls in /proc/sys). Unlike [Handle.OpenThreadSelf],
// [Handle.OpenSelf], and [Handle.OpenPid], the procfs handle used internally
// for this operation will never use "subset=pid", which makes it a more juicy
// target for [CVE-2024-21626]-style attacks (and doing something like opening
// a directory with OpenRoot effectively leaks [OpenUnsafeProcRoot] as long as
// the file descriptor is open).
//
// [CVE-2024-21626]: https://github.com/opencontainers/runc/security/advisories/GHSA-xr7r-f8xq-vfvv
func (proc *Handle) OpenRoot(subpath string) (*os.File, error) {
	return proc.inner.OpenRoot(subpath, unix.O_PATH|unix.O_NOFOLLOW)
}

// OpenPid returns a handle to /proc/$pid/<subpath> (pid can be a pid or tid).
// This is mainly intended for usage when operating on other processes.
//
// You should not use this for the current thread, as special handling is
// needed for /proc/thread-self (or /proc/self/task/<tid>) when dealing with
// goroutine scheduling -- use [Handle.OpenThreadSelf] instead.
//
// To refer to the current thread-group, you should use prefer
// [Handle.OpenSelf] to passing os.Getpid as the pid argument.
func (proc *Handle) OpenPid(pid int, subpath string) (*os.File, error) {
	return proc.inner.OpenPid(pid, subpath, unix.O_PATH|unix.O_NOFOLLOW)
}

// ProcSelfFdReadlink gets the real path of the given file by looking at
// /proc/self/fd/<fd> with [readlink]. It is effectively just shorthand for
// something along the lines of:
//
//	proc, err := procfs.OpenProcRoot()
//	if err != nil {
//		return err
//	}
//	link, err := proc.OpenThreadSelf(fmt.Sprintf("fd/%d", f.Fd()))
//	if err != nil {
//		return err
//	}
//	defer link.Close()
//	var buf [4096]byte
//	n, err := unix.Readlinkat(int(link.Fd()), "", buf[:])
//	if err != nil {
//		return err
//	}
//	pathname := buf[:n]
//
// [readlink]: https://pkg.go.dev/golang.org/x/sys/unix#Readlinkat
func ProcSelfFdReadlink(f *os.File) (string, error) {
	proc, err := procfs.Open()
	if err != nil {
		return "", err
	}
	defer proc.Close() //nolint:errcheck // close failures aren't critical here

	fdPath := "fd/" + strconv.Itoa(int(f.Fd()))
	return proc.Readlink(procfs.ProcThreadSelf, fdPath)
}
