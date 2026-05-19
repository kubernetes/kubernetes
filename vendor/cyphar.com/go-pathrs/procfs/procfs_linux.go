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

// Package procfs provides a safe API for operating on /proc on Linux.
package procfs

import (
	"os"
	"runtime"

	"cyphar.com/go-pathrs/internal/fdutils"
	"cyphar.com/go-pathrs/internal/libpathrs"
)

// ProcBase is used with [ProcReadlink] and related functions to indicate what
// /proc subpath path operations should be done relative to.
type ProcBase struct {
	inner libpathrs.ProcBase
}

var (
	// ProcRoot indicates to use /proc. Note that this mode may be more
	// expensive because we have to take steps to try to avoid leaking unmasked
	// procfs handles, so you should use [ProcBaseSelf] if you can.
	ProcRoot = ProcBase{inner: libpathrs.ProcRoot}
	// ProcSelf indicates to use /proc/self. For most programs, this is the
	// standard choice.
	ProcSelf = ProcBase{inner: libpathrs.ProcSelf}
	// ProcThreadSelf indicates to use /proc/thread-self. In multi-threaded
	// programs where one thread has a different CLONE_FS, it is possible for
	// /proc/self to point the wrong thread and so /proc/thread-self may be
	// necessary.
	ProcThreadSelf = ProcBase{inner: libpathrs.ProcThreadSelf}
)

// ProcPid returns a ProcBase which indicates to use /proc/$pid for the given
// PID (or TID). Be aware that due to PID recycling, using this is generally
// not safe except in certain circumstances. Namely:
//
//   - PID 1 (the init process), as that PID cannot ever get recycled.
//   - Your current PID (though you should just use [ProcBaseSelf]).
//   - Your current TID if you have used [runtime.LockOSThread] (though you
//     should just use [ProcBaseThreadSelf]).
//   - PIDs of child processes (as long as you are sure that no other part of
//     your program incorrectly catches or ignores SIGCHLD, and that you do it
//     *before* you call wait(2)or any equivalent method that could reap
//     zombies).
func ProcPid(pid int) ProcBase {
	if pid < 0 || pid >= 1<<31 {
		panic("invalid ProcBasePid value") // TODO: should this be an error?
	}
	return ProcBase{inner: libpathrs.ProcPid(uint32(pid))}
}

// ThreadCloser is a callback that needs to be called when you are done
// operating on an [os.File] fetched using [Handle.OpenThreadSelf].
//
// [os.File]: https://pkg.go.dev/os#File
type ThreadCloser func()

// Handle is a wrapper around an *os.File handle to "/proc", which can be
// used to do further procfs-related operations in a safe way.
type Handle struct {
	inner *os.File
}

// Close releases all internal resources for this [Handle].
//
// Note that if the handle is actually the global cached handle, this operation
// is a no-op.
func (proc *Handle) Close() error {
	var err error
	if proc.inner != nil {
		err = proc.inner.Close()
	}
	return err
}

// OpenOption is a configuration function passed as an argument to [Open].
type OpenOption func(*libpathrs.ProcfsOpenHow) error

// UnmaskedProcRoot can be passed to [Open] to request an unmasked procfs
// handle be created.
//
//	procfs, err := procfs.OpenRoot(procfs.UnmaskedProcRoot)
func UnmaskedProcRoot(how *libpathrs.ProcfsOpenHow) error {
	*how.Flags() |= libpathrs.ProcfsNewUnmasked
	return nil
}

// Open creates a new [Handle] to a safe "/proc", based on the passed
// configuration options (in the form of a series of [OpenOption]s).
func Open(opts ...OpenOption) (*Handle, error) {
	var how libpathrs.ProcfsOpenHow
	for _, opt := range opts {
		if err := opt(&how); err != nil {
			return nil, err
		}
	}
	fd, err := libpathrs.ProcfsOpen(&how)
	if err != nil {
		return nil, err
	}
	var procFile *os.File
	if int(fd) >= 0 {
		procFile = os.NewFile(fd, "/proc")
	}
	// TODO: Check that fd == PATHRS_PROC_DEFAULT_ROOTFD in the <0 case?
	return &Handle{inner: procFile}, nil
}

// TODO: Switch to something fdutils.WithFileFd-like.
func (proc *Handle) fd() int {
	if proc.inner != nil {
		return int(proc.inner.Fd())
	}
	return libpathrs.ProcDefaultRootFd
}

// TODO: Should we expose open?
func (proc *Handle) open(base ProcBase, path string, flags int) (_ *os.File, Closer ThreadCloser, Err error) {
	var closer ThreadCloser
	if base == ProcThreadSelf {
		runtime.LockOSThread()
		closer = runtime.UnlockOSThread
	}
	defer func() {
		if closer != nil && Err != nil {
			closer()
			Closer = nil
		}
	}()

	fd, err := libpathrs.ProcOpenat(proc.fd(), base.inner, path, flags)
	if err != nil {
		return nil, nil, err
	}
	file, err := fdutils.MkFile(fd)
	return file, closer, err
}

// OpenRoot safely opens a given path from inside /proc/.
//
// This function must only be used for accessing global information from procfs
// (such as /proc/cpuinfo) or information about other processes (such as
// /proc/1). Accessing your own process information should be done using
// [Handle.OpenSelf] or [Handle.OpenThreadSelf].
func (proc *Handle) OpenRoot(path string, flags int) (*os.File, error) {
	file, closer, err := proc.open(ProcRoot, path, flags)
	if closer != nil {
		// should not happen
		panic("non-zero closer returned from procOpen(ProcRoot)")
	}
	return file, err
}

// OpenSelf safely opens a given path from inside /proc/self/.
//
// This method is recommend for getting process information about the current
// process for almost all Go processes *except* for cases where there are
// [runtime.LockOSThread] threads that have changed some aspect of their state
// (such as through unshare(CLONE_FS) or changing namespaces).
//
// For such non-heterogeneous processes, /proc/self may reference to a task
// that has different state from the current goroutine and so it may be
// preferable to use [Handle.OpenThreadSelf]. The same is true if a user
// really wants to inspect the current OS thread's information (such as
// /proc/thread-self/stack or /proc/thread-self/status which is always uniquely
// per-thread).
//
// Unlike [Handle.OpenThreadSelf], this method does not involve locking
// the goroutine to the current OS thread and so is simpler to use and
// theoretically has slightly less overhead.
//
// [runtime.LockOSThread]: https://pkg.go.dev/runtime#LockOSThread
func (proc *Handle) OpenSelf(path string, flags int) (*os.File, error) {
	file, closer, err := proc.open(ProcSelf, path, flags)
	if closer != nil {
		// should not happen
		panic("non-zero closer returned from procOpen(ProcSelf)")
	}
	return file, err
}

// OpenPid safely opens a given path from inside /proc/$pid/, where pid can be
// either a PID or TID.
//
// This is effectively equivalent to calling [Handle.OpenRoot] with the
// pid prefixed to the subpath.
//
// Be aware that due to PID recycling, using this is generally not safe except
// in certain circumstances. See the documentation of [ProcPid] for more
// details.
func (proc *Handle) OpenPid(pid int, path string, flags int) (*os.File, error) {
	file, closer, err := proc.open(ProcPid(pid), path, flags)
	if closer != nil {
		// should not happen
		panic("non-zero closer returned from procOpen(ProcPidOpen)")
	}
	return file, err
}

// OpenThreadSelf safely opens a given path from inside /proc/thread-self/.
//
// Most Go processes have heterogeneous threads (all threads have most of the
// same kernel state such as CLONE_FS) and so [Handle.OpenSelf] is
// preferable for most users.
//
// For non-heterogeneous threads, or users that actually want thread-specific
// information (such as /proc/thread-self/stack or /proc/thread-self/status),
// this method is necessary.
//
// Because Go can change the running OS thread of your goroutine without notice
// (and then subsequently kill the old thread), this method will lock the
// current goroutine to the OS thread (with [runtime.LockOSThread]) and the
// caller is responsible for unlocking the the OS thread with the
// [ThreadCloser] callback once they are done using the returned file. This
// callback MUST be called AFTER you have finished using the returned
// [os.File]. This callback is completely separate to [os.File.Close], so it
// must be called regardless of how you close the handle.
//
// [runtime.LockOSThread]: https://pkg.go.dev/runtime#LockOSThread
// [os.File]: https://pkg.go.dev/os#File
// [os.File.Close]: https://pkg.go.dev/os#File.Close
func (proc *Handle) OpenThreadSelf(path string, flags int) (*os.File, ThreadCloser, error) {
	return proc.open(ProcThreadSelf, path, flags)
}

// Readlink safely reads the contents of a symlink from the given procfs base.
//
// This is effectively equivalent to doing an Open*(O_PATH|O_NOFOLLOW) of the
// path and then doing unix.Readlinkat(fd, ""), but with the benefit that
// thread locking is not necessary for [ProcThreadSelf].
func (proc *Handle) Readlink(base ProcBase, path string) (string, error) {
	return libpathrs.ProcReadlinkat(proc.fd(), base.inner, path)
}
