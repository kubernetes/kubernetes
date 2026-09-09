// SPDX-License-Identifier: MPL-2.0

//go:build linux

// Copyright (C) 2024-2025 Aleksa Sarai <cyphar@cyphar.com>
// Copyright (C) 2024-2025 SUSE LLC
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// Package procfs provides a safe API for operating on /proc on Linux. Note
// that this is the *internal* procfs API, mainy needed due to Go's
// restrictions on cyclic dependencies and its incredibly minimal visibility
// system without making a separate internal/ package.
package procfs

import (
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"strconv"

	"golang.org/x/sys/unix"

	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal"
	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/assert"
	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/fd"
	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/gocompat"
	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/linux"
)

// The kernel guarantees that the root inode of a procfs mount has an
// f_type of PROC_SUPER_MAGIC and st_ino of PROC_ROOT_INO.
const (
	procSuperMagic = 0x9fa0 // PROC_SUPER_MAGIC
	procRootIno    = 1      // PROC_ROOT_INO
)

// verifyProcHandle checks that the handle is from a procfs filesystem.
// Contrast this to [verifyProcRoot], which also verifies that the handle is
// the root of a procfs mount.
func verifyProcHandle(procHandle fd.Fd) error {
	if statfs, err := fd.Fstatfs(procHandle); err != nil {
		return err
	} else if statfs.Type != procSuperMagic {
		return fmt.Errorf("%w: incorrect procfs root filesystem type 0x%x", errUnsafeProcfs, statfs.Type)
	}
	return nil
}

// verifyProcRoot verifies that the handle is the root of a procfs filesystem.
// Contrast this to [verifyProcHandle], which only verifies if the handle is
// some file on procfs (regardless of what file it is).
func verifyProcRoot(procRoot fd.Fd) error {
	if err := verifyProcHandle(procRoot); err != nil {
		return err
	}
	if stat, err := fd.Fstat(procRoot); err != nil {
		return err
	} else if stat.Ino != procRootIno {
		return fmt.Errorf("%w: incorrect procfs root inode number %d", errUnsafeProcfs, stat.Ino)
	}
	return nil
}

type procfsFeatures struct {
	// hasSubsetPid was added in Linux 5.8, along with hidepid=ptraceable (and
	// string-based hidepid= values). Before this patchset, it was not really
	// safe to try to modify procfs superblock flags because the superblock was
	// shared -- so if this feature is not available, **you should not set any
	// superblock flags**.
	//
	// 6814ef2d992a ("proc: add option to mount only a pids subset")
	// fa10fed30f25 ("proc: allow to mount many instances of proc in one pid namespace")
	// 24a71ce5c47f ("proc: instantiate only pids that we can ptrace on 'hidepid=4' mount option")
	// 1c6c4d112e81 ("proc: use human-readable values for hidepid")
	// 9ff7258575d5 ("Merge branch 'proc-linus' of git://git.kernel.org/pub/scm/linux/kernel/git/ebiederm/user-namespace")
	hasSubsetPid bool
}

var getProcfsFeatures = gocompat.SyncOnceValue(func() procfsFeatures {
	if !linux.HasNewMountAPI() {
		return procfsFeatures{}
	}
	procfsCtx, err := fd.Fsopen("proc", unix.FSOPEN_CLOEXEC)
	if err != nil {
		return procfsFeatures{}
	}
	defer procfsCtx.Close() //nolint:errcheck // close failures aren't critical here

	return procfsFeatures{
		hasSubsetPid: unix.FsconfigSetString(int(procfsCtx.Fd()), "subset", "pid") == nil,
	}
})

func newPrivateProcMount(subset bool) (_ *Handle, Err error) {
	procfsCtx, err := fd.Fsopen("proc", unix.FSOPEN_CLOEXEC)
	if err != nil {
		return nil, err
	}
	defer procfsCtx.Close() //nolint:errcheck // close failures aren't critical here

	if subset && getProcfsFeatures().hasSubsetPid {
		// Try to configure hidepid=ptraceable,subset=pid if possible, but
		// ignore errors.
		_ = unix.FsconfigSetString(int(procfsCtx.Fd()), "hidepid", "ptraceable")
		_ = unix.FsconfigSetString(int(procfsCtx.Fd()), "subset", "pid")
	}

	// Get an actual handle.
	if err := unix.FsconfigCreate(int(procfsCtx.Fd())); err != nil {
		return nil, os.NewSyscallError("fsconfig create procfs", err)
	}
	// TODO: Output any information from the fscontext log to debug logs.
	procRoot, err := fd.Fsmount(procfsCtx, unix.FSMOUNT_CLOEXEC, unix.MS_NODEV|unix.MS_NOEXEC|unix.MS_NOSUID)
	if err != nil {
		return nil, err
	}
	defer func() {
		if Err != nil {
			_ = procRoot.Close()
		}
	}()
	return newHandle(procRoot)
}

func clonePrivateProcMount() (_ *Handle, Err error) {
	// Try to make a clone without using AT_RECURSIVE if we can. If this works,
	// we can be sure there are no over-mounts and so if the root is valid then
	// we're golden. Otherwise, we have to deal with over-mounts.
	procRoot, err := fd.OpenTree(nil, "/proc", unix.OPEN_TREE_CLONE)
	if err != nil || hookForcePrivateProcRootOpenTreeAtRecursive(procRoot) {
		procRoot, err = fd.OpenTree(nil, "/proc", unix.OPEN_TREE_CLONE|unix.AT_RECURSIVE)
	}
	if err != nil {
		return nil, fmt.Errorf("creating a detached procfs clone: %w", err)
	}
	defer func() {
		if Err != nil {
			_ = procRoot.Close()
		}
	}()
	return newHandle(procRoot)
}

func privateProcRoot(subset bool) (*Handle, error) {
	if !linux.HasNewMountAPI() || hookForceGetProcRootUnsafe() {
		return nil, fmt.Errorf("new mount api: %w", unix.ENOTSUP)
	}
	// Try to create a new procfs mount from scratch if we can. This ensures we
	// can get a procfs mount even if /proc is fake (for whatever reason).
	procRoot, err := newPrivateProcMount(subset)
	if err != nil || hookForcePrivateProcRootOpenTree(procRoot) {
		// Try to clone /proc then...
		procRoot, err = clonePrivateProcMount()
	}
	return procRoot, err
}

func unsafeHostProcRoot() (_ *Handle, Err error) {
	procRoot, err := os.OpenFile("/proc", unix.O_PATH|unix.O_NOFOLLOW|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	defer func() {
		if Err != nil {
			_ = procRoot.Close()
		}
	}()
	return newHandle(procRoot)
}

// Handle is a wrapper around an *os.File handle to "/proc", which can be used
// to do further procfs-related operations in a safe way.
type Handle struct {
	Inner fd.Fd
	// Does this handle have subset=pid set?
	isSubset bool
}

func newHandle(procRoot fd.Fd) (*Handle, error) {
	if err := verifyProcRoot(procRoot); err != nil {
		// This is only used in methods that
		_ = procRoot.Close()
		return nil, err
	}
	proc := &Handle{Inner: procRoot}
	// With subset=pid we can be sure that /proc/uptime will not exist.
	if err := fd.Faccessat(proc.Inner, "uptime", unix.F_OK, unix.AT_SYMLINK_NOFOLLOW); err != nil {
		proc.isSubset = errors.Is(err, os.ErrNotExist)
	}
	return proc, nil
}

// Close closes the underlying file for the Handle.
func (proc *Handle) Close() error { return proc.Inner.Close() }

var getCachedProcRoot = gocompat.SyncOnceValue(func() *Handle {
	procRoot, err := getProcRoot(true)
	if err != nil {
		return nil // just don't cache if we see an error
	}
	if !procRoot.isSubset {
		return nil // we only cache verified subset=pid handles
	}

	// Disarm (*Handle).Close() to stop someone from accidentally closing
	// the global handle.
	procRoot.Inner = fd.NopCloser(procRoot.Inner)
	return procRoot
})

// OpenProcRoot tries to open a "safer" handle to "/proc".
func OpenProcRoot() (*Handle, error) {
	if proc := getCachedProcRoot(); proc != nil {
		return proc, nil
	}
	return getProcRoot(true)
}

// OpenUnsafeProcRoot opens a handle to "/proc" without any overmounts or
// masked paths (but also without "subset=pid").
func OpenUnsafeProcRoot() (*Handle, error) { return getProcRoot(false) }

func getProcRoot(subset bool) (*Handle, error) {
	proc, err := privateProcRoot(subset)
	if err != nil {
		// Fall back to using a /proc handle if making a private mount failed.
		// If we have openat2, at least we can avoid some kinds of over-mount
		// attacks, but without openat2 there's not much we can do.
		proc, err = unsafeHostProcRoot()
	}
	return proc, err
}

var hasProcThreadSelf = gocompat.SyncOnceValue(func() bool {
	return unix.Access("/proc/thread-self/", unix.F_OK) == nil
})

var errUnsafeProcfs = errors.New("unsafe procfs detected")

// lookup is a very minimal wrapper around [procfsLookupInRoot] which is
// intended to be called from the external API.
func (proc *Handle) lookup(subpath string) (*os.File, error) {
	handle, err := procfsLookupInRoot(proc.Inner, subpath)
	if err != nil {
		return nil, err
	}
	return handle, nil
}

// procfsBase is an enum indicating the prefix of a subpath in operations
// involving [Handle]s.
type procfsBase string

const (
	// ProcRoot refers to the root of the procfs (i.e., "/proc/<subpath>").
	ProcRoot procfsBase = "/proc"
	// ProcSelf refers to the current process' subdirectory (i.e.,
	// "/proc/self/<subpath>").
	ProcSelf procfsBase = "/proc/self"
	// ProcThreadSelf refers to the current thread's subdirectory (i.e.,
	// "/proc/thread-self/<subpath>"). In multi-threaded programs (i.e., all Go
	// programs) where one thread has a different CLONE_FS, it is possible for
	// "/proc/self" to point the wrong thread and so "/proc/thread-self" may be
	// necessary. Note that on pre-3.17 kernels, "/proc/thread-self" doesn't
	// exist and so a fallback will be used in that case.
	ProcThreadSelf procfsBase = "/proc/thread-self"
	// TODO: Switch to an interface setup so we can have a more type-safe
	// version of ProcPid and remove the need to worry about invalid string
	// values.
)

// prefix returns a prefix that can be used with the given [Handle].
func (base procfsBase) prefix(proc *Handle) (string, error) {
	switch base {
	case ProcRoot:
		return ".", nil
	case ProcSelf:
		return "self", nil
	case ProcThreadSelf:
		threadSelf := "thread-self"
		if !hasProcThreadSelf() || hookForceProcSelfTask() {
			// Pre-3.17 kernels don't have /proc/thread-self, so do it
			// manually.
			threadSelf = "self/task/" + strconv.Itoa(unix.Gettid())
			if err := fd.Faccessat(proc.Inner, threadSelf, unix.F_OK, unix.AT_SYMLINK_NOFOLLOW); err != nil || hookForceProcSelf() {
				// In this case, we running in a pid namespace that doesn't
				// match the /proc mount we have. This can happen inside runc.
				//
				// Unfortunately, there is no nice way to get the correct TID
				// to use here because of the age of the kernel, so we have to
				// just use /proc/self and hope that it works.
				threadSelf = "self"
			}
		}
		return threadSelf, nil
	}
	return "", fmt.Errorf("invalid procfs base %q", base)
}

// ProcThreadSelfCloser is a callback that needs to be called when you are done
// operating on an [os.File] fetched using [ProcThreadSelf].
//
// [os.File]: https://pkg.go.dev/os#File
type ProcThreadSelfCloser func()

// open is the core lookup operation for [Handle]. It returns a handle to
// "/proc/<base>/<subpath>". If the returned [ProcThreadSelfCloser] is non-nil,
// you should call it after you are done interacting with the returned handle.
//
// In general you should use prefer to use the other helpers, as they remove
// the need to interact with [procfsBase] and do not return a nil
// [ProcThreadSelfCloser] for [procfsBase] values other than [ProcThreadSelf]
// where it is necessary.
func (proc *Handle) open(base procfsBase, subpath string) (_ *os.File, closer ProcThreadSelfCloser, Err error) {
	prefix, err := base.prefix(proc)
	if err != nil {
		return nil, nil, err
	}
	subpath = prefix + "/" + subpath

	switch base {
	case ProcRoot:
		file, err := proc.lookup(subpath)
		if errors.Is(err, os.ErrNotExist) {
			// The Handle handle in use might be a subset=pid one, which will
			// result in spurious errors. In this case, just open a temporary
			// unmasked procfs handle for this operation.
			proc, err2 := OpenUnsafeProcRoot() // !subset=pid
			if err2 != nil {
				return nil, nil, err
			}
			defer proc.Close() //nolint:errcheck // close failures aren't critical here

			file, err = proc.lookup(subpath)
		}
		return file, nil, err

	case ProcSelf:
		file, err := proc.lookup(subpath)
		return file, nil, err

	case ProcThreadSelf:
		// We need to lock our thread until the caller is done with the handle
		// because between getting the handle and using it we could get
		// interrupted by the Go runtime and hit the case where the underlying
		// thread is swapped out and the original thread is killed, resulting
		// in pull-your-hair-out-hard-to-debug issues in the caller.
		runtime.LockOSThread()
		defer func() {
			if Err != nil {
				runtime.UnlockOSThread()
				closer = nil
			}
		}()

		file, err := proc.lookup(subpath)
		return file, runtime.UnlockOSThread, err
	}
	// should never be reached
	return nil, nil, fmt.Errorf("[internal error] invalid procfs base %q", base)
}

// OpenThreadSelf returns a handle to "/proc/thread-self/<subpath>" (or an
// equivalent handle on older kernels where "/proc/thread-self" doesn't exist).
// Once finished with the handle, you must call the returned closer function
// (runtime.UnlockOSThread). You must not pass the returned *os.File to other
// Go threads or use the handle after calling the closer.
func (proc *Handle) OpenThreadSelf(subpath string) (_ *os.File, _ ProcThreadSelfCloser, Err error) {
	return proc.open(ProcThreadSelf, subpath)
}

// OpenSelf returns a handle to /proc/self/<subpath>.
func (proc *Handle) OpenSelf(subpath string) (*os.File, error) {
	file, closer, err := proc.open(ProcSelf, subpath)
	assert.Assert(closer == nil, "closer for ProcSelf must be nil")
	return file, err
}

// OpenRoot returns a handle to /proc/<subpath>.
func (proc *Handle) OpenRoot(subpath string) (*os.File, error) {
	file, closer, err := proc.open(ProcRoot, subpath)
	assert.Assert(closer == nil, "closer for ProcRoot must be nil")
	return file, err
}

// OpenPid returns a handle to /proc/$pid/<subpath> (pid can be a pid or tid).
// This is mainly intended for usage when operating on other processes.
func (proc *Handle) OpenPid(pid int, subpath string) (*os.File, error) {
	return proc.OpenRoot(strconv.Itoa(pid) + "/" + subpath)
}

// checkSubpathOvermount checks if the dirfd and path combination is on the
// same mount as the given root.
func checkSubpathOvermount(root, dir fd.Fd, path string) error {
	// Get the mntID of our procfs handle.
	expectedMountID, err := fd.GetMountID(root, "")
	if err != nil {
		return fmt.Errorf("get root mount id: %w", err)
	}
	// Get the mntID of the target magic-link.
	gotMountID, err := fd.GetMountID(dir, path)
	if err != nil {
		return fmt.Errorf("get subpath mount id: %w", err)
	}
	// As long as the directory mount is alive, even with wrapping mount IDs,
	// we would expect to see a different mount ID here. (Of course, if we're
	// using unsafeHostProcRoot() then an attaker could change this after we
	// did this check.)
	if expectedMountID != gotMountID {
		return fmt.Errorf("%w: subpath %s/%s has an overmount obscuring the real path (mount ids do not match %d != %d)",
			errUnsafeProcfs, dir.Name(), path, expectedMountID, gotMountID)
	}
	return nil
}

// Readlink performs a readlink operation on "/proc/<base>/<subpath>" in a way
// that should be free from race attacks. This is most commonly used to get the
// real path of a file by looking at "/proc/self/fd/$n", with the same safety
// protections as [Open] (as well as some additional checks against
// overmounts).
func (proc *Handle) Readlink(base procfsBase, subpath string) (string, error) {
	link, closer, err := proc.open(base, subpath)
	if closer != nil {
		defer closer()
	}
	if err != nil {
		return "", fmt.Errorf("get safe %s/%s handle: %w", base, subpath, err)
	}
	defer link.Close() //nolint:errcheck // close failures aren't critical here

	// Try to detect if there is a mount on top of the magic-link. This should
	// be safe in general (a mount on top of the path afterwards would not
	// affect the handle itself) and will definitely be safe if we are using
	// privateProcRoot() (at least since Linux 5.12[1], when anonymous mount
	// namespaces were completely isolated from external mounts including mount
	// propagation events).
	//
	// [1]: Linux commit ee2e3f50629f ("mount: fix mounting of detached mounts
	// onto targets that reside on shared mounts").
	if err := checkSubpathOvermount(proc.Inner, link, ""); err != nil {
		return "", fmt.Errorf("check safety of %s/%s magiclink: %w", base, subpath, err)
	}

	// readlinkat implies AT_EMPTY_PATH since Linux 2.6.39. See Linux commit
	// 65cfc6722361 ("readlinkat(), fchownat() and fstatat() with empty
	// relative pathnames").
	return fd.Readlinkat(link, "")
}

// ProcSelfFdReadlink gets the real path of the given file by looking at
// readlink(/proc/thread-self/fd/$n).
//
// This is just a wrapper around [Handle.Readlink].
func ProcSelfFdReadlink(fd fd.Fd) (string, error) {
	procRoot, err := OpenProcRoot() // subset=pid
	if err != nil {
		return "", err
	}
	defer procRoot.Close() //nolint:errcheck // close failures aren't critical here

	fdPath := "fd/" + strconv.Itoa(int(fd.Fd()))
	return procRoot.Readlink(ProcThreadSelf, fdPath)
}

// CheckProcSelfFdPath returns whether the given file handle matches the
// expected path. (This is inherently racy.)
func CheckProcSelfFdPath(path string, file fd.Fd) error {
	if err := fd.IsDeadInode(file); err != nil {
		return err
	}
	actualPath, err := ProcSelfFdReadlink(file)
	if err != nil {
		return fmt.Errorf("get path of handle: %w", err)
	}
	if actualPath != path {
		return fmt.Errorf("%w: handle path %q doesn't match expected path %q", internal.ErrPossibleBreakout, actualPath, path)
	}
	return nil
}

// ReopenFd takes an existing file descriptor and "re-opens" it through
// /proc/thread-self/fd/<fd>. This allows for O_PATH file descriptors to be
// upgraded to regular file descriptors, as well as changing the open mode of a
// regular file descriptor. Some filesystems have unique handling of open(2)
// which make this incredibly useful (such as /dev/ptmx).
func ReopenFd(handle fd.Fd, flags int) (*os.File, error) {
	procRoot, err := OpenProcRoot() // subset=pid
	if err != nil {
		return nil, err
	}
	defer procRoot.Close() //nolint:errcheck // close failures aren't critical here

	// We can't operate on /proc/thread-self/fd/$n directly when doing a
	// re-open, so we need to open /proc/thread-self/fd and then open a single
	// final component.
	procFdDir, closer, err := procRoot.OpenThreadSelf("fd/")
	if err != nil {
		return nil, fmt.Errorf("get safe /proc/thread-self/fd handle: %w", err)
	}
	defer procFdDir.Close() //nolint:errcheck // close failures aren't critical here
	defer closer()

	// Try to detect if there is a mount on top of the magic-link we are about
	// to open. If we are using unsafeHostProcRoot(), this could change after
	// we check it (and there's nothing we can do about that) but for
	// privateProcRoot() this should be guaranteed to be safe (at least since
	// Linux 5.12[1], when anonymous mount namespaces were completely isolated
	// from external mounts including mount propagation events).
	//
	// [1]: Linux commit ee2e3f50629f ("mount: fix mounting of detached mounts
	// onto targets that reside on shared mounts").
	fdStr := strconv.Itoa(int(handle.Fd()))
	if err := checkSubpathOvermount(procRoot.Inner, procFdDir, fdStr); err != nil {
		return nil, fmt.Errorf("check safety of /proc/thread-self/fd/%s magiclink: %w", fdStr, err)
	}

	flags |= unix.O_CLOEXEC
	// Rather than just wrapping fd.Openat, open-code it so we can copy
	// handle.Name().
	reopenFd, err := unix.Openat(int(procFdDir.Fd()), fdStr, flags, 0)
	if err != nil {
		return nil, fmt.Errorf("reopen fd %d: %w", handle.Fd(), err)
	}
	return os.NewFile(uintptr(reopenFd), handle.Name()), nil
}

// Test hooks used in the procfs tests to verify that the fallback logic works.
// See testing_mocks_linux_test.go and procfs_linux_test.go for more details.
var (
	hookForcePrivateProcRootOpenTree            = hookDummyFile
	hookForcePrivateProcRootOpenTreeAtRecursive = hookDummyFile
	hookForceGetProcRootUnsafe                  = hookDummy

	hookForceProcSelfTask = hookDummy
	hookForceProcSelf     = hookDummy
)

func hookDummy() bool                { return false }
func hookDummyFile(_ io.Closer) bool { return false }
