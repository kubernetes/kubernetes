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

package pathrs

import (
	"errors"
	"fmt"
	"os"
	"syscall"

	"cyphar.com/go-pathrs/internal/fdutils"
	"cyphar.com/go-pathrs/internal/libpathrs"
)

// Root is a handle to the root of a directory tree to resolve within. The only
// purpose of this "root handle" is to perform operations within the directory
// tree, or to get a [Handle] to inodes within the directory tree.
//
// At time of writing, it is considered a *VERY BAD IDEA* to open a [Root]
// inside a possibly-attacker-controlled directory tree. While we do have
// protections that should defend against it, it's far more dangerous than just
// opening a directory tree which is not inside a potentially-untrusted
// directory.
type Root struct {
	inner *os.File
}

// OpenRoot creates a new [Root] handle to the directory at the given path.
func OpenRoot(path string) (*Root, error) {
	fd, err := libpathrs.OpenRoot(path)
	if err != nil {
		return nil, err
	}
	file, err := fdutils.MkFile(fd)
	if err != nil {
		return nil, err
	}
	return &Root{inner: file}, nil
}

// RootFromFile creates a new [Root] handle from an [os.File] referencing a
// directory. The provided file will be duplicated, so the original file should
// still be closed by the caller.
//
// This is effectively the inverse operation of [Root.IntoFile].
//
// [os.File]: https://pkg.go.dev/os#File
func RootFromFile(file *os.File) (*Root, error) {
	newFile, err := fdutils.DupFile(file)
	if err != nil {
		return nil, fmt.Errorf("duplicate root fd: %w", err)
	}
	return &Root{inner: newFile}, nil
}

// Resolve resolves the given path within the [Root]'s directory tree, and
// returns a [Handle] to the resolved path. The path must already exist,
// otherwise an error will occur.
//
// All symlinks (including trailing symlinks) are followed, but they are
// resolved within the rootfs. If you wish to open a handle to the symlink
// itself, use [ResolveNoFollow].
func (r *Root) Resolve(path string) (*Handle, error) {
	return fdutils.WithFileFd(r.inner, func(rootFd uintptr) (*Handle, error) {
		handleFd, err := libpathrs.InRootResolve(rootFd, path)
		if err != nil {
			return nil, err
		}
		handleFile, err := fdutils.MkFile(handleFd)
		if err != nil {
			return nil, err
		}
		return &Handle{inner: handleFile}, nil
	})
}

// ResolveNoFollow is effectively an O_NOFOLLOW version of [Resolve]. Their
// behaviour is identical, except that *trailing* symlinks will not be
// followed. If the final component is a trailing symlink, an O_PATH|O_NOFOLLOW
// handle to the symlink itself is returned.
func (r *Root) ResolveNoFollow(path string) (*Handle, error) {
	return fdutils.WithFileFd(r.inner, func(rootFd uintptr) (*Handle, error) {
		handleFd, err := libpathrs.InRootResolveNoFollow(rootFd, path)
		if err != nil {
			return nil, err
		}
		handleFile, err := fdutils.MkFile(handleFd)
		if err != nil {
			return nil, err
		}
		return &Handle{inner: handleFile}, nil
	})
}

// Open is effectively shorthand for [Resolve] followed by [Handle.Open], but
// can be slightly more efficient (it reduces CGo overhead and the number of
// syscalls used when using the openat2-based resolver) and is arguably more
// ergonomic to use.
//
// This is effectively equivalent to [os.Open].
//
// [os.Open]: https://pkg.go.dev/os#Open
func (r *Root) Open(path string) (*os.File, error) {
	return r.OpenFile(path, os.O_RDONLY)
}

// OpenFile is effectively shorthand for [Resolve] followed by
// [Handle.OpenFile], but can be slightly more efficient (it reduces CGo
// overhead and the number of syscalls used when using the openat2-based
// resolver) and is arguably more ergonomic to use.
//
// However, if flags contains os.O_NOFOLLOW and the path is a symlink, then
// OpenFile's behaviour will match that of openat2. In most cases an error will
// be returned, but if os.O_PATH is provided along with os.O_NOFOLLOW then a
// file equivalent to [ResolveNoFollow] will be returned instead.
//
// This is effectively equivalent to [os.OpenFile], except that os.O_CREAT is
// not supported.
//
// [os.OpenFile]: https://pkg.go.dev/os#OpenFile
func (r *Root) OpenFile(path string, flags int) (*os.File, error) {
	return fdutils.WithFileFd(r.inner, func(rootFd uintptr) (*os.File, error) {
		fd, err := libpathrs.InRootOpen(rootFd, path, flags)
		if err != nil {
			return nil, err
		}
		return fdutils.MkFile(fd)
	})
}

// Create creates a file within the [Root]'s directory tree at the given path,
// and returns a handle to the file. The provided mode is used for the new file
// (the process's umask applies).
//
// Unlike [os.Create], if the file already exists an error is created rather
// than the file being opened and truncated.
//
// [os.Create]: https://pkg.go.dev/os#Create
func (r *Root) Create(path string, flags int, mode os.FileMode) (*os.File, error) {
	unixMode, err := toUnixMode(mode, false)
	if err != nil {
		return nil, err
	}
	return fdutils.WithFileFd(r.inner, func(rootFd uintptr) (*os.File, error) {
		handleFd, err := libpathrs.InRootCreat(rootFd, path, flags, unixMode)
		if err != nil {
			return nil, err
		}
		return fdutils.MkFile(handleFd)
	})
}

// Rename two paths within a [Root]'s directory tree. The flags argument is
// identical to the RENAME_* flags to the renameat2(2) system call.
func (r *Root) Rename(src, dst string, flags uint) error {
	_, err := fdutils.WithFileFd(r.inner, func(rootFd uintptr) (struct{}, error) {
		err := libpathrs.InRootRename(rootFd, src, dst, flags)
		return struct{}{}, err
	})
	return err
}

// RemoveDir removes the named empty directory within a [Root]'s directory
// tree.
func (r *Root) RemoveDir(path string) error {
	_, err := fdutils.WithFileFd(r.inner, func(rootFd uintptr) (struct{}, error) {
		err := libpathrs.InRootRmdir(rootFd, path)
		return struct{}{}, err
	})
	return err
}

// RemoveFile removes the named file within a [Root]'s directory tree.
func (r *Root) RemoveFile(path string) error {
	_, err := fdutils.WithFileFd(r.inner, func(rootFd uintptr) (struct{}, error) {
		err := libpathrs.InRootUnlink(rootFd, path)
		return struct{}{}, err
	})
	return err
}

// Remove removes the named file or (empty) directory within a [Root]'s
// directory tree.
//
// This is effectively equivalent to [os.Remove].
//
// [os.Remove]: https://pkg.go.dev/os#Remove
func (r *Root) Remove(path string) error {
	// In order to match os.Remove's implementation we need to also do both
	// syscalls unconditionally and adjust the error based on whether
	// pathrs_inroot_rmdir() returned ENOTDIR.
	unlinkErr := r.RemoveFile(path)
	if unlinkErr == nil {
		return nil
	}
	rmdirErr := r.RemoveDir(path)
	if rmdirErr == nil {
		return nil
	}
	// Both failed, adjust the error in the same way that os.Remove does.
	err := rmdirErr
	if errors.Is(err, syscall.ENOTDIR) {
		err = unlinkErr
	}
	return err
}

// RemoveAll recursively deletes a path and all of its children.
//
// This is effectively equivalent to [os.RemoveAll].
//
// [os.RemoveAll]: https://pkg.go.dev/os#RemoveAll
func (r *Root) RemoveAll(path string) error {
	_, err := fdutils.WithFileFd(r.inner, func(rootFd uintptr) (struct{}, error) {
		err := libpathrs.InRootRemoveAll(rootFd, path)
		return struct{}{}, err
	})
	return err
}

// Mkdir creates a directory within a [Root]'s directory tree. The provided
// mode is used for the new directory (the process's umask applies).
//
// This is effectively equivalent to [os.Mkdir].
//
// [os.Mkdir]: https://pkg.go.dev/os#Mkdir
func (r *Root) Mkdir(path string, mode os.FileMode) error {
	unixMode, err := toUnixMode(mode, false)
	if err != nil {
		return err
	}

	_, err = fdutils.WithFileFd(r.inner, func(rootFd uintptr) (struct{}, error) {
		err := libpathrs.InRootMkdir(rootFd, path, unixMode)
		return struct{}{}, err
	})
	return err
}

// MkdirAll creates a directory (and any parent path components if they don't
// exist) within a [Root]'s directory tree. The provided mode is used for any
// directories created by this function (the process's umask applies).
//
// This is effectively equivalent to [os.MkdirAll].
//
// [os.MkdirAll]: https://pkg.go.dev/os#MkdirAll
func (r *Root) MkdirAll(path string, mode os.FileMode) (*Handle, error) {
	unixMode, err := toUnixMode(mode, false)
	if err != nil {
		return nil, err
	}

	return fdutils.WithFileFd(r.inner, func(rootFd uintptr) (*Handle, error) {
		handleFd, err := libpathrs.InRootMkdirAll(rootFd, path, unixMode)
		if err != nil {
			return nil, err
		}
		handleFile, err := fdutils.MkFile(handleFd)
		if err != nil {
			return nil, err
		}
		return &Handle{inner: handleFile}, err
	})
}

// Mknod creates a new device inode of the given type within a [Root]'s
// directory tree. The provided mode is used for the new directory (the
// process's umask applies).
//
// This is effectively equivalent to [unix.Mknod].
//
// [unix.Mknod]: https://pkg.go.dev/golang.org/x/sys/unix#Mknod
func (r *Root) Mknod(path string, mode os.FileMode, dev uint64) error {
	unixMode, err := toUnixMode(mode, true)
	if err != nil {
		return err
	}

	_, err = fdutils.WithFileFd(r.inner, func(rootFd uintptr) (struct{}, error) {
		err := libpathrs.InRootMknod(rootFd, path, unixMode, dev)
		return struct{}{}, err
	})
	return err
}

// Symlink creates a symlink within a [Root]'s directory tree. The symlink is
// created at path and is a link to target.
//
// This is effectively equivalent to [os.Symlink].
//
// [os.Symlink]: https://pkg.go.dev/os#Symlink
func (r *Root) Symlink(path, target string) error {
	_, err := fdutils.WithFileFd(r.inner, func(rootFd uintptr) (struct{}, error) {
		err := libpathrs.InRootSymlink(rootFd, path, target)
		return struct{}{}, err
	})
	return err
}

// Hardlink creates a hardlink within a [Root]'s directory tree. The hardlink
// is created at path and is a link to target. Both paths are within the
// [Root]'s directory tree (you cannot hardlink to a different [Root] or the
// host).
//
// This is effectively equivalent to [os.Link].
//
// [os.Link]: https://pkg.go.dev/os#Link
func (r *Root) Hardlink(path, target string) error {
	_, err := fdutils.WithFileFd(r.inner, func(rootFd uintptr) (struct{}, error) {
		err := libpathrs.InRootHardlink(rootFd, path, target)
		return struct{}{}, err
	})
	return err
}

// Readlink returns the target of a symlink with a [Root]'s directory tree.
//
// This is effectively equivalent to [os.Readlink].
//
// [os.Readlink]: https://pkg.go.dev/os#Readlink
func (r *Root) Readlink(path string) (string, error) {
	return fdutils.WithFileFd(r.inner, func(rootFd uintptr) (string, error) {
		return libpathrs.InRootReadlink(rootFd, path)
	})
}

// IntoFile unwraps the [Root] into its underlying [os.File].
//
// It is critical that you do not operate on this file descriptor yourself,
// because the security properties of libpathrs depend on users doing all
// relevant filesystem operations through libpathrs.
//
// This operation returns the internal [os.File] of the [Root] directly, so
// calling [Root.Close] will also close any copies of the returned [os.File].
// If you want to get an independent copy, use [Root.Clone] followed by
// [Root.IntoFile] on the cloned [Root].
//
// [os.File]: https://pkg.go.dev/os#File
func (r *Root) IntoFile() *os.File {
	// TODO: Figure out if we really don't want to make a copy.
	// TODO: We almost certainly want to clear r.inner here, but we can't do
	//       that easily atomically (we could use atomic.Value but that'll make
	//       things quite a bit uglier).
	return r.inner
}

// Clone creates a copy of a [Root] handle, such that it has a separate
// lifetime to the original (while referring to the same underlying directory).
func (r *Root) Clone() (*Root, error) {
	return RootFromFile(r.inner)
}

// Close frees all of the resources used by the [Root] handle.
func (r *Root) Close() error {
	return r.inner.Close()
}
