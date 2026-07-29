// SPDX-License-Identifier: BSD-3-Clause

// Copyright (C) 2017-2024 SUSE LLC. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package securejoin

import "os"

// In future this should be moved into a separate package, because now there
// are several projects (umoci and go-mtree) that are using this sort of
// interface.

// VFS is the minimal interface necessary to use [SecureJoinVFS]. A nil VFS is
// equivalent to using the standard [os].* family of functions. This is mainly
// used for the purposes of mock testing, but also can be used to otherwise use
// [SecureJoinVFS] with VFS-like system.
type VFS interface {
	// Lstat returns an [os.FileInfo] describing the named file. If the
	// file is a symbolic link, the returned [os.FileInfo] describes the
	// symbolic link. Lstat makes no attempt to follow the link.
	// The semantics are identical to [os.Lstat].
	Lstat(name string) (os.FileInfo, error)

	// Readlink returns the destination of the named symbolic link.
	// The semantics are identical to [os.Readlink].
	Readlink(name string) (string, error)
}

// osVFS is the "nil" VFS, in that it just passes everything through to the os
// module.
type osVFS struct{}

func (o osVFS) Lstat(name string) (os.FileInfo, error) { return os.Lstat(name) }

func (o osVFS) Readlink(name string) (string, error) { return os.Readlink(name) }
