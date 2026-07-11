// SPDX-License-Identifier: BSD-3-Clause

// Copyright (C) 2014-2015 Docker Inc & Go Authors. All rights reserved.
// Copyright (C) 2017-2025 SUSE LLC. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package securejoin

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/cyphar/filepath-securejoin/internal/consts"
)

// IsNotExist tells you if err is an error that implies that either the path
// accessed does not exist (or path components don't exist). This is
// effectively a more broad version of [os.IsNotExist].
func IsNotExist(err error) bool {
	// Check that it's not actually an ENOTDIR, which in some cases is a more
	// convoluted case of ENOENT (usually involving weird paths).
	return errors.Is(err, os.ErrNotExist) || errors.Is(err, syscall.ENOTDIR) || errors.Is(err, syscall.ENOENT)
}

// errUnsafeRoot is returned if the user provides SecureJoinVFS with a path
// that contains ".." components.
var errUnsafeRoot = errors.New("root path provided to SecureJoin contains '..' components")

// stripVolume just gets rid of the Windows volume included in a path. Based on
// some godbolt tests, the Go compiler is smart enough to make this a no-op on
// Linux.
func stripVolume(path string) string {
	return path[len(filepath.VolumeName(path)):]
}

// hasDotDot checks if the path contains ".." components in a platform-agnostic
// way.
func hasDotDot(path string) bool {
	// If we are on Windows, strip any volume letters. It turns out that
	// C:..\foo may (or may not) be a valid pathname and we need to handle that
	// leading "..".
	path = stripVolume(path)
	// Look for "/../" in the path, but we need to handle leading and trailing
	// ".."s by adding separators. Doing this with filepath.Separator is ugly
	// so just convert to Unix-style "/" first.
	path = filepath.ToSlash(path)
	return strings.Contains("/"+path+"/", "/../")
}

// SecureJoinVFS joins the two given path components (similar to
// [filepath.Join]) except that the returned path is guaranteed to be scoped
// inside the provided root path (when evaluated). Any symbolic links in the
// path are evaluated with the given root treated as the root of the
// filesystem, similar to a chroot. The filesystem state is evaluated through
// the given [VFS] interface (if nil, the standard [os].* family of functions
// are used).
//
// Note that the guarantees provided by this function only apply if the path
// components in the returned string are not modified (in other words are not
// replaced with symlinks on the filesystem) after this function has returned.
// Such a symlink race is necessarily out-of-scope of SecureJoinVFS.
//
// NOTE: Due to the above limitation, Linux users are strongly encouraged to
// use [OpenInRoot] instead, which does safely protect against these kinds of
// attacks. There is no way to solve this problem with SecureJoinVFS because
// the API is fundamentally wrong (you cannot return a "safe" path string and
// guarantee it won't be modified afterwards).
//
// Volume names in unsafePath are always discarded, regardless if they are
// provided via direct input or when evaluating symlinks. Therefore:
//
// "C:\Temp" + "D:\path\to\file.txt" results in "C:\Temp\path\to\file.txt"
//
// If the provided root is not [filepath.Clean] then an error will be returned,
// as such root paths are bordering on somewhat unsafe and using such paths is
// not best practice. We also strongly suggest that any root path is first
// fully resolved using [filepath.EvalSymlinks] or otherwise constructed to
// avoid containing symlink components. Of course, the root also *must not* be
// attacker-controlled.
func SecureJoinVFS(root, unsafePath string, vfs VFS) (string, error) { //nolint:revive // name is part of public API
	// The root path must not contain ".." components, otherwise when we join
	// the subpath we will end up with a weird path. We could work around this
	// in other ways but users shouldn't be giving us non-lexical root paths in
	// the first place.
	if hasDotDot(root) {
		return "", errUnsafeRoot
	}

	// Use the os.* VFS implementation if none was specified.
	if vfs == nil {
		vfs = osVFS{}
	}

	unsafePath = filepath.FromSlash(unsafePath)
	var (
		currentPath   string
		remainingPath = unsafePath
		linksWalked   int
	)
	for remainingPath != "" {
		// On Windows, if we managed to end up at a path referencing a volume,
		// drop the volume to make sure we don't end up with broken paths or
		// escaping the root volume.
		remainingPath = stripVolume(remainingPath)

		// Get the next path component.
		var part string
		if i := strings.IndexRune(remainingPath, filepath.Separator); i == -1 {
			part, remainingPath = remainingPath, ""
		} else {
			part, remainingPath = remainingPath[:i], remainingPath[i+1:]
		}

		// Apply the component lexically to the path we are building.
		// currentPath does not contain any symlinks, and we are lexically
		// dealing with a single component, so it's okay to do a filepath.Clean
		// here.
		nextPath := filepath.Join(string(filepath.Separator), currentPath, part)
		if nextPath == string(filepath.Separator) {
			currentPath = ""
			continue
		}
		fullPath := root + string(filepath.Separator) + nextPath

		// Figure out whether the path is a symlink.
		fi, err := vfs.Lstat(fullPath)
		if err != nil && !IsNotExist(err) {
			return "", err
		}
		// Treat non-existent path components the same as non-symlinks (we
		// can't do any better here).
		if IsNotExist(err) || fi.Mode()&os.ModeSymlink == 0 {
			currentPath = nextPath
			continue
		}

		// It's a symlink, so get its contents and expand it by prepending it
		// to the yet-unparsed path.
		linksWalked++
		if linksWalked > consts.MaxSymlinkLimit {
			return "", &os.PathError{Op: "SecureJoin", Path: root + string(filepath.Separator) + unsafePath, Err: syscall.ELOOP}
		}

		dest, err := vfs.Readlink(fullPath)
		if err != nil {
			return "", err
		}
		remainingPath = dest + string(filepath.Separator) + remainingPath
		// Absolute symlinks reset any work we've already done.
		if filepath.IsAbs(dest) {
			currentPath = ""
		}
	}

	// There should be no lexical components like ".." left in the path here,
	// but for safety clean up the path before joining it to the root.
	finalPath := filepath.Join(string(filepath.Separator), currentPath)
	return filepath.Join(root, finalPath), nil
}

// SecureJoin is a wrapper around [SecureJoinVFS] that just uses the [os].* library
// of functions as the [VFS]. If in doubt, use this function over [SecureJoinVFS].
func SecureJoin(root, unsafePath string) (string, error) {
	return SecureJoinVFS(root, unsafePath, nil)
}
