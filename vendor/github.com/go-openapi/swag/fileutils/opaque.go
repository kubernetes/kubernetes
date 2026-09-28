// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package fileutils

import (
	"io/fs"
	"path"
)

// OpaqueDirFS is a file system that entirely owns some of its directories.
//
// [OverlayFS] neither merges an opaque directory with the layers below it,
// nor resolves any name under it against them:
// what the owning layer holds replaces what they hold under the same name.
//
// [OpaqueFS] is the implementation provided by this package.
type OpaqueDirFS interface {
	fs.FS

	// IsOpaqueDir reports whether a directory was declared as entirely owned by this file system.
	//
	// It answers from the declarations given to [NewOpaqueFS], and does not check that the file
	// system holds the directory: a name declared opaque reports true either way.
	// [OverlayFS] consults it only for a layer that holds the directory.
	IsOpaqueDir(name string) bool
}

// OpaqueDirsAll declares that a file system owns every directory it holds, except its root.
//
// It is the only pattern recognized by [NewOpaqueFS]: directories are otherwise matched
// by their exact name.
const OpaqueDirsAll = "*"

// OpaqueFS makes a [fs.FS] into an [OpaqueDirFS], by declaring the directories that it owns.
type OpaqueFS struct {
	fs.FS

	opaqueDirs map[string]struct{}
}

// NewOpaqueFS declares the directories that a file system entirely owns,
// so that they shadow the layers below when it is stacked in an [OverlayFS].
//
// Directories are slash-separated paths, as accepted by [fs.ValidPath], and are cleaned.
// Declaring "." makes the whole file system opaque, down from its root.
// Declaring [OpaqueDirsAll] makes every directory opaque but the root,
// so that the root still merges and the layers below keep contributing what they hold beside it.
//
// Declaring a directory that this file system does not hold changes nothing once it is stacked.
// [OverlayFS] skips a layer that does not hold the directory before it consults opacity,
// so the layers below keep resolving that directory and everything under it.
func NewOpaqueFS(base fs.FS, dirs ...string) *OpaqueFS {
	opaqueDirs := make(map[string]struct{}, len(dirs))
	for _, dir := range dirs {
		opaqueDirs[path.Clean(dir)] = struct{}{}
	}

	return &OpaqueFS{
		FS:         base,
		opaqueDirs: opaqueDirs,
	}
}

// IsOpaqueDir reports whether a directory was declared as entirely owned by this file system.
//
// It answers from the declarations given to [NewOpaqueFS], and does not check that the file
// system holds the directory: a name declared opaque reports true either way.
// [OverlayFS] consults it only for a layer that holds the directory.
func (f *OpaqueFS) IsOpaqueDir(name string) bool {
	if _, isOpaque := f.opaqueDirs[name]; isOpaque {
		return true
	}

	if name == "." {
		// the root is owned only when it is declared explicitly
		return false
	}

	_, ownsAll := f.opaqueDirs[OpaqueDirsAll]

	return ownsAll
}

// ReadFile reads a file from the wrapped file system.
func (f *OpaqueFS) ReadFile(name string) ([]byte, error) {
	return fs.ReadFile(f.FS, name)
}

// Stat returns the [fs.FileInfo] of a name in the wrapped file system.
func (f *OpaqueFS) Stat(name string) (fs.FileInfo, error) {
	return fs.Stat(f.FS, name)
}

// ReadDir lists a directory of the wrapped file system.
func (f *OpaqueFS) ReadDir(name string) ([]fs.DirEntry, error) {
	return fs.ReadDir(f.FS, name)
}
