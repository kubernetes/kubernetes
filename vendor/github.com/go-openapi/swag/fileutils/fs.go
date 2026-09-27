// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package fileutils

import (
	"fmt"
	"io/fs"
	"os"
)

// OsFS exposes package os features as an [fs.FS], without having to use [os.Root].
//
// Existing alternatives from the standard library are [os.DirFS], which requires a base directory,
// and [os.Root.FS], which requires a root.
// [OsFS] is intended to be used when none of these alternatives are workable,
// that is when the caller does not know which root it should run in.
//
// Names are passed to the os package unchanged,
// so [OsFS] accepts absolute and relative paths, which a conforming [fs.FS] rejects.
// It offers no containment: every file that the process may read is reachable.
//
// [OsFS] implements [fs.FS], [fs.ReadFileFS] and [fs.ReadDirFS].
type OsFS struct {
}

// NewReadOnlyOsFS builds an [OsFS], a read-only view of the os file system.
func NewReadOnlyOsFS() *OsFS {
	return &OsFS{}
}

// Open opens the named file for reading.
func (f *OsFS) Open(name string) (fs.File, error) {
	return os.Open(name)
}

// ReadFile reads the named file and returns its content.
func (f *OsFS) ReadFile(name string) ([]byte, error) {
	return os.ReadFile(name)
}

// ReadDir reads the named directory and returns its entries sorted by file name.
func (f *OsFS) ReadDir(name string) ([]fs.DirEntry, error) {
	return os.ReadDir(name)
}

// FileReaderFS makes a [fs.FS] into a [fs.ReadFileFS], with a [FileReaderFS.ReadFile] method.
type FileReaderFS struct {
	fs.FS
}

// NewFileReaderFS transforms a [fs.FS] into a [fs.ReadFileFS].
func NewFileReaderFS(base fs.FS) *FileReaderFS {
	return &FileReaderFS{
		FS: base,
	}
}

// ReadFile reads the named file from the base file system and returns its content.
func (f *FileReaderFS) ReadFile(name string) ([]byte, error) {
	return fs.ReadFile(f.FS, name)
}

// GlobOsFS is an [OsFS] that also implements [fs.GlobFS], with a [GlobOsFS.Glob] method.
type GlobOsFS struct {
	*OsFS
}

// NewGlobOsFS is like [NewReadOnlyOsFS], augmented to match the [fs.GlobFS] interface.
func NewGlobOsFS() *GlobOsFS {
	return &GlobOsFS{
		OsFS: NewReadOnlyOsFS(),
	}
}

// Glob returns the names matching pattern, sorted in lexical order.
//
// It returns a nil slice and no error when nothing matches,
// and [path.ErrBadPattern] when the pattern is malformed.
func (f *GlobOsFS) Glob(pattern string) ([]string, error) {
	return fs.Glob(f.OsFS, pattern)
}

// MustSub re-roots a file system at one of its directories, and panics when it cannot.
//
// It is [fs.Sub] for the cases where the directory is a constant of the program, such as a folder
// of an [embed.FS] assembled at initialization time: there, a failure means the program is wrong,
// not that its input is.
//
// Use [fs.Sub] itself whenever the directory comes from the outside, such as a flag,
// a configuration file or a request, so that an invalid one is reported rather than fatal.
//
// It is meant to be composed inline:
//
//	assets := NewOverlayFS(
//		MustSub(embedded, "templates"),
//		MustSub(embedded, "templates/contrib/mine"),
//	)
func MustSub(fsys fs.FS, dir string) fs.FS {
	subFS, err := fs.Sub(fsys, dir)
	if err != nil {
		panic(fmt.Errorf("fileutils.MustSub: cannot re-root at %q: %w", dir, err))
	}

	return subFS
}
