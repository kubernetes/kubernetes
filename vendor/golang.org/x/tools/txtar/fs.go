// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package txtar

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"path"
	"slices"
	"time"
)

// FS returns the file system form of an Archive.
// It returns an error if any of the file names in the archive
// are not valid file system names.
// The archive must not be modified while the FS is in use.
//
// If the file system detects that it has been modified, calls to the
// file system return an ErrModified error.
func FS(a *Archive) (fs.FS, error) {
	// Create a filesystem with a root directory.
	root := &node{fileinfo: fileinfo{path: ".", mode: readOnlyDir}}
	fsys := &filesystem{a, map[string]*node{root.path: root}}

	if err := initFiles(fsys); err != nil {
		return nil, fmt.Errorf("cannot create fs.FS from txtar.Archive: %s", err)
	}
	return fsys, nil
}

const (
	readOnly    fs.FileMode = 0o444 // read only mode
	readOnlyDir             = readOnly | fs.ModeDir
)

// ErrModified indicates that file system returned by FS
// noticed that the underlying archive has been modified
// since the call to FS. Detection of modification is best effort,
// to help diagnose misuse of the API, and is not guaranteed.
var ErrModified error = errors.New("txtar.Archive has been modified during txtar.FS")

// A filesystem is a simple in-memory file system for txtar archives,
// represented as a map from valid path names to information about the
// files or directories they represent.
//
// File system operations are read only. Modifications to the underlying
// *Archive may race. To help prevent this, the filesystem tries
// to detect modification during Open and return ErrModified if it
// is able to detect a modification.
type filesystem struct {
	ar    *Archive
	nodes map[string]*node
}

// node is a file or directory in the tree of a filesystem.
type node struct {
	fileinfo               // fs.FileInfo and fs.DirEntry implementation
	idx      int           // index into ar.Files (for files)
	entries  []fs.DirEntry // subdirectories and files (for directories)
}

var _ fs.FS = (*filesystem)(nil)
var _ fs.DirEntry = (*node)(nil)

// initFiles initializes fsys from fsys.ar.Files. Returns an error if there are any
// invalid file names or collisions between file or directories.
func initFiles(fsys *filesystem) error {
	for idx, file := range fsys.ar.Files {
		name := file.Name
		if !fs.ValidPath(name) {
			return fmt.Errorf("file %q is an invalid path", name)
		}

		n := &node{idx: idx, fileinfo: fileinfo{path: name, size: len(file.Data), mode: readOnly}}
		if err := insert(fsys, n); err != nil {
			return err
		}
	}
	return nil
}

// insert adds node n as an entry to its parent directory within the filesystem.
func insert(fsys *filesystem, n *node) error {
	if m := fsys.nodes[n.path]; m != nil {
		return fmt.Errorf("duplicate path %q", n.path)
	}
	fsys.nodes[n.path] = n

	// fsys.nodes contains "." to prevent infinite loops.
	parent, err := directory(fsys, path.Dir(n.path))
	if err != nil {
		return err
	}
	parent.entries = append(parent.entries, n)
	return nil
}

// directory returns the directory node with the path dir and lazily-creates it
// if it does not exist.
func directory(fsys *filesystem, dir string) (*node, error) {
	if m := fsys.nodes[dir]; m != nil && m.IsDir() {
		return m, nil // pre-existing directory
	}

	n := &node{fileinfo: fileinfo{path: dir, mode: readOnlyDir}}
	if err := insert(fsys, n); err != nil {
		return nil, err
	}
	return n, nil
}

// dataOf returns the data associated with the file t.
// May return ErrModified if fsys.ar has been modified.
func dataOf(fsys *filesystem, n *node) ([]byte, error) {
	if n.idx >= len(fsys.ar.Files) {
		return nil, ErrModified
	}

	f := fsys.ar.Files[n.idx]
	if f.Name != n.path || len(f.Data) != n.size {
		return nil, ErrModified
	}
	return f.Data, nil
}

func (fsys *filesystem) Open(name string) (fs.File, error) {
	if !fs.ValidPath(name) {
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrInvalid}
	}

	n := fsys.nodes[name]
	switch {
	case n == nil:
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
	case n.IsDir():
		return &openDir{fileinfo: n.fileinfo, entries: n.entries}, nil
	default:
		data, err := dataOf(fsys, n)
		if err != nil {
			return nil, err
		}
		return &openFile{fileinfo: n.fileinfo, data: data}, nil
	}
}

func (fsys *filesystem) ReadFile(name string) ([]byte, error) {
	file, err := fsys.Open(name)
	if err != nil {
		return nil, err
	}
	if file, ok := file.(*openFile); ok {
		return slices.Clone(file.data), nil
	}
	return nil, &fs.PathError{Op: "read", Path: name, Err: fs.ErrInvalid}
}

// A fileinfo implements fs.FileInfo and fs.DirEntry for a given archive file.
type fileinfo struct {
	path string // unique path to the file or directory within a filesystem
	size int
	mode fs.FileMode
}

var _ fs.FileInfo = (*fileinfo)(nil)
var _ fs.DirEntry = (*fileinfo)(nil)

func (i *fileinfo) Name() string               { return path.Base(i.path) }
func (i *fileinfo) Size() int64                { return int64(i.size) }
func (i *fileinfo) Mode() fs.FileMode          { return i.mode }
func (i *fileinfo) Type() fs.FileMode          { return i.mode.Type() }
func (i *fileinfo) ModTime() time.Time         { return time.Time{} }
func (i *fileinfo) IsDir() bool                { return i.mode&fs.ModeDir != 0 }
func (i *fileinfo) Sys() any                   { return nil }
func (i *fileinfo) Info() (fs.FileInfo, error) { return i, nil }

// An openFile is a regular (non-directory) fs.File open for reading.
type openFile struct {
	fileinfo
	data   []byte
	offset int64
}

var _ fs.File = (*openFile)(nil)

func (f *openFile) Stat() (fs.FileInfo, error) { return &f.fileinfo, nil }
func (f *openFile) Close() error               { return nil }
func (f *openFile) Read(b []byte) (int, error) {
	if f.offset >= int64(len(f.data)) {
		return 0, io.EOF
	}
	if f.offset < 0 {
		return 0, &fs.PathError{Op: "read", Path: f.path, Err: fs.ErrInvalid}
	}
	n := copy(b, f.data[f.offset:])
	f.offset += int64(n)
	return n, nil
}

func (f *openFile) Seek(offset int64, whence int) (int64, error) {
	switch whence {
	case 0:
		// offset += 0
	case 1:
		offset += f.offset
	case 2:
		offset += int64(len(f.data))
	}
	if offset < 0 || offset > int64(len(f.data)) {
		return 0, &fs.PathError{Op: "seek", Path: f.path, Err: fs.ErrInvalid}
	}
	f.offset = offset
	return offset, nil
}

func (f *openFile) ReadAt(b []byte, offset int64) (int, error) {
	if offset < 0 || offset > int64(len(f.data)) {
		return 0, &fs.PathError{Op: "read", Path: f.path, Err: fs.ErrInvalid}
	}
	n := copy(b, f.data[offset:])
	if n < len(b) {
		return n, io.EOF
	}
	return n, nil
}

// A openDir is a directory fs.File (so also an fs.ReadDirFile) open for reading.
type openDir struct {
	fileinfo
	entries []fs.DirEntry
	offset  int
}

var _ fs.ReadDirFile = (*openDir)(nil)

func (d *openDir) Stat() (fs.FileInfo, error) { return &d.fileinfo, nil }
func (d *openDir) Close() error               { return nil }
func (d *openDir) Read(b []byte) (int, error) {
	return 0, &fs.PathError{Op: "read", Path: d.path, Err: fs.ErrInvalid}
}

func (d *openDir) ReadDir(count int) ([]fs.DirEntry, error) {
	n := len(d.entries) - d.offset
	if n == 0 && count > 0 {
		return nil, io.EOF
	}
	if count > 0 && n > count {
		n = count
	}
	list := make([]fs.DirEntry, n)
	copy(list, d.entries[d.offset:d.offset+n])
	d.offset += n
	return list, nil
}
