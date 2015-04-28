// Copyright (c) 2014 ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ql

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/cznic/mathutil"
)

var (
	_ http.FileSystem = (*HTTPFS)(nil)
	_ http.File       = (*HTTPFile)(nil)
	_ os.FileInfo     = (*HTTPFile)(nil)
	_ os.FileInfo     = (*dirEntry)(nil)
)

type dirEntry string

func (d dirEntry) Name() string       { return string(d) }
func (d dirEntry) Size() int64        { return -1 }
func (d dirEntry) Mode() os.FileMode  { return os.ModeDir }
func (d dirEntry) ModTime() time.Time { return time.Time{} }
func (d dirEntry) IsDir() bool        { return true }
func (d dirEntry) Sys() interface{}   { return interface{}(nil) }

// A HTTPFile is returned by the HTTPFS's Open method and can be served by the
// http.FileServer implementation.
type HTTPFile struct {
	closed     bool
	content    []byte
	dirEntries []os.FileInfo
	isFile     bool
	name       string
	off        int
	sz         int
}

// Close implements http.File.
func (f *HTTPFile) Close() error {
	if f.closed {
		return os.ErrInvalid
	}

	f.closed = true
	return nil
}

// IsDir implements os.FileInfo
func (f *HTTPFile) IsDir() bool { return !f.isFile }

// Mode implements os.FileInfo
func (f *HTTPFile) Mode() os.FileMode {
	switch f.isFile {
	case false:
		return os.FileMode(0444)
	default:
		return os.ModeDir
	}
}

// ModTime implements os.FileInfo
func (f *HTTPFile) ModTime() time.Time {
	return time.Time{}
}

// Name implements os.FileInfo
func (f *HTTPFile) Name() string { return path.Base(f.name) }

// Size implements os.FileInfo
func (f *HTTPFile) Size() int64 {
	switch f.isFile {
	case false:
		return -1
	default:
		return int64(len(f.content))
	}
}

// Stat implements http.File.
func (f *HTTPFile) Stat() (os.FileInfo, error) { return f, nil }

// Sys implements os.FileInfo
func (f *HTTPFile) Sys() interface{} { return interface{}(nil) }

// Readdir implements http.File.
func (f *HTTPFile) Readdir(count int) ([]os.FileInfo, error) {
	if f.isFile {
		return nil, fmt.Errorf("not a directory: %s", f.name)
	}

	if count <= 0 {
		r := f.dirEntries
		f.dirEntries = f.dirEntries[:0]
		return r, nil
	}

	rq := mathutil.Min(count, len(f.dirEntries))
	r := f.dirEntries[:rq]
	f.dirEntries = f.dirEntries[rq:]
	if len(r) != 0 {
		return r, nil
	}

	return nil, io.EOF
}

// Read implements http.File.
func (f *HTTPFile) Read(b []byte) (int, error) {
	if f.closed {
		return 0, os.ErrInvalid
	}

	n := copy(b, f.content[f.off:])
	f.off += n
	if n != 0 {
		return n, nil
	}

	return 0, io.EOF
}

// Seek implements http.File.
func (f *HTTPFile) Seek(offset int64, whence int) (int64, error) {
	if f.closed {
		return 0, os.ErrInvalid
	}

	if offset < 0 {
		return int64(f.off), fmt.Errorf("cannot seek before start of file")
	}

	switch whence {
	case 0:
		noff := int64(f.off) + offset
		if noff > mathutil.MaxInt {
			return int64(f.off), fmt.Errorf("seek target overflows int: %d", noff)
		}

		f.off = mathutil.Min(int(offset), len(f.content))
		if f.off == int(offset) {
			return offset, nil
		}

		return int64(f.off), io.EOF
	case 1:
		noff := int64(f.off) + offset
		if noff > mathutil.MaxInt {
			return int64(f.off), fmt.Errorf("seek target overflows int: %d", noff)
		}

		off := mathutil.Min(f.off+int(offset), len(f.content))
		if off == f.off+int(offset) {
			f.off = off
			return int64(off), nil
		}

		f.off = off
		return int64(off), io.EOF
	case 2:
		noff := int64(f.off) - offset
		if noff < 0 {
			return int64(f.off), fmt.Errorf("cannot seek before start of file")
		}

		f.off = len(f.content) - int(offset)
		return int64(f.off), nil
	default:
		return int64(f.off), fmt.Errorf("seek: invalid whence %d", whence)
	}
}

// HTTPFS implements a http.FileSystem backed by data in a DB.
type HTTPFS struct {
	db       *DB
	dir, get List
}

// NewHTTPFS returns a http.FileSystem backed by a result record set of query.
// The record set provides two mandatory fields: path and content (the field
// names are case sensitive). Type of name must be string and type of content
// must be blob (ie. []byte). Field 'path' value is the "file" pathname, which
// must be rooted; and field 'content' value is its "data".
func (db *DB) NewHTTPFS(query string) (*HTTPFS, error) {
	if _, err := Compile(query); err != nil {
		return nil, err
	}

	dir, err := Compile(fmt.Sprintf("SELECT path FROM (%s) WHERE hasPrefix(path, $1)", query))
	if err != nil {
		return nil, err
	}

	get, err := Compile(fmt.Sprintf("SELECT content FROM (%s) WHERE path == $1", query))
	if err != nil {
		return nil, err
	}

	return &HTTPFS{db: db, dir: dir, get: get}, nil
}

// Open implements http.FileSystem. The name parameter represents a file path.
// The elements in a file path are separated by slash ('/', U+002F) characters,
// regardless of host operating system convention.
func (f *HTTPFS) Open(name string) (http.File, error) {
	if filepath.Separator != '/' && strings.IndexRune(name, filepath.Separator) >= 0 ||
		strings.Contains(name, "\x00") {
		return nil, fmt.Errorf("invalid character in file path: %q", name)
	}

	name = path.Clean("/" + name)
	rs, _, err := f.db.Execute(nil, f.get, name)
	if err != nil {
		return nil, err
	}

	n := 0
	var fdata []byte
	if err = rs[0].Do(false, func(data []interface{}) (more bool, err error) {
		switch n {
		case 0:
			var ok bool
			fdata, ok = data[0].([]byte)
			if !ok {
				return false, fmt.Errorf("open: expected blob, got %T", data[0])
			}
			n++
			return true, nil
		default:
			return false, fmt.Errorf("open: more than one result was returned for %s", name)
		}
	}); err != nil {
		return nil, err
	}

	if n == 1 { // file found
		return &HTTPFile{name: name, isFile: true, content: fdata}, nil
	}

	dirName := name
	if dirName[len(dirName)-1] != filepath.Separator {
		dirName += string(filepath.Separator)
	}
	// Open("/a/b"): {/a/b/c.x,/a/b/d.x,/a/e.x,/a/b/f/g.x} -> {c.x,d.x,f}
	rs, _, err = f.db.Execute(nil, f.dir, dirName)
	if err != nil {
		return nil, err
	}

	n = 0
	r := &HTTPFile{name: dirName}
	m := map[string]bool{}
	x := len(dirName)
	if err = rs[0].Do(false, func(data []interface{}) (more bool, err error) {
		n++
		switch name := data[0].(type) {
		case string:
			if filepath.Separator != '/' && strings.IndexRune(name, filepath.Separator) >= 0 ||
				strings.Contains(name, "\x00") {
				return false, fmt.Errorf("invalid character in file path: %q", name)
			}

			name = path.Clean("/" + name)
			rest := name[x:]
			parts := strings.Split(rest, "/")
			if len(parts) == 0 {
				return true, nil
			}

			nm := parts[0]
			switch len(parts) {
			case 1: // file
				r.dirEntries = append(r.dirEntries, &HTTPFile{isFile: true, name: nm})
			default: // directory
				if !m[nm] {
					r.dirEntries = append(r.dirEntries, dirEntry(nm))
				}
				m[nm] = true
			}
			return true, nil
		default:
			return false, fmt.Errorf("expected string path, got %T(%v)", name, name)
		}
	}); err != nil {
		return nil, err
	}

	if n != 0 {
		return r, nil
	}

	return nil, os.ErrNotExist
}
