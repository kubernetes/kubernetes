// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package fileutils

import (
	"bytes"
	"errors"
	"io"
	"io/fs"
	"path"
	"slices"
	"strings"
	"time"
)

// Default modes of the entries of a [MapFS].
//
// A [MapFS] is read-only, so its files and directories are readable and never writable.
const (
	// DefaultFileMode is the mode reported by a file with no [MapFile.Mode] of its own.
	DefaultFileMode fs.FileMode = 0o444

	// DefaultDirMode is the mode reported by the directories of a [MapFS].
	DefaultDirMode fs.FileMode = fs.ModeDir | 0o555
)

var (
	// errNameIsFileAndDir is reported when the same name is held as a file and as a parent directory.
	errNameIsFileAndDir = errors.New("name is held both as a file and as a directory")

	// errNotDir is reported when a regular file is listed as a directory.
	errNotDir = errors.New("not a directory")
)

// MapFile is a file held by a [MapFS].
//
// Only [MapFile.Data] is required. [NewMapFS] fills the remaining fields with presets when they
// are left to their zero value.
type MapFile struct {
	// Data is the content of the file.
	Data []byte

	// Mode is the file mode reported by [fs.FileInfo.Mode]. Zero means [DefaultFileMode].
	Mode fs.FileMode

	// ModTime is the modification time reported by [fs.FileInfo.ModTime].
	//
	// The zero value is left as is, so that a [MapFS] built from the same input twice
	// reports the same metadata.
	ModTime time.Time

	// Sys is the opaque value reported by [fs.FileInfo.Sys].
	Sys any
}

// MapFS is a read-only in-memory [fs.FS], built from a map of file names to content.
//
// It is intended for the cases where the files to serve are held in memory rather than on disk:
// an overlay assembled from raw bytes, assets that a configuration provides, or a fixture in a test.
//
// Names are slash-separated paths, as accepted by [fs.ValidPath]. [NewMapFS] cleans them,
// so a name may be given with a leading "./" or "/", or with redundant elements.
// A name that remains invalid once cleaned, such as one climbing above the root, is reported
// as an error rather than dropped.
//
// Separators are never translated, so that the same input yields the same file system on every
// platform: a caller holding os paths converts them with [path/filepath.ToSlash] beforehand.
//
// Directories are implied by the names of the files, and are indexed once, when the file system
// is built: a name holds a file, and every one of its parents holds a directory.
// The root "." always exists, even when the file system holds no file at all.
//
// [MapFS] implements [fs.FS], [fs.ReadFileFS], [fs.StatFS] and [fs.ReadDirFS], but not [fs.GlobFS]:
// [fs.Glob] resolves against it all the same, through [MapFS.ReadDir].
type MapFS struct {
	files map[string]MapFile
	dirs  map[string][]fs.DirEntry
}

// NewMapFS builds an in-memory file system from a map of file names to content.
//
// Names are normalized, and reported as an error when they remain invalid, or when the same name
// is held both as a file and as the parent directory of another one.
// A [MapFile] left with a zero [MapFile.Mode] reports [DefaultFileMode].
//
// The map is copied, so adding or removing an entry afterwards leaves the file system alone.
// The contents are not: a caller that writes to a [MapFile.Data] slice it still holds changes
// what the file system serves. Hand over a slice nothing else keeps, or copy it first.
func NewMapFS(files map[string]MapFile) (*MapFS, error) {
	normalized := make(map[string]MapFile, len(files))

	for name, file := range files {
		clean, err := normalizeMapName(name)
		if err != nil {
			return nil, err
		}

		if _, isDuplicate := normalized[clean]; isDuplicate {
			return nil, &fs.PathError{Op: "newmapfs", Path: clean, Err: fs.ErrExist}
		}

		if file.Mode == 0 {
			file.Mode = DefaultFileMode
		}

		normalized[clean] = file
	}

	dirs, err := indexMapDirs(normalized)
	if err != nil {
		return nil, err
	}

	return &MapFS{
		files: normalized,
		dirs:  dirs,
	}, nil
}

// FromRawMap builds the files of a [MapFS] from raw contents, leaving every metadata field to its preset.
//
// It is the shortest way to a [MapFS] when all the caller holds is bytes:
//
//	mapFS, err := NewMapFS(FromRawMap(map[string][]byte{
//		"folder/file1": raw1,
//		"folder/file2": raw2,
//	}))
func FromRawMap(raw map[string][]byte) map[string]MapFile {
	files := make(map[string]MapFile, len(raw))
	for name, data := range raw {
		files[name] = MapFile{Data: data}
	}

	return files
}

// Open opens the named file or directory.
//
// Opening a directory yields a [fs.ReadDirFile] reporting the same entries as [MapFS.ReadDir].
func (f *MapFS) Open(name string) (fs.File, error) {
	if !fs.ValidPath(name) {
		return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrInvalid}
	}

	if file, isFile := f.files[name]; isFile {
		return &openMapFile{
			info:   mapFileInfo{name: path.Base(name), file: file},
			reader: bytes.NewReader(file.Data),
		}, nil
	}

	entries, isDir := f.dirs[name]
	if !isDir {
		return nil, notFound("open", name)
	}

	return &openMapDir{info: mapDirInfo{name: path.Base(name)}, entries: entries}, nil
}

// ReadFile reads the named file and returns a copy of its content.
func (f *MapFS) ReadFile(name string) ([]byte, error) {
	if !fs.ValidPath(name) {
		return nil, &fs.PathError{Op: "read", Path: name, Err: fs.ErrInvalid}
	}

	file, isFile := f.files[name]
	if !isFile {
		if _, isDir := f.dirs[name]; isDir {
			return nil, &fs.PathError{Op: "read", Path: name, Err: errIsDir}
		}

		return nil, notFound("read", name)
	}

	// a caller mutating the result would otherwise mutate what the file system serves
	return slices.Clone(file.Data), nil
}

// Stat returns the [fs.FileInfo] of the named file or directory.
func (f *MapFS) Stat(name string) (fs.FileInfo, error) {
	if !fs.ValidPath(name) {
		return nil, &fs.PathError{Op: "stat", Path: name, Err: fs.ErrInvalid}
	}

	if file, isFile := f.files[name]; isFile {
		return mapFileInfo{name: path.Base(name), file: file}, nil
	}

	if _, isDir := f.dirs[name]; isDir {
		return mapDirInfo{name: path.Base(name)}, nil
	}

	return nil, notFound("stat", name)
}

// ReadDir lists the named directory, with its entries sorted by file name.
func (f *MapFS) ReadDir(name string) ([]fs.DirEntry, error) {
	if !fs.ValidPath(name) {
		return nil, &fs.PathError{Op: "readdir", Path: name, Err: fs.ErrInvalid}
	}

	entries, isDir := f.dirs[name]
	if !isDir {
		if _, isFile := f.files[name]; isFile {
			return nil, &fs.PathError{Op: "readdir", Path: name, Err: errNotDir}
		}

		return nil, notFound("readdir", name)
	}

	// the index is shared by every caller, so it must not escape
	return slices.Clone(entries), nil
}

// normalizeMapName turns the name of a file into the cleaned form that [fs.ValidPath] accepts.
//
// Separators are left alone: an [fs.FS] name is slash-separated by definition, and translating
// them here would resolve the same input differently depending on the platform.
func normalizeMapName(name string) (string, error) {
	clean := path.Clean(strings.TrimPrefix(name, "/"))

	if !fs.ValidPath(clean) || clean == "." {
		return "", &fs.PathError{Op: "newmapfs", Path: name, Err: fs.ErrInvalid}
	}

	return clean, nil
}

// indexMapDirs builds the directory index of a [MapFS], once, from the names of its files.
//
// Every parent of a file name holds a directory, up to the root, which always exists.
func indexMapDirs(files map[string]MapFile) (map[string][]fs.DirEntry, error) {
	children := map[string]map[string]bool{".": {}} // directory -> child base name -> is a directory

	for name := range files {
		dir := path.Dir(name)
		addMapChild(children, dir, path.Base(name), false)

		// every ancestor of the file holds a directory
		for dir != "." {
			parent := path.Dir(dir)
			addMapChild(children, parent, path.Base(dir), true)
			dir = parent
		}
	}

	dirs := make(map[string][]fs.DirEntry, len(children))
	for dir, names := range children {
		if _, isFile := files[dir]; isFile {
			return nil, &fs.PathError{Op: "newmapfs", Path: dir, Err: errNameIsFileAndDir}
		}

		entries := make([]fs.DirEntry, 0, len(names))
		for base, isDir := range names {
			if isDir {
				entries = append(entries, mapDirInfo{name: base})

				continue
			}

			entries = append(entries, mapFileInfo{name: base, file: files[path.Join(dir, base)]})
		}

		slices.SortFunc(entries, func(a, b fs.DirEntry) int {
			return strings.Compare(a.Name(), b.Name())
		})
		dirs[dir] = entries
	}

	return dirs, nil
}

// addMapChild records that a directory holds an entry.
func addMapChild(children map[string]map[string]bool, dir, base string, isDir bool) {
	entries, exists := children[dir]
	if !exists {
		entries = make(map[string]bool)
		children[dir] = entries
	}

	entries[base] = isDir
}

// mapFileInfo reports the metadata of a file of a [MapFS]. It is both a [fs.FileInfo] and a [fs.DirEntry].
type mapFileInfo struct {
	name string
	file MapFile
}

func (i mapFileInfo) Name() string               { return i.name }
func (i mapFileInfo) Size() int64                { return int64(len(i.file.Data)) }
func (i mapFileInfo) Mode() fs.FileMode          { return i.file.Mode }
func (i mapFileInfo) Type() fs.FileMode          { return i.file.Mode.Type() }
func (i mapFileInfo) ModTime() time.Time         { return i.file.ModTime }
func (i mapFileInfo) IsDir() bool                { return false }
func (i mapFileInfo) Sys() any                   { return i.file.Sys }
func (i mapFileInfo) Info() (fs.FileInfo, error) { return i, nil }

// mapDirInfo reports the metadata of a directory of a [MapFS]. It is both a [fs.FileInfo] and a [fs.DirEntry].
//
// Directories are implied by the names of the files, so they carry no metadata of their own.
type mapDirInfo struct {
	name string
}

func (i mapDirInfo) Name() string               { return i.name }
func (i mapDirInfo) Size() int64                { return 0 }
func (i mapDirInfo) Mode() fs.FileMode          { return DefaultDirMode }
func (i mapDirInfo) Type() fs.FileMode          { return fs.ModeDir }
func (i mapDirInfo) ModTime() time.Time         { return time.Time{} }
func (i mapDirInfo) IsDir() bool                { return true }
func (i mapDirInfo) Sys() any                   { return nil }
func (i mapDirInfo) Info() (fs.FileInfo, error) { return i, nil }

// openMapFile is the [fs.File] returned when opening a file of a [MapFS].
type openMapFile struct {
	info   mapFileInfo
	reader *bytes.Reader
}

func (f *openMapFile) Stat() (fs.FileInfo, error) { return f.info, nil }

func (f *openMapFile) Close() error { return nil }

func (f *openMapFile) Read(p []byte) (int, error) { return f.reader.Read(p) }

func (f *openMapFile) Seek(offset int64, whence int) (int64, error) {
	return f.reader.Seek(offset, whence)
}

// openMapDir is the [fs.ReadDirFile] returned when opening a directory of a [MapFS].
type openMapDir struct {
	info    mapDirInfo
	entries []fs.DirEntry
	offset  int
}

func (d *openMapDir) Stat() (fs.FileInfo, error) { return d.info, nil }

func (d *openMapDir) Close() error { return nil }

func (d *openMapDir) Read([]byte) (int, error) {
	return 0, &fs.PathError{Op: "read", Path: d.info.name, Err: errIsDir}
}

// ReadDir returns the next n entries, or all the remaining ones when n is not positive.
func (d *openMapDir) ReadDir(n int) ([]fs.DirEntry, error) {
	remaining := len(d.entries) - d.offset
	if n <= 0 {
		entries := slices.Clone(d.entries[d.offset:])
		d.offset = len(d.entries)

		return entries, nil
	}

	if remaining == 0 {
		return nil, io.EOF
	}

	n = min(n, remaining)
	entries := slices.Clone(d.entries[d.offset : d.offset+n])
	d.offset += n

	return entries, nil
}
