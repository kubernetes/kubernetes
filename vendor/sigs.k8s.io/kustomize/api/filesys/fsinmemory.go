// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filesys

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

var _ FileSystem = &fsInMemory{}

// fsInMemory implements FileSystem using a in-memory filesystem
// primarily for use in tests.
type fsInMemory struct {
	m map[string]*fileInMemory
}

// MakeFsInMemory returns an instance of fsInMemory with no files in it.
func MakeFsInMemory() FileSystem {
	result := &fsInMemory{m: map[string]*fileInMemory{}}
	result.Mkdir(separator)
	return result
}

const (
	separator = string(filepath.Separator)
	doubleSep = separator + separator
)

// Create assures a fake file appears in the in-memory file system.
func (fs *fsInMemory) Create(name string) (File, error) {
	f := &fileInMemory{}
	f.open = true
	fs.m[name] = f
	return fs.m[name], nil
}

// Mkdir assures a fake directory appears in the in-memory file system.
func (fs *fsInMemory) Mkdir(name string) error {
	fs.m[name] = makeDir(name)
	return nil
}

// MkdirAll delegates to Mkdir
func (fs *fsInMemory) MkdirAll(name string) error {
	return fs.Mkdir(name)
}

// RemoveAll presumably does rm -r on a path.
// There's no error.
func (fs *fsInMemory) RemoveAll(name string) error {
	var toRemove []string
	for k := range fs.m {
		if strings.HasPrefix(k, name) {
			toRemove = append(toRemove, k)
		}
	}
	for _, k := range toRemove {
		delete(fs.m, k)
	}
	return nil
}

// Open returns a fake file in the open state.
func (fs *fsInMemory) Open(name string) (File, error) {
	if _, found := fs.m[name]; !found {
		return nil, fmt.Errorf("file %q cannot be opened", name)
	}
	return fs.m[name], nil
}

// CleanedAbs cannot fail.
func (fs *fsInMemory) CleanedAbs(path string) (ConfirmedDir, string, error) {
	if fs.IsDir(path) {
		return ConfirmedDir(path), "", nil
	}
	d := filepath.Dir(path)
	if d == path {
		return ConfirmedDir(d), "", nil
	}
	return ConfirmedDir(d), filepath.Base(path), nil
}

// Exists returns true if file is known.
func (fs *fsInMemory) Exists(name string) bool {
	_, found := fs.m[name]
	return found
}

// Glob returns the list of matching files
func (fs *fsInMemory) Glob(pattern string) ([]string, error) {
	var result []string
	for p := range fs.m {
		if fs.pathMatch(p, pattern) {
			result = append(result, p)
		}
	}
	sort.Strings(result)
	return result, nil
}

// IsDir returns true if the file exists and is a directory.
func (fs *fsInMemory) IsDir(name string) bool {
	f, found := fs.m[name]
	if found && f.dir {
		return true
	}
	if !strings.HasSuffix(name, separator) {
		name = name + separator
	}
	for k := range fs.m {
		if strings.HasPrefix(k, name) {
			return true
		}
	}
	return false
}

// ReadFile always returns an empty bytes and error depending on content of m.
func (fs *fsInMemory) ReadFile(name string) ([]byte, error) {
	if ff, found := fs.m[name]; found {
		return ff.content, nil
	}
	return nil, fmt.Errorf("cannot read file %q", name)
}

// WriteFile always succeeds and does nothing.
func (fs *fsInMemory) WriteFile(name string, c []byte) error {
	ff := &fileInMemory{}
	ff.Write(c)
	fs.m[name] = ff
	return nil
}

// Walk implements filepath.Walk using the fake filesystem.
func (fs *fsInMemory) Walk(path string, walkFn filepath.WalkFunc) error {
	info, err := fs.lstat(path)
	if err != nil {
		err = walkFn(path, info, err)
	} else {
		err = fs.walk(path, info, walkFn)
	}
	if err == filepath.SkipDir {
		return nil
	}
	return err
}

func (fs *fsInMemory) pathMatch(path, pattern string) bool {
	match, _ := filepath.Match(pattern, path)
	return match
}

func (fs *fsInMemory) lstat(path string) (*fileInfo, error) {
	f, found := fs.m[path]
	if !found {
		return nil, os.ErrNotExist
	}
	return &fileInfo{f}, nil
}

func (fs *fsInMemory) join(elem ...string) string {
	for i, e := range elem {
		if e != "" {
			return strings.Replace(
				strings.Join(elem[i:], separator), doubleSep, separator, -1)
		}
	}
	return ""
}

func (fs *fsInMemory) readDirNames(path string) []string {
	var names []string
	if !strings.HasSuffix(path, separator) {
		path += separator
	}
	pathSegments := strings.Count(path, separator)
	for name := range fs.m {
		if name == path {
			continue
		}
		if strings.Count(name, separator) > pathSegments {
			continue
		}
		if strings.HasPrefix(name, path) {
			names = append(names, filepath.Base(name))
		}
	}
	sort.Strings(names)
	return names
}

func (fs *fsInMemory) walk(path string, info os.FileInfo, walkFn filepath.WalkFunc) error {
	if !info.IsDir() {
		return walkFn(path, info, nil)
	}

	names := fs.readDirNames(path)
	if err := walkFn(path, info, nil); err != nil {
		return err
	}
	for _, name := range names {
		filename := fs.join(path, name)
		fileInfo, err := fs.lstat(filename)
		if err != nil {
			if err := walkFn(filename, fileInfo, os.ErrNotExist); err != nil && err != filepath.SkipDir {
				return err
			}
		} else {
			err = fs.walk(filename, fileInfo, walkFn)
			if err != nil {
				if !fileInfo.IsDir() || err != filepath.SkipDir {
					return err
				}
			}
		}
	}
	return nil
}
