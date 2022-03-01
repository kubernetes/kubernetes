// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filesys

import (
	"path/filepath"
)

const (
	Separator = string(filepath.Separator)
	SelfDir   = "."
	ParentDir = ".."
)

// FileSystem groups basic os filesystem methods.
// It's supposed be functional subset of https://golang.org/pkg/os
type FileSystem interface {
	// Create a file.
	Create(path string) (File, error)
	// MkDir makes a directory.
	Mkdir(path string) error
	// MkDirAll makes a directory path, creating intervening directories.
	MkdirAll(path string) error
	// RemoveAll removes path and any children it contains.
	RemoveAll(path string) error
	// Open opens the named file for reading.
	Open(path string) (File, error)
	// IsDir returns true if the path is a directory.
	IsDir(path string) bool
	// ReadDir returns a list of files and directories within a directory.
	ReadDir(path string) ([]string, error)
	// CleanedAbs converts the given path into a
	// directory and a file name, where the directory
	// is represented as a ConfirmedDir and all that implies.
	// If the entire path is a directory, the file component
	// is an empty string.
	CleanedAbs(path string) (ConfirmedDir, string, error)
	// Exists is true if the path exists in the file system.
	Exists(path string) bool
	// Glob returns the list of matching files,
	// emulating https://golang.org/pkg/path/filepath/#Glob
	Glob(pattern string) ([]string, error)
	// ReadFile returns the contents of the file at the given path.
	ReadFile(path string) ([]byte, error)
	// WriteFile writes the data to a file at the given path,
	// overwriting anything that's already there.
	WriteFile(path string, data []byte) error
	// Walk walks the file system with the given WalkFunc.
	Walk(path string, walkFn filepath.WalkFunc) error
}

// FileSystemOrOnDisk satisfies the FileSystem interface by forwarding
// all of its method calls to the given FileSystem whenever it's not nil.
// If it's nil, the call is forwarded to the OS's underlying file system.
type FileSystemOrOnDisk struct {
	FileSystem FileSystem
}

// Set sets the given FileSystem as the target for all the FileSystem method calls.
func (fs *FileSystemOrOnDisk) Set(f FileSystem) { fs.FileSystem = f }

func (fs FileSystemOrOnDisk) fs() FileSystem {
	if fs.FileSystem != nil {
		return fs.FileSystem
	}
	return MakeFsOnDisk()
}

func (fs FileSystemOrOnDisk) Create(path string) (File, error) {
	return fs.fs().Create(path)
}

func (fs FileSystemOrOnDisk) Mkdir(path string) error {
	return fs.fs().Mkdir(path)
}

func (fs FileSystemOrOnDisk) MkdirAll(path string) error {
	return fs.fs().MkdirAll(path)
}

func (fs FileSystemOrOnDisk) RemoveAll(path string) error {
	return fs.fs().RemoveAll(path)
}

func (fs FileSystemOrOnDisk) Open(path string) (File, error) {
	return fs.fs().Open(path)
}

func (fs FileSystemOrOnDisk) IsDir(path string) bool {
	return fs.fs().IsDir(path)
}

func (fs FileSystemOrOnDisk) ReadDir(path string) ([]string, error) {
	return fs.fs().ReadDir(path)
}

func (fs FileSystemOrOnDisk) CleanedAbs(path string) (ConfirmedDir, string, error) {
	return fs.fs().CleanedAbs(path)
}

func (fs FileSystemOrOnDisk) Exists(path string) bool {
	return fs.fs().Exists(path)
}

func (fs FileSystemOrOnDisk) Glob(pattern string) ([]string, error) {
	return fs.fs().Glob(pattern)
}

func (fs FileSystemOrOnDisk) ReadFile(path string) ([]byte, error) {
	return fs.fs().ReadFile(path)
}

func (fs FileSystemOrOnDisk) WriteFile(path string, data []byte) error {
	return fs.fs().WriteFile(path, data)
}

func (fs FileSystemOrOnDisk) Walk(path string, walkFn filepath.WalkFunc) error {
	return fs.fs().Walk(path, walkFn)
}
