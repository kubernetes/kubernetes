// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filesys

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"sigs.k8s.io/kustomize/kyaml/errors"
)

var _ FileSystem = fsOnDisk{}

// fsOnDisk implements FileSystem using the local filesystem.
type fsOnDisk struct{}

// MakeFsOnDisk makes an instance of fsOnDisk.
func MakeFsOnDisk() FileSystem {
	return fsOnDisk{}
}

// Create delegates to os.Create.
func (fsOnDisk) Create(name string) (File, error) { return os.Create(name) }

// Mkdir delegates to os.Mkdir.
func (fsOnDisk) Mkdir(name string) error {
	return os.Mkdir(name, 0777|os.ModeDir)
}

// MkdirAll delegates to os.MkdirAll.
func (fsOnDisk) MkdirAll(name string) error {
	return os.MkdirAll(name, 0777|os.ModeDir)
}

// RemoveAll delegates to os.RemoveAll.
func (fsOnDisk) RemoveAll(name string) error {
	return os.RemoveAll(name)
}

// Open delegates to os.Open.
func (fsOnDisk) Open(name string) (File, error) { return os.Open(name) }

// CleanedAbs converts the given path into a
// directory and a file name, where the directory
// is represented as a ConfirmedDir and all that implies.
// If the entire path is a directory, the file component
// is an empty string.
func (x fsOnDisk) CleanedAbs(
	path string) (ConfirmedDir, string, error) {
	absRoot, err := filepath.Abs(path)
	if err != nil {
		return "", "", fmt.Errorf(
			"abs path error on '%s' : %v", path, err)
	}
	deLinked, err := filepath.EvalSymlinks(absRoot)
	if err != nil {
		return "", "", fmt.Errorf(
			"evalsymlink failure on '%s' : %w", path, err)
	}
	if x.IsDir(deLinked) {
		return ConfirmedDir(deLinked), "", nil
	}
	d := filepath.Dir(deLinked)
	if !x.IsDir(d) {
		// Programmer/assumption error.
		log.Fatalf("first part of '%s' not a directory", deLinked)
	}
	if d == deLinked {
		// Programmer/assumption error.
		log.Fatalf("d '%s' should be a subset of deLinked", d)
	}
	f := filepath.Base(deLinked)
	if filepath.Join(d, f) != deLinked {
		// Programmer/assumption error.
		log.Fatalf("these should be equal: '%s', '%s'",
			filepath.Join(d, f), deLinked)
	}
	return ConfirmedDir(d), f, nil
}

// Exists returns true if os.Stat succeeds.
func (fsOnDisk) Exists(name string) bool {
	_, err := os.Stat(name)
	return err == nil
}

// Glob returns the list of matching files
func (fsOnDisk) Glob(pattern string) ([]string, error) {
	var result []string
	allFilePaths, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	if IsHiddenFilePath(pattern) {
		result = allFilePaths
	} else {
		result = RemoveHiddenFiles(allFilePaths)
	}
	return result, nil
}

// IsDir delegates to os.Stat and FileInfo.IsDir
func (fsOnDisk) IsDir(name string) bool {
	info, err := os.Stat(name)
	if err != nil {
		return false
	}
	return info.IsDir()
}

// ReadDir delegates to os.ReadDir
func (fsOnDisk) ReadDir(name string) ([]string, error) {
	dirEntries, err := os.ReadDir(name)
	if err != nil {
		return nil, err
	}
	result := make([]string, len(dirEntries))
	for i := range dirEntries {
		result[i] = dirEntries[i].Name()
	}
	return result, nil
}

// ReadFile delegates to os.ReadFile.
func (fsOnDisk) ReadFile(name string) ([]byte, error) {
	content, err := os.ReadFile(name)
	return content, errors.Wrap(err)
}

// WriteFile delegates to os.WriteFile with read/write permissions.
func (fsOnDisk) WriteFile(name string, c []byte) error {
	return errors.Wrap(os.WriteFile(name, c, 0666)) //nolint:gosec
}

// Walk delegates to filepath.Walk.
func (fsOnDisk) Walk(path string, walkFn filepath.WalkFunc) error {
	return filepath.Walk(path, walkFn)
}
