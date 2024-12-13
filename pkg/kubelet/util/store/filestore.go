/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package store

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

const (
	// Name prefix for the temporary files.
	tmpPrefix = "."

	// The default permission bits to set on the filestore directory.
	directoryPerm = 0700
)

// FileStore is an implementation of the Store interface which stores data in files.
type FileStore struct {
	// Absolute path to the base directory for storing data files.
	directoryPath string

	// filesystem to use.
	filesystem utilfs.Filesystem
}

// NewFileStore returns an instance of FileStore.
func NewFileStore(path string, fs utilfs.Filesystem) (Store, error) {
	if err := fs.MkdirAll(path, directoryPerm); err != nil {
		return nil, err
	}
	return &FileStore{directoryPath: path, filesystem: fs}, nil
}

// Write writes the given data to a file named key.
func (f *FileStore) Write(key string, data []byte) error {
	if err := ValidateKey(key); err != nil {
		return err
	}
	if err := f.filesystem.MkdirAll(f.directoryPath, directoryPerm); err != nil {
		return err
	}

	return writeFile(f.filesystem, f.getPathByKey(key), data)
}

// Read reads the data from the file named key.
func (f *FileStore) Read(key string) ([]byte, error) {
	if err := ValidateKey(key); err != nil {
		return nil, err
	}
	bytes, err := f.filesystem.ReadFile(f.getPathByKey(key))
	if os.IsNotExist(err) {
		return bytes, ErrKeyNotFound
	}
	return bytes, err
}

// Delete deletes the key file.
func (f *FileStore) Delete(key string) error {
	if err := ValidateKey(key); err != nil {
		return err
	}
	return removePath(f.filesystem, f.getPathByKey(key))
}

// List returns all keys in the store.
func (f *FileStore) List() ([]string, error) {
	keys := make([]string, 0)
	files, err := f.filesystem.ReadDir(f.directoryPath)
	if err != nil {
		return keys, err
	}
	for _, f := range files {
		if !strings.HasPrefix(f.Name(), tmpPrefix) {
			keys = append(keys, f.Name())
		}
	}
	return keys, nil
}

// getPathByKey returns the full path of the file for the key.
func (f *FileStore) getPathByKey(key string) string {
	return filepath.Join(f.directoryPath, key)
}

// writeFile writes data to path in a single transaction.
func writeFile(fs utilfs.Filesystem, path string, data []byte) (retErr error) {
	// Create a temporary file in the base directory of `path` with a prefix.
	tmpFile, err := fs.TempFile(filepath.Dir(path), tmpPrefix)
	if err != nil {
		return err
	}

	tmpPath := tmpFile.Name()
	shouldClose := true

	defer func() {
		// Close the file.
		if shouldClose {
			if err := tmpFile.Close(); err != nil {
				if retErr == nil {
					retErr = fmt.Errorf("close error: %v", err)
				} else {
					retErr = fmt.Errorf("failed to close temp file after error %v; close error: %v", retErr, err)
				}
			}
		}

		// Clean up the temp file on error.
		if retErr != nil && tmpPath != "" {
			if err := removePath(fs, tmpPath); err != nil {
				retErr = fmt.Errorf("failed to remove the temporary file (%q) after error %v; remove error: %v", tmpPath, retErr, err)
			}
		}
	}()

	// Write data.
	if _, err := tmpFile.Write(data); err != nil {
		return err
	}

	// Sync file.
	if err := tmpFile.Sync(); err != nil {
		return err
	}

	// Closing the file before renaming.
	err = tmpFile.Close()
	shouldClose = false
	if err != nil {
		return err
	}

	return fs.Rename(tmpPath, path)
}

func removePath(fs utilfs.Filesystem, path string) error {
	if err := fs.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}
