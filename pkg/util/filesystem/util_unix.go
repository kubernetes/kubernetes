//go:build freebsd || linux || darwin

/*
Copyright 2023 The Kubernetes Authors.

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

package filesystem

import (
	"fmt"
	"os"
	"path/filepath"
)

// IsUnixDomainSocket returns whether a given file is a AF_UNIX socket file
func IsUnixDomainSocket(filePath string) (bool, error) {
	fi, err := os.Stat(filePath)
	if err != nil {
		return false, fmt.Errorf("stat file %s failed: %v", filePath, err)
	}
	if fi.Mode()&os.ModeSocket == 0 {
		return false, nil
	}
	return true, nil
}

// Chmod is the same as os.Chmod on Unix.
func Chmod(name string, mode os.FileMode) error {
	return os.Chmod(name, mode)
}

// MkdirAll is same as os.MkdirAll on Unix.
func MkdirAll(path string, perm os.FileMode) error {
	return os.MkdirAll(path, perm)
}

// IsAbs is same as filepath.IsAbs on Unix.
func IsAbs(path string) bool {
	return filepath.IsAbs(path)
}

// IsPathValidForMount checks if the given `path` is a valid mount point.
// Return whether the path is valid and error encountered if any
func IsPathValidForMount(path string) (bool, error) {
	dir, err := os.Lstat(path)
	if err != nil {
		if !os.IsNotExist(err) {
			return false, err
		}

		return false, nil
	}

	// for Unix/Linux OS, check if the path is directory.
	if dir.IsDir() {
		return true, nil
	}

	return false, fmt.Errorf("path %v exists but is not a mounted path", path)
}
