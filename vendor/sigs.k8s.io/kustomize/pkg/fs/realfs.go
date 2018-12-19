/*
Copyright 2018 The Kubernetes Authors.

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

package fs

import (
	"io/ioutil"
	"os"
	"path/filepath"
)

var _ FileSystem = realFS{}

// realFS implements FileSystem using the local filesystem.
type realFS struct{}

// MakeRealFS makes an instance of realFS.
func MakeRealFS() FileSystem {
	return realFS{}
}

// Create delegates to os.Create.
func (realFS) Create(name string) (File, error) { return os.Create(name) }

// Mkdir delegates to os.Mkdir.
func (realFS) Mkdir(name string) error {
	return os.Mkdir(name, 0777|os.ModeDir)
}

// MkdirAll delegates to os.MkdirAll.
func (realFS) MkdirAll(name string) error {
	return os.MkdirAll(name, 0777|os.ModeDir)
}

// RemoveAll delegates to os.RemoveAll.
func (realFS) RemoveAll(name string) error {
	return os.RemoveAll(name)
}

// Open delegates to os.Open.
func (realFS) Open(name string) (File, error) { return os.Open(name) }

// Exists returns true if os.Stat succeeds.
func (realFS) Exists(name string) bool {
	_, err := os.Stat(name)
	return err == nil
}

// Glob returns the list of matching files
func (realFS) Glob(pattern string) ([]string, error) {
	return filepath.Glob(pattern)
}

// IsDir delegates to os.Stat and FileInfo.IsDir
func (realFS) IsDir(name string) bool {
	info, err := os.Stat(name)
	if err != nil {
		return false
	}
	return info.IsDir()
}

// ReadFile delegates to ioutil.ReadFile.
func (realFS) ReadFile(name string) ([]byte, error) { return ioutil.ReadFile(name) }

// WriteFile delegates to ioutil.WriteFile with read/write permissions.
func (realFS) WriteFile(name string, c []byte) error {
	return ioutil.WriteFile(name, c, 0666)
}
