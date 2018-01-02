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

package temp

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
)

// Directory is an interface to a temporary directory, in which you can
// create new files.
type Directory interface {
	// NewFile creates a new file in that directory. Calling NewFile
	// with the same filename twice will result in an error.
	NewFile(name string) (io.WriteCloser, error)
	// Delete removes the directory and its content.
	Delete() error
}

// TempDir is wrapping an temporary directory on disk.
type TempDir struct {
	// Name is the name (full path) of the created directory.
	Name string
}

var _ Directory = &TempDir{}

// CreateTempDir returns a new Directory wrapping a temporary directory
// on disk.
func CreateTempDir(prefix string) (*TempDir, error) {
	name, err := ioutil.TempDir("", fmt.Sprintf("%s-", prefix))
	if err != nil {
		return nil, err
	}

	return &TempDir{
		Name: name,
	}, nil
}

// NewFile creates a new file in the specified directory.
func (d *TempDir) NewFile(name string) (io.WriteCloser, error) {
	return os.OpenFile(
		filepath.Join(d.Name, name),
		os.O_WRONLY|os.O_CREATE|os.O_TRUNC|os.O_EXCL,
		0700,
	)
}

// Delete the underlying directory, and all of its content.
func (d *TempDir) Delete() error {
	return os.RemoveAll(d.Name)
}
