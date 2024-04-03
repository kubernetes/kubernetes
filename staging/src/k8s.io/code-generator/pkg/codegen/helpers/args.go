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

package helpers

import (
	"errors"
	"os"
	"path"
	"path/filepath"

	"k8s.io/code-generator/pkg/fs/current"
)

var (
	// ErrInputDirIsRequired is returned when the input directory is not specified.
	ErrInputDirIsRequired = errors.New("--input-dir is required to be a valid directory")
	// ErrBoilerplateIsntReadable is returned when the boilerplate file is not readable.
	ErrBoilerplateIsntReadable = errors.New("--boilerplate needs to point to a readable file")
)

// Args are the arguments for the helper generator.
type Args struct {
	InputDir      string   `doc:"The root directory under which to search for Go files which request code to be generated. This must be a local path, not a Go package."`
	Boilerplate   string   `doc:"An optional override for the header file to insert into generated files."`
	ExtraPeerDirs []string `doc:"An optional list (this flag may be specified multiple times) of \"extra\" directories to consider during conversion generation."`
}

// Validate the arguments.
func (a *Args) Validate() error {
	if len(a.InputDir) == 0 {
		return ErrInputDirIsRequired
	}
	if fp, err := filepath.Abs(a.InputDir); err != nil {
		return err
	} else {
		a.InputDir = fp
	}
	if !isDir(a.InputDir) {
		return ErrInputDirIsRequired
	}
	if len(a.Boilerplate) == 0 {
		if b, err := defaultBoilerplate(); err != nil {
			return err
		} else {
			a.Boilerplate = b
		}
	}
	if len(a.Boilerplate) > 0 && !isReadable(a.Boilerplate) {
		return ErrBoilerplateIsntReadable
	}
	if fp, err := filepath.Abs(a.Boilerplate); err != nil {
		return err
	} else {
		a.Boilerplate = fp
	}
	return nil
}

func defaultBoilerplate() (string, error) {
	dir, err := current.Dir()
	if err != nil {
		return "", err
	}
	codegenRoot := path.Dir(path.Dir(path.Dir(dir)))
	return path.Join(codegenRoot, "examples", "hack", "boilerplate.go.txt"), nil
}

func isReadable(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func isDir(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return info.IsDir()
}
