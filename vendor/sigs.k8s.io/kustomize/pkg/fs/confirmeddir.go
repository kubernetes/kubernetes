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
	"path/filepath"
	"strings"
)

// ConfirmedDir is a clean, absolute, delinkified path
// that was confirmed to point to an existing directory.
type ConfirmedDir string

// Return a temporary dir, else error.
// The directory is cleaned, no symlinks, etc. so its
// returned as a ConfirmedDir.
func NewTmpConfirmedDir() (ConfirmedDir, error) {
	n, err := ioutil.TempDir("", "kustomize-")
	if err != nil {
		return "", err
	}
	return ConfirmedDir(n), nil
}

// HasPrefix returns true if the directory argument
// is a prefix of self (d) from the point of view of
// a file system.
//
// I.e., it's true if the argument equals or contains
// self (d) in a file path sense.
//
// HasPrefix emulates the semantics of strings.HasPrefix
// such that the following are true:
//
//   strings.HasPrefix("foobar", "foobar")
//   strings.HasPrefix("foobar", "foo")
//   strings.HasPrefix("foobar", "")
//
//   d := fSys.ConfirmDir("/foo/bar")
//   d.HasPrefix("/foo/bar")
//   d.HasPrefix("/foo")
//   d.HasPrefix("/")
//
// Not contacting a file system here to check for
// actual path existence.
//
// This is tested on linux, but will have trouble
// on other operating systems.
// TODO(monopole) Refactor when #golang/go/18358 closes.
// See also:
//   https://github.com/golang/go/issues/18358
//   https://github.com/golang/dep/issues/296
//   https://github.com/golang/dep/blob/master/internal/fs/fs.go#L33
//   https://codereview.appspot.com/5712045
func (d ConfirmedDir) HasPrefix(path ConfirmedDir) bool {
	if path.String() == string(filepath.Separator) || path == d {
		return true
	}
	return strings.HasPrefix(
		string(d),
		string(path)+string(filepath.Separator))
}

func (d ConfirmedDir) Join(path string) string {
	return filepath.Join(string(d), path)
}

func (d ConfirmedDir) String() string {
	return string(d)
}
