// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filesys

import (
	"io/ioutil"
	"path/filepath"
	"strings"
)

// ConfirmedDir is a clean, absolute, delinkified path
// that was confirmed to point to an existing directory.
type ConfirmedDir string

// NewTmpConfirmedDir returns a temporary dir, else error.
// The directory is cleaned, no symlinks, etc. so it's
// returned as a ConfirmedDir.
func NewTmpConfirmedDir() (ConfirmedDir, error) {
	n, err := ioutil.TempDir("", "kustomize-")
	if err != nil {
		return "", err
	}

	// In MacOs `ioutil.TempDir` creates a directory
	// with root in the `/var` folder, which is in turn
	// a symlinked path to `/private/var`.
	// Function `filepath.EvalSymlinks`is used to
	// resolve the real absolute path.
	deLinked, err := filepath.EvalSymlinks(n)
	return ConfirmedDir(deLinked), err
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
