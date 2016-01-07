/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package container

import (
	"os"
)

// OSInterface collects system level operations that need to be mocked out
// during tests.
type OSInterface interface {
	Mkdir(path string, perm os.FileMode) error
	Symlink(oldname string, newname string) error
}

// RealOS is used to dispatch the real system level operaitons.
type RealOS struct{}

// MkDir will will call os.Mkdir to create a directory.
func (RealOS) Mkdir(path string, perm os.FileMode) error {
	return os.Mkdir(path, perm)
}

// Symlink will call os.Symlink to create a symbolic link.
func (RealOS) Symlink(oldname string, newname string) error {
	return os.Symlink(oldname, newname)
}

// FakeOS mocks out certain OS calls to avoid perturbing the filesystem
// on the test machine.
type FakeOS struct{}

// MkDir is a fake call that just returns nil.
func (FakeOS) Mkdir(path string, perm os.FileMode) error {
	return nil
}

// Symlink is a fake call that just returns nil.
func (FakeOS) Symlink(oldname string, newname string) error {
	return nil
}
