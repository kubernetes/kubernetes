// Copyright © 2018 Steve Francia <spf@spf13.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package afero

import (
	"errors"
)

// Symlinker is an optional interface in Afero. It is only implemented by the
// filesystems saying so.
// It indicates support for 3 symlink related interfaces that implement the
// behaviors of the os methods:
//    - Lstat
//    - Symlink, and
//    - Readlink
type Symlinker interface {
	Lstater
	Linker
	LinkReader
}

// Linker is an optional interface in Afero. It is only implemented by the
// filesystems saying so.
// It will call Symlink if the filesystem itself is, or it delegates to, the os filesystem,
// or the filesystem otherwise supports Symlink's.
type Linker interface {
	SymlinkIfPossible(oldname, newname string) error
}

// ErrNoSymlink is the error that will be wrapped in an os.LinkError if a file system
// does not support Symlink's either directly or through its delegated filesystem.
// As expressed by support for the Linker interface.
var ErrNoSymlink = errors.New("symlink not supported")

// LinkReader is an optional interface in Afero. It is only implemented by the
// filesystems saying so.
type LinkReader interface {
	ReadlinkIfPossible(name string) (string, error)
}

// ErrNoReadlink is the error that will be wrapped in an os.Path if a file system
// does not support the readlink operation either directly or through its delegated filesystem.
// As expressed by support for the LinkReader interface.
var ErrNoReadlink = errors.New("readlink not supported")
