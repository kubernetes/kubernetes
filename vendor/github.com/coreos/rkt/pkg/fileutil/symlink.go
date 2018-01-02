// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package fileutil

import (
	"errors"
	"os"
	"path/filepath"
)

// The following code was taken from "src/path/filepath/symlink.go" from Go 1.7.4.
// Modifications include:
// - remove Windows specific code
// - continue evaluatig paths, even if they don't exist

// Original copyright notice:
//
// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

func isRoot(path string) bool {
	return path == "/"
}

func walkLink(path string, linksWalked *int) (newpath string, islink bool, err error) {
	if *linksWalked > 255 {
		return "", false, errors.New("EvalSymlinks: too many links")
	}
	fi, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return path, false, nil
	}
	if err != nil {
		return "", false, err
	}
	if fi.Mode()&os.ModeSymlink == 0 {
		return path, false, nil
	}
	newpath, err = os.Readlink(path)
	if err != nil {
		return "", false, err
	}
	*linksWalked++
	return newpath, true, nil
}

func walkLinks(path string, linksWalked *int) (string, error) {
	dir, file := filepath.Split(path)

	switch {
	case dir == "":
		newpath, _, err := walkLink(file, linksWalked)
		return newpath, err
	case file == "":
		if os.IsPathSeparator(dir[len(dir)-1]) {
			if isRoot(dir) {
				return dir, nil
			}
			return walkLinks(dir[:len(dir)-1], linksWalked)
		}
		newpath, _, err := walkLink(dir, linksWalked)
		return newpath, err
	default:
		newdir, err := walkLinks(dir, linksWalked)
		if err != nil {
			return "", err
		}
		newpath, islink, err := walkLink(filepath.Join(newdir, file), linksWalked)
		if err != nil {
			return "", err
		}
		if !islink {
			return newpath, nil
		}
		if filepath.IsAbs(newpath) || os.IsPathSeparator(newpath[0]) {
			return newpath, nil
		}
		return filepath.Join(newdir, newpath), nil
	}
}

func walkSymlinks(path string) (string, error) {
	if path == "" {
		return path, nil
	}
	var linksWalked int // to protect against cycles
	for {
		i := linksWalked
		newpath, err := walkLinks(path, &linksWalked)
		if err != nil {
			return "", err
		}
		if i == linksWalked {
			return filepath.Clean(newpath), nil
		}
		path = newpath
	}
}

// EvalSymlinksAlways behaves like filepath.EvalSymlinks
// but does not require that all path components must exist.
//
// While filepath.EvalSymlink behaves like `readlink -e`
// this function behaves like `readlink -m`.
//
// Unlike `readlink` EvalSymlinksAlways might return a relative path.
func EvalSymlinksAlways(path string) (string, error) {
	return walkSymlinks(path)
}
