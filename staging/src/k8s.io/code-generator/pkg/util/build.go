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

package util

import (
	gobuild "go/build"
	"path/filepath"
	"strings"
)

// CurrentPackage returns the go package of the current directory, or "" if it cannot
// be derived from the GOPATH.
func CurrentPackage() string {
	for _, root := range gobuild.Default.SrcDirs() {
		if pkg, ok := hasSubdir(root, "."); ok {
			return pkg
		}
	}
	return ""
}

func hasSubdir(root, dir string) (rel string, ok bool) {
	// ensure a tailing separator to properly compare on word-boundaries
	const sep = string(filepath.Separator)
	root = filepath.Clean(root)
	if !strings.HasSuffix(root, sep) {
		root += sep
	}

	// check whether root dir starts with root
	dir = filepath.Clean(dir)
	if !strings.HasPrefix(dir, root) {
		return "", false
	}

	// cut off root
	return filepath.ToSlash(dir[len(root):]), true
}
