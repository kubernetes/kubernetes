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
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"golang.org/x/tools/go/packages"
)

type empty struct{}

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

// BoilerplatePath returns the path to the boilerplate file in code-generator,
// or "" if the default boilerplate.go.txt file cannot be located.
func BoilerplatePath() string {
	// set up paths to check
	paths := []string{
		// works when run from root of $GOPATH containing k8s.io/code-generator
		filepath.Join(reflect.TypeOf(empty{}).PkgPath(), "/../../hack/boilerplate.go.txt"),
		// works when run from root of module vendoring k8s.io/code-generator
		"vendor/k8s.io/code-generator/hack/boilerplate.go.txt",
		// works when run from root of $GOPATH containing k8s.io/kubernetes
		"k8s.io/kubernetes/vendor/k8s.io/code-generator/hack/boilerplate.go.txt",
	}

	// see if we can locate the module directory and add that to the list
	config := packages.Config{Mode: packages.NeedModule}
	if loadedPackages, err := packages.Load(&config, "k8s.io/code-generator/pkg/util"); err == nil {
		for _, loadedPackage := range loadedPackages {
			if loadedPackage.Module != nil && loadedPackage.Module.Dir != "" {
				paths = append(paths, filepath.Join(loadedPackage.Module.Dir, "hack/boilerplate.go.txt"))
			}
		}
	}

	// try all paths and return the first that exists
	for _, path := range paths {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	// cannot be located, invoker will have to explicitly specify boilerplate file
	return ""
}

// Vendorless trims vendor prefix from a package path to make it canonical
func Vendorless(p string) string {
	if pos := strings.LastIndex(p, "/vendor/"); pos != -1 {
		return p[pos+len("/vendor/"):]
	}
	return p
}
