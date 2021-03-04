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

package main

import (
	"fmt"
	"go/build"
	"os"
	"path"
	"sort"
)

// VisitFunc is a function called by WalkPkg to examine a single package.
type VisitFunc func(importPath string, absPath string) error

// ErrSkipPkg can be returned by a VisitFunc to indicate that the package in
// question should not be walked any further.
var ErrSkipPkg = fmt.Errorf("package skipped")

// WalkPkg recursively visits all packages under pkgName.  This is similar
// to filepath.Walk, except that it follows symlinks.  A package is always
// visited before the children of that package.  If visit returns ErrSkipPkg,
// pkgName will not be walked.
func WalkPkg(pkgName string, visit VisitFunc) error {
	// Visit the package itself.
	pkg, err := findPackage(pkgName)
	if err != nil {
		return err
	}
	if err := visit(pkg.ImportPath, pkg.Dir); err == ErrSkipPkg {
		return nil
	} else if err != nil {
		return err
	}

	// Read all of the child dirents and find sub-packages.
	infos, err := readDirInfos(pkg.Dir)
	if err != nil {
		return err
	}
	for _, info := range infos {
		if !info.IsDir() {
			continue
		}
		name := info.Name()
		if name[0] == '_' || (len(name) > 1 && name[0] == '.') || name == "testdata" {
			continue
		}
		// Don't use path.Join() because it drops leading `./` via path.Clean().
		err := WalkPkg(pkgName+"/"+name, visit)
		if err != nil {
			return err
		}
	}
	return nil
}

// findPackage finds a Go package.
func findPackage(pkgName string) (*build.Package, error) {
	debug("find", pkgName)
	pkg, err := build.Import(pkgName, getwd(), build.FindOnly)
	if err != nil {
		return nil, err
	}
	return pkg, nil
}

// readDirInfos returns a list of os.FileInfo structures for the dirents under
// dirPath.  The result list is sorted by name.  This is very similar to
// ioutil.ReadDir, except that it follows symlinks.
func readDirInfos(dirPath string) ([]os.FileInfo, error) {
	names, err := readDirNames(dirPath)
	if err != nil {
		return nil, err
	}
	sort.Strings(names)

	infos := make([]os.FileInfo, 0, len(names))
	for _, n := range names {
		info, err := os.Stat(path.Join(dirPath, n))
		if err != nil {
			return nil, err
		}
		infos = append(infos, info)
	}
	return infos, nil
}

// readDirNames returns a list of all dirents in dirPath.  The result list is
// not sorted or filtered.
func readDirNames(dirPath string) ([]string, error) {
	d, err := os.Open(dirPath)
	if err != nil {
		return nil, err
	}
	defer d.Close()

	names, err := d.Readdirnames(-1)
	if err != nil {
		return nil, err
	}
	return names, nil
}
