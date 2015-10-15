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

package generator

import (
	"path/filepath"
	"strings"

	"k8s.io/kubernetes/cmd/libs/go2idl/types"
)

// ImportTracker may be passed to a namer.RawNamer, to track the imports needed
// for the types it names.
//
// TODO: pay attention to the package name (instead of renaming every package).
// TODO: Figure out the best way to make names for packages that collide.
type ImportTracker struct {
	pathToName map[string]string
	// forbidden names are in here. (e.g. "go" is a directory in which
	// there is code, but "go" is not a legal name for a package, so we put
	// it here to prevent us from naming any package "go")
	nameToPath map[string]string
}

func NewImportTracker(types ...*types.Type) *ImportTracker {
	tracker := &ImportTracker{
		pathToName: map[string]string{},
		nameToPath: map[string]string{
			"go": "",
			// Add other forbidden keywords that also happen to be
			// package names here.
		},
	}
	tracker.AddTypes(types...)
	return tracker
}

func (tracker *ImportTracker) AddTypes(types ...*types.Type) {
	for _, t := range types {
		tracker.AddType(t)
	}
}
func (tracker *ImportTracker) AddType(t *types.Type) {
	path := t.Name.Package
	if path == "" {
		return
	}
	if _, ok := tracker.pathToName[path]; ok {
		return
	}
	dirs := strings.Split(path, string(filepath.Separator))
	for n := len(dirs) - 1; n >= 0; n-- {
		// TODO: bikeshed about whether it's more readable to have an
		// _, something else, or nothing between directory names.
		name := strings.Join(dirs[n:], "_")
		// These characters commonly appear in import paths for go
		// packages, but aren't legal go names. So we'll sanitize.
		name = strings.Replace(name, ".", "_", -1)
		name = strings.Replace(name, "-", "_", -1)
		if _, found := tracker.nameToPath[name]; found {
			// This name collides with some other package
			continue
		}
		tracker.nameToPath[name] = path
		tracker.pathToName[path] = name
		return
	}
	panic("can't find import for " + path)
}

func (tracker *ImportTracker) ImportLines() []string {
	out := []string{}
	for path, name := range tracker.pathToName {
		out = append(out, name+" \""+path+"\"")
	}
	return out
}

// LocalNameOf returns the name you would use to refer to the package at the
// specified path within the body of a file.
func (tracker *ImportTracker) LocalNameOf(path string) string {
	return tracker.pathToName[path]
}
