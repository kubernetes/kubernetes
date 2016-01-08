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

package protobuf

import (
	"fmt"
	"sort"

	"k8s.io/kubernetes/cmd/libs/go2idl/types"
)

// ImportTracker handles Protobuf package imports
//
// TODO: pay attention to the package name (instead of renaming every package).
// TODO: Figure out the best way to make names for packages that collide.
type ImportTracker struct {
	pathToName map[string]string
	// forbidden names are in here. (e.g. "go" is a directory in which
	// there is code, but "go" is not a legal name for a package, so we put
	// it here to prevent us from naming any package "go")
	nameToPath map[string]string
	local      types.Name
}

func NewImportTracker(local types.Name, types ...*types.Type) *ImportTracker {
	tracker := &ImportTracker{
		local:      local,
		pathToName: map[string]string{},
		nameToPath: map[string]string{
		// Add other forbidden keywords that also happen to be
		// package names here.
		},
	}
	tracker.AddTypes(types...)
	return tracker
}

// AddNullable ensures that support for the nullable Gogo-protobuf extension is added.
func (tracker *ImportTracker) AddNullable() {
	tracker.AddType(&types.Type{
		Kind: typesKindProtobuf,
		Name: types.Name{
			Name:    "nullable",
			Package: "gogoproto",
			Path:    "github.com/gogo/protobuf/gogoproto/gogo.proto",
		},
	})
}

func (tracker *ImportTracker) AddTypes(types ...*types.Type) {
	for _, t := range types {
		tracker.AddType(t)
	}
}
func (tracker *ImportTracker) AddType(t *types.Type) {
	if tracker.local.Package == t.Name.Package {
		return
	}
	// Golang type
	if t.Kind != typesKindProtobuf {
		// ignore built in package
		if t.Kind == types.Builtin {
			return
		}
		if _, ok := tracker.nameToPath[t.Name.Package]; !ok {
			tracker.nameToPath[t.Name.Package] = ""
		}
		return
	}
	// ignore default proto package
	if len(t.Name.Package) == 0 {
		return
	}
	path := t.Name.Path
	if len(path) == 0 {
		panic(fmt.Sprintf("imported proto package %s must also have a path", t.Name.Package))
	}
	if _, ok := tracker.pathToName[path]; ok {
		return
	}
	tracker.nameToPath[t.Name.Package] = path
	tracker.pathToName[path] = t.Name.Package
}

func (tracker *ImportTracker) ImportLines() []string {
	for k, v := range tracker.nameToPath {
		if len(v) == 0 {
			panic(fmt.Sprintf("tracking import Go package %s from %s, but no matching proto path set", k, tracker.local.Package))
		}
	}
	out := []string{}
	for path := range tracker.pathToName {
		out = append(out, path)
	}
	sort.Sort(sort.StringSlice(out))
	return out
}

// LocalNameOf returns the name you would use to refer to the package at the
// specified path within the body of a file.
func (tracker *ImportTracker) LocalNameOf(path string) string {
	return tracker.pathToName[path]
}
