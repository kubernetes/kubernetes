/*
Copyright 2015 The Kubernetes Authors.

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

	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"
)

func NewImportTracker(typesToAdd ...*types.Type) namer.ImportTracker {
	tracker := namer.NewDefaultImportTracker(types.Name{})
	tracker.IsInvalidType = func(*types.Type) bool { return false }
	tracker.LocalName = func(name types.Name) string { return golangTrackerLocalName(&tracker, name) }
	tracker.PrintImport = func(path, name string) string { return name + " \"" + path + "\"" }

	tracker.AddTypes(typesToAdd...)
	return &tracker

}

func golangTrackerLocalName(tracker namer.ImportTracker, t types.Name) string {
	path := t.Package
	dirs := strings.Split(path, string(filepath.Separator))
	for n := len(dirs) - 1; n >= 0; n-- {
		// TODO: bikeshed about whether it's more readable to have an
		// _, something else, or nothing between directory names.
		name := strings.Join(dirs[n:], "_")
		// These characters commonly appear in import paths for go
		// packages, but aren't legal go names. So we'll sanitize.
		name = strings.Replace(name, ".", "_", -1)
		name = strings.Replace(name, "-", "_", -1)
		if _, found := tracker.PathOf(name); found {
			// This name collides with some other package
			continue
		}
		return name
	}
	panic("can't find import for " + path)
}
