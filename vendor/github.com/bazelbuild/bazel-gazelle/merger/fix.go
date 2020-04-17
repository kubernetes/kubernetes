/* Copyright 2017 The Bazel Authors. All rights reserved.

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

package merger

import (
	"fmt"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/rule"
)

// FixLoads removes loads of unused go rules and adds loads of newly used rules.
// This should be called after FixFile and MergeFile, since symbols
// may be introduced that aren't loaded.
//
// This function calls File.Sync before processing loads.
func FixLoads(f *rule.File, knownLoads []rule.LoadInfo) {
	knownFiles := make(map[string]bool)
	knownKinds := make(map[string]string)
	for _, l := range knownLoads {
		knownFiles[l.Name] = true
		for _, k := range l.Symbols {
			knownKinds[k] = l.Name
		}
	}

	// Sync the file. We need File.Loads and File.Rules to contain inserted
	// statements and not deleted statements.
	f.Sync()

	// Scan load statements in the file. Keep track of loads of known files,
	// since these may be changed. Keep track of symbols loaded from unknown
	// files; we will not add loads for these.
	var loads []*rule.Load
	otherLoadedKinds := make(map[string]bool)
	for _, l := range f.Loads {
		if knownFiles[l.Name()] {
			loads = append(loads, l)
			continue
		}
		for _, sym := range l.Symbols() {
			otherLoadedKinds[sym] = true
		}
	}

	// Make a map of all the symbols from known files used in this file.
	usedKinds := make(map[string]map[string]bool)
	for _, r := range f.Rules {
		kind := r.Kind()
		if file, ok := knownKinds[kind]; ok && !otherLoadedKinds[kind] {
			if usedKinds[file] == nil {
				usedKinds[file] = make(map[string]bool)
			}
			usedKinds[file][kind] = true
		}
	}

	// Fix the load statements. The order is important, so we iterate over
	// knownLoads instead of knownFiles.
	for _, known := range knownLoads {
		file := known.Name
		first := true
		for _, l := range loads {
			if l.Name() != file {
				continue
			}
			if first {
				fixLoad(l, file, usedKinds[file], knownKinds)
				first = false
			} else {
				fixLoad(l, file, nil, knownKinds)
			}
			if l.IsEmpty() {
				l.Delete()
			}
		}
		if first {
			load := fixLoad(nil, file, usedKinds[file], knownKinds)
			if load != nil {
				index := newLoadIndex(f, known.After)
				load.Insert(f, index)
			}
		}
	}
}

// fixLoad updates a load statement with the given symbols. If load is nil,
// a new load may be created and returned. Symbols in kinds will be added
// to the load if they're not already present. Known symbols not in kinds
// will be removed if present. Other symbols will be preserved. If load is
// empty, nil is returned.
func fixLoad(load *rule.Load, file string, kinds map[string]bool, knownKinds map[string]string) *rule.Load {
	if load == nil {
		if len(kinds) == 0 {
			return nil
		}
		load = rule.NewLoad(file)
	}

	for k := range kinds {
		load.Add(k)
	}
	for _, k := range load.Symbols() {
		if knownKinds[k] != "" && !kinds[k] {
			load.Remove(k)
		}
	}
	return load
}

// newLoadIndex returns the index in stmts where a new load statement should
// be inserted. after is a list of function names that the load should not
// be inserted before.
func newLoadIndex(f *rule.File, after []string) int {
	if len(after) == 0 {
		return 0
	}
	index := 0
	for _, r := range f.Rules {
		for _, a := range after {
			if r.Kind() == a && r.Index() >= index {
				index = r.Index() + 1
			}
		}
	}
	return index
}

// CheckGazelleLoaded searches the given WORKSPACE file for a repository named
// "bazel_gazelle". If no such repository is found *and* the repo is not
// declared with a directive *and* at least one load statement mentions
// the repository, a descriptive error will be returned.
//
// This should be called after modifications have been made to WORKSPACE
// (i.e., after FixLoads) before writing it to disk.
func CheckGazelleLoaded(f *rule.File) error {
	needGazelle := false
	for _, l := range f.Loads {
		if strings.HasPrefix(l.Name(), "@bazel_gazelle//") {
			needGazelle = true
		}
	}
	if !needGazelle {
		return nil
	}
	for _, r := range f.Rules {
		if r.Name() == "bazel_gazelle" {
			return nil
		}
	}
	for _, d := range f.Directives {
		if d.Key != "repo" {
			continue
		}
		if fs := strings.Fields(d.Value); len(fs) > 0 && fs[0] == "bazel_gazelle" {
			return nil
		}
	}
	return fmt.Errorf(`%s: error: bazel_gazelle is not declared in WORKSPACE.
Without this repository, Gazelle cannot safely modify the WORKSPACE file.
See the instructions at https://github.com/bazelbuild/bazel-gazelle.
If the bazel_gazelle is declared inside a macro, you can suppress this error
by adding a comment like this to WORKSPACE:
    # gazelle:repo bazel_gazelle
`, f.Path)
}
