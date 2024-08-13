// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packages

import (
	"encoding/json"
	"path/filepath"

	"golang.org/x/tools/internal/gocommand"
)

// determineRootDirs returns a mapping from absolute directories that could
// contain code to their corresponding import path prefixes.
func (state *golistState) determineRootDirs() (map[string]string, error) {
	env, err := state.getEnv()
	if err != nil {
		return nil, err
	}
	if env["GOMOD"] != "" {
		state.rootsOnce.Do(func() {
			state.rootDirs, state.rootDirsError = state.determineRootDirsModules()
		})
	} else {
		state.rootsOnce.Do(func() {
			state.rootDirs, state.rootDirsError = state.determineRootDirsGOPATH()
		})
	}
	return state.rootDirs, state.rootDirsError
}

func (state *golistState) determineRootDirsModules() (map[string]string, error) {
	// List all of the modules--the first will be the directory for the main
	// module. Any replaced modules will also need to be treated as roots.
	// Editing files in the module cache isn't a great idea, so we don't
	// plan to ever support that.
	out, err := state.invokeGo("list", "-m", "-json", "all")
	if err != nil {
		// 'go list all' will fail if we're outside of a module and
		// GO111MODULE=on. Try falling back without 'all'.
		var innerErr error
		out, innerErr = state.invokeGo("list", "-m", "-json")
		if innerErr != nil {
			return nil, err
		}
	}
	roots := map[string]string{}
	modules := map[string]string{}
	var i int
	for dec := json.NewDecoder(out); dec.More(); {
		mod := new(gocommand.ModuleJSON)
		if err := dec.Decode(mod); err != nil {
			return nil, err
		}
		if mod.Dir != "" && mod.Path != "" {
			// This is a valid module; add it to the map.
			absDir, err := filepath.Abs(mod.Dir)
			if err != nil {
				return nil, err
			}
			modules[absDir] = mod.Path
			// The first result is the main module.
			if i == 0 || mod.Replace != nil && mod.Replace.Path != "" {
				roots[absDir] = mod.Path
			}
		}
		i++
	}
	return roots, nil
}

func (state *golistState) determineRootDirsGOPATH() (map[string]string, error) {
	m := map[string]string{}
	for _, dir := range filepath.SplitList(state.mustGetEnv()["GOPATH"]) {
		absDir, err := filepath.Abs(dir)
		if err != nil {
			return nil, err
		}
		m[filepath.Join(absDir, "src")] = ""
	}
	return m, nil
}
