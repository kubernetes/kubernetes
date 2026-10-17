// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modindex

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"golang.org/x/mod/semver"
	"golang.org/x/tools/internal/gopathwalk"
)

type directory struct {
	path       string // relative to GOMODCACHE
	importPath string
	version    string // semantic version
}

// bestDirByImportPath returns the best directory for each import
// path, where "best" means most recent semantic version. These import
// paths are inferred from the GOMODCACHE-relative dir names in dirs.
func bestDirByImportPath(dirs []string) (map[string]directory, error) {
	dirsByPath := make(map[string]directory)
	for _, dir := range dirs {
		importPath, version, err := dirToImportPathVersion(dir)
		if err != nil {
			return nil, err
		}
		new := directory{
			path:       dir,
			importPath: importPath,
			version:    version,
		}
		if old, ok := dirsByPath[importPath]; !ok || compareDirectory(new, old) < 0 {
			dirsByPath[importPath] = new
		}
	}
	return dirsByPath, nil
}

// compareDirectory defines an ordering of path@version directories,
// by descending version, then by ascending path.
func compareDirectory(x, y directory) int {
	if sign := -semver.Compare(x.version, y.version); sign != 0 {
		return sign // latest first
	}
	return strings.Compare(string(x.path), string(y.path))
}

// modCacheRegexp splits a relpathpath into module, module version, and package.
var modCacheRegexp = regexp.MustCompile(`(.*)@([^/\\]*)(.*)`)

// dirToImportPathVersion computes import path and semantic version
// from a GOMODCACHE-relative directory name.
func dirToImportPathVersion(dir string) (string, string, error) {
	m := modCacheRegexp.FindStringSubmatch(string(dir))
	// m[1] is the module path
	// m[2] is the version major.minor.patch(-<pre release identifier)
	// m[3] is the rest of the package path
	if len(m) != 4 {
		return "", "", fmt.Errorf("bad dir %s", dir)
	}
	if !semver.IsValid(m[2]) {
		return "", "", fmt.Errorf("bad semantic version %s", m[2])
	}
	// ToSlash is required to convert Windows file paths
	// into Go package import paths.
	return filepath.ToSlash(m[1] + m[3]), m[2], nil
}

// findDirs returns an unordered list of relevant package directories,
// relative to the specified module cache root. The result includes only
// module dirs whose mtime is within (start, end).
func findDirs(root string, start, end time.Time) []string {
	var (
		resMu sync.Mutex
		res   []string
	)

	addDir := func(root gopathwalk.Root, dir string) {
		// TODO(pjw): do we need to check times?
		resMu.Lock()
		defer resMu.Unlock()
		res = append(res, relative(root.Path, dir))
	}

	skipDir := func(_ gopathwalk.Root, dir string) bool {
		// The cache directory is already ignored in gopathwalk.
		if filepath.Base(dir) == "internal" {
			return true
		}

		// Skip toolchains.
		if strings.Contains(dir, "toolchain@") {
			return true
		}

		// Don't look inside @ directories that are too old/new.
		if strings.Contains(filepath.Base(dir), "@") {
			st, err := os.Stat(dir)
			if err != nil {
				log.Printf("can't stat dir %s %v", dir, err)
				return true
			}
			mtime := st.ModTime()
			return mtime.Before(start) || mtime.After(end)
		}

		return false
	}

	// TODO(adonovan): parallelize this. Even with a hot buffer cache,
	//   find $(go env GOMODCACHE) -type d
	// can easily take up a minute.
	roots := []gopathwalk.Root{{Path: root, Type: gopathwalk.RootModuleCache}}
	gopathwalk.WalkSkip(roots, addDir, skipDir, gopathwalk.Options{
		ModulesEnabled: true,
		Concurrency:    1, // TODO(pjw): adjust concurrency
		// Logf: log.Printf,
	})

	return res
}
