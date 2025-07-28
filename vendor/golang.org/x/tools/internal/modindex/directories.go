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
	"slices"
	"strings"
	"sync"
	"time"

	"golang.org/x/mod/semver"
	"golang.org/x/tools/internal/gopathwalk"
)

type directory struct {
	path       Relpath
	importPath string
	version    string // semantic version
	syms       []symbol
}

// byImportPath groups the directories by import path,
// sorting the ones with the same import path by semantic version,
// most recent first.
func byImportPath(dirs []Relpath) (map[string][]*directory, error) {
	ans := make(map[string][]*directory) // key is import path
	for _, d := range dirs {
		ip, sv, err := DirToImportPathVersion(d)
		if err != nil {
			return nil, err
		}
		ans[ip] = append(ans[ip], &directory{
			path:       d,
			importPath: ip,
			version:    sv,
		})
	}
	for k, v := range ans {
		semanticSort(v)
		ans[k] = v
	}
	return ans, nil
}

// sort the directories by semantic version, latest first
func semanticSort(v []*directory) {
	slices.SortFunc(v, func(l, r *directory) int {
		if n := semver.Compare(l.version, r.version); n != 0 {
			return -n // latest first
		}
		return strings.Compare(string(l.path), string(r.path))
	})
}

// modCacheRegexp splits a relpathpath into module, module version, and package.
var modCacheRegexp = regexp.MustCompile(`(.*)@([^/\\]*)(.*)`)

// DirToImportPathVersion computes import path and semantic version
func DirToImportPathVersion(dir Relpath) (string, string, error) {
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
	// ToSlash is required for Windows.
	return filepath.ToSlash(m[1] + m[3]), m[2], nil
}

// a region controls what directories to look at, for
// updating the index incrementally, and for testing that.
// (for testing one builds an index as of A, incrementally
// updates it to B, and compares the result to an index build
// as of B.)
type region struct {
	onlyAfter, onlyBefore time.Time
	sync.Mutex
	ans []Relpath
}

func findDirs(root string, onlyAfter, onlyBefore time.Time) []Relpath {
	roots := []gopathwalk.Root{{Path: root, Type: gopathwalk.RootModuleCache}}
	// TODO(PJW): adjust concurrency
	opts := gopathwalk.Options{ModulesEnabled: true, Concurrency: 1 /* ,Logf: log.Printf*/}
	betw := &region{
		onlyAfter:  onlyAfter,
		onlyBefore: onlyBefore,
	}
	gopathwalk.WalkSkip(roots, betw.addDir, betw.skipDir, opts)
	return betw.ans
}

func (r *region) addDir(rt gopathwalk.Root, dir string) {
	// do we need to check times?
	r.Lock()
	defer r.Unlock()
	x := filepath.ToSlash(string(toRelpath(Abspath(rt.Path), dir)))
	r.ans = append(r.ans, toRelpath(Abspath(rt.Path), x))
}

func (r *region) skipDir(_ gopathwalk.Root, dir string) bool {
	// The cache directory is already ignored in gopathwalk\
	if filepath.Base(dir) == "internal" {
		return true
	}
	if strings.Contains(dir, "toolchain@") {
		return true
	}
	// don't look inside @ directories that are too old
	if strings.Contains(filepath.Base(dir), "@") {
		st, err := os.Stat(dir)
		if err != nil {
			log.Printf("can't stat dir %s %v", dir, err)
			return true
		}
		if st.ModTime().Before(r.onlyAfter) {
			return true
		}
		if st.ModTime().After(r.onlyBefore) {
			return true
		}
	}
	return false
}
