// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgtree

import (
	"sort"
	"strings"
)

// ReachMap maps a set of import paths (keys) to the sets of transitively
// reachable tree-internal packages, and all the tree-external packages
// reachable through those internal packages.
//
// See PackageTree.ToReachMap() for more information.
type ReachMap map[string]struct {
	Internal, External []string
}

// Eliminate import paths with any elements having leading dots, leading
// underscores, or testdata. If these are internally reachable (which is
// a no-no, but possible), any external imports will have already been
// pulled up through ExternalReach. The key here is that we don't want
// to treat such packages as themselves being sources.
func pkgFilter(pkg string) bool {
	for _, elem := range strings.Split(pkg, "/") {
		if strings.HasPrefix(elem, ".") || strings.HasPrefix(elem, "_") || elem == "testdata" {
			return false
		}
	}
	return true
}

// FlattenFn flattens a reachmap into a sorted, deduplicated list of all the
// external imports named by its contained packages, but excludes imports coming
// from packages with disallowed patterns in their names: any path element with
// a leading dot, a leading underscore, with the name "testdata".
//
// Imports for which exclude returns true will be left out.
func (rm ReachMap) FlattenFn(exclude func(string) bool) []string {
	exm := make(map[string]struct{})
	for pkg, ie := range rm {
		if pkgFilter(pkg) {
			for _, ex := range ie.External {
				if exclude != nil && exclude(ex) {
					continue
				}
				exm[ex] = struct{}{}
			}
		}
	}

	if len(exm) == 0 {
		return []string{}
	}

	ex := make([]string, 0, len(exm))
	for p := range exm {
		ex = append(ex, p)
	}

	sort.Strings(ex)
	return ex
}
