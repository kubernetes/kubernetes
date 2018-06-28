// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"sort"

	"github.com/armon/go-radix"
	"github.com/golang/dep/gps/pkgtree"
)

// rootdata holds static data and constraining rules from the root project for
// use in solving.
type rootdata struct {
	// Path to the root of the project on which gps is operating.
	dir string

	// Ruleset for ignored import paths.
	ir *pkgtree.IgnoredRuleset

	// Map of packages to require.
	req map[string]bool

	// A ProjectConstraints map containing the validated (guaranteed non-empty)
	// overrides declared by the root manifest.
	ovr ProjectConstraints

	// A map of the ProjectRoot (local names) that should be allowed to change
	chng map[ProjectRoot]struct{}

	// Flag indicating all projects should be allowed to change, without regard
	// for lock.
	chngall bool

	// A map of the project names listed in the root's lock.
	rlm map[ProjectRoot]LockedProject

	// A defensively copied instance of the root manifest.
	rm SimpleManifest

	// A defensively copied instance of the root lock.
	rl safeLock

	// A defensively copied instance of params.RootPackageTree
	rpt pkgtree.PackageTree

	// The ProjectAnalyzer to use for all GetManifestAndLock calls.
	an ProjectAnalyzer
}

// externalImportList returns a list of the unique imports from the root data.
// Ignores and requires are taken into consideration, stdlib is excluded, and
// errors within the local set of package are not backpropagated.
func (rd rootdata) externalImportList(stdLibFn func(string) bool) []string {
	rm, _ := rd.rpt.ToReachMap(true, true, false, rd.ir)
	reach := rm.FlattenFn(stdLibFn)

	// If there are any requires, slide them into the reach list, as well.
	if len(rd.req) > 0 {
		// Make a map of imports that are both in the import path list and the
		// required list to avoid duplication.
		skip := make(map[string]bool, len(rd.req))
		for _, r := range reach {
			if rd.req[r] {
				skip[r] = true
			}
		}

		for r := range rd.req {
			if !skip[r] {
				reach = append(reach, r)
			}
		}
	}

	sort.Strings(reach)
	return reach
}

func (rd rootdata) getApplicableConstraints(stdLibFn func(string) bool) []workingConstraint {
	pc := rd.rm.DependencyConstraints()

	// Ensure that overrides which aren't in the combined pc map already make it
	// in. Doing so makes input hashes equal in more useful cases.
	for pr, pp := range rd.ovr {
		if _, has := pc[pr]; !has {
			cpp := ProjectProperties{
				Constraint: pp.Constraint,
				Source:     pp.Source,
			}
			if cpp.Constraint == nil {
				cpp.Constraint = anyConstraint{}
			}

			pc[pr] = cpp
		}
	}

	// Now override them all to produce a consolidated workingConstraint slice
	combined := rd.ovr.overrideAll(pc)

	type wccount struct {
		count int
		wc    workingConstraint
	}
	xt := radix.New()
	for _, wc := range combined {
		xt.Insert(string(wc.Ident.ProjectRoot), wccount{wc: wc})
	}

	// Walk all dep import paths we have to consider and mark the corresponding
	// wc entry in the trie, if any
	for _, im := range rd.externalImportList(stdLibFn) {
		if stdLibFn(im) {
			continue
		}

		if pre, v, match := xt.LongestPrefix(im); match && isPathPrefixOrEqual(pre, im) {
			wcc := v.(wccount)
			wcc.count++
			xt.Insert(pre, wcc)
		}
	}

	var ret []workingConstraint

	xt.Walk(func(s string, v interface{}) bool {
		wcc := v.(wccount)
		if wcc.count > 0 {
			ret = append(ret, wcc.wc)
		}
		return false
	})

	return ret
}

func (rd rootdata) combineConstraints() []workingConstraint {
	return rd.ovr.overrideAll(rd.rm.DependencyConstraints())
}

// needVersionListFor indicates whether we need a version list for a given
// project root, based solely on general solver inputs (no constraint checking
// required). Assuming the argument is not the root project itself, this will be
// true if any of the following conditions hold:
//
//  - ChangeAll is on
//  - The project is not in the lock
//  - The project is in the lock, but is also in the list of projects to change
func (rd rootdata) needVersionsFor(pr ProjectRoot) bool {
	if rd.isRoot(pr) {
		return false
	}

	if rd.chngall {
		return true
	}

	if _, has := rd.rlm[pr]; !has {
		// not in the lock
		return true
	}

	if _, has := rd.chng[pr]; has {
		// in the lock, but marked for change
		return true
	}
	// in the lock, not marked for change
	return false

}

func (rd rootdata) isRoot(pr ProjectRoot) bool {
	return pr == ProjectRoot(rd.rpt.ImportRoot)
}

// rootAtom creates an atomWithPackages that represents the root project.
func (rd rootdata) rootAtom() atomWithPackages {
	a := atom{
		id: ProjectIdentifier{
			ProjectRoot: ProjectRoot(rd.rpt.ImportRoot),
		},
		// This is a hack so that the root project doesn't have a nil version.
		// It's sort of OK because the root never makes it out into the results.
		// We may need a more elegant solution if we discover other side
		// effects, though.
		v: rootRev,
	}

	list := make([]string, 0, len(rd.rpt.Packages))
	for path, pkg := range rd.rpt.Packages {
		if pkg.Err != nil && !rd.ir.IsIgnored(path) {
			list = append(list, path)
		}
	}
	sort.Strings(list)

	return atomWithPackages{
		a:  a,
		pl: list,
	}
}
