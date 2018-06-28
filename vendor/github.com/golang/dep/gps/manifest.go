// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import "github.com/golang/dep/gps/pkgtree"

// Manifest represents manifest-type data for a project at a particular version.
// The constraints expressed in a manifest determine the set of versions that
// are acceptable to try for a given project.
//
// Expressing a constraint in a manifest does not guarantee that a particular
// dependency will be present. It only guarantees that if packages in the
// project specified by the dependency are discovered through static analysis of
// the (transitive) import graph, then they will conform to the constraint.
//
// This does entail that manifests can express constraints on projects they do
// not themselves import. This is by design, but its implications are complex.
// See the gps docs for more information: https://github.com/sdboyer/gps/wiki
type Manifest interface {
	// Returns a list of project-level constraints.
	DependencyConstraints() ProjectConstraints
}

// RootManifest extends Manifest to add special controls over solving that are
// only afforded to the root project.
type RootManifest interface {
	Manifest

	// Overrides returns a list of ProjectConstraints that will unconditionally
	// supersede any ProjectConstraint declarations made in either the root
	// manifest, or in any dependency's manifest.
	//
	// Overrides are a special control afforded only to root manifests. Tool
	// users should be encouraged to use them only as a last resort; they do not
	// "play well with others" (that is their express goal), and overreliance on
	// them can harm the ecosystem as a whole.
	Overrides() ProjectConstraints

	// IngoredPackages returns a pkgtree.IgnoredRuleset, which comprises a set
	// of import paths, or import path patterns, that are to be ignored during
	// solving. These ignored import paths can be within the root project, or
	// part of other projects. Ignoring a package means that both it and its
	// (unique) imports will be disregarded by all relevant solver operations.
	//
	// It is an error to include a package in both the ignored and required
	// sets.
	IgnoredPackages() *pkgtree.IgnoredRuleset

	// RequiredPackages returns a set of import paths to require. These packages
	// are required to be present in any solution. The list can include main
	// packages.
	//
	// It is meaningless to specify packages that are within the
	// PackageTree of the ProjectRoot (though not an error, because the
	// RootManifest itself does not report a ProjectRoot).
	//
	// It is an error to include a package in both the ignored and required
	// sets.
	RequiredPackages() map[string]bool
}

// SimpleManifest is a helper for tools to enumerate manifest data. It's
// generally intended for ephemeral manifests, such as those Analyzers create on
// the fly for projects with no manifest metadata, or metadata through a foreign
// tool's idioms.
type SimpleManifest struct {
	Deps ProjectConstraints
}

var _ Manifest = SimpleManifest{}

// DependencyConstraints returns the project's dependencies.
func (m SimpleManifest) DependencyConstraints() ProjectConstraints {
	return m.Deps
}

// simpleRootManifest exists so that we have a safe value to swap into solver
// params when a nil Manifest is provided.
type simpleRootManifest struct {
	c, ovr ProjectConstraints
	ig     *pkgtree.IgnoredRuleset
	req    map[string]bool
}

func (m simpleRootManifest) DependencyConstraints() ProjectConstraints {
	return m.c
}
func (m simpleRootManifest) Overrides() ProjectConstraints {
	return m.ovr
}
func (m simpleRootManifest) IgnoredPackages() *pkgtree.IgnoredRuleset {
	return m.ig
}
func (m simpleRootManifest) RequiredPackages() map[string]bool {
	return m.req
}
func (m simpleRootManifest) dup() simpleRootManifest {
	m2 := simpleRootManifest{
		c:   make(ProjectConstraints, len(m.c)),
		ovr: make(ProjectConstraints, len(m.ovr)),
		req: make(map[string]bool, len(m.req)),
	}

	for k, v := range m.c {
		m2.c[k] = v
	}
	for k, v := range m.ovr {
		m2.ovr[k] = v
	}
	for k, v := range m.req {
		m2.req[k] = v
	}

	// IgnoredRulesets are immutable, and safe to reuse.
	m2.ig = m.ig

	return m2
}

// prepManifest ensures a manifest is prepared and safe for use by the solver.
// This is mostly about ensuring that no outside routine can modify the manifest
// while the solver is in-flight, but it also filters out any empty
// ProjectProperties.
//
// This is achieved by copying the manifest's data into a new SimpleManifest.
func prepManifest(m Manifest) SimpleManifest {
	if m == nil {
		return SimpleManifest{}
	}

	deps := m.DependencyConstraints()

	rm := SimpleManifest{
		Deps: make(ProjectConstraints, len(deps)),
	}

	for k, d := range deps {
		// A zero-value ProjectProperties is equivalent to one with an
		// anyConstraint{} in terms of how the solver will treat it. However, we
		// normalize between these two by omitting such instances entirely, as
		// it negates some possibility for false mismatches in input hashing.
		if d.Constraint == nil {
			if d.Source == "" {
				continue
			}
			d.Constraint = anyConstraint{}
		}

		rm.Deps[k] = d
	}

	return rm
}
