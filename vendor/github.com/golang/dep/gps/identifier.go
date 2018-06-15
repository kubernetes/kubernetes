// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"fmt"
	"math/rand"
	"strconv"
)

// ProjectRoot is the topmost import path in a tree of other import paths - the
// root of the tree. In gps' current design, ProjectRoots have to correspond to
// a repository root (mostly), but their real purpose is to identify the root
// import path of a "project", logically encompassing all child packages.
//
// Projects are a crucial unit of operation in gps. Constraints are declared by
// a project's manifest, and apply to all packages in a ProjectRoot's tree.
// Solving itself mostly proceeds on a project-by-project basis.
//
// Aliasing string types is usually a bit of an anti-pattern. gps does it here
// as a means of clarifying API intent. This is important because Go's package
// management domain has lots of different path-ish strings floating around:
//
//  actual directories:
//	/home/sdboyer/go/src/github.com/sdboyer/gps/example
//  URLs:
//	https://github.com/sdboyer/gps
//  import paths:
//	github.com/sdboyer/gps/example
//  portions of import paths that refer to a package:
//	example
//  portions that could not possibly refer to anything sane:
//	github.com/sdboyer
//  portions that correspond to a repository root:
//	github.com/sdboyer/gps
//
// While not a panacea, having ProjectRoot allows gps to clearly indicate via
// the type system when a path-ish string must have particular semantics.
type ProjectRoot string

// A ProjectIdentifier provides the name and source location of a dependency. It
// is related to, but differs in two key ways from, a plain import path.
//
// First, ProjectIdentifiers do not identify a single package. Rather, they
// encompass the whole tree of packages, including tree's root - the
// ProjectRoot. In gps' current design, this ProjectRoot almost always
// corresponds to the root of a repository.
//
// Second, ProjectIdentifiers can optionally carry a Source, which
// identifies where the underlying source code can be located on the network.
// These can be either a full URL, including protocol, or plain import paths.
// So, these are all valid data for Source:
//
//  github.com/sdboyer/gps
//  github.com/fork/gps
//  git@github.com:sdboyer/gps
//  https://github.com/sdboyer/gps
//
// With plain import paths, network addresses are derived purely through an
// algorithm. By having an explicit network name, it becomes possible to, for
// example, transparently substitute a fork for the original upstream source
// repository.
//
// Note that gps makes no guarantees about the actual import paths contained in
// a repository aligning with ImportRoot. If tools, or their users, specify an
// alternate Source that contains a repository with incompatible internal
// import paths, gps' solving operations will error. (gps does no import
// rewriting.)
//
// Also note that if different projects' manifests report a different
// Source for a given ImportRoot, it is a solve failure. Everyone has to
// agree on where a given import path should be sourced from.
//
// If Source is not explicitly set, gps will derive the network address from
// the ImportRoot using a similar algorithm to that utilized by `go get`.
type ProjectIdentifier struct {
	ProjectRoot ProjectRoot
	Source      string
}

// Less compares by ProjectRoot then normalized Source.
func (i ProjectIdentifier) Less(j ProjectIdentifier) bool {
	if i.ProjectRoot < j.ProjectRoot {
		return true
	}
	if j.ProjectRoot < i.ProjectRoot {
		return false
	}
	return i.normalizedSource() < j.normalizedSource()
}

func (i ProjectIdentifier) eq(j ProjectIdentifier) bool {
	if i.ProjectRoot != j.ProjectRoot {
		return false
	}
	if i.Source == j.Source {
		return true
	}

	if (i.Source == "" && j.Source == string(j.ProjectRoot)) ||
		(j.Source == "" && i.Source == string(i.ProjectRoot)) {
		return true
	}

	return false
}

// equiv will check if the two identifiers are "equivalent," under special
// rules.
//
// Given that the ProjectRoots are equal (==), equivalency occurs if:
//
// 1. The Sources are equal (==), OR
// 2. The LEFT (the receiver) Source is non-empty, and the right
// Source is empty.
//
// *This is asymmetry in this binary relation is intentional.* It facilitates
// the case where we allow for a ProjectIdentifier with an explicit Source
// to match one without.
func (i ProjectIdentifier) equiv(j ProjectIdentifier) bool {
	if i.ProjectRoot != j.ProjectRoot {
		return false
	}
	if i.Source == j.Source {
		return true
	}

	if i.Source != "" && j.Source == "" {
		return true
	}

	return false
}

func (i ProjectIdentifier) normalizedSource() string {
	if i.Source == "" {
		return string(i.ProjectRoot)
	}
	return i.Source
}

func (i ProjectIdentifier) String() string {
	if i.Source == "" || i.Source == string(i.ProjectRoot) {
		return string(i.ProjectRoot)
	}
	return fmt.Sprintf("%s (from %s)", i.ProjectRoot, i.Source)
}

func (i ProjectIdentifier) normalize() ProjectIdentifier {
	if i.Source == "" {
		i.Source = string(i.ProjectRoot)
	}

	return i
}

// ProjectProperties comprise the properties that can be attached to a
// ProjectRoot.
//
// In general, these are declared in the context of a map of ProjectRoot to its
// ProjectProperties; they make little sense without their corresponding
// ProjectRoot.
type ProjectProperties struct {
	Source     string
	Constraint Constraint
}

// bimodalIdentifiers are used to track work to be done in the unselected queue.
type bimodalIdentifier struct {
	id ProjectIdentifier
	// List of packages required within/under the ProjectIdentifier
	pl []string
	// prefv is used to indicate a 'preferred' version. This is expected to be
	// derived from a dep's lock data, or else is empty.
	prefv Version
	// Indicates that the bmi came from the root project originally
	fromRoot bool
}

type atom struct {
	id ProjectIdentifier
	v  Version
}

// With a random revision and no name, collisions are...unlikely
var nilpa = atom{
	v: Revision(strconv.FormatInt(rand.Int63(), 36)),
}

type atomWithPackages struct {
	a  atom
	pl []string
}

// bmi converts an atomWithPackages into a bimodalIdentifier.
//
// This is mostly intended for (read-only) trace use, so the package list slice
// is not copied. It is the callers responsibility to not modify the pl slice,
// lest that backpropagate and cause inconsistencies.
func (awp atomWithPackages) bmi() bimodalIdentifier {
	return bimodalIdentifier{
		id: awp.a.id,
		pl: awp.pl,
	}
}

// completeDep (name hopefully to change) provides the whole picture of a
// dependency - the root (repo and project, since currently we assume the two
// are the same) name, a constraint, and the actual packages needed that are
// under that root.
type completeDep struct {
	// The base workingConstraint
	workingConstraint
	// The specific packages required from the ProjectDep
	pl []string
}

// dependency represents an incomplete edge in the depgraph. It has a
// fully-realized atom as the depender (the tail/source of the edge), and a set
// of requirements that any atom to be attached at the head/target must satisfy.
type dependency struct {
	depender atom
	dep      completeDep
}
