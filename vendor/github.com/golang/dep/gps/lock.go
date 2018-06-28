// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"bytes"
	"fmt"
	"sort"
)

// Lock represents data from a lock file (or however the implementing tool
// chooses to store it) at a particular version that is relevant to the
// satisfiability solving process.
//
// In general, the information produced by gps on finding a successful
// solution is all that would be necessary to constitute a lock file, though
// tools can include whatever other information they want in their storage.
type Lock interface {
	// The hash digest of inputs to gps that resulted in this lock data.
	InputsDigest() []byte

	// Projects returns the list of LockedProjects contained in the lock data.
	Projects() []LockedProject
}

// LocksAreEq checks if two locks are equivalent. This checks that
// all contained LockedProjects are equal, and optionally (if the third
// parameter is true) whether the locks' input hashes are equal.
func LocksAreEq(l1, l2 Lock, checkHash bool) bool {
	// Cheapest ops first
	if checkHash && !bytes.Equal(l1.InputsDigest(), l2.InputsDigest()) {
		return false
	}

	p1, p2 := l1.Projects(), l2.Projects()
	if len(p1) != len(p2) {
		return false
	}

	p1 = sortedLockedProjects(p1)
	p2 = sortedLockedProjects(p2)

	for k, lp := range p1 {
		if !lp.Eq(p2[k]) {
			return false
		}
	}
	return true
}

// sortedLockedProjects returns a sorted copy of lps, or itself if already sorted.
func sortedLockedProjects(lps []LockedProject) []LockedProject {
	if len(lps) <= 1 || sort.SliceIsSorted(lps, func(i, j int) bool {
		return lps[i].Ident().Less(lps[j].Ident())
	}) {
		return lps
	}
	cp := make([]LockedProject, len(lps))
	copy(cp, lps)
	sort.Slice(cp, func(i, j int) bool {
		return cp[i].Ident().Less(cp[j].Ident())
	})
	return cp
}

// LockedProject is a single project entry from a lock file. It expresses the
// project's name, one or both of version and underlying revision, the network
// URI for accessing it, the path at which it should be placed within a vendor
// directory, and the packages that are used in it.
type LockedProject struct {
	pi   ProjectIdentifier
	v    UnpairedVersion
	r    Revision
	pkgs []string
}

// SimpleLock is a helper for tools to easily describe lock data when they know
// that no hash, or other complex information, is available.
type SimpleLock []LockedProject

var _ Lock = SimpleLock{}

// InputsDigest always returns an empty string for SimpleLock. This makes it useless
// as a stable lock to be written to disk, but still useful for some ephemeral
// purposes.
func (SimpleLock) InputsDigest() []byte {
	return nil
}

// Projects returns the entire contents of the SimpleLock.
func (l SimpleLock) Projects() []LockedProject {
	return l
}

// NewLockedProject creates a new LockedProject struct with a given
// ProjectIdentifier (name and optional upstream source URL), version. and list
// of packages required from the project.
//
// Note that passing a nil version will cause a panic. This is a correctness
// measure to ensure that the solver is never exposed to a version-less lock
// entry. Such a case would be meaningless - the solver would have no choice but
// to simply dismiss that project. By creating a hard failure case via panic
// instead, we are trying to avoid inflicting the resulting pain on the user by
// instead forcing a decision on the Analyzer implementation.
func NewLockedProject(id ProjectIdentifier, v Version, pkgs []string) LockedProject {
	if v == nil {
		panic("must provide a non-nil version to create a LockedProject")
	}

	lp := LockedProject{
		pi:   id,
		pkgs: pkgs,
	}

	switch tv := v.(type) {
	case Revision:
		lp.r = tv
	case branchVersion:
		lp.v = tv
	case semVersion:
		lp.v = tv
	case plainVersion:
		lp.v = tv
	case versionPair:
		lp.r = tv.r
		lp.v = tv.v
	}

	return lp
}

// Ident returns the identifier describing the project. This includes both the
// local name (the root name by which the project is referenced in import paths)
// and the network name, where the upstream source lives.
func (lp LockedProject) Ident() ProjectIdentifier {
	return lp.pi
}

// Version assembles together whatever version and/or revision data is
// available into a single Version.
func (lp LockedProject) Version() Version {
	if lp.r == "" {
		return lp.v
	}

	if lp.v == nil {
		return lp.r
	}

	return lp.v.Pair(lp.r)
}

// Eq checks if two LockedProject instances are equal.
func (lp LockedProject) Eq(lp2 LockedProject) bool {
	if lp.pi != lp2.pi {
		return false
	}

	if lp.r != lp2.r {
		return false
	}

	if len(lp.pkgs) != len(lp2.pkgs) {
		return false
	}

	for k, v := range lp.pkgs {
		if lp2.pkgs[k] != v {
			return false
		}
	}

	v1n := lp.v == nil
	v2n := lp2.v == nil

	if v1n != v2n {
		return false
	}

	if !v1n && !lp.v.Matches(lp2.v) {
		return false
	}

	return true
}

// Packages returns the list of packages from within the LockedProject that are
// actually used in the import graph. Some caveats:
//
//  * The names given are relative to the root import path for the project. If
//    the root package itself is imported, it's represented as ".".
//  * Just because a package path isn't included in this list doesn't mean it's
//    safe to remove - it could contain C files, or other assets, that can't be
//    safely removed.
//  * The slice is not a copy. If you need to modify it, copy it first.
func (lp LockedProject) Packages() []string {
	return lp.pkgs
}

func (lp LockedProject) String() string {
	return fmt.Sprintf("%s@%s with packages: %v",
		lp.Ident(), lp.Version(), lp.pkgs)
}

type safeLock struct {
	h []byte
	p []LockedProject
}

func (sl safeLock) InputsDigest() []byte {
	return sl.h
}

func (sl safeLock) Projects() []LockedProject {
	return sl.p
}

// prepLock ensures a lock is prepared and safe for use by the solver. This is
// mostly about defensively ensuring that no outside routine can modify the lock
// while the solver is in-flight.
//
// This is achieved by copying the lock's data into a new safeLock.
func prepLock(l Lock) safeLock {
	pl := l.Projects()

	rl := safeLock{
		h: l.InputsDigest(),
		p: make([]LockedProject, len(pl)),
	}
	copy(rl.p, pl)

	return rl
}
