// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import "github.com/golang/dep/gps/internal/pb"

// versionUnifier facilitates cross-type version comparison and set operations.
type versionUnifier struct {
	b   sourceBridge
	mtr *metrics
}

// pairVersion takes an UnpairedVersion and attempts to pair it with an
// underlying Revision in the context of the provided ProjectIdentifier by
// consulting the canonical version list.
func (vu versionUnifier) pairVersion(id ProjectIdentifier, v UnpairedVersion) PairedVersion {
	vl, err := vu.b.listVersions(id)
	if err != nil {
		return nil
	}

	vu.mtr.push("b-pair-version")
	// doing it like this is a bit sloppy
	for _, v2 := range vl {
		if p, ok := v2.(PairedVersion); ok {
			if p.Matches(v) {
				vu.mtr.pop()
				return p
			}
		}
	}

	vu.mtr.pop()
	return nil
}

// pairRevision takes a Revision  and attempts to pair it with all possible
// versionsby consulting the canonical version list of the provided
// ProjectIdentifier.
func (vu versionUnifier) pairRevision(id ProjectIdentifier, r Revision) []Version {
	vl, err := vu.b.listVersions(id)
	if err != nil {
		return nil
	}

	vu.mtr.push("b-pair-rev")
	p := []Version{r}
	// doing it like this is a bit sloppy
	for _, v2 := range vl {
		if pv, ok := v2.(PairedVersion); ok {
			if pv.Matches(r) {
				p = append(p, pv)
			}
		}
	}

	vu.mtr.pop()
	return p
}

// matches performs a typical match check between the provided version and
// constraint. If that basic check fails and the provided version is incomplete
// (e.g. an unpaired version or bare revision), it will attempt to gather more
// information on one or the other and re-perform the comparison.
func (vu versionUnifier) matches(id ProjectIdentifier, c Constraint, v Version) bool {
	if c.Matches(v) {
		return true
	}

	vu.mtr.push("b-matches")
	// This approach is slightly wasteful, but just SO much less verbose, and
	// more easily understood.
	vtu := vu.createTypeUnion(id, v)

	var uc Constraint
	if cv, ok := c.(Version); ok {
		uc = vu.createTypeUnion(id, cv)
	} else {
		uc = c
	}

	vu.mtr.pop()
	return uc.Matches(vtu)
}

// matchesAny is the authoritative version of Constraint.MatchesAny.
func (vu versionUnifier) matchesAny(id ProjectIdentifier, c1, c2 Constraint) bool {
	if c1.MatchesAny(c2) {
		return true
	}

	vu.mtr.push("b-matches-any")
	// This approach is slightly wasteful, but just SO much less verbose, and
	// more easily understood.
	var uc1, uc2 Constraint
	if v1, ok := c1.(Version); ok {
		uc1 = vu.createTypeUnion(id, v1)
	} else {
		uc1 = c1
	}

	if v2, ok := c2.(Version); ok {
		uc2 = vu.createTypeUnion(id, v2)
	} else {
		uc2 = c2
	}

	vu.mtr.pop()
	return uc1.MatchesAny(uc2)
}

// intersect is the authoritative version of Constraint.Intersect.
func (vu versionUnifier) intersect(id ProjectIdentifier, c1, c2 Constraint) Constraint {
	rc := c1.Intersect(c2)
	if rc != none {
		return rc
	}

	vu.mtr.push("b-intersect")
	// This approach is slightly wasteful, but just SO much less verbose, and
	// more easily understood.
	var uc1, uc2 Constraint
	if v1, ok := c1.(Version); ok {
		uc1 = vu.createTypeUnion(id, v1)
	} else {
		uc1 = c1
	}

	if v2, ok := c2.(Version); ok {
		uc2 = vu.createTypeUnion(id, v2)
	} else {
		uc2 = c2
	}

	vu.mtr.pop()
	return uc1.Intersect(uc2)
}

// createTypeUnion creates a versionTypeUnion for the provided version.
//
// This union may (and typically will) end up being nothing more than the single
// input version, but creating a versionTypeUnion guarantees that 'local'
// constraint checks (direct method calls) are authoritative.
func (vu versionUnifier) createTypeUnion(id ProjectIdentifier, v Version) versionTypeUnion {
	switch tv := v.(type) {
	case Revision:
		return versionTypeUnion(vu.pairRevision(id, tv))
	case PairedVersion:
		return versionTypeUnion(vu.pairRevision(id, tv.Revision()))
	case UnpairedVersion:
		pv := vu.pairVersion(id, tv)
		if pv == nil {
			return versionTypeUnion{tv}
		}

		return versionTypeUnion(vu.pairRevision(id, pv.Revision()))
	}

	return nil
}

// versionTypeUnion represents a set of versions that are, within the scope of
// this solver run, equivalent.
//
// The simple case here is just a pair - a normal version plus its underlying
// revision - but if a tag or branch point at the same rev, then we consider
// them equivalent. Again, however, this equivalency is short-lived; it must be
// re-assessed during every solver run.
//
// The union members are treated as being OR'd together:  all constraint
// operations attempt each member, and will take the most open/optimistic
// answer.
//
// This technically does allow tags to match branches - something we otherwise
// try hard to avoid - but because the original input constraint never actually
// changes (and is never written out in the Solution), there's no harmful case
// of a user suddenly riding a branch when they expected a fixed tag.
type versionTypeUnion []Version

// This should generally not be called, but is required for the interface. If it
// is called, we have a bigger problem (the type has escaped the solver); thus,
// panic.
func (vtu versionTypeUnion) String() string {
	panic("versionTypeUnion should never be turned into a string; it is solver internal-only")
}

// This should generally not be called, but is required for the interface. If it
// is called, we have a bigger problem (the type has escaped the solver); thus,
// panic.
func (vtu versionTypeUnion) ImpliedCaretString() string {
	panic("versionTypeUnion should never be turned into a string; it is solver internal-only")
}

func (vtu versionTypeUnion) typedString() string {
	panic("versionTypeUnion should never be turned into a string; it is solver internal-only")
}

// This should generally not be called, but is required for the interface. If it
// is called, we have a bigger problem (the type has escaped the solver); thus,
// panic.
func (vtu versionTypeUnion) Type() VersionType {
	panic("versionTypeUnion should never need to answer a Type() call; it is solver internal-only")
}

// Matches takes a version, and returns true if that version matches any version
// contained in the union.
//
// This DOES allow tags to match branches, albeit indirectly through a revision.
func (vtu versionTypeUnion) Matches(v Version) bool {
	vtu2, otherIs := v.(versionTypeUnion)

	for _, v1 := range vtu {
		if otherIs {
			for _, v2 := range vtu2 {
				if v1.Matches(v2) {
					return true
				}
			}
		} else if v1.Matches(v) {
			return true
		}
	}

	return false
}

// MatchesAny returns true if any of the contained versions (which are also
// constraints) in the union successfully MatchAny with the provided
// constraint.
func (vtu versionTypeUnion) MatchesAny(c Constraint) bool {
	vtu2, otherIs := c.(versionTypeUnion)

	for _, v1 := range vtu {
		if otherIs {
			for _, v2 := range vtu2 {
				if v1.MatchesAny(v2) {
					return true
				}
			}
		} else if v1.MatchesAny(c) {
			return true
		}
	}

	return false
}

// Intersect takes a constraint, and attempts to intersect it with all the
// versions contained in the union until one returns non-none. If that never
// happens, then none is returned.
//
// In order to avoid weird version floating elsewhere in the solver, the union
// always returns the input constraint. (This is probably obviously correct, but
// is still worth noting.)
func (vtu versionTypeUnion) Intersect(c Constraint) Constraint {
	vtu2, otherIs := c.(versionTypeUnion)

	for _, v1 := range vtu {
		if otherIs {
			for _, v2 := range vtu2 {
				if rc := v1.Intersect(v2); rc != none {
					return rc
				}
			}
		} else if rc := v1.Intersect(c); rc != none {
			return rc
		}
	}

	return none
}

func (vtu versionTypeUnion) identical(c Constraint) bool {
	vtu2, ok := c.(versionTypeUnion)
	if !ok {
		return false
	}
	if len(vtu) != len(vtu2) {
		return false
	}
	used := make([]bool, len(vtu))
outter:
	for _, v := range vtu {
		for i, v2 := range vtu2 {
			if used[i] {
				continue
			}
			if v.identical(v2) {
				used[i] = true
				continue outter
			}
		}
		return false
	}
	return true
}

func (vtu versionTypeUnion) copyTo(*pb.Constraint) {
	panic("versionTypeUnion should never be serialized; it is solver internal-only")
}
