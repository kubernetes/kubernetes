// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"fmt"
	"sort"

	"github.com/Masterminds/semver"
	"github.com/golang/dep/gps/internal/pb"
)

var (
	none = noneConstraint{}
	any  = anyConstraint{}
)

// A Constraint provides structured limitations on the versions that are
// admissible for a given project.
//
// As with Version, it has a private method because the gps's internal
// implementation of the problem is complete, and the system relies on type
// magic to operate.
type Constraint interface {
	fmt.Stringer

	// ImpliedCaretString converts the Constraint to a string in the same manner
	// as String(), but treats the empty operator as equivalent to ^, rather
	// than =.
	//
	// In the same way that String() is the inverse of NewConstraint(), this
	// method is the inverse of NewSemverConstraintIC().
	ImpliedCaretString() string

	// Matches indicates if the provided Version is allowed by the Constraint.
	Matches(Version) bool

	// MatchesAny indicates if the intersection of the Constraint with the
	// provided Constraint would yield a Constraint that could allow *any*
	// Version.
	MatchesAny(Constraint) bool

	// Intersect computes the intersection of the Constraint with the provided
	// Constraint.
	Intersect(Constraint) Constraint

	// typedString emits the normal stringified representation of the provided
	// constraint, prefixed with a string that uniquely identifies the type of
	// the constraint.
	//
	// It also forces Constraint to be a private/sealed interface, which is a
	// design goal of the system.
	typedString() string

	// copyTo copies fields into a serializable representation which can be
	// converted back into an identical Constraint with constraintFromCache.
	copyTo(*pb.Constraint)

	// identical returns true if the constraints are identical.
	//
	// Identical Constraints behave identically for all methods defined by the
	// interface. A Constraint is always identical to itself.
	//
	// Constraints serialized for caching are de-serialized into identical instances.
	identical(Constraint) bool
}

// constraintFromCache returns a Constraint identical to the one which produced m.
func constraintFromCache(m *pb.Constraint) (Constraint, error) {
	switch m.Type {
	case pb.Constraint_Revision:
		return Revision(m.Value), nil
	case pb.Constraint_Branch:
		return NewBranch(m.Value), nil
	case pb.Constraint_DefaultBranch:
		return newDefaultBranch(m.Value), nil
	case pb.Constraint_Version:
		return plainVersion(m.Value), nil
	case pb.Constraint_Semver:
		return NewSemverConstraint(m.Value)

	default:
		return nil, fmt.Errorf("unrecognized Constraint type: %#v", m)
	}
}

// unpairedVersionFromCache returns an UnpairedVersion identical to the one which produced m.
func unpairedVersionFromCache(m *pb.Constraint) (UnpairedVersion, error) {
	switch m.Type {
	case pb.Constraint_Branch:
		return NewBranch(m.Value), nil
	case pb.Constraint_DefaultBranch:
		return newDefaultBranch(m.Value), nil
	case pb.Constraint_Version:
		return plainVersion(m.Value), nil
	case pb.Constraint_Semver:
		sv, err := semver.NewVersion(m.Value)
		if err != nil {
			return nil, err
		}
		return semVersion{sv: sv}, nil

	default:
		return nil, fmt.Errorf("unrecognized UnpairedVersion type: %#v", m)
	}
}

// NewSemverConstraint attempts to construct a semver Constraint object from the
// input string.
//
// If the input string cannot be made into a valid semver Constraint, an error
// is returned.
func NewSemverConstraint(body string) (Constraint, error) {
	c, err := semver.NewConstraint(body)
	if err != nil {
		return nil, err
	}
	// If we got a simple semver.Version, simplify by returning our
	// corresponding type
	if sv, ok := c.(semver.Version); ok {
		return semVersion{sv: sv}, nil
	}
	return semverConstraint{c: c}, nil
}

// NewSemverConstraintIC attempts to construct a semver Constraint object from the
// input string, defaulting to a caret, ^, when no operator is specified. Put
// differently, ^ is the default operator for NewSemverConstraintIC, while =
// is the default operator for NewSemverConstraint.
//
// If the input string cannot be made into a valid semver Constraint, an error
// is returned.
func NewSemverConstraintIC(body string) (Constraint, error) {
	c, err := semver.NewConstraintIC(body)
	if err != nil {
		return nil, err
	}
	// If we got a simple semver.Version, simplify by returning our
	// corresponding type
	if sv, ok := c.(semver.Version); ok {
		return semVersion{sv: sv}, nil
	}
	return semverConstraint{c: c}, nil
}

type semverConstraint struct {
	c semver.Constraint
}

func (c semverConstraint) String() string {
	return c.c.String()
}

// ImpliedCaretString converts the Constraint to a string in the same manner
// as String(), but treats the empty operator as equivalent to ^, rather
// than =.
//
// In the same way that String() is the inverse of NewConstraint(), this
// method is the inverse of NewSemverConstraintIC().
func (c semverConstraint) ImpliedCaretString() string {
	return c.c.ImpliedCaretString()
}

func (c semverConstraint) typedString() string {
	return fmt.Sprintf("svc-%s", c.c.String())
}

func (c semverConstraint) Matches(v Version) bool {
	switch tv := v.(type) {
	case versionTypeUnion:
		for _, elem := range tv {
			if c.Matches(elem) {
				return true
			}
		}
	case semVersion:
		return c.c.Matches(tv.sv) == nil
	case versionPair:
		if tv2, ok := tv.v.(semVersion); ok {
			return c.c.Matches(tv2.sv) == nil
		}
	}

	return false
}

func (c semverConstraint) MatchesAny(c2 Constraint) bool {
	return c.Intersect(c2) != none
}

func (c semverConstraint) Intersect(c2 Constraint) Constraint {
	switch tc := c2.(type) {
	case anyConstraint:
		return c
	case versionTypeUnion:
		for _, elem := range tc {
			if rc := c.Intersect(elem); rc != none {
				return rc
			}
		}
	case semverConstraint:
		rc := c.c.Intersect(tc.c)
		if !semver.IsNone(rc) {
			return semverConstraint{c: rc}
		}
	case semVersion:
		rc := c.c.Intersect(tc.sv)
		if !semver.IsNone(rc) {
			// If single version intersected with constraint, we know the result
			// must be the single version, so just return it back out
			return c2
		}
	case versionPair:
		if tc2, ok := tc.v.(semVersion); ok {
			rc := c.c.Intersect(tc2.sv)
			if !semver.IsNone(rc) {
				// same reasoning as previous case
				return c2
			}
		}
	}

	return none
}

func (c semverConstraint) identical(c2 Constraint) bool {
	sc2, ok := c2.(semverConstraint)
	if !ok {
		return false
	}
	return c.c.String() == sc2.c.String()
}

func (c semverConstraint) copyTo(msg *pb.Constraint) {
	msg.Type = pb.Constraint_Semver
	msg.Value = c.String()
}

// IsAny indicates if the provided constraint is the wildcard "Any" constraint.
func IsAny(c Constraint) bool {
	_, ok := c.(anyConstraint)
	return ok
}

// Any returns a constraint that will match anything.
func Any() Constraint {
	return anyConstraint{}
}

// anyConstraint is an unbounded constraint - it matches all other types of
// constraints. It mirrors the behavior of the semver package's any type.
type anyConstraint struct{}

func (anyConstraint) String() string {
	return "*"
}

func (anyConstraint) ImpliedCaretString() string {
	return "*"
}

func (anyConstraint) typedString() string {
	return "any-*"
}

func (anyConstraint) Matches(Version) bool {
	return true
}

func (anyConstraint) MatchesAny(Constraint) bool {
	return true
}

func (anyConstraint) Intersect(c Constraint) Constraint {
	return c
}

func (anyConstraint) identical(c Constraint) bool {
	return IsAny(c)
}

func (anyConstraint) copyTo(*pb.Constraint) {
	panic("anyConstraint should never be serialized; it is solver internal-only")
}

// noneConstraint is the empty set - it matches no versions. It mirrors the
// behavior of the semver package's none type.
type noneConstraint struct{}

func (noneConstraint) String() string {
	return ""
}

func (noneConstraint) ImpliedCaretString() string {
	return ""
}

func (noneConstraint) typedString() string {
	return "none-"
}

func (noneConstraint) Matches(Version) bool {
	return false
}

func (noneConstraint) MatchesAny(Constraint) bool {
	return false
}

func (noneConstraint) Intersect(Constraint) Constraint {
	return none
}

func (noneConstraint) identical(c Constraint) bool {
	_, ok := c.(noneConstraint)
	return ok
}

func (noneConstraint) copyTo(*pb.Constraint) {
	panic("noneConstraint should never be serialized; it is solver internal-only")
}

// A ProjectConstraint combines a ProjectIdentifier with a Constraint. It
// indicates that, if packages contained in the ProjectIdentifier enter the
// depgraph, they must do so at a version that is allowed by the Constraint.
type ProjectConstraint struct {
	Ident      ProjectIdentifier
	Constraint Constraint
}

// ProjectConstraints is a map of projects, as identified by their import path
// roots (ProjectRoots) to the corresponding ProjectProperties.
//
// They are the standard form in which Manifests declare their required
// dependency properties - constraints and network locations - as well as the
// form in which RootManifests declare their overrides.
type ProjectConstraints map[ProjectRoot]ProjectProperties

type workingConstraint struct {
	Ident                     ProjectIdentifier
	Constraint                Constraint
	overrNet, overrConstraint bool
}

func pcSliceToMap(l []ProjectConstraint, r ...[]ProjectConstraint) ProjectConstraints {
	final := make(ProjectConstraints)

	for _, pc := range l {
		final[pc.Ident.ProjectRoot] = ProjectProperties{
			Source:     pc.Ident.Source,
			Constraint: pc.Constraint,
		}
	}

	for _, pcs := range r {
		for _, pc := range pcs {
			if pp, exists := final[pc.Ident.ProjectRoot]; exists {
				// Technically this should be done through a bridge for
				// cross-version-type matching...but this is a one off for root and
				// that's just ridiculous for this.
				pp.Constraint = pp.Constraint.Intersect(pc.Constraint)
				final[pc.Ident.ProjectRoot] = pp
			} else {
				final[pc.Ident.ProjectRoot] = ProjectProperties{
					Source:     pc.Ident.Source,
					Constraint: pc.Constraint,
				}
			}
		}
	}

	return final
}

func (m ProjectConstraints) asSortedSlice() []ProjectConstraint {
	pcs := make([]ProjectConstraint, len(m))

	k := 0
	for pr, pp := range m {
		pcs[k] = ProjectConstraint{
			Ident: ProjectIdentifier{
				ProjectRoot: pr,
				Source:      pp.Source,
			},
			Constraint: pp.Constraint,
		}
		k++
	}

	sort.SliceStable(pcs, func(i, j int) bool {
		return pcs[i].Ident.Less(pcs[j].Ident)
	})
	return pcs
}

// overrideAll treats the receiver ProjectConstraints map as a set of override
// instructions, and applies overridden values to the ProjectConstraints.
//
// A slice of workingConstraint is returned, allowing differentiation between
// values that were or were not overridden.
func (m ProjectConstraints) overrideAll(pcm ProjectConstraints) (out []workingConstraint) {
	out = make([]workingConstraint, len(pcm))
	k := 0
	for pr, pp := range pcm {
		out[k] = m.override(pr, pp)
		k++
	}

	sort.SliceStable(out, func(i, j int) bool {
		return out[i].Ident.Less(out[j].Ident)
	})
	return
}

// override replaces a single ProjectConstraint with a workingConstraint,
// overriding its values if a corresponding entry exists in the
// ProjectConstraints map.
func (m ProjectConstraints) override(pr ProjectRoot, pp ProjectProperties) workingConstraint {
	wc := workingConstraint{
		Ident: ProjectIdentifier{
			ProjectRoot: pr,
			Source:      pp.Source,
		},
		Constraint: pp.Constraint,
	}

	if opp, has := m[pr]; has {
		// The rule for overrides is that *any* non-zero value for the prop
		// should be considered an override, even if it's equal to what's
		// already there.
		if opp.Constraint != nil {
			wc.Constraint = opp.Constraint
			wc.overrConstraint = true
		}

		// This may appear incorrect, because the solver encodes meaning into
		// the empty string for NetworkName (it means that it would use the
		// import path by default, but could be coerced into using an alternate
		// URL). However, that 'coercion' can only happen if there's a
		// disagreement between projects on where a dependency should be sourced
		// from. Such disagreement is exactly what overrides preclude, so
		// there's no need to preserve the meaning of "" here - thus, we can
		// treat it as a zero value and ignore it, rather than applying it.
		if opp.Source != "" {
			wc.Ident.Source = opp.Source
			wc.overrNet = true
		}
	}

	return wc
}
