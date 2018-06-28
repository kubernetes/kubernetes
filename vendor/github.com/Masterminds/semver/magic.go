package semver

import "errors"

var errNone = errors.New("The 'None' constraint admits no versions.")

// Any is a constraint that is satisfied by any valid semantic version.
type any struct{}

// Any creates a constraint that will match any version.
func Any() Constraint {
	return any{}
}

func (any) String() string {
	return "*"
}

func (any) ImpliedCaretString() string {
	return "*"
}

// Matches checks that a version satisfies the constraint. As all versions
// satisfy Any, this always returns nil.
func (any) Matches(v Version) error {
	return nil
}

// Intersect computes the intersection between two constraints.
//
// As Any is the set of all possible versions, any intersection with that
// infinite set will necessarily be the entirety of the second set. Thus, this
// simply returns the passed constraint.
func (any) Intersect(c Constraint) Constraint {
	return c
}

// MatchesAny indicates whether there exists any version that can satisfy both
// this constraint, and the passed constraint. As all versions
// satisfy Any, this is always true - unless none is passed.
func (any) MatchesAny(c Constraint) bool {
	if _, ok := c.(none); ok {
		return false
	}
	return true
}

func (any) Union(c Constraint) Constraint {
	return Any()
}

func (any) _private() {}

// None is an unsatisfiable constraint - it represents the empty set.
type none struct{}

// None creates a constraint that matches no versions (the empty set).
func None() Constraint {
	return none{}
}

func (none) String() string {
	return ""
}

func (none) ImpliedCaretString() string {
	return ""
}

// Matches checks that a version satisfies the constraint. As no version can
// satisfy None, this always fails (returns an error).
func (none) Matches(v Version) error {
	return errNone
}

// Intersect computes the intersection between two constraints.
//
// None is the empty set of versions, and any intersection with the empty set is
// necessarily the empty set. Thus, this always returns None.
func (none) Intersect(Constraint) Constraint {
	return None()
}

func (none) Union(c Constraint) Constraint {
	return c
}

// MatchesAny indicates whether there exists any version that can satisfy the
// constraint. As no versions satisfy None, this is always false.
func (none) MatchesAny(c Constraint) bool {
	return false
}

func (none) _private() {}

// IsNone indicates if a constraint will match no versions - that is, the
// constraint represents the empty set.
func IsNone(c Constraint) bool {
	_, ok := c.(none)
	return ok
}

// IsAny indicates if a constraint will match any and all versions.
func IsAny(c Constraint) bool {
	_, ok := c.(any)
	return ok
}
