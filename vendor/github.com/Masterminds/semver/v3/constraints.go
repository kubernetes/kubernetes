package semver

import (
	"bytes"
	"errors"
	"fmt"
	"regexp"
	"strings"
)

// Constraints is one or more constraint that a semantic version can be
// checked against.
type Constraints struct {
	constraints [][]*constraint
}

// NewConstraint returns a Constraints instance that a Version instance can
// be checked against. If there is a parse error it will be returned.
func NewConstraint(c string) (*Constraints, error) {

	// Rewrite - ranges into a comparison operation.
	c = rewriteRange(c)

	ors := strings.Split(c, "||")
	or := make([][]*constraint, len(ors))
	for k, v := range ors {

		// TODO: Find a way to validate and fetch all the constraints in a simpler form

		// Validate the segment
		if !validConstraintRegex.MatchString(v) {
			return nil, fmt.Errorf("improper constraint: %s", v)
		}

		cs := findConstraintRegex.FindAllString(v, -1)
		if cs == nil {
			cs = append(cs, v)
		}
		result := make([]*constraint, len(cs))
		for i, s := range cs {
			pc, err := parseConstraint(s)
			if err != nil {
				return nil, err
			}

			result[i] = pc
		}
		or[k] = result
	}

	o := &Constraints{constraints: or}
	return o, nil
}

// Check tests if a version satisfies the constraints.
func (cs Constraints) Check(v *Version) bool {
	// loop over the ORs and check the inner ANDs
	for _, o := range cs.constraints {
		joy := true
		for _, c := range o {
			if !c.check(v) {
				joy = false
				break
			}
		}

		if joy {
			return true
		}
	}

	return false
}

// Validate checks if a version satisfies a constraint. If not a slice of
// reasons for the failure are returned in addition to a bool.
func (cs Constraints) Validate(v *Version) (bool, []error) {
	// loop over the ORs and check the inner ANDs
	var e []error

	// Capture the prerelease message only once. When it happens the first time
	// this var is marked
	var prerelesase bool
	for _, o := range cs.constraints {
		joy := true
		for _, c := range o {
			// Before running the check handle the case there the version is
			// a prerelease and the check is not searching for prereleases.
			if c.con.pre == "" && v.pre != "" {
				if !prerelesase {
					em := fmt.Errorf("%s is a prerelease version and the constraint is only looking for release versions", v)
					e = append(e, em)
					prerelesase = true
				}
				joy = false

			} else {

				if !c.check(v) {
					em := fmt.Errorf(constraintMsg[c.origfunc], v, c.orig)
					e = append(e, em)
					joy = false
				}
			}
		}

		if joy {
			return true, []error{}
		}
	}

	return false, e
}

func (cs Constraints) String() string {
	buf := make([]string, len(cs.constraints))
	var tmp bytes.Buffer

	for k, v := range cs.constraints {
		tmp.Reset()
		vlen := len(v)
		for kk, c := range v {
			tmp.WriteString(c.string())

			// Space separate the AND conditions
			if vlen > 1 && kk < vlen-1 {
				tmp.WriteString(" ")
			}
		}
		buf[k] = tmp.String()
	}

	return strings.Join(buf, " || ")
}

var constraintOps map[string]cfunc
var constraintMsg map[string]string
var constraintRegex *regexp.Regexp
var constraintRangeRegex *regexp.Regexp

// Used to find individual constraints within a multi-constraint string
var findConstraintRegex *regexp.Regexp

// Used to validate an segment of ANDs is valid
var validConstraintRegex *regexp.Regexp

const cvRegex string = `v?([0-9|x|X|\*]+)(\.[0-9|x|X|\*]+)?(\.[0-9|x|X|\*]+)?` +
	`(-([0-9A-Za-z\-]+(\.[0-9A-Za-z\-]+)*))?` +
	`(\+([0-9A-Za-z\-]+(\.[0-9A-Za-z\-]+)*))?`

func init() {
	constraintOps = map[string]cfunc{
		"":   constraintTildeOrEqual,
		"=":  constraintTildeOrEqual,
		"!=": constraintNotEqual,
		">":  constraintGreaterThan,
		"<":  constraintLessThan,
		">=": constraintGreaterThanEqual,
		"=>": constraintGreaterThanEqual,
		"<=": constraintLessThanEqual,
		"=<": constraintLessThanEqual,
		"~":  constraintTilde,
		"~>": constraintTilde,
		"^":  constraintCaret,
	}

	constraintMsg = map[string]string{
		"":   "%s is not equal to %s",
		"=":  "%s is not equal to %s",
		"!=": "%s is equal to %s",
		">":  "%s is less than or equal to %s",
		"<":  "%s is greater than or equal to %s",
		">=": "%s is less than %s",
		"=>": "%s is less than %s",
		"<=": "%s is greater than %s",
		"=<": "%s is greater than %s",
		"~":  "%s does not have same major and minor version as %s",
		"~>": "%s does not have same major and minor version as %s",
		"^":  "%s does not have same major version as %s",
	}

	ops := make([]string, 0, len(constraintOps))
	for k := range constraintOps {
		ops = append(ops, regexp.QuoteMeta(k))
	}

	constraintRegex = regexp.MustCompile(fmt.Sprintf(
		`^\s*(%s)\s*(%s)\s*$`,
		strings.Join(ops, "|"),
		cvRegex))

	constraintRangeRegex = regexp.MustCompile(fmt.Sprintf(
		`\s*(%s)\s+-\s+(%s)\s*`,
		cvRegex, cvRegex))

	findConstraintRegex = regexp.MustCompile(fmt.Sprintf(
		`(%s)\s*(%s)`,
		strings.Join(ops, "|"),
		cvRegex))

	validConstraintRegex = regexp.MustCompile(fmt.Sprintf(
		`^(\s*(%s)\s*(%s)\s*\,?)+$`,
		strings.Join(ops, "|"),
		cvRegex))
}

// An individual constraint
type constraint struct {
	// The version used in the constraint check. For example, if a constraint
	// is '<= 2.0.0' the con a version instance representing 2.0.0.
	con *Version

	// The original parsed version (e.g., 4.x from != 4.x)
	orig string

	// The original operator for the constraint
	origfunc string

	// When an x is used as part of the version (e.g., 1.x)
	minorDirty bool
	dirty      bool
	patchDirty bool
}

// Check if a version meets the constraint
func (c *constraint) check(v *Version) bool {
	return constraintOps[c.origfunc](v, c)
}

// String prints an individual constraint into a string
func (c *constraint) string() string {
	return c.origfunc + c.orig
}

type cfunc func(v *Version, c *constraint) bool

func parseConstraint(c string) (*constraint, error) {
	if len(c) > 0 {
		m := constraintRegex.FindStringSubmatch(c)
		if m == nil {
			return nil, fmt.Errorf("improper constraint: %s", c)
		}

		cs := &constraint{
			orig:     m[2],
			origfunc: m[1],
		}

		ver := m[2]
		minorDirty := false
		patchDirty := false
		dirty := false
		if isX(m[3]) || m[3] == "" {
			ver = "0.0.0"
			dirty = true
		} else if isX(strings.TrimPrefix(m[4], ".")) || m[4] == "" {
			minorDirty = true
			dirty = true
			ver = fmt.Sprintf("%s.0.0%s", m[3], m[6])
		} else if isX(strings.TrimPrefix(m[5], ".")) || m[5] == "" {
			dirty = true
			patchDirty = true
			ver = fmt.Sprintf("%s%s.0%s", m[3], m[4], m[6])
		}

		con, err := NewVersion(ver)
		if err != nil {

			// The constraintRegex should catch any regex parsing errors. So,
			// we should never get here.
			return nil, errors.New("constraint Parser Error")
		}

		cs.con = con
		cs.minorDirty = minorDirty
		cs.patchDirty = patchDirty
		cs.dirty = dirty

		return cs, nil
	}

	// The rest is the special case where an empty string was passed in which
	// is equivalent to * or >=0.0.0
	con, err := StrictNewVersion("0.0.0")
	if err != nil {

		// The constraintRegex should catch any regex parsing errors. So,
		// we should never get here.
		return nil, errors.New("constraint Parser Error")
	}

	cs := &constraint{
		con:        con,
		orig:       c,
		origfunc:   "",
		minorDirty: false,
		patchDirty: false,
		dirty:      true,
	}
	return cs, nil
}

// Constraint functions
func constraintNotEqual(v *Version, c *constraint) bool {
	if c.dirty {

		// If there is a pre-release on the version but the constraint isn't looking
		// for them assume that pre-releases are not compatible. See issue 21 for
		// more details.
		if v.Prerelease() != "" && c.con.Prerelease() == "" {
			return false
		}

		if c.con.Major() != v.Major() {
			return true
		}
		if c.con.Minor() != v.Minor() && !c.minorDirty {
			return true
		} else if c.minorDirty {
			return false
		} else if c.con.Patch() != v.Patch() && !c.patchDirty {
			return true
		} else if c.patchDirty {
			// Need to handle prereleases if present
			if v.Prerelease() != "" || c.con.Prerelease() != "" {
				return comparePrerelease(v.Prerelease(), c.con.Prerelease()) != 0
			}
			return false
		}
	}

	return !v.Equal(c.con)
}

func constraintGreaterThan(v *Version, c *constraint) bool {

	// If there is a pre-release on the version but the constraint isn't looking
	// for them assume that pre-releases are not compatible. See issue 21 for
	// more details.
	if v.Prerelease() != "" && c.con.Prerelease() == "" {
		return false
	}

	if !c.dirty {
		return v.Compare(c.con) == 1
	}

	if v.Major() > c.con.Major() {
		return true
	} else if v.Major() < c.con.Major() {
		return false
	} else if c.minorDirty {
		// This is a range case such as >11. When the version is something like
		// 11.1.0 is it not > 11. For that we would need 12 or higher
		return false
	} else if c.patchDirty {
		// This is for ranges such as >11.1. A version of 11.1.1 is not greater
		// which one of 11.2.1 is greater
		return v.Minor() > c.con.Minor()
	}

	// If we have gotten here we are not comparing pre-preleases and can use the
	// Compare function to accomplish that.
	return v.Compare(c.con) == 1
}

func constraintLessThan(v *Version, c *constraint) bool {
	// If there is a pre-release on the version but the constraint isn't looking
	// for them assume that pre-releases are not compatible. See issue 21 for
	// more details.
	if v.Prerelease() != "" && c.con.Prerelease() == "" {
		return false
	}

	return v.Compare(c.con) < 0
}

func constraintGreaterThanEqual(v *Version, c *constraint) bool {

	// If there is a pre-release on the version but the constraint isn't looking
	// for them assume that pre-releases are not compatible. See issue 21 for
	// more details.
	if v.Prerelease() != "" && c.con.Prerelease() == "" {
		return false
	}

	return v.Compare(c.con) >= 0
}

func constraintLessThanEqual(v *Version, c *constraint) bool {
	// If there is a pre-release on the version but the constraint isn't looking
	// for them assume that pre-releases are not compatible. See issue 21 for
	// more details.
	if v.Prerelease() != "" && c.con.Prerelease() == "" {
		return false
	}

	if !c.dirty {
		return v.Compare(c.con) <= 0
	}

	if v.Major() > c.con.Major() {
		return false
	} else if v.Minor() > c.con.Minor() && !c.minorDirty {
		return false
	}

	return true
}

// ~*, ~>* --> >= 0.0.0 (any)
// ~2, ~2.x, ~2.x.x, ~>2, ~>2.x ~>2.x.x --> >=2.0.0, <3.0.0
// ~2.0, ~2.0.x, ~>2.0, ~>2.0.x --> >=2.0.0, <2.1.0
// ~1.2, ~1.2.x, ~>1.2, ~>1.2.x --> >=1.2.0, <1.3.0
// ~1.2.3, ~>1.2.3 --> >=1.2.3, <1.3.0
// ~1.2.0, ~>1.2.0 --> >=1.2.0, <1.3.0
func constraintTilde(v *Version, c *constraint) bool {
	// If there is a pre-release on the version but the constraint isn't looking
	// for them assume that pre-releases are not compatible. See issue 21 for
	// more details.
	if v.Prerelease() != "" && c.con.Prerelease() == "" {
		return false
	}

	if v.LessThan(c.con) {
		return false
	}

	// ~0.0.0 is a special case where all constraints are accepted. It's
	// equivalent to >= 0.0.0.
	if c.con.Major() == 0 && c.con.Minor() == 0 && c.con.Patch() == 0 &&
		!c.minorDirty && !c.patchDirty {
		return true
	}

	if v.Major() != c.con.Major() {
		return false
	}

	if v.Minor() != c.con.Minor() && !c.minorDirty {
		return false
	}

	return true
}

// When there is a .x (dirty) status it automatically opts in to ~. Otherwise
// it's a straight =
func constraintTildeOrEqual(v *Version, c *constraint) bool {
	// If there is a pre-release on the version but the constraint isn't looking
	// for them assume that pre-releases are not compatible. See issue 21 for
	// more details.
	if v.Prerelease() != "" && c.con.Prerelease() == "" {
		return false
	}

	if c.dirty {
		return constraintTilde(v, c)
	}

	return v.Equal(c.con)
}

// ^*      -->  (any)
// ^1.2.3  -->  >=1.2.3 <2.0.0
// ^1.2    -->  >=1.2.0 <2.0.0
// ^1      -->  >=1.0.0 <2.0.0
// ^0.2.3  -->  >=0.2.3 <0.3.0
// ^0.2    -->  >=0.2.0 <0.3.0
// ^0.0.3  -->  >=0.0.3 <0.0.4
// ^0.0    -->  >=0.0.0 <0.1.0
// ^0      -->  >=0.0.0 <1.0.0
func constraintCaret(v *Version, c *constraint) bool {
	// If there is a pre-release on the version but the constraint isn't looking
	// for them assume that pre-releases are not compatible. See issue 21 for
	// more details.
	if v.Prerelease() != "" && c.con.Prerelease() == "" {
		return false
	}

	// This less than handles prereleases
	if v.LessThan(c.con) {
		return false
	}

	// ^ when the major > 0 is >=x.y.z < x+1
	if c.con.Major() > 0 || c.minorDirty {

		// ^ has to be within a major range for > 0. Everything less than was
		// filtered out with the LessThan call above. This filters out those
		// that greater but not within the same major range.
		return v.Major() == c.con.Major()
	}

	// ^ when the major is 0 and minor > 0 is >=0.y.z < 0.y+1
	// If the con Minor is > 0 it is not dirty
	if c.con.Minor() > 0 || c.patchDirty {
		return v.Minor() == c.con.Minor()
	}

	// At this point the major is 0 and the minor is 0 and not dirty. The patch
	// is not dirty so we need to check if they are equal. If they are not equal
	return c.con.Patch() == v.Patch()
}

func isX(x string) bool {
	switch x {
	case "x", "*", "X":
		return true
	default:
		return false
	}
}

func rewriteRange(i string) string {
	m := constraintRangeRegex.FindAllStringSubmatch(i, -1)
	if m == nil {
		return i
	}
	o := i
	for _, v := range m {
		t := fmt.Sprintf(">= %s, <= %s", v[1], v[11])
		o = strings.Replace(o, v[0], t, 1)
	}

	return o
}
