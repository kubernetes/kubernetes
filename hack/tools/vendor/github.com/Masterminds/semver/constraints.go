package semver

import (
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
		cs := strings.Split(v, ",")
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
					em := fmt.Errorf(c.msg, v, c.orig)
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

var constraintOps map[string]cfunc
var constraintMsg map[string]string
var constraintRegex *regexp.Regexp

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
}

// An individual constraint
type constraint struct {
	// The callback function for the restraint. It performs the logic for
	// the constraint.
	function cfunc

	msg string

	// The version used in the constraint check. For example, if a constraint
	// is '<= 2.0.0' the con a version instance representing 2.0.0.
	con *Version

	// The original parsed version (e.g., 4.x from != 4.x)
	orig string

	// When an x is used as part of the version (e.g., 1.x)
	minorDirty bool
	dirty      bool
	patchDirty bool
}

// Check if a version meets the constraint
func (c *constraint) check(v *Version) bool {
	return c.function(v, c)
}

type cfunc func(v *Version, c *constraint) bool

func parseConstraint(c string) (*constraint, error) {
	m := constraintRegex.FindStringSubmatch(c)
	if m == nil {
		return nil, fmt.Errorf("improper constraint: %s", c)
	}

	ver := m[2]
	orig := ver
	minorDirty := false
	patchDirty := false
	dirty := false
	if isX(m[3]) {
		ver = "0.0.0"
		dirty = true
	} else if isX(strings.TrimPrefix(m[4], ".")) || m[4] == "" {
		minorDirty = true
		dirty = true
		ver = fmt.Sprintf("%s.0.0%s", m[3], m[6])
	} else if isX(strings.TrimPrefix(m[5], ".")) {
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

	cs := &constraint{
		function:   constraintOps[m[1]],
		msg:        constraintMsg[m[1]],
		con:        con,
		orig:       orig,
		minorDirty: minorDirty,
		patchDirty: patchDirty,
		dirty:      dirty,
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
		}

		return false
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

	return v.Compare(c.con) == 1
}

func constraintLessThan(v *Version, c *constraint) bool {
	// If there is a pre-release on the version but the constraint isn't looking
	// for them assume that pre-releases are not compatible. See issue 21 for
	// more details.
	if v.Prerelease() != "" && c.con.Prerelease() == "" {
		return false
	}

	if !c.dirty {
		return v.Compare(c.con) < 0
	}

	if v.Major() > c.con.Major() {
		return false
	} else if v.Minor() > c.con.Minor() && !c.minorDirty {
		return false
	}

	return true
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
		c.msg = constraintMsg["~"]
		return constraintTilde(v, c)
	}

	return v.Equal(c.con)
}

// ^* --> (any)
// ^2, ^2.x, ^2.x.x --> >=2.0.0, <3.0.0
// ^2.0, ^2.0.x --> >=2.0.0, <3.0.0
// ^1.2, ^1.2.x --> >=1.2.0, <2.0.0
// ^1.2.3 --> >=1.2.3, <2.0.0
// ^1.2.0 --> >=1.2.0, <2.0.0
func constraintCaret(v *Version, c *constraint) bool {
	// If there is a pre-release on the version but the constraint isn't looking
	// for them assume that pre-releases are not compatible. See issue 21 for
	// more details.
	if v.Prerelease() != "" && c.con.Prerelease() == "" {
		return false
	}

	if v.LessThan(c.con) {
		return false
	}

	if v.Major() != c.con.Major() {
		return false
	}

	return true
}

var constraintRangeRegex *regexp.Regexp

const cvRegex string = `v?([0-9|x|X|\*]+)(\.[0-9|x|X|\*]+)?(\.[0-9|x|X|\*]+)?` +
	`(-([0-9A-Za-z\-]+(\.[0-9A-Za-z\-]+)*))?` +
	`(\+([0-9A-Za-z\-]+(\.[0-9A-Za-z\-]+)*))?`

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
