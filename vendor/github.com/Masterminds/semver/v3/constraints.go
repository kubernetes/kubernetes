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
	containsPre []bool

	// IncludePrerelease specifies if pre-releases should be included in
	// the results. Note, if a constraint range has a prerelease than
	// prereleases will be included for that AND group even if this is
	// set to false.
	IncludePrerelease bool
}

// NewConstraint returns a Constraints instance that a Version instance can
// be checked against. If there is a parse error it will be returned.
func NewConstraint(c string) (*Constraints, error) {

	// Rewrite - ranges into a comparison operation.
	c = rewriteRange(c)

	ors := strings.Split(c, "||")
	lenors := len(ors)
	or := make([][]*constraint, lenors)
	hasPre := make([]bool, lenors)
	for k, v := range ors {
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

			// If one of the constraints has a prerelease record this.
			// This information is used when checking all in an "and"
			// group to ensure they all check for prereleases.
			if pc.con.pre != "" {
				hasPre[k] = true
			}

			result[i] = pc
		}
		or[k] = result
	}

	o := &Constraints{
		constraints: or,
		containsPre: hasPre,
	}
	return o, nil
}

// Check tests if a version satisfies the constraints.
func (cs Constraints) Check(v *Version) bool {
	// TODO(mattfarina): For v4 of this library consolidate the Check and Validate
	// functions as the underlying functions make that possible now.
	// loop over the ORs and check the inner ANDs
	for i, o := range cs.constraints {
		joy := true
		for _, c := range o {
			if check, _ := c.check(v, (cs.IncludePrerelease || cs.containsPre[i])); !check {
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
	for i, o := range cs.constraints {
		joy := true
		for _, c := range o {
			// Before running the check handle the case there the version is
			// a prerelease and the check is not searching for prereleases.
			if !(cs.IncludePrerelease || cs.containsPre[i]) && v.pre != "" {
				if !prerelesase {
					em := fmt.Errorf("%s is a prerelease version and the constraint is only looking for release versions", v)
					e = append(e, em)
					prerelesase = true
				}
				joy = false

			} else {

				if _, err := c.check(v, (cs.IncludePrerelease || cs.containsPre[i])); err != nil {
					e = append(e, err)
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

// UnmarshalText implements the encoding.TextUnmarshaler interface.
func (cs *Constraints) UnmarshalText(text []byte) error {
	temp, err := NewConstraint(string(text))
	if err != nil {
		return err
	}

	*cs = *temp

	return nil
}

// MarshalText implements the encoding.TextMarshaler interface.
func (cs Constraints) MarshalText() ([]byte, error) {
	return []byte(cs.String()), nil
}

var constraintOps map[string]cfunc
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

	ops := `=||!=|>|<|>=|=>|<=|=<|~|~>|\^`

	constraintRegex = regexp.MustCompile(fmt.Sprintf(
		`^\s*(%s)\s*(%s)\s*$`,
		ops,
		cvRegex))

	constraintRangeRegex = regexp.MustCompile(fmt.Sprintf(
		`\s*(%s)\s+-\s+(%s)\s*`,
		cvRegex, cvRegex))

	findConstraintRegex = regexp.MustCompile(fmt.Sprintf(
		`(%s)\s*(%s)`,
		ops,
		cvRegex))

	// The first time a constraint shows up will look slightly different from
	// future times it shows up due to a leading space or comma in a given
	// string.
	validConstraintRegex = regexp.MustCompile(fmt.Sprintf(
		`^(\s*(%s)\s*(%s)\s*)((?:\s+|,\s*)(%s)\s*(%s)\s*)*$`,
		ops,
		cvRegex,
		ops,
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
func (c *constraint) check(v *Version, includePre bool) (bool, error) {
	return constraintOps[c.origfunc](v, c, includePre)
}

// String prints an individual constraint into a string
func (c *constraint) string() string {
	return c.origfunc + c.orig
}

type cfunc func(v *Version, c *constraint, includePre bool) (bool, error)

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
			ver = fmt.Sprintf("0.0.0%s", m[6])
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
			return nil, errors.New("constraint parser error")
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
		return nil, errors.New("constraint parser error")
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
func constraintNotEqual(v *Version, c *constraint, includePre bool) (bool, error) {
	// The existence of prereleases is checked at the group level and passed in.
	// Exit early if the version has a prerelease but those are to be ignored.
	if v.Prerelease() != "" && !includePre {
		return false, fmt.Errorf("%s is a prerelease version and the constraint is only looking for release versions", v)
	}

	if c.dirty {
		if c.con.Major() != v.Major() {
			return true, nil
		}
		if c.con.Minor() != v.Minor() && !c.minorDirty {
			return true, nil
		} else if c.minorDirty {
			return false, fmt.Errorf("%s is equal to %s", v, c.orig)
		} else if c.con.Patch() != v.Patch() && !c.patchDirty {
			return true, nil
		} else if c.patchDirty {
			// Need to handle prereleases if present
			if v.Prerelease() != "" || c.con.Prerelease() != "" {
				eq := comparePrerelease(v.Prerelease(), c.con.Prerelease()) != 0
				if eq {
					return true, nil
				}
				return false, fmt.Errorf("%s is equal to %s", v, c.orig)
			}
			return false, fmt.Errorf("%s is equal to %s", v, c.orig)
		}
	}

	eq := v.Equal(c.con)
	if eq {
		return false, fmt.Errorf("%s is equal to %s", v, c.orig)
	}

	return true, nil
}

func constraintGreaterThan(v *Version, c *constraint, includePre bool) (bool, error) {

	// The existence of prereleases is checked at the group level and passed in.
	// Exit early if the version has a prerelease but those are to be ignored.
	if v.Prerelease() != "" && !includePre {
		return false, fmt.Errorf("%s is a prerelease version and the constraint is only looking for release versions", v)
	}

	var eq bool

	if !c.dirty {
		eq = v.Compare(c.con) == 1
		if eq {
			return true, nil
		}
		return false, fmt.Errorf("%s is less than or equal to %s", v, c.orig)
	}

	if v.Major() > c.con.Major() {
		return true, nil
	} else if v.Major() < c.con.Major() {
		return false, fmt.Errorf("%s is less than or equal to %s", v, c.orig)
	} else if c.minorDirty {
		// This is a range case such as >11. When the version is something like
		// 11.1.0 is it not > 11. For that we would need 12 or higher
		return false, fmt.Errorf("%s is less than or equal to %s", v, c.orig)
	} else if c.patchDirty {
		// This is for ranges such as >11.1. A version of 11.1.1 is not greater
		// which one of 11.2.1 is greater
		eq = v.Minor() > c.con.Minor()
		if eq {
			return true, nil
		}
		return false, fmt.Errorf("%s is less than or equal to %s", v, c.orig)
	}

	// If we have gotten here we are not comparing pre-preleases and can use the
	// Compare function to accomplish that.
	eq = v.Compare(c.con) == 1
	if eq {
		return true, nil
	}
	return false, fmt.Errorf("%s is less than or equal to %s", v, c.orig)
}

func constraintLessThan(v *Version, c *constraint, includePre bool) (bool, error) {
	// The existence of prereleases is checked at the group level and passed in.
	// Exit early if the version has a prerelease but those are to be ignored.
	if v.Prerelease() != "" && !includePre {
		return false, fmt.Errorf("%s is a prerelease version and the constraint is only looking for release versions", v)
	}

	eq := v.Compare(c.con) < 0
	if eq {
		return true, nil
	}
	return false, fmt.Errorf("%s is greater than or equal to %s", v, c.orig)
}

func constraintGreaterThanEqual(v *Version, c *constraint, includePre bool) (bool, error) {

	// The existence of prereleases is checked at the group level and passed in.
	// Exit early if the version has a prerelease but those are to be ignored.
	if v.Prerelease() != "" && !includePre {
		return false, fmt.Errorf("%s is a prerelease version and the constraint is only looking for release versions", v)
	}

	eq := v.Compare(c.con) >= 0
	if eq {
		return true, nil
	}
	return false, fmt.Errorf("%s is less than %s", v, c.orig)
}

func constraintLessThanEqual(v *Version, c *constraint, includePre bool) (bool, error) {
	// The existence of prereleases is checked at the group level and passed in.
	// Exit early if the version has a prerelease but those are to be ignored.
	if v.Prerelease() != "" && !includePre {
		return false, fmt.Errorf("%s is a prerelease version and the constraint is only looking for release versions", v)
	}

	var eq bool

	if !c.dirty {
		eq = v.Compare(c.con) <= 0
		if eq {
			return true, nil
		}
		return false, fmt.Errorf("%s is greater than %s", v, c.orig)
	}

	if v.Major() > c.con.Major() {
		return false, fmt.Errorf("%s is greater than %s", v, c.orig)
	} else if v.Major() == c.con.Major() && v.Minor() > c.con.Minor() && !c.minorDirty {
		return false, fmt.Errorf("%s is greater than %s", v, c.orig)
	}

	return true, nil
}

// ~*, ~>* --> >= 0.0.0 (any)
// ~2, ~2.x, ~2.x.x, ~>2, ~>2.x ~>2.x.x --> >=2.0.0, <3.0.0
// ~2.0, ~2.0.x, ~>2.0, ~>2.0.x --> >=2.0.0, <2.1.0
// ~1.2, ~1.2.x, ~>1.2, ~>1.2.x --> >=1.2.0, <1.3.0
// ~1.2.3, ~>1.2.3 --> >=1.2.3, <1.3.0
// ~1.2.0, ~>1.2.0 --> >=1.2.0, <1.3.0
func constraintTilde(v *Version, c *constraint, includePre bool) (bool, error) {
	// The existence of prereleases is checked at the group level and passed in.
	// Exit early if the version has a prerelease but those are to be ignored.
	if v.Prerelease() != "" && !includePre {
		return false, fmt.Errorf("%s is a prerelease version and the constraint is only looking for release versions", v)
	}

	if v.LessThan(c.con) {
		return false, fmt.Errorf("%s is less than %s", v, c.orig)
	}

	// ~0.0.0 is a special case where all constraints are accepted. It's
	// equivalent to >= 0.0.0.
	if c.con.Major() == 0 && c.con.Minor() == 0 && c.con.Patch() == 0 &&
		!c.minorDirty && !c.patchDirty {
		return true, nil
	}

	if v.Major() != c.con.Major() {
		return false, fmt.Errorf("%s does not have same major version as %s", v, c.orig)
	}

	if v.Minor() != c.con.Minor() && !c.minorDirty {
		return false, fmt.Errorf("%s does not have same major and minor version as %s", v, c.orig)
	}

	return true, nil
}

// When there is a .x (dirty) status it automatically opts in to ~. Otherwise
// it's a straight =
func constraintTildeOrEqual(v *Version, c *constraint, includePre bool) (bool, error) {
	// The existence of prereleases is checked at the group level and passed in.
	// Exit early if the version has a prerelease but those are to be ignored.
	if v.Prerelease() != "" && !includePre {
		return false, fmt.Errorf("%s is a prerelease version and the constraint is only looking for release versions", v)
	}

	if c.dirty {
		return constraintTilde(v, c, includePre)
	}

	eq := v.Equal(c.con)
	if eq {
		return true, nil
	}

	return false, fmt.Errorf("%s is not equal to %s", v, c.orig)
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
func constraintCaret(v *Version, c *constraint, includePre bool) (bool, error) {
	// The existence of prereleases is checked at the group level and passed in.
	// Exit early if the version has a prerelease but those are to be ignored.
	if v.Prerelease() != "" && !includePre {
		return false, fmt.Errorf("%s is a prerelease version and the constraint is only looking for release versions", v)
	}

	// This less than handles prereleases
	if v.LessThan(c.con) {
		return false, fmt.Errorf("%s is less than %s", v, c.orig)
	}

	var eq bool

	// ^ when the major > 0 is >=x.y.z < x+1
	if c.con.Major() > 0 || c.minorDirty {

		// ^ has to be within a major range for > 0. Everything less than was
		// filtered out with the LessThan call above. This filters out those
		// that greater but not within the same major range.
		eq = v.Major() == c.con.Major()
		if eq {
			return true, nil
		}
		return false, fmt.Errorf("%s does not have same major version as %s", v, c.orig)
	}

	// ^ when the major is 0 and minor > 0 is >=0.y.z < 0.y+1
	if c.con.Major() == 0 && v.Major() > 0 {
		return false, fmt.Errorf("%s does not have same major version as %s", v, c.orig)
	}
	// If the con Minor is > 0 it is not dirty
	if c.con.Minor() > 0 || c.patchDirty {
		eq = v.Minor() == c.con.Minor()
		if eq {
			return true, nil
		}
		return false, fmt.Errorf("%s does not have same minor version as %s. Expected minor versions to match when constraint major version is 0", v, c.orig)
	}
	// ^ when the minor is 0 and minor > 0 is =0.0.z
	if c.con.Minor() == 0 && v.Minor() > 0 {
		return false, fmt.Errorf("%s does not have same minor version as %s", v, c.orig)
	}

	// At this point the major is 0 and the minor is 0 and not dirty. The patch
	// is not dirty so we need to check if they are equal. If they are not equal
	eq = c.con.Patch() == v.Patch()
	if eq {
		return true, nil
	}
	return false, fmt.Errorf("%s does not equal %s. Expect version and constraint to equal when major and minor versions are 0", v, c.orig)
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
		t := fmt.Sprintf(">= %s, <= %s ", v[1], v[11])
		o = strings.Replace(o, v[0], t, 1)
	}

	return o
}
