package semver

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// The compiled version of the regex created at init() is cached here so it
// only needs to be created once.
var versionRegex *regexp.Regexp
var validPrereleaseRegex *regexp.Regexp

var (
	// ErrInvalidSemVer is returned a version is found to be invalid when
	// being parsed.
	ErrInvalidSemVer = errors.New("Invalid Semantic Version")

	// ErrInvalidMetadata is returned when the metadata is an invalid format
	ErrInvalidMetadata = errors.New("Invalid Metadata string")

	// ErrInvalidPrerelease is returned when the pre-release is an invalid format
	ErrInvalidPrerelease = errors.New("Invalid Prerelease string")
)

// SemVerRegex is the regular expression used to parse a semantic version.
const SemVerRegex string = `v?([0-9]+)(\.[0-9]+)?(\.[0-9]+)?` +
	`(-([0-9A-Za-z\-]+(\.[0-9A-Za-z\-]+)*))?` +
	`(\+([0-9A-Za-z\-]+(\.[0-9A-Za-z\-]+)*))?`

// ValidPrerelease is the regular expression which validates
// both prerelease and metadata values.
const ValidPrerelease string = `^([0-9A-Za-z\-]+(\.[0-9A-Za-z\-]+)*)$`

// Version represents a single semantic version.
type Version struct {
	major, minor, patch int64
	pre                 string
	metadata            string
	original            string
}

func init() {
	versionRegex = regexp.MustCompile("^" + SemVerRegex + "$")
	validPrereleaseRegex = regexp.MustCompile(ValidPrerelease)
}

// NewVersion parses a given version and returns an instance of Version or
// an error if unable to parse the version.
func NewVersion(v string) (*Version, error) {
	m := versionRegex.FindStringSubmatch(v)
	if m == nil {
		return nil, ErrInvalidSemVer
	}

	sv := &Version{
		metadata: m[8],
		pre:      m[5],
		original: v,
	}

	var temp int64
	temp, err := strconv.ParseInt(m[1], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("Error parsing version segment: %s", err)
	}
	sv.major = temp

	if m[2] != "" {
		temp, err = strconv.ParseInt(strings.TrimPrefix(m[2], "."), 10, 64)
		if err != nil {
			return nil, fmt.Errorf("Error parsing version segment: %s", err)
		}
		sv.minor = temp
	} else {
		sv.minor = 0
	}

	if m[3] != "" {
		temp, err = strconv.ParseInt(strings.TrimPrefix(m[3], "."), 10, 64)
		if err != nil {
			return nil, fmt.Errorf("Error parsing version segment: %s", err)
		}
		sv.patch = temp
	} else {
		sv.patch = 0
	}

	return sv, nil
}

// MustParse parses a given version and panics on error.
func MustParse(v string) *Version {
	sv, err := NewVersion(v)
	if err != nil {
		panic(err)
	}
	return sv
}

// String converts a Version object to a string.
// Note, if the original version contained a leading v this version will not.
// See the Original() method to retrieve the original value. Semantic Versions
// don't contain a leading v per the spec. Instead it's optional on
// implementation.
func (v *Version) String() string {
	var buf bytes.Buffer

	fmt.Fprintf(&buf, "%d.%d.%d", v.major, v.minor, v.patch)
	if v.pre != "" {
		fmt.Fprintf(&buf, "-%s", v.pre)
	}
	if v.metadata != "" {
		fmt.Fprintf(&buf, "+%s", v.metadata)
	}

	return buf.String()
}

// Original returns the original value passed in to be parsed.
func (v *Version) Original() string {
	return v.original
}

// Major returns the major version.
func (v *Version) Major() int64 {
	return v.major
}

// Minor returns the minor version.
func (v *Version) Minor() int64 {
	return v.minor
}

// Patch returns the patch version.
func (v *Version) Patch() int64 {
	return v.patch
}

// Prerelease returns the pre-release version.
func (v *Version) Prerelease() string {
	return v.pre
}

// Metadata returns the metadata on the version.
func (v *Version) Metadata() string {
	return v.metadata
}

// originalVPrefix returns the original 'v' prefix if any.
func (v *Version) originalVPrefix() string {

	// Note, only lowercase v is supported as a prefix by the parser.
	if v.original != "" && v.original[:1] == "v" {
		return v.original[:1]
	}
	return ""
}

// IncPatch produces the next patch version.
// If the current version does not have prerelease/metadata information,
// it unsets metadata and prerelease values, increments patch number.
// If the current version has any of prerelease or metadata information,
// it unsets both values and keeps curent patch value
func (v Version) IncPatch() Version {
	vNext := v
	// according to http://semver.org/#spec-item-9
	// Pre-release versions have a lower precedence than the associated normal version.
	// according to http://semver.org/#spec-item-10
	// Build metadata SHOULD be ignored when determining version precedence.
	if v.pre != "" {
		vNext.metadata = ""
		vNext.pre = ""
	} else {
		vNext.metadata = ""
		vNext.pre = ""
		vNext.patch = v.patch + 1
	}
	vNext.original = v.originalVPrefix() + "" + vNext.String()
	return vNext
}

// IncMinor produces the next minor version.
// Sets patch to 0.
// Increments minor number.
// Unsets metadata.
// Unsets prerelease status.
func (v Version) IncMinor() Version {
	vNext := v
	vNext.metadata = ""
	vNext.pre = ""
	vNext.patch = 0
	vNext.minor = v.minor + 1
	vNext.original = v.originalVPrefix() + "" + vNext.String()
	return vNext
}

// IncMajor produces the next major version.
// Sets patch to 0.
// Sets minor to 0.
// Increments major number.
// Unsets metadata.
// Unsets prerelease status.
func (v Version) IncMajor() Version {
	vNext := v
	vNext.metadata = ""
	vNext.pre = ""
	vNext.patch = 0
	vNext.minor = 0
	vNext.major = v.major + 1
	vNext.original = v.originalVPrefix() + "" + vNext.String()
	return vNext
}

// SetPrerelease defines the prerelease value.
// Value must not include the required 'hypen' prefix.
func (v Version) SetPrerelease(prerelease string) (Version, error) {
	vNext := v
	if len(prerelease) > 0 && !validPrereleaseRegex.MatchString(prerelease) {
		return vNext, ErrInvalidPrerelease
	}
	vNext.pre = prerelease
	vNext.original = v.originalVPrefix() + "" + vNext.String()
	return vNext, nil
}

// SetMetadata defines metadata value.
// Value must not include the required 'plus' prefix.
func (v Version) SetMetadata(metadata string) (Version, error) {
	vNext := v
	if len(metadata) > 0 && !validPrereleaseRegex.MatchString(metadata) {
		return vNext, ErrInvalidMetadata
	}
	vNext.metadata = metadata
	vNext.original = v.originalVPrefix() + "" + vNext.String()
	return vNext, nil
}

// LessThan tests if one version is less than another one.
func (v *Version) LessThan(o *Version) bool {
	return v.Compare(o) < 0
}

// GreaterThan tests if one version is greater than another one.
func (v *Version) GreaterThan(o *Version) bool {
	return v.Compare(o) > 0
}

// Equal tests if two versions are equal to each other.
// Note, versions can be equal with different metadata since metadata
// is not considered part of the comparable version.
func (v *Version) Equal(o *Version) bool {
	return v.Compare(o) == 0
}

// Compare compares this version to another one. It returns -1, 0, or 1 if
// the version smaller, equal, or larger than the other version.
//
// Versions are compared by X.Y.Z. Build metadata is ignored. Prerelease is
// lower than the version without a prerelease.
func (v *Version) Compare(o *Version) int {
	// Compare the major, minor, and patch version for differences. If a
	// difference is found return the comparison.
	if d := compareSegment(v.Major(), o.Major()); d != 0 {
		return d
	}
	if d := compareSegment(v.Minor(), o.Minor()); d != 0 {
		return d
	}
	if d := compareSegment(v.Patch(), o.Patch()); d != 0 {
		return d
	}

	// At this point the major, minor, and patch versions are the same.
	ps := v.pre
	po := o.Prerelease()

	if ps == "" && po == "" {
		return 0
	}
	if ps == "" {
		return 1
	}
	if po == "" {
		return -1
	}

	return comparePrerelease(ps, po)
}

// UnmarshalJSON implements JSON.Unmarshaler interface.
func (v *Version) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	temp, err := NewVersion(s)
	if err != nil {
		return err
	}
	v.major = temp.major
	v.minor = temp.minor
	v.patch = temp.patch
	v.pre = temp.pre
	v.metadata = temp.metadata
	v.original = temp.original
	temp = nil
	return nil
}

// MarshalJSON implements JSON.Marshaler interface.
func (v *Version) MarshalJSON() ([]byte, error) {
	return json.Marshal(v.String())
}

func compareSegment(v, o int64) int {
	if v < o {
		return -1
	}
	if v > o {
		return 1
	}

	return 0
}

func comparePrerelease(v, o string) int {

	// split the prelease versions by their part. The separator, per the spec,
	// is a .
	sparts := strings.Split(v, ".")
	oparts := strings.Split(o, ".")

	// Find the longer length of the parts to know how many loop iterations to
	// go through.
	slen := len(sparts)
	olen := len(oparts)

	l := slen
	if olen > slen {
		l = olen
	}

	// Iterate over each part of the prereleases to compare the differences.
	for i := 0; i < l; i++ {
		// Since the lentgh of the parts can be different we need to create
		// a placeholder. This is to avoid out of bounds issues.
		stemp := ""
		if i < slen {
			stemp = sparts[i]
		}

		otemp := ""
		if i < olen {
			otemp = oparts[i]
		}

		d := comparePrePart(stemp, otemp)
		if d != 0 {
			return d
		}
	}

	// Reaching here means two versions are of equal value but have different
	// metadata (the part following a +). They are not identical in string form
	// but the version comparison finds them to be equal.
	return 0
}

func comparePrePart(s, o string) int {
	// Fastpath if they are equal
	if s == o {
		return 0
	}

	// When s or o are empty we can use the other in an attempt to determine
	// the response.
	if s == "" {
		if o != "" {
			return -1
		}
		return 1
	}

	if o == "" {
		if s != "" {
			return 1
		}
		return -1
	}

	// When comparing strings "99" is greater than "103". To handle
	// cases like this we need to detect numbers and compare them. According
	// to the semver spec, numbers are always positive. If there is a - at the
	// start like -99 this is to be evaluated as an alphanum. numbers always
	// have precedence over alphanum. Parsing as Uints because negative numbers
	// are ignored.

	oi, n1 := strconv.ParseUint(o, 10, 64)
	si, n2 := strconv.ParseUint(s, 10, 64)

	// The case where both are strings compare the strings
	if n1 != nil && n2 != nil {
		if s > o {
			return 1
		}
		return -1
	} else if n1 != nil {
		// o is a string and s is a number
		return -1
	} else if n2 != nil {
		// s is a string and o is a number
		return 1
	}
	// Both are numbers
	if si > oi {
		return 1
	}
	return -1

}
