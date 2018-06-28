package semver

import (
	"bytes"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"sync"
)

// The compiled version of the regex created at init() is cached here so it
// only needs to be created once.
var versionRegex *regexp.Regexp

var (
	// ErrInvalidSemVer is returned a version is found to be invalid when
	// being parsed.
	ErrInvalidSemVer = errors.New("Invalid Semantic Version")
)

// Error type; lets us defer string interpolation
type badVersionSegment struct {
	e error
}

func (b badVersionSegment) Error() string {
	return fmt.Sprintf("Error parsing version segment: %s", b.e)
}

// CacheVersions controls whether or not parsed constraints are cached. Defaults
// to true.
var CacheVersions = true
var versionCache = make(map[string]vcache)
var versionCacheLock sync.RWMutex

type vcache struct {
	v   Version
	err error
}

// SemVerRegex id the regular expression used to parse a semantic version.
const SemVerRegex string = `v?([0-9]+)(\.[0-9]+)?(\.[0-9]+)?` +
	`(-([0-9A-Za-z\-]+(\.[0-9A-Za-z\-]+)*))?` +
	`(\+([0-9A-Za-z\-]+(\.[0-9A-Za-z\-]+)*))?`

type specialVersion uint8

const (
	notSpecial specialVersion = iota
	zeroVersion
	infiniteVersion
)

// Version represents a single semantic version.
type Version struct {
	major, minor, patch uint64
	pre                 string
	metadata            string
	original            string
	special             specialVersion
}

func init() {
	versionRegex = regexp.MustCompile("^" + SemVerRegex + "$")
}

// NewVersion parses a given version and returns an instance of Version or
// an error if unable to parse the version.
func NewVersion(v string) (Version, error) {
	if CacheVersions {
		versionCacheLock.RLock()
		if sv, exists := versionCache[v]; exists {
			versionCacheLock.RUnlock()
			return sv.v, sv.err
		}
		versionCacheLock.RUnlock()
	}

	m := versionRegex.FindStringSubmatch(v)
	if m == nil {
		if CacheVersions {
			versionCacheLock.Lock()
			versionCache[v] = vcache{err: ErrInvalidSemVer}
			versionCacheLock.Unlock()
		}
		return Version{}, ErrInvalidSemVer
	}

	sv := Version{
		metadata: m[8],
		pre:      m[5],
		original: v,
	}

	var temp uint64
	temp, err := strconv.ParseUint(m[1], 10, 32)
	if err != nil {
		bvs := badVersionSegment{e: err}
		if CacheVersions {
			versionCacheLock.Lock()
			versionCache[v] = vcache{err: bvs}
			versionCacheLock.Unlock()
		}

		return Version{}, bvs
	}
	sv.major = temp

	if m[2] != "" {
		temp, err = strconv.ParseUint(strings.TrimPrefix(m[2], "."), 10, 32)
		if err != nil {
			bvs := badVersionSegment{e: err}
			if CacheVersions {
				versionCacheLock.Lock()
				versionCache[v] = vcache{err: bvs}
				versionCacheLock.Unlock()
			}

			return Version{}, bvs
		}
		sv.minor = temp
	} else {
		sv.minor = 0
	}

	if m[3] != "" {
		temp, err = strconv.ParseUint(strings.TrimPrefix(m[3], "."), 10, 32)
		if err != nil {
			bvs := badVersionSegment{e: err}
			if CacheVersions {
				versionCacheLock.Lock()
				versionCache[v] = vcache{err: bvs}
				versionCacheLock.Unlock()
			}

			return Version{}, bvs
		}
		sv.patch = temp
	} else {
		sv.patch = 0
	}

	if CacheVersions {
		versionCacheLock.Lock()
		versionCache[v] = vcache{v: sv}
		versionCacheLock.Unlock()
	}

	return sv, nil
}

// String converts a Version object to a string.
// Note, if the original version contained a leading v this version will not.
// See the Original() method to retrieve the original value. Semantic Versions
// don't contain a leading v per the spec. Instead it's optional on
// impelementation.
func (v Version) String() string {
	return v.toString(false)
}

// ImpliedCaretString follows the same rules as String(), but in accordance with
// the Constraint interface will always print a leading "=", as all Versions,
// when acting as a Constraint, act as exact matches.
func (v Version) ImpliedCaretString() string {
	return v.toString(true)
}

func (v Version) toString(ic bool) string {
	var buf bytes.Buffer

	var base string
	if ic {
		base = "=%d.%d.%d"
	} else {
		base = "%d.%d.%d"
	}

	fmt.Fprintf(&buf, base, v.major, v.minor, v.patch)
	if v.pre != "" {
		fmt.Fprintf(&buf, "-%s", v.pre)
	}
	if v.metadata != "" {
		fmt.Fprintf(&buf, "+%s", v.metadata)
	}

	return buf.String()
}

// Original returns the original value passed in to be parsed.
func (v Version) Original() string {
	return v.original
}

// Major returns the major version.
func (v *Version) Major() uint64 {
	return v.major
}

// Minor returns the minor version.
func (v *Version) Minor() uint64 {
	return v.minor
}

// Patch returns the patch version.
func (v *Version) Patch() uint64 {
	return v.patch
}

// Prerelease returns the pre-release version.
func (v Version) Prerelease() string {
	return v.pre
}

// Metadata returns the metadata on the version.
func (v Version) Metadata() string {
	return v.metadata
}

// LessThan tests if one version is less than another one.
func (v Version) LessThan(o Version) bool {
	return v.Compare(o) < 0
}

// GreaterThan tests if one version is greater than another one.
func (v Version) GreaterThan(o Version) bool {
	return v.Compare(o) > 0
}

// Equal tests if two versions are equal to each other.
// Note, versions can be equal with different metadata since metadata
// is not considered part of the comparable version.
func (v Version) Equal(o Version) bool {
	return v.Compare(o) == 0
}

// Compare compares this version to another one. It returns -1, 0, or 1 if
// the version smaller, equal, or larger than the other version.
//
// Versions are compared by X.Y.Z. Build metadata is ignored. Prerelease is
// lower than the version without a prerelease.
func (v Version) Compare(o Version) int {
	// The special field supercedes all the other information. If it's not
	// equal, we can skip out early
	if v.special != o.special {
		switch v.special {
		case zeroVersion:
			return -1
		case notSpecial:
			if o.special == zeroVersion {
				return 1
			}
			return -1
		case infiniteVersion:
			return 1
		}
	} else if v.special != notSpecial {
		// If special fields are equal and not notSpecial, then they're
		// necessarily equal
		return 0
	}

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

// Matches checks that a verstions match. If they do not,
// an error is returned indcating the problem; if it does, the error is nil.
// This is part of the Constraint interface.
func (v Version) Matches(v2 Version) error {
	if v.Equal(v2) {
		return nil
	}

	return VersionMatchFailure{v: v, other: v2}
}

// MatchesAny checks if an instance of a version matches a constraint which can
// include anything matching the Constraint interface.
func (v Version) MatchesAny(c Constraint) bool {
	if v2, ok := c.(Version); ok {
		return v.Equal(v2)
	}

	// The other implementations all have specific handling for this; fall
	// back on theirs.
	return c.MatchesAny(v)
}

// Intersect computes the intersection between the receiving Constraint and
// passed Constraint, and returns a new Constraint representing the result.
// This is part of the Constraint interface.
func (v Version) Intersect(c Constraint) Constraint {
	if v2, ok := c.(Version); ok {
		if v.Equal(v2) {
			return v
		}
		return none{}
	}

	return c.Intersect(v)
}

// Union computes the union between the receiving Constraint and the passed
// Constraint, and returns a new Constraint representing the result.
// This is part of the Constraint interface.
func (v Version) Union(c Constraint) Constraint {
	if v2, ok := c.(Version); ok && v.Equal(v2) {
		return v
	}

	return Union(v, c)
}

func (Version) _private() {}
func (Version) _real()    {}

func compareSegment(v, o uint64) int {
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
		// Since the length of the parts can be different we need to create
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
	if o == "" {
		_, n := strconv.ParseUint(s, 10, 64)
		if n != nil {
			return -1
		}
		return 1
	}
	if s == "" {
		_, n := strconv.ParseUint(o, 10, 64)
		if n != nil {
			return 1
		}
		return -1
	}

	if s > o {
		return 1
	}
	return -1
}

func numPartsEq(v1, v2 Version) bool {
	if v1.special != v2.special {
		return false
	}
	if v1.special != notSpecial {
		// If special fields are equal and not notSpecial, then the versions are
		// necessarily equal, so their numeric parts are too.
		return true
	}

	if v1.major != v2.major {
		return false
	}
	if v1.minor != v2.minor {
		return false
	}
	if v1.patch != v2.patch {
		return false
	}
	return true
}
