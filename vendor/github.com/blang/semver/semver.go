package semver

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

const (
	numbers  string = "0123456789"
	alphas          = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-"
	alphanum        = alphas + numbers
)

// SpecVersion is the latest fully supported spec version of semver
var SpecVersion = Version{
	Major: 2,
	Minor: 0,
	Patch: 0,
}

// Version represents a semver compatible version
type Version struct {
	Major uint64
	Minor uint64
	Patch uint64
	Pre   []PRVersion
	Build []string //No Precendence
}

// Version to string
func (v Version) String() string {
	b := make([]byte, 0, 5)
	b = strconv.AppendUint(b, v.Major, 10)
	b = append(b, '.')
	b = strconv.AppendUint(b, v.Minor, 10)
	b = append(b, '.')
	b = strconv.AppendUint(b, v.Patch, 10)

	if len(v.Pre) > 0 {
		b = append(b, '-')
		b = append(b, v.Pre[0].String()...)

		for _, pre := range v.Pre[1:] {
			b = append(b, '.')
			b = append(b, pre.String()...)
		}
	}

	if len(v.Build) > 0 {
		b = append(b, '+')
		b = append(b, v.Build[0]...)

		for _, build := range v.Build[1:] {
			b = append(b, '.')
			b = append(b, build...)
		}
	}

	return string(b)
}

// Equals checks if v is equal to o.
func (v Version) Equals(o Version) bool {
	return (v.Compare(o) == 0)
}

// EQ checks if v is equal to o.
func (v Version) EQ(o Version) bool {
	return (v.Compare(o) == 0)
}

// NE checks if v is not equal to o.
func (v Version) NE(o Version) bool {
	return (v.Compare(o) != 0)
}

// GT checks if v is greater than o.
func (v Version) GT(o Version) bool {
	return (v.Compare(o) == 1)
}

// GTE checks if v is greater than or equal to o.
func (v Version) GTE(o Version) bool {
	return (v.Compare(o) >= 0)
}

// GE checks if v is greater than or equal to o.
func (v Version) GE(o Version) bool {
	return (v.Compare(o) >= 0)
}

// LT checks if v is less than o.
func (v Version) LT(o Version) bool {
	return (v.Compare(o) == -1)
}

// LTE checks if v is less than or equal to o.
func (v Version) LTE(o Version) bool {
	return (v.Compare(o) <= 0)
}

// LE checks if v is less than or equal to o.
func (v Version) LE(o Version) bool {
	return (v.Compare(o) <= 0)
}

// Compare compares Versions v to o:
// -1 == v is less than o
// 0 == v is equal to o
// 1 == v is greater than o
func (v Version) Compare(o Version) int {
	if v.Major != o.Major {
		if v.Major > o.Major {
			return 1
		}
		return -1
	}
	if v.Minor != o.Minor {
		if v.Minor > o.Minor {
			return 1
		}
		return -1
	}
	if v.Patch != o.Patch {
		if v.Patch > o.Patch {
			return 1
		}
		return -1
	}

	// Quick comparison if a version has no prerelease versions
	if len(v.Pre) == 0 && len(o.Pre) == 0 {
		return 0
	} else if len(v.Pre) == 0 && len(o.Pre) > 0 {
		return 1
	} else if len(v.Pre) > 0 && len(o.Pre) == 0 {
		return -1
	}

	i := 0
	for ; i < len(v.Pre) && i < len(o.Pre); i++ {
		if comp := v.Pre[i].Compare(o.Pre[i]); comp == 0 {
			continue
		} else if comp == 1 {
			return 1
		} else {
			return -1
		}
	}

	// If all pr versions are the equal but one has further prversion, this one greater
	if i == len(v.Pre) && i == len(o.Pre) {
		return 0
	} else if i == len(v.Pre) && i < len(o.Pre) {
		return -1
	} else {
		return 1
	}

}

// Validate validates v and returns error in case
func (v Version) Validate() error {
	// Major, Minor, Patch already validated using uint64

	for _, pre := range v.Pre {
		if !pre.IsNum { //Numeric prerelease versions already uint64
			if len(pre.VersionStr) == 0 {
				return fmt.Errorf("Prerelease can not be empty %q", pre.VersionStr)
			}
			if !containsOnly(pre.VersionStr, alphanum) {
				return fmt.Errorf("Invalid character(s) found in prerelease %q", pre.VersionStr)
			}
		}
	}

	for _, build := range v.Build {
		if len(build) == 0 {
			return fmt.Errorf("Build meta data can not be empty %q", build)
		}
		if !containsOnly(build, alphanum) {
			return fmt.Errorf("Invalid character(s) found in build meta data %q", build)
		}
	}

	return nil
}

// New is an alias for Parse and returns a pointer, parses version string and returns a validated Version or error
func New(s string) (vp *Version, err error) {
	v, err := Parse(s)
	vp = &v
	return
}

// Make is an alias for Parse, parses version string and returns a validated Version or error
func Make(s string) (Version, error) {
	return Parse(s)
}

// ParseTolerant allows for certain version specifications that do not strictly adhere to semver
// specs to be parsed by this library. It does so by normalizing versions before passing them to
// Parse(). It currently trims spaces, removes a "v" prefix, and adds a 0 patch number to versions
// with only major and minor components specified
func ParseTolerant(s string) (Version, error) {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "v")

	// Split into major.minor.(patch+pr+meta)
	parts := strings.SplitN(s, ".", 3)
	if len(parts) < 3 {
		if strings.ContainsAny(parts[len(parts)-1], "+-") {
			return Version{}, errors.New("Short version cannot contain PreRelease/Build meta data")
		}
		for len(parts) < 3 {
			parts = append(parts, "0")
		}
		s = strings.Join(parts, ".")
	}

	return Parse(s)
}

// Parse parses version string and returns a validated Version or error
func Parse(s string) (Version, error) {
	if len(s) == 0 {
		return Version{}, errors.New("Version string empty")
	}

	// Split into major.minor.(patch+pr+meta)
	parts := strings.SplitN(s, ".", 3)
	if len(parts) != 3 {
		return Version{}, errors.New("No Major.Minor.Patch elements found")
	}

	// Major
	if !containsOnly(parts[0], numbers) {
		return Version{}, fmt.Errorf("Invalid character(s) found in major number %q", parts[0])
	}
	if hasLeadingZeroes(parts[0]) {
		return Version{}, fmt.Errorf("Major number must not contain leading zeroes %q", parts[0])
	}
	major, err := strconv.ParseUint(parts[0], 10, 64)
	if err != nil {
		return Version{}, err
	}

	// Minor
	if !containsOnly(parts[1], numbers) {
		return Version{}, fmt.Errorf("Invalid character(s) found in minor number %q", parts[1])
	}
	if hasLeadingZeroes(parts[1]) {
		return Version{}, fmt.Errorf("Minor number must not contain leading zeroes %q", parts[1])
	}
	minor, err := strconv.ParseUint(parts[1], 10, 64)
	if err != nil {
		return Version{}, err
	}

	v := Version{}
	v.Major = major
	v.Minor = minor

	var build, prerelease []string
	patchStr := parts[2]

	if buildIndex := strings.IndexRune(patchStr, '+'); buildIndex != -1 {
		build = strings.Split(patchStr[buildIndex+1:], ".")
		patchStr = patchStr[:buildIndex]
	}

	if preIndex := strings.IndexRune(patchStr, '-'); preIndex != -1 {
		prerelease = strings.Split(patchStr[preIndex+1:], ".")
		patchStr = patchStr[:preIndex]
	}

	if !containsOnly(patchStr, numbers) {
		return Version{}, fmt.Errorf("Invalid character(s) found in patch number %q", patchStr)
	}
	if hasLeadingZeroes(patchStr) {
		return Version{}, fmt.Errorf("Patch number must not contain leading zeroes %q", patchStr)
	}
	patch, err := strconv.ParseUint(patchStr, 10, 64)
	if err != nil {
		return Version{}, err
	}

	v.Patch = patch

	// Prerelease
	for _, prstr := range prerelease {
		parsedPR, err := NewPRVersion(prstr)
		if err != nil {
			return Version{}, err
		}
		v.Pre = append(v.Pre, parsedPR)
	}

	// Build meta data
	for _, str := range build {
		if len(str) == 0 {
			return Version{}, errors.New("Build meta data is empty")
		}
		if !containsOnly(str, alphanum) {
			return Version{}, fmt.Errorf("Invalid character(s) found in build meta data %q", str)
		}
		v.Build = append(v.Build, str)
	}

	return v, nil
}

// MustParse is like Parse but panics if the version cannot be parsed.
func MustParse(s string) Version {
	v, err := Parse(s)
	if err != nil {
		panic(`semver: Parse(` + s + `): ` + err.Error())
	}
	return v
}

// PRVersion represents a PreRelease Version
type PRVersion struct {
	VersionStr string
	VersionNum uint64
	IsNum      bool
}

// NewPRVersion creates a new valid prerelease version
func NewPRVersion(s string) (PRVersion, error) {
	if len(s) == 0 {
		return PRVersion{}, errors.New("Prerelease is empty")
	}
	v := PRVersion{}
	if containsOnly(s, numbers) {
		if hasLeadingZeroes(s) {
			return PRVersion{}, fmt.Errorf("Numeric PreRelease version must not contain leading zeroes %q", s)
		}
		num, err := strconv.ParseUint(s, 10, 64)

		// Might never be hit, but just in case
		if err != nil {
			return PRVersion{}, err
		}
		v.VersionNum = num
		v.IsNum = true
	} else if containsOnly(s, alphanum) {
		v.VersionStr = s
		v.IsNum = false
	} else {
		return PRVersion{}, fmt.Errorf("Invalid character(s) found in prerelease %q", s)
	}
	return v, nil
}

// IsNumeric checks if prerelease-version is numeric
func (v PRVersion) IsNumeric() bool {
	return v.IsNum
}

// Compare compares two PreRelease Versions v and o:
// -1 == v is less than o
// 0 == v is equal to o
// 1 == v is greater than o
func (v PRVersion) Compare(o PRVersion) int {
	if v.IsNum && !o.IsNum {
		return -1
	} else if !v.IsNum && o.IsNum {
		return 1
	} else if v.IsNum && o.IsNum {
		if v.VersionNum == o.VersionNum {
			return 0
		} else if v.VersionNum > o.VersionNum {
			return 1
		} else {
			return -1
		}
	} else { // both are Alphas
		if v.VersionStr == o.VersionStr {
			return 0
		} else if v.VersionStr > o.VersionStr {
			return 1
		} else {
			return -1
		}
	}
}

// PreRelease version to string
func (v PRVersion) String() string {
	if v.IsNum {
		return strconv.FormatUint(v.VersionNum, 10)
	}
	return v.VersionStr
}

func containsOnly(s string, set string) bool {
	return strings.IndexFunc(s, func(r rune) bool {
		return !strings.ContainsRune(set, r)
	}) == -1
}

func hasLeadingZeroes(s string) bool {
	return len(s) > 1 && s[0] == '0'
}

// NewBuildVersion creates a new valid build version
func NewBuildVersion(s string) (string, error) {
	if len(s) == 0 {
		return "", errors.New("Buildversion is empty")
	}
	if !containsOnly(s, alphanum) {
		return "", fmt.Errorf("Invalid character(s) found in build meta data %q", s)
	}
	return s, nil
}
