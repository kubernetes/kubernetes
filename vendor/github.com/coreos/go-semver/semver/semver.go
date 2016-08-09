package semver

import (
	"bytes"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

type Version struct {
	Major      int64
	Minor      int64
	Patch      int64
	PreRelease PreRelease
	Metadata   string
}

type PreRelease string

func splitOff(input *string, delim string) (val string) {
	parts := strings.SplitN(*input, delim, 2)

	if len(parts) == 2 {
		*input = parts[0]
		val = parts[1]
	}

	return val
}

func NewVersion(version string) (*Version, error) {
	v := Version{}

	dotParts := strings.SplitN(version, ".", 3)

	if len(dotParts) != 3 {
		return nil, errors.New(fmt.Sprintf("%s is not in dotted-tri format", version))
	}

	v.Metadata = splitOff(&dotParts[2], "+")
	v.PreRelease = PreRelease(splitOff(&dotParts[2], "-"))

	parsed := make([]int64, 3, 3)

	for i, v := range dotParts[:3] {
		val, err := strconv.ParseInt(v, 10, 64)
		parsed[i] = val
		if err != nil {
			return nil, err
		}
	}

	v.Major = parsed[0]
	v.Minor = parsed[1]
	v.Patch = parsed[2]

	return &v, nil
}

func Must(v *Version, err error) *Version {
	if err != nil {
		panic(err)
	}
	return v
}

func (v *Version) String() string {
	var buffer bytes.Buffer

	base := fmt.Sprintf("%d.%d.%d", v.Major, v.Minor, v.Patch)
	buffer.WriteString(base)

	if v.PreRelease != "" {
		buffer.WriteString(fmt.Sprintf("-%s", v.PreRelease))
	}

	if v.Metadata != "" {
		buffer.WriteString(fmt.Sprintf("+%s", v.Metadata))
	}

	return buffer.String()
}

func (v *Version) LessThan(versionB Version) bool {
	versionA := *v
	cmp := recursiveCompare(versionA.Slice(), versionB.Slice())

	if cmp == 0 {
		cmp = preReleaseCompare(versionA, versionB)
	}

	if cmp == -1 {
		return true
	}

	return false
}

/* Slice converts the comparable parts of the semver into a slice of strings */
func (v *Version) Slice() []int64 {
	return []int64{v.Major, v.Minor, v.Patch}
}

func (p *PreRelease) Slice() []string {
	preRelease := string(*p)
	return strings.Split(preRelease, ".")
}

func preReleaseCompare(versionA Version, versionB Version) int {
	a := versionA.PreRelease
	b := versionB.PreRelease

	/* Handle the case where if two versions are otherwise equal it is the
	 * one without a PreRelease that is greater */
	if len(a) == 0 && (len(b) > 0) {
		return 1
	} else if len(b) == 0 && (len(a) > 0) {
		return -1
	}

	// If there is a prelease, check and compare each part.
	return recursivePreReleaseCompare(a.Slice(), b.Slice())
}

func recursiveCompare(versionA []int64, versionB []int64) int {
	if len(versionA) == 0 {
		return 0
	}

	a := versionA[0]
	b := versionB[0]

	if a > b {
		return 1
	} else if a < b {
		return -1
	}

	return recursiveCompare(versionA[1:], versionB[1:])
}

func recursivePreReleaseCompare(versionA []string, versionB []string) int {
	// Handle slice length disparity.
	if len(versionA) == 0 {
		// Nothing to compare too, so we return 0
		return 0
	} else if len(versionB) == 0 {
		// We're longer than versionB so return 1.
		return 1
	}

	a := versionA[0]
	b := versionB[0]

	aInt := false; bInt := false

	aI, err := strconv.Atoi(versionA[0])
	if err == nil {
		aInt = true
	}

	bI, err := strconv.Atoi(versionB[0])
	if err == nil {
		bInt = true
	}

	// Handle Integer Comparison
	if aInt && bInt {
		if aI > bI {
			return 1
		} else if aI < bI {
			return -1
		}
	}

	// Handle String Comparison
	if a > b {
		return 1
	} else if a < b {
		return -1
	}

	return recursivePreReleaseCompare(versionA[1:], versionB[1:])
}

// BumpMajor increments the Major field by 1 and resets all other fields to their default values
func (v *Version) BumpMajor() {
	v.Major += 1
	v.Minor = 0
	v.Patch = 0
	v.PreRelease = PreRelease("")
	v.Metadata = ""
}

// BumpMinor increments the Minor field by 1 and resets all other fields to their default values
func (v *Version) BumpMinor() {
	v.Minor += 1
	v.Patch = 0
	v.PreRelease = PreRelease("")
	v.Metadata = ""
}

// BumpPatch increments the Patch field by 1 and resets all other fields to their default values
func (v *Version) BumpPatch() {
	v.Patch += 1
	v.PreRelease = PreRelease("")
	v.Metadata = ""
}
