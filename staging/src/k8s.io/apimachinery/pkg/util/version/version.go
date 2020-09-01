/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package version

import (
	"bytes"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// Version is an opaque representation of a version number
type Version struct {
	components    []uint
	semver        bool
	preRelease    string
	buildMetadata string
}

var (
	// versionMatchRE splits a version string into numeric and "extra" parts
	versionMatchRE = regexp.MustCompile(`^\s*v?([0-9]+(?:\.[0-9]+)*)(.*)*$`)
	// extraMatchRE splits the "extra" part of versionMatchRE into semver pre-release and build metadata; it does not validate the "no leading zeroes" constraint for pre-release
	extraMatchRE = regexp.MustCompile(`^(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?\s*$`)
)

func parse(str string, semver bool) (*Version, error) {
	parts := versionMatchRE.FindStringSubmatch(str)
	if parts == nil {
		return nil, fmt.Errorf("could not parse %q as version", str)
	}
	numbers, extra := parts[1], parts[2]

	components := strings.Split(numbers, ".")
	if (semver && len(components) != 3) || (!semver && len(components) < 2) {
		return nil, fmt.Errorf("illegal version string %q", str)
	}

	v := &Version{
		components: make([]uint, len(components)),
		semver:     semver,
	}
	for i, comp := range components {
		if (i == 0 || semver) && strings.HasPrefix(comp, "0") && comp != "0" {
			return nil, fmt.Errorf("illegal zero-prefixed version component %q in %q", comp, str)
		}
		num, err := strconv.ParseUint(comp, 10, 0)
		if err != nil {
			return nil, fmt.Errorf("illegal non-numeric version component %q in %q: %v", comp, str, err)
		}
		v.components[i] = uint(num)
	}

	if semver && extra != "" {
		extraParts := extraMatchRE.FindStringSubmatch(extra)
		if extraParts == nil {
			return nil, fmt.Errorf("could not parse pre-release/metadata (%s) in version %q", extra, str)
		}
		v.preRelease, v.buildMetadata = extraParts[1], extraParts[2]

		for _, comp := range strings.Split(v.preRelease, ".") {
			if _, err := strconv.ParseUint(comp, 10, 0); err == nil {
				if strings.HasPrefix(comp, "0") && comp != "0" {
					return nil, fmt.Errorf("illegal zero-prefixed version component %q in %q", comp, str)
				}
			}
		}
	}

	return v, nil
}

// ParseGeneric parses a "generic" version string. The version string must consist of two
// or more dot-separated numeric fields (the first of which can't have leading zeroes),
// followed by arbitrary uninterpreted data (which need not be separated from the final
// numeric field by punctuation). For convenience, leading and trailing whitespace is
// ignored, and the version can be preceded by the letter "v". See also ParseSemantic.
func ParseGeneric(str string) (*Version, error) {
	return parse(str, false)
}

// MustParseGeneric is like ParseGeneric except that it panics on error
func MustParseGeneric(str string) *Version {
	v, err := ParseGeneric(str)
	if err != nil {
		panic(err)
	}
	return v
}

// ParseSemantic parses a version string that exactly obeys the syntax and semantics of
// the "Semantic Versioning" specification (http://semver.org/) (although it ignores
// leading and trailing whitespace, and allows the version to be preceded by "v"). For
// version strings that are not guaranteed to obey the Semantic Versioning syntax, use
// ParseGeneric.
func ParseSemantic(str string) (*Version, error) {
	return parse(str, true)
}

// MustParseSemantic is like ParseSemantic except that it panics on error
func MustParseSemantic(str string) *Version {
	v, err := ParseSemantic(str)
	if err != nil {
		panic(err)
	}
	return v
}

// Major returns the major release number
func (v *Version) Major() uint {
	return v.components[0]
}

// Minor returns the minor release number
func (v *Version) Minor() uint {
	return v.components[1]
}

// Patch returns the patch release number if v is a Semantic Version, or 0
func (v *Version) Patch() uint {
	if len(v.components) < 3 {
		return 0
	}
	return v.components[2]
}

// BuildMetadata returns the build metadata, if v is a Semantic Version, or ""
func (v *Version) BuildMetadata() string {
	return v.buildMetadata
}

// PreRelease returns the prerelease metadata, if v is a Semantic Version, or ""
func (v *Version) PreRelease() string {
	return v.preRelease
}

// Components returns the version number components
func (v *Version) Components() []uint {
	return v.components
}

// WithMajor returns copy of the version object with requested major number
func (v *Version) WithMajor(major uint) *Version {
	result := *v
	result.components = []uint{major, v.Minor(), v.Patch()}
	return &result
}

// WithMinor returns copy of the version object with requested minor number
func (v *Version) WithMinor(minor uint) *Version {
	result := *v
	result.components = []uint{v.Major(), minor, v.Patch()}
	return &result
}

// WithPatch returns copy of the version object with requested patch number
func (v *Version) WithPatch(patch uint) *Version {
	result := *v
	result.components = []uint{v.Major(), v.Minor(), patch}
	return &result
}

// WithPreRelease returns copy of the version object with requested prerelease
func (v *Version) WithPreRelease(preRelease string) *Version {
	result := *v
	result.components = []uint{v.Major(), v.Minor(), v.Patch()}
	result.preRelease = preRelease
	return &result
}

// WithBuildMetadata returns copy of the version object with requested buildMetadata
func (v *Version) WithBuildMetadata(buildMetadata string) *Version {
	result := *v
	result.components = []uint{v.Major(), v.Minor(), v.Patch()}
	result.buildMetadata = buildMetadata
	return &result
}

// String converts a Version back to a string; note that for versions parsed with
// ParseGeneric, this will not include the trailing uninterpreted portion of the version
// number.
func (v *Version) String() string {
	var buffer bytes.Buffer

	for i, comp := range v.components {
		if i > 0 {
			buffer.WriteString(".")
		}
		buffer.WriteString(fmt.Sprintf("%d", comp))
	}
	if v.preRelease != "" {
		buffer.WriteString("-")
		buffer.WriteString(v.preRelease)
	}
	if v.buildMetadata != "" {
		buffer.WriteString("+")
		buffer.WriteString(v.buildMetadata)
	}

	return buffer.String()
}

// compareInternal returns -1 if v is less than other, 1 if it is greater than other, or 0
// if they are equal
func (v *Version) compareInternal(other *Version) int {

	vLen := len(v.components)
	oLen := len(other.components)
	for i := 0; i < vLen && i < oLen; i++ {
		switch {
		case other.components[i] < v.components[i]:
			return 1
		case other.components[i] > v.components[i]:
			return -1
		}
	}

	// If components are common but one has more items and they are not zeros, it is bigger
	switch {
	case oLen < vLen && !onlyZeros(v.components[oLen:]):
		return 1
	case oLen > vLen && !onlyZeros(other.components[vLen:]):
		return -1
	}

	if !v.semver || !other.semver {
		return 0
	}

	switch {
	case v.preRelease == "" && other.preRelease != "":
		return 1
	case v.preRelease != "" && other.preRelease == "":
		return -1
	case v.preRelease == other.preRelease: // includes case where both are ""
		return 0
	}

	vPR := strings.Split(v.preRelease, ".")
	oPR := strings.Split(other.preRelease, ".")
	for i := 0; i < len(vPR) && i < len(oPR); i++ {
		vNum, err := strconv.ParseUint(vPR[i], 10, 0)
		if err == nil {
			oNum, err := strconv.ParseUint(oPR[i], 10, 0)
			if err == nil {
				switch {
				case oNum < vNum:
					return 1
				case oNum > vNum:
					return -1
				default:
					continue
				}
			}
		}
		if oPR[i] < vPR[i] {
			return 1
		} else if oPR[i] > vPR[i] {
			return -1
		}
	}

	switch {
	case len(oPR) < len(vPR):
		return 1
	case len(oPR) > len(vPR):
		return -1
	}

	return 0
}

// returns false if array contain any non-zero element
func onlyZeros(array []uint) bool {
	for _, num := range array {
		if num != 0 {
			return false
		}
	}
	return true
}

// AtLeast tests if a version is at least equal to a given minimum version. If both
// Versions are Semantic Versions, this will use the Semantic Version comparison
// algorithm. Otherwise, it will compare only the numeric components, with non-present
// components being considered "0" (ie, "1.4" is equal to "1.4.0").
func (v *Version) AtLeast(min *Version) bool {
	return v.compareInternal(min) != -1
}

// LessThan tests if a version is less than a given version. (It is exactly the opposite
// of AtLeast, for situations where asking "is v too old?" makes more sense than asking
// "is v new enough?".)
func (v *Version) LessThan(other *Version) bool {
	return v.compareInternal(other) == -1
}

// Compare compares v against a version string (which will be parsed as either Semantic
// or non-Semantic depending on v). On success it returns -1 if v is less than other, 1 if
// it is greater than other, or 0 if they are equal.
func (v *Version) Compare(other string) (int, error) {
	ov, err := parse(other, v.semver)
	if err != nil {
		return 0, err
	}
	return v.compareInternal(ov), nil
}
