/*
Copyright 2020 The Kubernetes Authors.

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

package dependencies

import (
	"github.com/blang/semver"
	"github.com/pkg/errors"
)

// Version is the internal representation of a Version as a string and a scheme
type Version struct {
	Version string
	Scheme  VersionScheme
}

// VersionScheme informs us on how to compare two versions
type VersionScheme string

const (
	// Semver [Semantic versioning](https://semver.org/), default
	Semver VersionScheme = "semver"
	// Alpha Alphanumeric, will use standard string sorting
	Alpha VersionScheme = "alpha"
	// Random when releases do not support sorting (e.g. hashes)
	Random VersionScheme = "random"
)

// VersionSensitivity informs us on how to compare whether a version is more
// recent than another, for example to only notify on new major versions
// Only applicable to Semver versioning
type VersionSensitivity string

const (
	// Patch version, e.g. 1.1.1 -> 1.1.2, default
	Patch VersionSensitivity = "patch"
	// Minor version, e.g. 1.1.1 -> 1.2.0
	Minor VersionSensitivity = "minor"
	// Major version, e.g. 1.1.1 -> 2.0.0
	Major VersionSensitivity = "major"
)

// MoreRecentThan checks whether a given version is more recent than another one.
//
// If the VersionScheme is "random", then it will return true if a != b.
func (a Version) MoreRecentThan(b Version) (bool, error) {
	return a.MoreSensitivelyRecentThan(b, Patch)
}

// MoreSensitivelyRecentThan checks whether a given version is more recent than
// another one, accepting a VersionSensitivity argument
//
// If the VersionScheme is "random", then it will return true if a != b.
func (a Version) MoreSensitivelyRecentThan(b Version, sensitivity VersionSensitivity) (bool, error) {
	// Default to a Patch-level sensitivity
	if sensitivity == "" {
		sensitivity = Patch
	}

	if a.Scheme != b.Scheme {
		return false, errors.Errorf("trying to compare incompatible 'Version' schemes: %s and %s", a.Scheme, b.Scheme)
	}

	switch a.Scheme {
	case Semver:
		aSemver, err := semver.Parse(a.Version)
		if err != nil {
			return false, err
		}

		bSemver, err := semver.Parse(b.Version)
		if err != nil {
			return false, err
		}

		return semverCompare(aSemver, bSemver, sensitivity)
	case Alpha:
		// Alphanumeric comparison (basic string compare)
		return a.Version > b.Version, nil
	case Random:
		// When identifiers are random (e.g. hashes), the newer version will just be a different version
		return a.Version != b.Version, nil
	default:
		return false, errors.Errorf("unknown version scheme: %s", a.Scheme)
	}
}

// semverCompare compares two semver versions depending on a sensitivity level
func semverCompare(a, b semver.Version, sensitivity VersionSensitivity) (bool, error) {
	switch sensitivity {
	case Major:
		return a.Major > b.Major, nil
	case Minor:
		return a.Major > b.Major || (a.Major == b.Major && a.Minor > b.Minor), nil
	case Patch:
		return a.GT(b), nil
	default:
		return false, errors.Errorf("unknown version sensitivity: %s", sensitivity)
	}
}
