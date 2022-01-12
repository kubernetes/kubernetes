/*
Copyright 2021 The Kubernetes Authors.

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

package api

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/component-base/version"
)

type Version struct {
	major  int
	minor  int
	latest bool
}

func (v Version) String() string {
	if v.latest {
		return "latest"
	}
	return fmt.Sprintf("v%d.%d", v.major, v.minor)
}

// Older returns true if this version v is older than the other.
func (v *Version) Older(other Version) bool {
	if v.latest { // Latest is always consider newer, even than future versions.
		return false
	}
	if other.latest {
		return true
	}
	if v.major != other.major {
		return v.major < other.major
	}
	return v.minor < other.minor
}

func (v *Version) Major() int {
	return v.major
}
func (v *Version) Minor() int {
	return v.minor
}
func (v *Version) Latest() bool {
	return v.latest
}

func MajorMinorVersion(major, minor int) Version {
	return Version{major: major, minor: minor}
}

// GetAPIVersion get the version of apiServer and return the version major and minor
func GetAPIVersion() Version {
	var err error
	v := Version{}
	apiVersion := version.Get()
	major, err := strconv.Atoi(apiVersion.Major)
	if err != nil {
		return v
	}
	// split the "normal" + and - for semver stuff to get the leading minor number
	minorString := strings.FieldsFunc(apiVersion.Minor, func(r rune) bool {
		return !unicode.IsDigit(r)
	})[0]
	minor, err := strconv.Atoi(minorString)
	if err != nil {
		return v
	}
	v = MajorMinorVersion(major, minor)
	return v
}

func LatestVersion() Version {
	return Version{latest: true}
}

// ParseLevel returns the level that should be evaluated.
// level must be "privileged", "baseline", or "restricted".
// if level does not match one of those strings, "restricted" and an error is returned.
func ParseLevel(level string) (Level, error) {
	switch Level(level) {
	case LevelPrivileged, LevelBaseline, LevelRestricted:
		return Level(level), nil
	default:
		return LevelRestricted, fmt.Errorf(`must be one of %s`, strings.Join(validLevels, ", "))
	}
}

// Valid checks whether the level l is a valid level.
func (l *Level) Valid() bool {
	switch *l {
	case LevelPrivileged, LevelBaseline, LevelRestricted:
		return true
	default:
		return false
	}
}

var versionRegexp = regexp.MustCompile(`^v1\.([0-9]|[1-9][0-9]*)$`)

// ParseVersion returns the policy version that should be evaluated.
// version must be "latest" or "v1.x".
// If version does not match one of those patterns, the latest version and an error is returned.
func ParseVersion(version string) (Version, error) {
	if version == "latest" {
		return Version{latest: true}, nil
	}
	match := versionRegexp.FindStringSubmatch(version)
	if len(match) != 2 {
		return Version{latest: true}, fmt.Errorf(`must be "latest" or "v1.x"`)
	}
	versionNumber, err := strconv.Atoi(match[1])
	if err != nil || versionNumber < 0 {
		return Version{latest: true}, fmt.Errorf(`must be "latest" or "v1.x"`)
	}
	return Version{major: 1, minor: versionNumber}, nil
}

type LevelVersion struct {
	Level
	Version
}

func (lv LevelVersion) String() string {
	return fmt.Sprintf("%s:%s", lv.Level, lv.Version)
}

type Policy struct {
	Enforce LevelVersion
	Audit   LevelVersion
	Warn    LevelVersion
}

// PolicyToEvaluate resolves the PodSecurity namespace labels to the policy for that namespace,
// falling back to the provided defaults when a label is unspecified. A valid policy is always
// returned, even when an error is returned. If labels cannot be parsed correctly, the values of
// "restricted" and "latest" are used for level and version respectively.
func PolicyToEvaluate(labels map[string]string, defaults Policy) (Policy, field.ErrorList) {
	var (
		err  error
		errs field.ErrorList

		p = defaults
	)
	if len(labels) == 0 {
		return p, nil
	}
	if level, ok := labels[EnforceLevelLabel]; ok {
		p.Enforce.Level, err = ParseLevel(level)
		errs = appendErr(errs, err, EnforceLevelLabel, level)
	}
	if version, ok := labels[EnforceVersionLabel]; ok {
		p.Enforce.Version, err = ParseVersion(version)
		errs = appendErr(errs, err, EnforceVersionLabel, version)
	}
	if level, ok := labels[AuditLevelLabel]; ok {
		p.Audit.Level, err = ParseLevel(level)
		errs = appendErr(errs, err, AuditLevelLabel, level)
		if err != nil {
			p.Audit.Level = LevelPrivileged // Fail open for audit.
		}
	}
	if version, ok := labels[AuditVersionLabel]; ok {
		p.Audit.Version, err = ParseVersion(version)
		errs = appendErr(errs, err, AuditVersionLabel, version)
	}
	if level, ok := labels[WarnLevelLabel]; ok {
		p.Warn.Level, err = ParseLevel(level)
		errs = appendErr(errs, err, WarnLevelLabel, level)
		if err != nil {
			p.Warn.Level = LevelPrivileged // Fail open for warn.
		}
	}
	if version, ok := labels[WarnVersionLabel]; ok {
		p.Warn.Version, err = ParseVersion(version)
		errs = appendErr(errs, err, WarnVersionLabel, version)
	}
	return p, errs
}

// CompareLevels returns an integer comparing two levels by strictness. The result will be 0 if
// a==b, -1 if a is less strict than b, and +1 if a is more strict than b.
func CompareLevels(a, b Level) int {
	if a == b {
		return 0
	}
	switch a {
	case LevelPrivileged:
		return -1
	case LevelRestricted:
		return 1
	default:
		if b == LevelPrivileged {
			return 1
		} else if b == LevelRestricted {
			return -1
		}
	}
	// This should only happen if both a & b are invalid levels.
	return 0
}

var labelsPath = field.NewPath("metadata", "labels")

// appendErr is a helper function to collect label-specific errors.
func appendErr(errs field.ErrorList, err error, label, value string) field.ErrorList {
	if err != nil {
		return append(errs, field.Invalid(labelsPath.Key(label), value, err.Error()))
	}
	return errs
}
