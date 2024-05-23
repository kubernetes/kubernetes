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
	"fmt"
	"reflect"
	"testing"
)

type testItem struct {
	version    string
	unparsed   string
	equalsPrev bool
}

func testOne(v *Version, item, prev testItem) error {
	str := v.String()
	if item.unparsed == "" {
		if str != item.version {
			return fmt.Errorf("bad round-trip: %q -> %q", item.version, str)
		}
	} else {
		if str != item.unparsed {
			return fmt.Errorf("bad unparse: %q -> %q, expected %q", item.version, str, item.unparsed)
		}
	}

	if prev.version != "" {
		cmp, err := v.Compare(prev.version)
		if err != nil {
			return fmt.Errorf("unexpected parse error: %v", err)
		}
		rv, err := parse(prev.version, v.semver)
		if err != nil {
			return fmt.Errorf("unexpected parse error: %v", err)
		}
		rcmp, err := rv.Compare(item.version)
		if err != nil {
			return fmt.Errorf("unexpected parse error: %v", err)
		}

		switch {
		case cmp == -1:
			return fmt.Errorf("unexpected ordering %q < %q", item.version, prev.version)
		case cmp == 0 && !item.equalsPrev:
			return fmt.Errorf("unexpected comparison %q == %q", item.version, prev.version)
		case cmp == 1 && item.equalsPrev:
			return fmt.Errorf("unexpected comparison %q != %q", item.version, prev.version)
		case cmp != -rcmp:
			return fmt.Errorf("unexpected reverse comparison %q <=> %q %v %v %v %v", item.version, prev.version, cmp, rcmp, v.Components(), rv.Components())
		}
	}

	return nil
}

func TestSemanticVersions(t *testing.T) {
	tests := []testItem{
		// This is every version string that appears in the 2.0 semver spec,
		// sorted in strictly increasing order except as noted.
		{version: "0.1.0"},
		{version: "1.0.0-0.3.7"},
		{version: "1.0.0-alpha"},
		{version: "1.0.0-alpha+001", equalsPrev: true},
		{version: "1.0.0-alpha.1"},
		{version: "1.0.0-alpha.beta"},
		{version: "1.0.0-beta"},
		{version: "1.0.0-beta+exp.sha.5114f85", equalsPrev: true},
		{version: "1.0.0-beta.2"},
		{version: "1.0.0-beta.11"},
		{version: "1.0.0-rc.1"},
		{version: "1.0.0-x.7.z.92"},
		{version: "1.0.0"},
		{version: "1.0.0+20130313144700", equalsPrev: true},
		{version: "1.8.0-alpha.3"},
		{version: "1.8.0-alpha.3.673+73326ef01d2d7c"},
		{version: "1.9.0"},
		{version: "1.10.0"},
		{version: "1.11.0"},
		{version: "2.0.0"},
		{version: "2.1.0"},
		{version: "2.1.1"},
		{version: "42.0.0"},

		// We also allow whitespace and "v" prefix
		{version: "   42.0.0", unparsed: "42.0.0", equalsPrev: true},
		{version: "\t42.0.0  ", unparsed: "42.0.0", equalsPrev: true},
		{version: "43.0.0-1", unparsed: "43.0.0-1"},
		{version: "43.0.0-1  ", unparsed: "43.0.0-1", equalsPrev: true},
		{version: "v43.0.0-1", unparsed: "43.0.0-1", equalsPrev: true},
		{version: "  v43.0.0", unparsed: "43.0.0"},
		{version: " 43.0.0 ", unparsed: "43.0.0", equalsPrev: true},
	}

	var prev testItem
	for _, item := range tests {
		v, err := ParseSemantic(item.version)
		if err != nil {
			t.Errorf("unexpected parse error: %v", err)
			continue
		}
		err = testOne(v, item, prev)
		if err != nil {
			t.Errorf("%v", err)
		}
		prev = item
	}
}

func TestBadSemanticVersions(t *testing.T) {
	tests := []string{
		// "MUST take the form X.Y.Z"
		"1",
		"1.2",
		"1.2.3.4",
		".2.3",
		"1..3",
		"1.2.",
		"",
		"..",
		// "where X, Y, and Z are non-negative integers"
		"-1.2.3",
		"1.-2.3",
		"1.2.-3",
		"1a.2.3",
		"1.2a.3",
		"1.2.3a",
		"a1.2.3",
		"a.b.c",
		"1 .2.3",
		"1. 2.3",
		// "and MUST NOT contain leading zeroes."
		"01.2.3",
		"1.02.3",
		"1.2.03",
		// "[pre-release] identifiers MUST comprise only ASCII alphanumerics and hyphen"
		"1.2.3-/",
		// "[pre-release] identifiers MUST NOT be empty"
		"1.2.3-",
		"1.2.3-.",
		"1.2.3-foo.",
		"1.2.3-.foo",
		// "Numeric [pre-release] identifiers MUST NOT include leading zeroes"
		"1.2.3-01",
		// "[build metadata] identifiers MUST comprise only ASCII alphanumerics and hyphen"
		"1.2.3+/",
		// "[build metadata] identifiers MUST NOT be empty"
		"1.2.3+",
		"1.2.3+.",
		"1.2.3+foo.",
		"1.2.3+.foo",

		// whitespace/"v"-prefix checks
		"v 1.2.3",
		"vv1.2.3",
	}

	for i := range tests {
		_, err := ParseSemantic(tests[i])
		if err == nil {
			t.Errorf("unexpected success parsing invalid semver %q", tests[i])
		}
	}
}

func TestGenericVersions(t *testing.T) {
	tests := []testItem{
		// This is all of the strings from TestSemanticVersions, plus some strings
		// from TestBadSemanticVersions that should parse as generic versions,
		// plus some additional strings.
		{version: "0.1.0", unparsed: "0.1.0"},
		{version: "1.0.0-0.3.7", unparsed: "1.0.0"},
		{version: "1.0.0-alpha", unparsed: "1.0.0", equalsPrev: true},
		{version: "1.0.0-alpha+001", unparsed: "1.0.0", equalsPrev: true},
		{version: "1.0.0-alpha.1", unparsed: "1.0.0", equalsPrev: true},
		{version: "1.0.0-alpha.beta", unparsed: "1.0.0", equalsPrev: true},
		{version: "1.0.0.beta", unparsed: "1.0.0", equalsPrev: true},
		{version: "1.0.0-beta+exp.sha.5114f85", unparsed: "1.0.0", equalsPrev: true},
		{version: "1.0.0.beta.2", unparsed: "1.0.0", equalsPrev: true},
		{version: "1.0.0.beta.11", unparsed: "1.0.0", equalsPrev: true},
		{version: "1.0.0.rc.1", unparsed: "1.0.0", equalsPrev: true},
		{version: "1.0.0-x.7.z.92", unparsed: "1.0.0", equalsPrev: true},
		{version: "1.0.0", unparsed: "1.0.0", equalsPrev: true},
		{version: "1.0.0+20130313144700", unparsed: "1.0.0", equalsPrev: true},
		{version: "1.2", unparsed: "1.2"},
		{version: "1.2a.3", unparsed: "1.2", equalsPrev: true},
		{version: "1.2.3", unparsed: "1.2.3"},
		{version: "1.2.3.0", unparsed: "1.2.3.0", equalsPrev: true},
		{version: "1.2.3a", unparsed: "1.2.3", equalsPrev: true},
		{version: "1.2.3-foo.", unparsed: "1.2.3", equalsPrev: true},
		{version: "1.2.3-.foo", unparsed: "1.2.3", equalsPrev: true},
		{version: "1.2.3-01", unparsed: "1.2.3", equalsPrev: true},
		{version: "1.2.3+", unparsed: "1.2.3", equalsPrev: true},
		{version: "1.2.3+foo.", unparsed: "1.2.3", equalsPrev: true},
		{version: "1.2.3+.foo", unparsed: "1.2.3", equalsPrev: true},
		{version: "1.02.3", unparsed: "1.2.3", equalsPrev: true},
		{version: "1.2.03", unparsed: "1.2.3", equalsPrev: true},
		{version: "1.2.003", unparsed: "1.2.3", equalsPrev: true},
		{version: "1.2.3.4", unparsed: "1.2.3.4"},
		{version: "1.2.3.4b3", unparsed: "1.2.3.4", equalsPrev: true},
		{version: "1.2.3.4.5", unparsed: "1.2.3.4.5"},
		{version: "1.9.0", unparsed: "1.9.0"},
		{version: "1.9.0.0.0.0.0.0", unparsed: "1.9.0.0.0.0.0.0", equalsPrev: true},
		{version: "1.10.0", unparsed: "1.10.0"},
		{version: "1.11.0", unparsed: "1.11.0"},
		{version: "1.11.0.0.5", unparsed: "1.11.0.0.5"},
		{version: "2.0.0", unparsed: "2.0.0"},
		{version: "2.1.0", unparsed: "2.1.0"},
		{version: "2.1.1", unparsed: "2.1.1"},
		{version: "42.0.0", unparsed: "42.0.0"},
		{version: "   42.0.0", unparsed: "42.0.0", equalsPrev: true},
		{version: "\t42.0.0  ", unparsed: "42.0.0", equalsPrev: true},
		{version: "42.0.0-1", unparsed: "42.0.0", equalsPrev: true},
		{version: "42.0.0-1  ", unparsed: "42.0.0", equalsPrev: true},
		{version: "v42.0.0-1", unparsed: "42.0.0", equalsPrev: true},
		{version: "  v43.0.0", unparsed: "43.0.0"},
		{version: " 43.0.0 ", unparsed: "43.0.0", equalsPrev: true},
	}

	var prev testItem
	for _, item := range tests {
		v, err := ParseGeneric(item.version)
		if err != nil {
			t.Errorf("unexpected parse error: %v", err)
			continue
		}
		err = testOne(v, item, prev)
		if err != nil {
			t.Errorf("%v", err)
		}
		prev = item
	}
}

func TestBadGenericVersions(t *testing.T) {
	tests := []string{
		"1",
		"01.2.3",
		"-1.2.3",
		"1.-2.3",
		".2.3",
		"1..3",
		"1a.2.3",
		"a1.2.3",
		"1 .2.3",
		"1. 2.3",
		"1.bob",
		"bob",
		"v 1.2.3",
		"vv1.2.3",
		"",
		".",
	}

	for i := range tests {
		_, err := ParseGeneric(tests[i])
		if err == nil {
			t.Errorf("unexpected success parsing invalid version %q", tests[i])
		}
	}
}

func TestComponents(t *testing.T) {

	var tests = []struct {
		version               string
		semver                bool
		expectedComponents    []uint
		expectedMajor         uint
		expectedMinor         uint
		expectedPatch         uint
		expectedPreRelease    string
		expectedBuildMetadata string
	}{
		{
			version:            "1.0.2",
			semver:             true,
			expectedComponents: []uint{1, 0, 2},
			expectedMajor:      1,
			expectedMinor:      0,
			expectedPatch:      2,
		},
		{
			version:               "1.0.2-alpha+001",
			semver:                true,
			expectedComponents:    []uint{1, 0, 2},
			expectedMajor:         1,
			expectedMinor:         0,
			expectedPatch:         2,
			expectedPreRelease:    "alpha",
			expectedBuildMetadata: "001",
		},
		{
			version:            "1.2",
			semver:             false,
			expectedComponents: []uint{1, 2},
			expectedMajor:      1,
			expectedMinor:      2,
		},
		{
			version:               "1.0.2-beta+exp.sha.5114f85",
			semver:                true,
			expectedComponents:    []uint{1, 0, 2},
			expectedMajor:         1,
			expectedMinor:         0,
			expectedPatch:         2,
			expectedPreRelease:    "beta",
			expectedBuildMetadata: "exp.sha.5114f85",
		},
	}

	for _, test := range tests {
		version, _ := parse(test.version, test.semver)
		if !reflect.DeepEqual(test.expectedComponents, version.Components()) {
			t.Error("parse returned un'expected components")
		}
		if test.expectedMajor != version.Major() {
			t.Errorf("parse returned version.Major %d, expected %d", test.expectedMajor, version.Major())
		}
		if test.expectedMinor != version.Minor() {
			t.Errorf("parse returned version.Minor %d, expected %d", test.expectedMinor, version.Minor())
		}
		if test.expectedPatch != version.Patch() {
			t.Errorf("parse returned version.Patch %d, expected %d", test.expectedPatch, version.Patch())
		}
		if test.expectedPreRelease != version.PreRelease() {
			t.Errorf("parse returned version.PreRelease %s, expected %s", test.expectedPreRelease, version.PreRelease())
		}
		if test.expectedBuildMetadata != version.BuildMetadata() {
			t.Errorf("parse returned version.BuildMetadata %s, expected %s", test.expectedBuildMetadata, version.BuildMetadata())
		}
	}
}

func TestHighestSupportedVersion(t *testing.T) {
	testCases := []struct {
		versions                        []string
		expectedHighestSupportedVersion string
		shouldFail                      bool
	}{
		{
			versions:                        []string{"v1.0.0"},
			expectedHighestSupportedVersion: "1.0.0",
			shouldFail:                      false,
		},
		{
			versions:   []string{"0.3.0"},
			shouldFail: true,
		},
		{
			versions:   []string{"0.2.0"},
			shouldFail: true,
		},
		{
			versions:                        []string{"1.0.0"},
			expectedHighestSupportedVersion: "1.0.0",
			shouldFail:                      false,
		},
		{
			versions:   []string{"v0.3.0"},
			shouldFail: true,
		},
		{
			versions:   []string{"v0.2.0"},
			shouldFail: true,
		},
		{
			versions:   []string{"0.2.0", "v0.3.0"},
			shouldFail: true,
		},
		{
			versions:                        []string{"0.2.0", "v1.0.0"},
			expectedHighestSupportedVersion: "1.0.0",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"0.2.0", "v1.2.3"},
			expectedHighestSupportedVersion: "1.2.3",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"v1.2.3", "v0.3.0"},
			expectedHighestSupportedVersion: "1.2.3",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"v1.2.3", "v0.3.0", "2.0.1"},
			expectedHighestSupportedVersion: "1.2.3",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"v1.2.3", "4.9.12", "v0.3.0", "2.0.1"},
			expectedHighestSupportedVersion: "1.2.3",
			shouldFail:                      false,
		},
		{
			versions:                        []string{"4.9.12", "2.0.1"},
			expectedHighestSupportedVersion: "",
			shouldFail:                      true,
		},
		{
			versions:                        []string{"v1.2.3", "boo", "v0.3.0", "2.0.1"},
			expectedHighestSupportedVersion: "1.2.3",
			shouldFail:                      false,
		},
		{
			versions:                        []string{},
			expectedHighestSupportedVersion: "",
			shouldFail:                      true,
		},
		{
			versions:                        []string{"var", "boo", "foo"},
			expectedHighestSupportedVersion: "",
			shouldFail:                      true,
		},
	}

	for _, tc := range testCases {
		// Arrange & Act
		actual, err := HighestSupportedVersion(tc.versions)

		// Assert
		if tc.shouldFail && err == nil {
			t.Fatalf("expecting highestSupportedVersion to fail, but got nil error for testcase: %#v", tc)
		}
		if !tc.shouldFail && err != nil {
			t.Fatalf("unexpected error during ValidatePlugin for testcase: %#v\r\n err:%v", tc, err)
		}
		if tc.expectedHighestSupportedVersion != "" {
			result, err := actual.Compare(tc.expectedHighestSupportedVersion)
			if err != nil {
				t.Fatalf("comparison failed with %v for testcase %#v", err, tc)
			}
			if result != 0 {
				t.Fatalf("expectedHighestSupportedVersion %v, but got %v for tc: %#v", tc.expectedHighestSupportedVersion, actual, tc)
			}
		}
	}
}

func TestSubtractMinor(t *testing.T) {
	var tests = []struct {
		version            string
		diff               uint
		expectedComponents []uint
	}{
		{
			version:            "1.0.2",
			diff:               3,
			expectedComponents: []uint{1, 0},
		},
		{
			version:            "1.3.2-alpha+001",
			diff:               2,
			expectedComponents: []uint{1, 1},
		},
		{
			version:            "1.3.2-alpha+001",
			diff:               3,
			expectedComponents: []uint{1, 0},
		},
		{
			version:            "1.20",
			diff:               5,
			expectedComponents: []uint{1, 15},
		},
	}

	for _, test := range tests {
		version, _ := ParseGeneric(test.version)
		if !reflect.DeepEqual(test.expectedComponents, version.SubtractMinor(test.diff).Components()) {
			t.Error("parse returned un'expected components")
		}
	}
}

func TestParse(t *testing.T) {

	var tests = []struct {
		version               string
		expectErr             bool
		expectedComponents    []uint
		expectedPreRelease    string
		expectedBuildMetadata string
	}{
		{
			version:            "1.0.2",
			expectedComponents: []uint{1, 0, 2},
		},
		{
			version:               "1.0.2-alpha+001",
			expectedComponents:    []uint{1, 0, 2},
			expectedPreRelease:    "alpha",
			expectedBuildMetadata: "001",
		},
		{
			version:            "1.2",
			expectedComponents: []uint{1, 2},
		},
		{
			version:               "1.0.2-beta+exp.sha.5114f85",
			expectedComponents:    []uint{1, 0, 2},
			expectedPreRelease:    "beta",
			expectedBuildMetadata: "exp.sha.5114f85",
		},
		{
			version:   "a.b.c",
			expectErr: true,
		},
	}

	for _, test := range tests {
		version, err := Parse(test.version)
		if test.expectErr {
			if err == nil {
				t.Fatalf("got no err, expected err")
			}
			continue
		}
		if !reflect.DeepEqual(test.expectedComponents, version.Components()) {
			t.Error("parse returned un'expected components")
		}
		if test.expectedPreRelease != version.PreRelease() {
			t.Errorf("parse returned version.PreRelease %s, expected %s", test.expectedPreRelease, version.PreRelease())
		}
		if test.expectedBuildMetadata != version.BuildMetadata() {
			t.Errorf("parse returned version.BuildMetadata %s, expected %s", test.expectedBuildMetadata, version.BuildMetadata())
		}
	}
}
