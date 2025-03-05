/*
Copyright 2024 The Kubernetes Authors.

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

package compatibility

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/version"
)

func TestValidate(t *testing.T) {
	tests := []struct {
		name                         string
		binaryVersion                string
		emulationVersion             string
		minCompatibilityVersion      string
		emulationVersionFloor        string
		minCompatibilityVersionFloor string
		expectErrors                 bool
	}{
		{
			name:                    "patch version diff ok",
			binaryVersion:           "v1.32.1",
			emulationVersion:        "v1.32.2",
			minCompatibilityVersion: "v1.32.5",
		},
		{
			name:                    "emulation version greater than binary not ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.33.0",
			minCompatibilityVersion: "v1.31.0",
			expectErrors:            true,
		},
		{
			name:                    "min compatibility version greater than emulation version not ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.31.0",
			minCompatibilityVersion: "v1.32.0",
			expectErrors:            true,
		},
		{
			name:                         "between floor and binary ok",
			binaryVersion:                "v1.32.1",
			emulationVersion:             "v1.31.0",
			minCompatibilityVersion:      "v1.30.0",
			emulationVersionFloor:        "v1.31.0",
			minCompatibilityVersionFloor: "v1.30.0",
		},
		{
			name:                         "emulation version less than floor not ok",
			binaryVersion:                "v1.32.1",
			emulationVersion:             "v1.30.0",
			minCompatibilityVersion:      "v1.30.0",
			emulationVersionFloor:        "v1.31.0",
			minCompatibilityVersionFloor: "v1.30.0",
			expectErrors:                 true,
		},
		{
			name:                         "min compatibility version less than floor not ok",
			binaryVersion:                "v1.32.1",
			emulationVersion:             "v1.31.0",
			minCompatibilityVersion:      "v1.29.0",
			emulationVersionFloor:        "v1.31.0",
			minCompatibilityVersionFloor: "v1.30.0",
			expectErrors:                 true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			effective := NewEffectiveVersionFromString(test.binaryVersion, test.emulationVersionFloor, test.minCompatibilityVersionFloor)
			emulationVersion := version.MustParseGeneric(test.emulationVersion)
			minCompatibilityVersion := version.MustParseGeneric(test.minCompatibilityVersion)
			effective.SetEmulationVersion(emulationVersion)
			effective.SetMinCompatibilityVersion(minCompatibilityVersion)

			errs := effective.Validate()
			if len(errs) > 0 && !test.expectErrors {
				t.Errorf("expected no errors, errors found %+v", errs)
			}

			if len(errs) == 0 && test.expectErrors {
				t.Errorf("expected errors, no errors found")
			}
		})
	}
}

func TestSetEmulationVersion(t *testing.T) {
	tests := []struct {
		name                          string
		binaryVersion                 string
		emulationVersion              string
		expectMinCompatibilityVersion string
		emulationVersionFloor         string
		minCompatibilityVersionFloor  string
	}{
		{
			name:                          "minCompatibilityVersion default to 1 minor less than emulationVersion",
			binaryVersion:                 "v1.34",
			emulationVersion:              "v1.32",
			expectMinCompatibilityVersion: "v1.31",
			emulationVersionFloor:         "v1.31",
			minCompatibilityVersionFloor:  "v1.31",
		},
		{
			name:                          "minCompatibilityVersion default to emulationVersion when hitting the floor",
			binaryVersion:                 "v1.34",
			emulationVersion:              "v1.31",
			expectMinCompatibilityVersion: "v1.31",
			emulationVersionFloor:         "v1.31",
			minCompatibilityVersionFloor:  "v1.31",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			effective := NewEffectiveVersionFromString(test.binaryVersion, test.emulationVersionFloor, test.minCompatibilityVersionFloor)

			emulationVersion := version.MustParseGeneric(test.emulationVersion)
			effective.SetEmulationVersion(emulationVersion)
			errs := effective.Validate()
			if len(errs) > 0 {
				t.Fatalf("expected no Validate errors, errors found %+v", errs)
				return
			}

			expectMinCompatibilityVersion := version.MustParseGeneric(test.expectMinCompatibilityVersion)
			if !effective.MinCompatibilityVersion().EqualTo(expectMinCompatibilityVersion) {
				t.Errorf("expected minCompatibilityVersion %s, got %s", expectMinCompatibilityVersion.String(), effective.MinCompatibilityVersion().String())
			}
		})
	}
}

func TestInfo(t *testing.T) {
	tests := []struct {
		name                          string
		binaryVersion                 string
		emulationVersion              string
		minCompatibilityVersion       string
		expectedMajor                 string
		expectedMinor                 string
		expectedEmulationMajor        string
		expectedEmulationMinor        string
		expectedMinCompatibilityMajor string
		expectedMinCompatibilityMinor string
	}{
		{
			name:                          "normal case",
			binaryVersion:                 "v1.34.0",
			emulationVersion:              "v1.32.0",
			minCompatibilityVersion:       "v1.31.0",
			expectedMajor:                 "1",
			expectedMinor:                 "34",
			expectedEmulationMajor:        "1",
			expectedEmulationMinor:        "32",
			expectedMinCompatibilityMajor: "1",
			expectedMinCompatibilityMinor: "31",
		},
		{
			name:             "default min compatibility version is emulation version - 1",
			binaryVersion:    "v1.34.0",
			emulationVersion: "v1.32.0",
			// minCompatibilityVersion not set, should default to v1.31.0
			expectedMajor:                 "1",
			expectedMinor:                 "34",
			expectedEmulationMajor:        "1",
			expectedEmulationMinor:        "32",
			expectedMinCompatibilityMajor: "1",
			expectedMinCompatibilityMinor: "31",
		},
		{
			name:             "emulation version same as binary version",
			binaryVersion:    "v1.34.0",
			emulationVersion: "v1.34.0",
			// minCompatibilityVersion not set, should default to v1.33.0
			expectedMajor:                 "1",
			expectedMinor:                 "34",
			expectedEmulationMajor:        "1",
			expectedEmulationMinor:        "34",
			expectedMinCompatibilityMajor: "1",
			expectedMinCompatibilityMinor: "33",
		},
		{
			name:          "empty binary version",
			binaryVersion: "",
		},
		{
			name:             "with pre-release and build metadata",
			binaryVersion:    "v1.34.0-alpha.1+abc123",
			emulationVersion: "v1.32.0",
			// minCompatibilityVersion not set, should default to v1.31.0
			expectedMajor:                 "1",
			expectedMinor:                 "34",
			expectedEmulationMajor:        "1",
			expectedEmulationMinor:        "32",
			expectedMinCompatibilityMajor: "1",
			expectedMinCompatibilityMinor: "31",
		},
		{
			name:                          "override default min compatibility version",
			binaryVersion:                 "v1.34.0",
			emulationVersion:              "v1.32.0",
			minCompatibilityVersion:       "v1.32.0", // explicitly set to same as emulation version
			expectedMajor:                 "1",
			expectedMinor:                 "34",
			expectedEmulationMajor:        "1",
			expectedEmulationMinor:        "32",
			expectedMinCompatibilityMajor: "1",
			expectedMinCompatibilityMinor: "32",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var effective MutableEffectiveVersion
			if test.binaryVersion == "" {
				effective = &effectiveVersion{}
			} else {
				effective = NewEffectiveVersionFromString(test.binaryVersion, "", "")
				if test.emulationVersion != "" {
					effective.SetEmulationVersion(version.MustParse(test.emulationVersion))
				}
				if test.minCompatibilityVersion != "" {
					effective.SetMinCompatibilityVersion(version.MustParse(test.minCompatibilityVersion))
				}
			}
			info := effective.Info()
			if info == nil {
				if test.expectedMajor != "" {
					t.Fatalf("expected info, got nil")
				}
				return
			}

			if info.Major != test.expectedMajor {
				t.Errorf("expected major %s, got %s", test.expectedMajor, info.Major)
			}
			if info.Minor != test.expectedMinor {
				t.Errorf("expected minor %s, got %s", test.expectedMinor, info.Minor)
			}
			if info.EmulationMajor != test.expectedEmulationMajor {
				t.Errorf("expected emulation major %s, got %s", test.expectedEmulationMajor, info.EmulationMajor)
			}
			if info.EmulationMinor != test.expectedEmulationMinor {
				t.Errorf("expected emulation minor %s, got %s", test.expectedEmulationMinor, info.EmulationMinor)
			}
			if info.MinCompatibilityMajor != test.expectedMinCompatibilityMajor {
				t.Errorf("expected min compatibility major %s, got %s", test.expectedMinCompatibilityMajor, info.MinCompatibilityMajor)
			}
			if info.MinCompatibilityMinor != test.expectedMinCompatibilityMinor {
				t.Errorf("expected min compatibility minor %s, got %s", test.expectedMinCompatibilityMinor, info.MinCompatibilityMinor)
			}
		})
	}
}
