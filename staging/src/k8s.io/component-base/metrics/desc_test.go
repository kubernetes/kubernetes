/*
Copyright 2019 The Kubernetes Authors.

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

package metrics

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/version"
)

func TestDescCreate(t *testing.T) {
	currentVersion := parseVersion(version.Info{
		Major:      "1",
		Minor:      "17",
		GitVersion: "v1.17.0-alpha-1.12345",
	})

	var tests = []struct {
		name              string
		fqName            string
		help              string
		stabilityLevel    StabilityLevel
		deprecatedVersion string

		shouldCreate          bool
		expectedAnnotatedHelp string
	}{
		{
			name:                  "alpha descriptor should be created",
			fqName:                "normal_alpha_descriptor",
			help:                  "this is an alpha descriptor",
			stabilityLevel:        ALPHA,
			deprecatedVersion:     "",
			shouldCreate:          true,
			expectedAnnotatedHelp: "[ALPHA] this is an alpha descriptor",
		},
		{
			name:                  "stable descriptor should be created",
			fqName:                "normal_stable_descriptor",
			help:                  "this is a stable descriptor",
			stabilityLevel:        STABLE,
			deprecatedVersion:     "",
			shouldCreate:          true,
			expectedAnnotatedHelp: "[STABLE] this is a stable descriptor",
		},
		{
			name:                  "deprecated descriptor should be created",
			fqName:                "deprecated_stable_descriptor",
			help:                  "this is a deprecated descriptor",
			stabilityLevel:        STABLE,
			deprecatedVersion:     "1.17.0",
			shouldCreate:          true,
			expectedAnnotatedHelp: "[STABLE] (Deprecated since 1.17.0) this is a deprecated descriptor",
		},
		{
			name:                  "hidden descriptor should not be created",
			fqName:                "hidden_stable_descriptor",
			help:                  "this is a hidden descriptor",
			stabilityLevel:        STABLE,
			deprecatedVersion:     "1.16.0",
			shouldCreate:          false,
			expectedAnnotatedHelp: "this is a hidden descriptor", // hidden descriptor shall not be annotated.
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.name, func(t *testing.T) {
			desc := NewDesc(tc.fqName, tc.help, nil, nil, tc.stabilityLevel, tc.deprecatedVersion)

			if desc.IsCreated() {
				t.Fatal("Descriptor should not be created by default.")
			}

			desc.create(&currentVersion)
			desc.create(&currentVersion) // we can safely create a descriptor over and over again.

			if desc.IsCreated() != tc.shouldCreate {
				t.Fatalf("expected create state: %v, but got: %v", tc.shouldCreate, desc.IsCreated())
			}

			if !strings.Contains(desc.String(), tc.expectedAnnotatedHelp) {
				t.Fatalf("expected annotated help: %s, but not in descriptor: %s", tc.expectedAnnotatedHelp, desc.String())
			}
		})
	}
}

func TestDescClearState(t *testing.T) {
	currentVersion := parseVersion(version.Info{
		Major:      "1",
		Minor:      "17",
		GitVersion: "v1.17.0-alpha-1.12345",
	})

	var tests = []struct {
		name              string
		fqName            string
		help              string
		stabilityLevel    StabilityLevel
		deprecatedVersion string
	}{
		{
			name:              "alpha descriptor",
			fqName:            "normal_alpha_descriptor",
			help:              "this is an alpha descriptor",
			stabilityLevel:    ALPHA,
			deprecatedVersion: "",
		},
		{
			name:              "stable descriptor",
			fqName:            "normal_stable_descriptor",
			help:              "this is a stable descriptor",
			stabilityLevel:    STABLE,
			deprecatedVersion: "",
		},
		{
			name:              "deprecated descriptor",
			fqName:            "deprecated_stable_descriptor",
			help:              "this is a deprecated descriptor",
			stabilityLevel:    STABLE,
			deprecatedVersion: "1.17.0",
		},
		{
			name:              "hidden descriptor",
			fqName:            "hidden_stable_descriptor",
			help:              "this is a hidden descriptor",
			stabilityLevel:    STABLE,
			deprecatedVersion: "1.16.0",
		},
	}

	for _, test := range tests {
		tc := test
		t.Run(tc.name, func(t *testing.T) {
			descA := NewDesc(tc.fqName, tc.help, nil, nil, tc.stabilityLevel, tc.deprecatedVersion)
			descB := NewDesc(tc.fqName, tc.help, nil, nil, tc.stabilityLevel, tc.deprecatedVersion)

			descA.create(&currentVersion)
			descA.ClearState()

			// create
			if !reflect.DeepEqual(*descA, *descB) {
				t.Fatal("descriptor state hasn't be cleaned up")
			}
		})
	}
}
