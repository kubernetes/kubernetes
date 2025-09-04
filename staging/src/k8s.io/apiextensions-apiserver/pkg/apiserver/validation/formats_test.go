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

package validation

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
)

func TestRegistryFormats(t *testing.T) {
	for _, sf := range supportedVersionedFormats {
		for f := range sf.formats {
			if !strfmt.Default.ContainsName(f) {
				t.Errorf("expected format %q in strfmt default registry", f)
			}
		}
	}
}

func TestSupportedFormats(t *testing.T) {
	vf := []versionedFormats{
		{
			introducedVersion: version.MajorMinor(1, 0),
			formats: sets.New(
				"A",
			),
		},
		{
			introducedVersion: version.MajorMinor(1, 1),
			formats: sets.New(
				"B",
				"C",
			),
		},
		// Version 1.2 has no new supported formats
		{
			introducedVersion: version.MajorMinor(1, 3),
			formats: sets.New(
				"D",
			),
		},
		{
			introducedVersion: version.MajorMinor(1, 3), // same version as previous entry
			formats: sets.New(
				"E",
			),
		},
		{
			introducedVersion: version.MajorMinor(1, 4),
			formats:           sets.New[string](),
		},
	}

	testCases := []struct {
		name            string
		version         *version.Version
		expectedFormats sets.Set[string]
	}{
		{
			name:            "version 1.0",
			version:         version.MajorMinor(1, 0),
			expectedFormats: sets.New("A"),
		},
		{
			name:            "version 1.1",
			version:         version.MajorMinor(1, 1),
			expectedFormats: sets.New("A", "B", "C"),
		},
		{
			name:            "version 1.2",
			version:         version.MajorMinor(1, 2),
			expectedFormats: sets.New("A", "B", "C"),
		},
		{
			name:            "version 1.3",
			version:         version.MajorMinor(1, 3),
			expectedFormats: sets.New("A", "B", "C", "D", "E"),
		},
		{
			name:            "version 1.4",
			version:         version.MajorMinor(1, 4),
			expectedFormats: sets.New("A", "B", "C", "D", "E"),
		},
	}
	allFormats := newFormatsAtVersion(version.MajorMinor(0, 0), vf)
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := newFormatsAtVersion(tc.version, vf)

			t.Run("newFormatsAtVersion", func(t *testing.T) {
				if !got.supported.Equal(tc.expectedFormats) {
					t.Errorf("expected %v, got %v", tc.expectedFormats, got.supported)
				}

				if len(got.supported.Difference(allFormats.supported)) == 0 {
					t.Errorf("expected allFormats to be a superset of all formats, but was missing %v", allFormats.supported)
				}
			})

			t.Run("StripUnsupportedFormatsPostProcessorForVersion", func(t *testing.T) {
				processor := StripUnsupportedFormatsPostProcessorForVersion(tc.version)
				for f := range allFormats.supported {
					schema := &spec.Schema{SchemaProps: spec.SchemaProps{Format: f}}
					err := processor(schema)
					if err != nil {
						t.Fatalf("Unexpected error: %v", err)
					}
					gotFormat := schema.Format
					if tc.expectedFormats.Has(f) {
						if gotFormat != f {
							t.Errorf("expected format %q, got %q", f, gotFormat)
						}
					} else {
						if gotFormat != "" {
							t.Errorf("expected format to be stripped out, got %q", gotFormat)
						}
					}
				}
			})
		})
	}
}

func TestGetUnrecognizedFormats(t *testing.T) {
	testCases := []struct {
		name                 string
		schema               *spec.Schema
		compatibilityVersion *version.Version
		expectedFormats      []string
	}{
		{
			name:                 "empty format",
			schema:               &spec.Schema{SchemaProps: spec.SchemaProps{Format: "", Type: []string{"string"}}},
			compatibilityVersion: version.MajorMinor(1, 0),
			expectedFormats:      []string{},
		},
		{
			name:                 "recognized format at version 1.0",
			schema:               &spec.Schema{SchemaProps: spec.SchemaProps{Format: "email", Type: []string{"string"}}},
			compatibilityVersion: version.MajorMinor(1, 0),
			expectedFormats:      []string{},
		},
		{
			name:                 "unrecognized format at version 1.0",
			schema:               &spec.Schema{SchemaProps: spec.SchemaProps{Format: "unknown-format", Type: []string{"string"}}},
			compatibilityVersion: version.MajorMinor(1, 0),
			expectedFormats:      []string{"unknown-format"},
		},
		{
			name:                 "recognized format with normalization",
			schema:               &spec.Schema{SchemaProps: spec.SchemaProps{Format: "k8s-short-name", Type: []string{"string"}}},
			compatibilityVersion: version.MajorMinor(1, 34),
			expectedFormats:      []string{},
		},
		{
			name:                 "unrecognized format with normalization",
			schema:               &spec.Schema{SchemaProps: spec.SchemaProps{Format: "k8s-long-name", Type: []string{"string"}}},
			compatibilityVersion: version.MajorMinor(1, 33),
			expectedFormats:      []string{"k8s-long-name"},
		},
		{
			name:                 "format introduced in later version",
			schema:               &spec.Schema{SchemaProps: spec.SchemaProps{Format: "k8s-short-name", Type: []string{"string"}}},
			compatibilityVersion: version.MajorMinor(1, 0),
			expectedFormats:      []string{"k8s-short-name"},
		},
		{
			name:                 "format with dash normalization",
			schema:               &spec.Schema{SchemaProps: spec.SchemaProps{Format: "k8sshortname", Type: []string{"string"}}},
			compatibilityVersion: version.MajorMinor(1, 34),
			expectedFormats:      []string{},
		},
		{
			name:                 "recognized format at exact version",
			schema:               &spec.Schema{SchemaProps: spec.SchemaProps{Format: "uuid", Type: []string{"string"}}},
			compatibilityVersion: version.MajorMinor(1, 0),
			expectedFormats:      []string{},
		},
		{
			name:                 "recognized format at higher version",
			schema:               &spec.Schema{SchemaProps: spec.SchemaProps{Format: "email", Type: []string{"string"}}},
			compatibilityVersion: version.MajorMinor(1, 35),
			expectedFormats:      []string{},
		},
		{
			name:                 "unrecognized format for integer type is not reported",
			schema:               &spec.Schema{SchemaProps: spec.SchemaProps{Format: "unknown-format", Type: []string{"integer"}}},
			compatibilityVersion: version.MajorMinor(1, 0),
			expectedFormats:      []string{},
		},
		{
			name:                 "unrecognized format for string,null type is not reported",
			schema:               &spec.Schema{SchemaProps: spec.SchemaProps{Format: "unknown-format", Type: []string{"string", "null"}}},
			compatibilityVersion: version.MajorMinor(1, 0),
			expectedFormats:      []string{},
		},
		{
			name:                 "unrecognized format for no type is not reported",
			schema:               &spec.Schema{SchemaProps: spec.SchemaProps{Format: "unknown-format"}},
			compatibilityVersion: version.MajorMinor(1, 0),
			expectedFormats:      []string{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := GetUnrecognizedFormats(tc.schema, tc.compatibilityVersion)

			if len(got) != len(tc.expectedFormats) {
				t.Errorf("expected %d unrecognized formats, got %d", len(tc.expectedFormats), len(got))
				return
			}

			// Convert to sets for comparison to handle order differences
			gotSet := sets.New(got...)
			expectedSet := sets.New(tc.expectedFormats...)

			if !gotSet.Equal(expectedSet) {
				t.Errorf("expected unrecognized formats %v, got %v", tc.expectedFormats, got)
			}
		})
	}
}
