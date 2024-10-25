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

package version

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/version"
)

func TestValidate(t *testing.T) {
	tests := []struct {
		name                    string
		binaryVersion           string
		emulationVersion        string
		minCompatibilityVersion string
		expectErrors            bool
	}{
		{
			name:                    "patch version diff ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.32.1",
			minCompatibilityVersion: "v1.31.5",
		},
		{
			name:                    "emulation version one minor lower than binary ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.31.0",
			minCompatibilityVersion: "v1.31.0",
		},
		{
			name:                    "emulation version two minor lower than binary not ok",
			binaryVersion:           "v1.33.2",
			emulationVersion:        "v1.31.0",
			minCompatibilityVersion: "v1.32.0",
			expectErrors:            true,
		},
		{
			name:                    "emulation version one minor higher than binary not ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.33.0",
			minCompatibilityVersion: "v1.31.0",
			expectErrors:            true,
		},
		{
			name:                    "emulation version two minor higher than binary not ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.34.0",
			minCompatibilityVersion: "v1.31.0",
			expectErrors:            true,
		},
		{
			name:                    "compatibility version same as binary not ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.32.0",
			minCompatibilityVersion: "v1.32.0",
			expectErrors:            true,
		},
		{
			name:                    "compatibility version two minor lower than binary not ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.32.0",
			minCompatibilityVersion: "v1.30.0",
			expectErrors:            true,
		},
		{
			name:                    "compatibility version one minor higher than binary not ok",
			binaryVersion:           "v1.32.2",
			emulationVersion:        "v1.32.0",
			minCompatibilityVersion: "v1.33.0",
			expectErrors:            true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			binaryVersion := version.MustParseGeneric(test.binaryVersion)
			effective := &effectiveVersion{}
			emulationVersion := version.MustParseGeneric(test.emulationVersion)
			minCompatibilityVersion := version.MustParseGeneric(test.minCompatibilityVersion)
			effective.Set(binaryVersion, emulationVersion, minCompatibilityVersion)

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
