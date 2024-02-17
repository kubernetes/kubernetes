/*
Copyright 2023 The Kubernetes Authors.

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
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"

	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestValidateWhenEmulationVersionEnabled(t *testing.T) {
	tests := []struct {
		name                    string
		featuresEnabled         []featuregate.Feature
		binaryVersion           string
		emulationVersion        string
		minCompatibilityVersion string
		expectErrors            bool
	}{
		{
			name:                    "patch version diff ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.30.1",
			minCompatibilityVersion: "v1.29.5",
		},
		{
			name:                    "emulation version one minor lower than binary ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.29.1",
			minCompatibilityVersion: "v1.29.0",
		},
		{
			name:                    "emulation version two minor lower than binary not ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.28.1",
			minCompatibilityVersion: "v1.29.0",
			expectErrors:            true,
		},
		{
			name:                    "emulation version one minor higher than binary not ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.31.1",
			minCompatibilityVersion: "v1.29.0",
			expectErrors:            true,
		},
		{
			name:                    "emulation version two minor higher than binary not ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.32.1",
			minCompatibilityVersion: "v1.29.0",
			expectErrors:            true,
		},
		{
			name:                    "compatibility version same as binary not ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.30.1",
			minCompatibilityVersion: "v1.30.0",
			expectErrors:            true,
		},
		{
			name:                    "compatibility version two minor lower than binary not ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.30.1",
			minCompatibilityVersion: "v1.28.0",
			expectErrors:            true,
		},
		{
			name:                    "compatibility version one minor higher than binary not ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.30.1",
			minCompatibilityVersion: "v1.31.0",
			expectErrors:            true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.EmulationVersion, true)()

			effective := &mutableEffectiveVersions{}
			binaryVersion := version.MustParseGeneric(test.binaryVersion)
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

func TestValidateWhenEmulationVersionDisabled(t *testing.T) {
	tests := []struct {
		name                    string
		featuresEnabled         []featuregate.Feature
		binaryVersion           string
		emulationVersion        string
		minCompatibilityVersion string
		expectErrors            bool
	}{
		{
			name:                    "patch version diff ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.30.1",
			minCompatibilityVersion: "v1.29.5",
		},
		{
			name:                    "emulation version one minor lower than binary not ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.29.1",
			minCompatibilityVersion: "v1.29",
			expectErrors:            true,
		},
		{
			name:                    "emulation version two minor lower than binary not ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.28.1",
			minCompatibilityVersion: "v1.29",
			expectErrors:            true,
		},
		{
			name:                    "emulation version one minor higher than binary not ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.31.1",
			minCompatibilityVersion: "v1.29",
			expectErrors:            true,
		},
		{
			name:                    "emulation version two minor higher than binary not ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.32.1",
			minCompatibilityVersion: "v1.29",
			expectErrors:            true,
		},
		{
			name:                    "compatibility version same as binary not ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.30.1",
			minCompatibilityVersion: "v1.30",
			expectErrors:            true,
		},
		{
			name:                    "compatibility version two minor lower than binary not ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.30.1",
			minCompatibilityVersion: "v1.28",
			expectErrors:            true,
		},
		{
			name:                    "compatibility version one minor higher than binary not ok",
			binaryVersion:           "v1.30.2",
			emulationVersion:        "v1.30.1",
			minCompatibilityVersion: "v1.31",
			expectErrors:            true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.EmulationVersion, false)()

			effective := &mutableEffectiveVersions{}
			binaryVersion := version.MustParseGeneric(test.binaryVersion)
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
