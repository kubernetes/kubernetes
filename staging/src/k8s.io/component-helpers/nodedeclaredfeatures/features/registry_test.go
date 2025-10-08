/*
Copyright 2025 The Kubernetes Authors.

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

package features

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/component-helpers/nodedeclaredfeatures"
	ndftesting "k8s.io/component-helpers/nodedeclaredfeatures/testing"
)

func TestFeatureValidation(t *testing.T) {
	// Defer a function to clean up the global registry after the test.
	originalFeatures := AllFeatures
	defer func() {
		AllFeatures = originalFeatures
	}()

	testCases := []struct {
		name          string
		features      []nodedeclaredfeatures.Feature
		expectErr     bool
		errorContains string
	}{
		{
			name: "valid features",
			features: []nodedeclaredfeatures.Feature{
				&ndftesting.MockFeature{NameFunc: func() string { return "ValidFeature" }},
				&ndftesting.MockFeature{NameFunc: func() string { return "MyFeature/MySubFeature" }},
				&ndftesting.MockFeature{NameFunc: func() string { return "MyFeature/mySubFeature" }},
			},
			expectErr: false,
		},
		{
			name: "duplicate feature",
			features: []nodedeclaredfeatures.Feature{
				&ndftesting.MockFeature{NameFunc: func() string { return "Duplicate" }},
				&ndftesting.MockFeature{NameFunc: func() string { return "Duplicate" }},
			},
			expectErr:     true,
			errorContains: "duplicate feature name \"Duplicate\"",
		},
		{
			name: "invalid name (kebab-case)",
			features: []nodedeclaredfeatures.Feature{
				&ndftesting.MockFeature{NameFunc: func() string { return "invalid-name" }},
			},
			expectErr:     true,
			errorContains: "invalid feature name \"invalid-name\"",
		},
		{
			name: "invalid name (starts with lowercase)",
			features: []nodedeclaredfeatures.Feature{
				&ndftesting.MockFeature{NameFunc: func() string { return "invalidName" }},
			},
			expectErr:     true,
			errorContains: "invalid feature name \"invalidName\"",
		},
		{
			name: "invalid name (ends with slash)",
			features: []nodedeclaredfeatures.Feature{
				&ndftesting.MockFeature{NameFunc: func() string { return "Invalid/" }},
			},
			expectErr:     true,
			errorContains: "invalid feature name \"Invalid/\"",
		},
		{
			name: "invalid name (too long)",
			features: []nodedeclaredfeatures.Feature{
				&ndftesting.MockFeature{NameFunc: func() string { return "a" + string(make([]byte, 253)) }},
			},
			expectErr:     true,
			errorContains: "must be no more than 253 characters",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Reset AllFeatures for each test case.
			AllFeatures = tc.features

			err := ValidateFeatures()

			if tc.expectErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.errorContains)
			} else {
				require.NoError(t, err)
			}
		})
	}
}
