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

package features

import (
	"reflect"
	"testing"

	"github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/ktesting"
)

// TestAddFeaturesToExistingFeatureGates ensures that
// the defaultVersionedKubernetesFeatureGates are added to a test feature gates registry.
func TestAddFeaturesToExistingFeatureGates(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeFeatureGates := &fakeRegistry{}
	tCtx.AssertNoError(AddFeaturesToExistingFeatureGates(fakeFeatureGates))
	tCtx.Assert(fakeFeatureGates.specs).Should(gomega.Equal(unversionedFeatureGates(defaultVersionedKubernetesFeatureGates)))
}

// TestAddVersionedFeaturesToExistingFeatureGates ensures that
// the defaultVersionedKubernetesFeatureGates are added to a versioned test feature gates registry.
func TestAddVersionedFeaturesToExistingFeatureGates(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeFeatureGates := &fakeVersionedRegistry{}
	tCtx.AssertNoError(AddVersionedFeaturesToExistingFeatureGates(fakeFeatureGates))
	tCtx.Assert(fakeFeatureGates.specs).Should(gomega.Equal(defaultVersionedKubernetesFeatureGates))
}

func TestUnversionedFeatureGates(t *testing.T) {
	testCases := []struct {
		name         string
		featureGates map[Feature]VersionedSpecs
		expected     map[Feature]FeatureSpec
	}{
		{
			name: "multiple features",
			featureGates: map[Feature]VersionedSpecs{
				"AlphaFeature": {
					{Version: version.MustParse("1.30"), Default: false, PreRelease: Alpha},
				},
				"BetaFeature": {
					{Version: version.MustParse("1.28"), Default: false, PreRelease: Alpha},
					{Version: version.MustParse("1.30"), Default: true, PreRelease: Beta},
				},
				"GAFeature": {
					{Version: version.MustParse("1.25"), Default: false, PreRelease: Alpha},
					{Version: version.MustParse("1.27"), Default: true, PreRelease: Beta},
					{Version: version.MustParse("1.29"), Default: true, PreRelease: GA, LockToDefault: true},
				},
			},
			expected: map[Feature]FeatureSpec{
				"AlphaFeature": {Default: false, PreRelease: Alpha},
				"BetaFeature":  {Default: true, PreRelease: Beta},
				"GAFeature":    {Default: true, PreRelease: GA, LockToDefault: true},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := unversionedFeatureGates(tc.featureGates)
			if !reflect.DeepEqual(actual, tc.expected) {
				t.Errorf("unversionedFeatureGates() = %v, want %v", actual, tc.expected)
			}
		})
	}
}

func TestReplaceFeatureGatesWithWarningIndicator(t *testing.T) {
	tCtx := ktesting.Init(t)
	defaultFeatureGates := FeatureGates()
	tCtx.Require(func() { defaultFeatureGates.Enabled("Foo") }).Should(gomega.Panic(), "reading an unregistered feature gate Foo should panic")

	if !replaceFeatureGatesWithWarningIndicator(defaultFeatureGates) {
		t.Error("replacing the default feature gates after reading a value hasn't produced a warning")
	}
}

type fakeRegistry struct {
	specs map[Feature]FeatureSpec
}

func (f *fakeRegistry) Add(specs map[Feature]FeatureSpec) error {
	f.specs = specs
	return nil
}

type fakeVersionedRegistry struct {
	specs map[Feature]VersionedSpecs
}

func (f *fakeVersionedRegistry) AddVersioned(specs map[Feature]VersionedSpecs) error {
	f.specs = specs
	return nil
}
