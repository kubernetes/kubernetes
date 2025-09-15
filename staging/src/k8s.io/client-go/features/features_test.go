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
	"testing"

	"github.com/stretchr/testify/require"
)

// TestAddFeaturesToExistingFeatureGates ensures that
// the defaultKubernetesFeatureGates are added to a test feature gates registry.
func TestAddFeaturesToExistingFeatureGates(t *testing.T) {
	fakeFeatureGates := &fakeRegistry{}
	require.NoError(t, AddFeaturesToExistingFeatureGates(fakeFeatureGates))
	require.Equal(t, defaultKubernetesFeatureGates, fakeFeatureGates.specs)
}

func TestReplaceFeatureGatesWithWarningIndicator(t *testing.T) {
	defaultFeatureGates := FeatureGates()
	require.Panics(t, func() { defaultFeatureGates.Enabled("Foo") }, "reading an unregistered feature gate Foo should panic")

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
