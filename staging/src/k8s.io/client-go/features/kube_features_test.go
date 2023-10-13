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

package features

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestDriveSetFeatureGates tests that we can store multiple implementations in a global var.
func TestDriveSetFeatureGates(t *testing.T) {
	SetFeatureGates(&fakeReader{})
	DefaultFeatureGates().Enabled("Foobar")
}

// TestDriveAddFeaturesToExistingFeatureGates ensures that
// the defaultKubernetesFeatureGates are added to a feature gate.
func TestDriveAddFeaturesToExistingFeatureGates(t *testing.T) {
	defaultKubernetesFeatureGates["MyFeature"] = FeatureSpec{
		Default:       true,
		LockToDefault: true,
		PreRelease:    "GA",
	}
	defer func() {
		delete(defaultKubernetesFeatureGates, "MyFeature")
	}()

	realFeatureGate := &fakeRegistry{}
	require.NoError(t, AddFeaturesToExistingFeatureGates(realFeatureGate))

	require.Equal(t, defaultKubernetesFeatureGates, realFeatureGate.specs)
}

type fakeReader struct{}

func (f *fakeReader) Enabled(key Feature) bool {
	return true
}

type fakeRegistry struct {
	specs map[Feature]FeatureSpec
}

func (f *fakeRegistry) Add(specs map[Feature]FeatureSpec) error {
	f.specs = specs
	return nil
}
