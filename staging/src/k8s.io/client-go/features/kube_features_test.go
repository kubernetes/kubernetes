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

	"k8s.io/component-base/featuregate"
)

// TestDriveSetFeatureGatesMultipleImplementations ensures that
// we can store multiple implementations in a global var.
func TestDriveSetFeatureGatesMultipleImplementations(t *testing.T) {
	featureGateA := &fakeFeatureGate{}
	SetFeatureGates(featureGateA)

	realFeatureGate := featuregate.NewFeatureGate()
	SetFeatureGates(realFeatureGate)
}

// TestDriveAddFeaturesToExistingFeatureGates ensures that
// the defaultKubernetesFeatureGates are added to a feature gate.
func TestDriveAddFeaturesToExistingFeatureGates(t *testing.T) {
	defaultKubernetesFeatureGates["MyFeature"] = featuregate.FeatureSpec{
		Default:       true,
		LockToDefault: true,
		PreRelease:    "GA",
	}
	defer func() {
		delete(defaultKubernetesFeatureGates, "MyFeature")
	}()

	realFeatureGate := featuregate.NewFeatureGate()
	require.NoError(t, AddFeaturesToExistingFeatureGates(realFeatureGate))

	require.True(t, realFeatureGate.Enabled("MyFeature"))
}

type fakeFeatureGate struct{}

func (f *fakeFeatureGate) Enabled(key featuregate.Feature) bool {
	return key == "FakeFeatureGate"
}
