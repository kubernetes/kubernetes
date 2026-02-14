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

package kubelet

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	ndflib "k8s.io/component-helpers/nodedeclaredfeatures"
	ndffeatures "k8s.io/component-helpers/nodedeclaredfeatures/features"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
)

func TestGuaranteedPodExclusiveCPUsFeatureDiscovery(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeDeclaredFeatures, true)
	const guaranteedCPUResizeDeclaredFeature = "GuaranteedQoSPodCPUResize"

	testcases := []struct {
		name                                string
		cpuManagerPolicy                    string
		inplacePodResizeExclusiveCPUsEnable bool
		expectFeaturePresent                bool
	}{
		{
			name:                                "feature enabled with static cpu manager policy",
			cpuManagerPolicy:                    "static",
			inplacePodResizeExclusiveCPUsEnable: true,
			expectFeaturePresent:                true,
		},
		{
			name:                                "feature enabled with none cpu manager policy",
			cpuManagerPolicy:                    "none",
			inplacePodResizeExclusiveCPUsEnable: true,
			expectFeaturePresent:                true,
		},
		{
			name:                                "feature disabled with static cpu manager policy",
			cpuManagerPolicy:                    "static",
			inplacePodResizeExclusiveCPUsEnable: false,
			expectFeaturePresent:                false,
		},
		{
			name:                                "feature disabled with none cpu manager policy",
			cpuManagerPolicy:                    "none",
			inplacePodResizeExclusiveCPUsEnable: false,
			expectFeaturePresent:                true,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingExclusiveCPUs, tc.inplacePodResizeExclusiveCPUsEnable)

			testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
			defer testKubelet.Cleanup()
			kubelet := testKubelet.kubelet
			fakeCM := cm.NewFakeContainerManagerWithNodeConfig(cm.NodeConfig{
				CPUManagerPolicy: tc.cpuManagerPolicy,
			})
			kubelet.containerManager = fakeCM
			framework, err := ndflib.New(ndffeatures.AllFeatures)
			require.NoError(t, err)
			kubelet.nodeDeclaredFeaturesFramework = framework

			features, err := kubelet.discoverNodeDeclaredFeatures()
			require.NoError(t, err)
			if tc.expectFeaturePresent {
				assert.Contains(t, features, guaranteedCPUResizeDeclaredFeature)
			} else {
				assert.NotContains(t, features, guaranteedCPUResizeDeclaredFeature)
			}
		})
	}
}
