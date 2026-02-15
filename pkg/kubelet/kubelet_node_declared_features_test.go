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
	"k8s.io/component-helpers/nodedeclaredfeatures/features/containerulimits"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestGuaranteedPodExclusiveCPUsFeatureDiscovery(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeDeclaredFeatures, true)

	testcases := []struct {
		name                                string
		cpuManagerPolicy                    string
		inplacePodResizeExclusiveCPUsEnable bool
		expectedFeature                     string
		expectFeature                       bool
	}{
		{
			name:                                "feature enabled with static cpu manager policy",
			cpuManagerPolicy:                    "static",
			inplacePodResizeExclusiveCPUsEnable: true,
			expectedFeature:                     "GuaranteedQoSPodCPUResize",
			expectFeature:                       true,
		},
		{
			name:                                "feature enabled with none cpu manager policy",
			cpuManagerPolicy:                    "none",
			inplacePodResizeExclusiveCPUsEnable: true,
			expectedFeature:                     "GuaranteedQoSPodCPUResize",
			expectFeature:                       true,
		},
		{
			name:                                "feature disabled with static cpu manager policy",
			cpuManagerPolicy:                    "static",
			inplacePodResizeExclusiveCPUsEnable: false,
			expectedFeature:                     "GuaranteedQoSPodCPUResize",
			expectFeature:                       false,
		},
		{
			name:                                "feature disabled with none cpu manager policy",
			cpuManagerPolicy:                    "none",
			inplacePodResizeExclusiveCPUsEnable: false,
			expectedFeature:                     "GuaranteedQoSPodCPUResize",
			expectFeature:                       true,
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

			features := kubelet.discoverNodeDeclaredFeatures()
			if tc.expectFeature {
				assert.Contains(t, features, tc.expectedFeature)
			} else {
				assert.NotContains(t, features, tc.expectedFeature)
			}
		})
	}
}

func TestContainerUlimitsFeatureDiscovery(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeDeclaredFeatures, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ContainerUlimits, true)

	testcases := []struct {
		name          string
		runtimeStatus *kubecontainer.RuntimeStatus
		expectFeature bool
	}{
		{
			name: "runtime supports container ulimits",
			runtimeStatus: &kubecontainer.RuntimeStatus{
				Features: &kubecontainer.RuntimeFeatures{ContainerUlimits: true},
			},
			expectFeature: true,
		},
		{
			name: "runtime does not support container ulimits",
			runtimeStatus: &kubecontainer.RuntimeStatus{
				Features: &kubecontainer.RuntimeFeatures{ContainerUlimits: false},
			},
			expectFeature: false,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
			defer testKubelet.Cleanup()
			kubelet := testKubelet.kubelet
			testKubelet.fakeRuntime.RuntimeStatus = tc.runtimeStatus
			kubelet.runtimeState.setRuntimeFeatures(tc.runtimeStatus.Features)

			framework, err := ndflib.New(ndffeatures.AllFeatures)
			require.NoError(t, err)
			kubelet.nodeDeclaredFeaturesFramework = framework

			discoveredFeatures := kubelet.discoverNodeDeclaredFeatures()
			if tc.expectFeature {
				assert.Contains(t, discoveredFeatures, containerulimits.ContainerUlimits)
			} else {
				assert.NotContains(t, discoveredFeatures, containerulimits.ContainerUlimits)
			}
		})
	}
}
