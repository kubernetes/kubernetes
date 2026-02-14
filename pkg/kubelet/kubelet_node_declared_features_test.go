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
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	ndf "k8s.io/component-helpers/nodedeclaredfeatures"
	ndftesting "k8s.io/component-helpers/nodedeclaredfeatures/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
)

func TestDeclaredFeatureDiscovery(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeDeclaredFeatures, true)

	podLevelResourcesIPPRFeatureGate := features.InPlacePodLevelResourcesVerticalScaling
	featureMaxVersion := version.MustParseSemantic("v1.36.0")
	createMockFeature := func(t *testing.T, name string, cfg *ndf.NodeConfiguration) *ndftesting.MockFeature {
		m := ndftesting.NewMockFeature(t)
		m.EXPECT().Name().Return(name).Maybe()
		m.EXPECT().MaxVersion().Return(featureMaxVersion).Maybe()
		m.EXPECT().Discover(cfg).Return(cfg.FeatureGates.Enabled(name)).Maybe()
		return m
	}

	testcases := []struct {
		name                         string
		podLevelResourcesIPPREnabled bool
		featureDiscovered            bool
		kubeletVersion               *version.Version
	}{
		{
			name:                         "feature gate enabled, feature discovered",
			podLevelResourcesIPPREnabled: true,
			featureDiscovered:            true,
			kubeletVersion:               featureMaxVersion.SubtractMinor(1),
		},
		{
			name:                         "feature gate enabled, feature not discovered as kubelet version higher than max version",
			podLevelResourcesIPPREnabled: true,
			featureDiscovered:            false,
			kubeletVersion:               featureMaxVersion.AddMinor(1),
		},
		{
			name:                         "feature disabled and not discovered",
			podLevelResourcesIPPREnabled: false,
			featureDiscovered:            false,
			kubeletVersion:               featureMaxVersion.SubtractMinor(1),
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, podLevelResourcesIPPRFeatureGate, tc.podLevelResourcesIPPREnabled)

			cfg := &ndf.NodeConfiguration{
				FeatureGates: FeatureGateAdapter{
					FeatureGate: utilfeature.DefaultFeatureGate,
				},
				Version: tc.kubeletVersion,
			}
			registeredFeatures := []ndf.Feature{
				createMockFeature(t, string(podLevelResourcesIPPRFeatureGate), cfg),
			}

			testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
			defer testKubelet.Cleanup()
			kubelet := testKubelet.kubelet
			fakeCM := cm.NewFakeContainerManagerWithNodeConfig(cm.NodeConfig{})
			kubelet.containerManager = fakeCM
			framework, err := ndf.New(registeredFeatures)
			require.NoError(t, err)
			kubelet.nodeDeclaredFeaturesFramework = framework
			kubelet.version = tc.kubeletVersion

			features := kubelet.discoverNodeDeclaredFeatures()
			if tc.featureDiscovered {
				assert.Contains(t, features, string(podLevelResourcesIPPRFeatureGate))
			} else {
				assert.NotContains(t, features, string(podLevelResourcesIPPRFeatureGate))
			}
		})
	}
}
