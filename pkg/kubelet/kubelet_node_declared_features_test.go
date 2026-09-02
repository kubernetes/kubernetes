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
	"maps"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	ndf "k8s.io/component-helpers/nodedeclaredfeatures"
	ndffeatures "k8s.io/component-helpers/nodedeclaredfeatures/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestRegisteredFeatureDiscovery(t *testing.T) {
	gateSpecs := utilfeature.DefaultFeatureGate.DeepCopy().GetAll()

	discover := func(t *testing.T, gates featuregatetesting.FeatureOverrides, runtimeFeatures *ndf.RuntimeFeatures) []string {
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, gates)
		testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
		defer testKubelet.Cleanup()
		kubelet := testKubelet.kubelet
		kubelet.nodeDeclaredFeaturesFramework = ndf.New(ndffeatures.AllFeatures)
		if runtimeFeatures != nil {
			kubelet.runtimeState.setRuntimeFeatures(&kubecontainer.RuntimeFeatures{
				UserNamespacesHostNetwork: runtimeFeatures.UserNamespacesHostNetwork,
				MountOptions:              runtimeFeatures.MountOptions,
				CgroupMountMode:           runtimeFeatures.CgroupMountMode,
			})
		}
		return kubelet.discoverNodeDeclaredFeatures()
	}

	for _, feature := range ndffeatures.AllFeatures {
		t.Run(feature.Name(), func(t *testing.T) {
			reqs := feature.Requirements()
			if rf := reqs.RequiredRuntimeFeatures; rf != nil {
				// Only the runtime features that discover sets on the kubelet's runtime state are
				// reported, so a feature requiring any other runtime feature can't be discovered.
				reported := ndf.RuntimeFeatures{UserNamespacesHostNetwork: rf.UserNamespacesHostNetwork, MountOptions: rf.MountOptions, CgroupMountMode: rf.CgroupMountMode}
				if *rf != reported {
					t.Skipf("the kubelet doesn't report all of the runtime features %s requires: %+v", feature.Name(), *rf)
				}
			}
			allGatesOn := featuregatetesting.FeatureOverrides{}
			for _, gate := range reqs.EnabledFeatureGates {
				require.Contains(t, gateSpecs, featuregate.Feature(gate), "unknown feature gate")
				allGatesOn[featuregate.Feature(gate)] = true
			}

			t.Run("requirements met", func(t *testing.T) {
				assert.Contains(t, discover(t, allGatesOn, reqs.RequiredRuntimeFeatures), feature.Name())
			})

			for _, gate := range reqs.EnabledFeatureGates {
				if gateSpecs[featuregate.Feature(gate)].LockToDefault {
					continue
				}
				t.Run(gate+" disabled", func(t *testing.T) {
					gates := maps.Clone(allGatesOn)
					gates[featuregate.Feature(gate)] = false
					assert.NotContains(t, discover(t, gates, reqs.RequiredRuntimeFeatures), feature.Name())
				})
			}

			if reqs.RequiredRuntimeFeatures != nil {
				t.Run("runtime features missing", func(t *testing.T) {
					assert.NotContains(t, discover(t, allGatesOn, nil), feature.Name())
				})
			}
		})
	}
}
