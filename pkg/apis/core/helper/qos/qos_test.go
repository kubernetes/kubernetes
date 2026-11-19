/*
Copyright The Kubernetes Authors.

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

package qos

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

func TestComputePodQOSPodLevelEphemeralStorage(t *testing.T) {
	guaranteedCPUAndMem := core.ResourceList{
		core.ResourceCPU:    resource.MustParse("100m"),
		core.ResourceMemory: resource.MustParse("100Mi"),
	}
	podOnlyEphemeralStorage := &core.ResourceRequirements{
		Requests: core.ResourceList{
			core.ResourceEphemeralStorage: resource.MustParse("1Gi"),
		},
		Limits: core.ResourceList{
			core.ResourceEphemeralStorage: resource.MustParse("1Gi"),
		},
	}

	testCases := []struct {
		name                                      string
		pod                                       *core.Pod
		podLevelResourcesEphemeralStorageEnabled  bool
		podLevelResourcesFixKubeletQOSClassEnable bool
		expected                                  core.PodQOSClass
	}{
		{
			name: "pod-level ephemeral-storage only falls back to container QoS when gate enabled",
			pod: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "plr-es-enabled"},
				Spec: core.PodSpec{
					Resources: podOnlyEphemeralStorage,
					Containers: []core.Container{
						{
							Name: "c1",
							Resources: core.ResourceRequirements{
								Requests: guaranteedCPUAndMem,
								Limits:   guaranteedCPUAndMem,
							},
						},
					},
				},
			},
			podLevelResourcesEphemeralStorageEnabled:  true,
			podLevelResourcesFixKubeletQOSClassEnable: true,
			expected: core.PodQOSGuaranteed,
		},
		{
			name: "pod-level ephemeral-storage only falls back to container QoS when gate disabled",
			pod: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "plr-es-disabled"},
				Spec: core.PodSpec{
					Resources: podOnlyEphemeralStorage,
					Containers: []core.Container{
						{
							Name: "c1",
							Resources: core.ResourceRequirements{
								Requests: guaranteedCPUAndMem,
								Limits:   guaranteedCPUAndMem,
							},
						},
					},
				},
			},
			podLevelResourcesEphemeralStorageEnabled:  false,
			podLevelResourcesFixKubeletQOSClassEnable: true,
			expected: core.PodQOSGuaranteed,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourcesFixKubeletQOSClass, tc.podLevelResourcesFixKubeletQOSClassEnable)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResourcesEphemeralStorage, tc.podLevelResourcesEphemeralStorageEnabled)

			if got := ComputePodQOS(tc.pod); got != tc.expected {
				t.Errorf("ComputePodQOS() = %v, want %v", got, tc.expected)
			}
		})
	}
}
