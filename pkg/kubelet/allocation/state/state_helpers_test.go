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

package state

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func TestUpdatePodFromCheckpoint(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodLevelResourcesVerticalScaling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingMemoryBackedVolumes, true)

	res100m100Mi := v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("100m"),
			v1.ResourceMemory: resource.MustParse("100Mi"),
		},
		Limits: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("200m"),
			v1.ResourceMemory: resource.MustParse("200Mi"),
		},
	}
	res200m200Mi := v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("200m"),
			v1.ResourceMemory: resource.MustParse("200Mi"),
		},
		Limits: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("400m"),
			v1.ResourceMemory: resource.MustParse("400Mi"),
		},
	}

	volLimit100Mi := resource.MustParse("100Mi")
	volLimit200Mi := resource.MustParse("200Mi")

	makePod := func(cRes, icRes, pRes *v1.ResourceRequirements, volLimit *resource.Quantity) *v1.Pod {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-pod",
				UID:  "test-uid",
			},
			Spec: v1.PodSpec{},
		}
		if cRes != nil {
			pod.Spec.Containers = []v1.Container{
				{
					Name:      "c1",
					Resources: *cRes.DeepCopy(),
				},
			}
		}
		if icRes != nil {
			pod.Spec.InitContainers = []v1.Container{
				{
					Name:      "ic1",
					Resources: *icRes.DeepCopy(),
				},
			}
		}
		if pRes != nil {
			pod.Spec.Resources = pRes.DeepCopy()
		}
		if volLimit != nil {
			lc := volLimit.DeepCopy()
			pod.Spec.Volumes = []v1.Volume{
				{
					Name: "vol1",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{
							Medium:    v1.StorageMediumMemory,
							SizeLimit: &lc,
						},
					},
				},
			}
		}
		return pod
	}

	tests := []struct {
		name          string
		pod           *v1.Pod
		storedPod     *v1.Pod
		expectUpdated bool
		expectedPod   *v1.Pod
	}{
		{
			name:          "nil incoming pod",
			pod:           nil,
			storedPod:     makePod(&res100m100Mi, nil, nil, nil),
			expectUpdated: false,
			expectedPod:   nil,
		},
		{
			name:          "nil stored pod",
			pod:           makePod(&res100m100Mi, nil, nil, nil),
			storedPod:     nil,
			expectUpdated: false,
			expectedPod:   makePod(&res100m100Mi, nil, nil, nil),
		},
		{
			name:          "identical pod and stored pod - no update",
			pod:           makePod(&res100m100Mi, &res100m100Mi, &res100m100Mi, &volLimit100Mi),
			storedPod:     makePod(&res100m100Mi, &res100m100Mi, &res100m100Mi, &volLimit100Mi),
			expectUpdated: false,
			expectedPod:   makePod(&res100m100Mi, &res100m100Mi, &res100m100Mi, &volLimit100Mi),
		},
		{
			name:          "container resources updated from stored pod",
			pod:           makePod(&res100m100Mi, nil, nil, nil),
			storedPod:     makePod(&res200m200Mi, nil, nil, nil),
			expectUpdated: true,
			expectedPod:   makePod(&res200m200Mi, nil, nil, nil),
		},
		{
			name:          "init container resources updated from stored pod",
			pod:           makePod(nil, &res100m100Mi, nil, nil),
			storedPod:     makePod(nil, &res200m200Mi, nil, nil),
			expectUpdated: true,
			expectedPod:   makePod(nil, &res200m200Mi, nil, nil),
		},
		{
			name:          "pod-level resources updated from stored pod",
			pod:           makePod(nil, nil, &res100m100Mi, nil),
			storedPod:     makePod(nil, nil, &res200m200Mi, nil),
			expectUpdated: true,
			expectedPod:   makePod(nil, nil, &res200m200Mi, nil),
		},
		{
			name:          "emptyDir volume limit updated from stored pod",
			pod:           makePod(nil, nil, nil, &volLimit100Mi),
			storedPod:     makePod(nil, nil, nil, &volLimit200Mi),
			expectUpdated: true,
			expectedPod:   makePod(nil, nil, nil, &volLimit200Mi),
		},
		{
			name:          "all resources updated together from stored pod",
			pod:           makePod(&res100m100Mi, &res100m100Mi, &res100m100Mi, &volLimit100Mi),
			storedPod:     makePod(&res200m200Mi, &res200m200Mi, &res200m200Mi, &volLimit200Mi),
			expectUpdated: true,
			expectedPod:   makePod(&res200m200Mi, &res200m200Mi, &res200m200Mi, &volLimit200Mi),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actualPod, updated := UpdatePodFromCheckpoint(tc.pod, tc.storedPod)
			assert.Equal(t, tc.expectUpdated, updated)
			if tc.expectedPod == nil {
				assert.Nil(t, actualPod)
				return
			}
			diff := cmp.Diff(tc.expectedPod, actualPod, cmp.Comparer(func(x, y resource.Quantity) bool {
				return x.Equal(y)
			}))
			if diff != "" {
				t.Errorf("pod mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
