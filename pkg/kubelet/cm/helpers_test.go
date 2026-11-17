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

package cm

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
)

// getResourceList returns a ResourceList with the
// specified cpu and memory resource values
func getResourceList(cpu, memory string) v1.ResourceList {
	res := v1.ResourceList{}
	if cpu != "" {
		res[v1.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[v1.ResourceMemory] = resource.MustParse(memory)
	}
	return res
}

// getResourceRequirements returns a ResourceRequirements object
func getResourceRequirements(requests, limits v1.ResourceList) v1.ResourceRequirements {
	res := v1.ResourceRequirements{}
	res.Requests = requests
	res.Limits = limits
	return res
}

func TestEffectivePodResources(t *testing.T) {
	restartAlways := v1.ContainerRestartPolicyAlways

	testCases := []struct {
		name                     string
		pod                      *v1.Pod
		podLevelResourcesEnabled bool
		draEnabled               bool
		expected                 *v1.ResourceRequirements
	}{
		{
			name: "besteffort pod returns non-nil empty requirements",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Name: "c1"}, {Name: "c2"}},
				},
			},
			podLevelResourcesEnabled: true,
			expected:                 &v1.ResourceRequirements{},
		},
		{
			name: "partial container limits omit the missing limit",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "c1", Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi"))},
						{Name: "c2", Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("300m", ""))},
					},
				},
			},
			podLevelResourcesEnabled: true,
			expected: &v1.ResourceRequirements{
				Requests: getResourceList("200m", "200Mi"),
				Limits:   getResourceList("500m", ""),
			},
		},
		{
			name: "init containers and restartable sidecars",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{Name: "init1", Resources: getResourceRequirements(getResourceList("500m", "100Mi"), getResourceList("600m", "200Mi"))},
						{Name: "sidecar", RestartPolicy: &restartAlways, Resources: getResourceRequirements(getResourceList("100m", "200Mi"), getResourceList("200m", "300Mi"))},
					},
					Containers: []v1.Container{
						{Name: "app", Resources: getResourceRequirements(getResourceList("200m", "200Mi"), getResourceList("300m", "300Mi"))},
					},
				},
			},
			podLevelResourcesEnabled: true,
			expected: &v1.ResourceRequirements{
				// max(init1, sidecar+app)
				Requests: getResourceList("500m", "400Mi"),
				Limits:   getResourceList("600m", "600Mi"),
			},
		},
		{
			name: "explicit pod-level resources",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: getResourceList("500m", "512Mi"),
						Limits:   getResourceList("1", "1Gi"),
					},
					Containers: []v1.Container{
						{Name: "c1", Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", ""))},
					},
				},
			},
			podLevelResourcesEnabled: true,
			expected: &v1.ResourceRequirements{
				Requests: getResourceList("500m", "512Mi"),
				Limits:   getResourceList("1", "1Gi"),
			},
		},
		{
			name: "partial explicit pod-level resources fall back to containers per resource",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: getResourceList("1", ""),
					},
					Containers: []v1.Container{
						{Name: "c1", Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi"))},
					},
				},
			},
			podLevelResourcesEnabled: true,
			expected: &v1.ResourceRequirements{
				Requests: getResourceList("1", "100Mi"),
				Limits:   getResourceList("200m", "200Mi"),
			},
		},
		{
			name: "explicit pod-level resources ignored when PodLevelResources disabled",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: getResourceList("500m", "512Mi"),
						Limits:   getResourceList("1", "1Gi"),
					},
					Containers: []v1.Container{
						{Name: "c1", Resources: getResourceRequirements(getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi"))},
					},
				},
			},
			podLevelResourcesEnabled: false,
			expected: &v1.ResourceRequirements{
				Requests: getResourceList("100m", "100Mi"),
				Limits:   getResourceList("200m", "200Mi"),
			},
		},
		{
			name: "excludes pod overhead and drops zero and non-cpu/memory resources",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Overhead: getResourceList("250m", "128Mi"),
					Containers: []v1.Container{{
						Name: "c1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:                   resource.MustParse("0"),
								v1.ResourceMemory:                resource.MustParse("200Mi"),
								v1.ResourceEphemeralStorage:      resource.MustParse("1Gi"),
								v1.ResourceName("hugepages-2Mi"): resource.MustParse("64Mi"),
							},
							Limits: v1.ResourceList{
								v1.ResourceCPU:                   resource.MustParse("300m"),
								v1.ResourceMemory:                resource.MustParse("400Mi"),
								v1.ResourceName("hugepages-2Mi"): resource.MustParse("64Mi"),
							},
						},
					}},
				},
			},
			podLevelResourcesEnabled: true,
			expected: &v1.ResourceRequirements{
				Requests: getResourceList("", "200Mi"),
				Limits:   getResourceList("300m", "400Mi"),
			},
		},
		{
			name: "includes extra node-allocatable resources when DRANodeAllocatableResources enabled",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{Name: "init1", Resources: getResourceRequirements(getResourceList("300m", "300Mi"), getResourceList("300m", "300Mi"))},
					},
					Containers: []v1.Container{
						{Name: "app", Resources: getResourceRequirements(getResourceList("500m", "500Mi"), getResourceList("500m", "500Mi"))},
					},
				},
				Status: v1.PodStatus{
					ExtraNodeAllocatableResources: []v1.ExtraNodeAllocatableResource{
						{
							ContainerName: "",
							Requests:      getResourceList("100m", "100Mi"),
							Limits:        getResourceList("100m", "100Mi"),
						},
						{
							ContainerName: "init1",
							Requests:      getResourceList("300m", "300Mi"),
							Limits:        getResourceList("300m", "300Mi"),
						},
					},
				},
			},
			podLevelResourcesEnabled: true,
			draEnabled:               true,
			expected: &v1.ResourceRequirements{
				// max(init1, app) + pod-level extra entry
				Requests: getResourceList("600m", "600Mi"),
				Limits:   getResourceList("600m", "600Mi"),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if !tc.podLevelResourcesEnabled {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.37"))
			}
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				pkgfeatures.PodLevelResources:           tc.podLevelResourcesEnabled,
				pkgfeatures.DRANodeAllocatableResources: tc.draEnabled,
			})
			actual := EffectivePodResources(tc.pod)
			if !apiequality.Semantic.DeepEqual(actual, tc.expected) {
				t.Errorf("EffectivePodResources() = %+v, want %+v", actual, tc.expected)
			}
		})
	}
}
