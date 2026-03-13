/*
Copyright 2016 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper/qos"
	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/features"
)

func TestComputePodQOS(t *testing.T) {
	testCases := []struct {
		pod                      *v1.Pod
		expected                 v1.PodQOSClass
		podLevelResourcesEnabled bool
	}{
		{
			pod: newPod("guaranteed", []v1.Container{
				newContainer("guaranteed", getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
			}),
			expected: v1.PodQOSGuaranteed,
		},
		{
			pod: newPod("guaranteed-guaranteed", []v1.Container{
				newContainer("guaranteed", getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
				newContainer("guaranteed", getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
			}),
			expected: v1.PodQOSGuaranteed,
		},
		{
			pod: newPod("best-effort-best-effort", []v1.Container{
				newContainer("best-effort", getResourceList("", ""), getResourceList("", "")),
				newContainer("best-effort", getResourceList("", ""), getResourceList("", "")),
			}),
			expected: v1.PodQOSBestEffort,
		},
		{
			pod: newPod("best-effort", []v1.Container{
				newContainer("best-effort", getResourceList("", ""), getResourceList("", "")),
			}),
			expected: v1.PodQOSBestEffort,
		},
		{
			pod: newPod("best-effort-burstable", []v1.Container{
				newContainer("best-effort", getResourceList("", ""), getResourceList("", "")),
				newContainer("burstable", getResourceList("1", ""), getResourceList("2", "")),
			}),
			expected: v1.PodQOSBurstable,
		},
		{
			pod: newPod("best-effort-guaranteed", []v1.Container{
				newContainer("best-effort", getResourceList("", ""), getResourceList("", "")),
				newContainer("guaranteed", getResourceList("10m", "100Mi"), getResourceList("10m", "100Mi")),
			}),
			expected: v1.PodQOSBurstable,
		},
		{
			pod: newPod("burstable-cpu-guaranteed-memory", []v1.Container{
				newContainer("burstable", getResourceList("", "100Mi"), getResourceList("", "100Mi")),
			}),
			expected: v1.PodQOSBurstable,
		},
		{
			pod: newPod("burstable-no-limits", []v1.Container{
				newContainer("burstable", getResourceList("100m", "100Mi"), getResourceList("", "")),
			}),
			expected: v1.PodQOSBurstable,
		},
		{
			pod: newPod("burstable-guaranteed", []v1.Container{
				newContainer("burstable", getResourceList("1", "100Mi"), getResourceList("2", "100Mi")),
				newContainer("guaranteed", getResourceList("100m", "100Mi"), getResourceList("100m", "100Mi")),
			}),
			expected: v1.PodQOSBurstable,
		},
		{
			pod: newPod("burstable-unbounded-but-requests-match-limits", []v1.Container{
				newContainer("burstable", getResourceList("100m", "100Mi"), getResourceList("200m", "200Mi")),
				newContainer("burstable-unbounded", getResourceList("100m", "100Mi"), getResourceList("", "")),
			}),
			expected: v1.PodQOSBurstable,
		},
		{
			pod: newPod("burstable-1", []v1.Container{
				newContainer("burstable", getResourceList("10m", "100Mi"), getResourceList("100m", "200Mi")),
			}),
			expected: v1.PodQOSBurstable,
		},
		{
			pod: newPod("burstable-2", []v1.Container{
				newContainer("burstable", getResourceList("0", "0"), getResourceList("100m", "200Mi")),
			}),
			expected: v1.PodQOSBurstable,
		},
		{
			pod: newPod("best-effort-hugepages", []v1.Container{
				newContainer("best-effort", addResource("hugepages-2Mi", "1Gi", getResourceList("0", "0")), addResource("hugepages-2Mi", "1Gi", getResourceList("0", "0"))),
			}),
			expected: v1.PodQOSBestEffort,
		},
		{
			pod: newPodWithInitContainers("init-container",
				[]v1.Container{
					newContainer("best-effort", getResourceList("", ""), getResourceList("", "")),
				},
				[]v1.Container{
					newContainer("burstable", getResourceList("10m", "100Mi"), getResourceList("100m", "200Mi")),
				}),
			expected: v1.PodQOSBurstable,
		},
		{
			pod: newPodWithResources(
				"guaranteed-with-pod-level-resources",
				[]v1.Container{
					newContainer("best-effort", getResourceList("", ""), getResourceList("", "")),
				},
				getResourceRequirements(getResourceList("10m", "100Mi"), getResourceList("10m", "100Mi")),
			),
			expected:                 v1.PodQOSGuaranteed,
			podLevelResourcesEnabled: true,
		},
		{
			pod: newPodWithResources(
				"guaranteed-with-pod-and-container-level-resources",
				[]v1.Container{
					newContainer("burstable", getResourceList("3m", "10Mi"), getResourceList("5m", "20Mi")),
				},
				getResourceRequirements(getResourceList("10m", "100Mi"), getResourceList("10m", "100Mi")),
			),
			expected:                 v1.PodQOSGuaranteed,
			podLevelResourcesEnabled: true,
		},
		{
			pod: newPodWithResources(
				"burstable-with-pod-level-resources",
				[]v1.Container{
					newContainer("best-effort", getResourceList("", ""), getResourceList("", "")),
				},
				getResourceRequirements(getResourceList("10m", "10Mi"), getResourceList("20m", "50Mi")),
			),
			expected:                 v1.PodQOSBurstable,
			podLevelResourcesEnabled: true,
		},
		{
			pod: newPodWithResources(
				"burstable-with-pod-and-container-level-resources",
				[]v1.Container{
					newContainer("burstable", getResourceList("5m", "10Mi"), getResourceList("5m", "10Mi")),
				},
				getResourceRequirements(getResourceList("10m", "10Mi"), getResourceList("20m", "50Mi")),
			),
			expected:                 v1.PodQOSBurstable,
			podLevelResourcesEnabled: true,
		},
		{
			pod: newPodWithResources(
				"burstable-with-pod-and-container-level-requests",
				[]v1.Container{
					newContainer("burstable", getResourceList("5m", "10Mi"), getResourceList("", "")),
				},
				getResourceRequirements(getResourceList("10m", "10Mi"), getResourceList("", "")),
			),
			expected:                 v1.PodQOSBurstable,
			podLevelResourcesEnabled: true,
		},
		{
			pod: newPodWithResources(
				"burstable-with-pod-and-container-level-resources-2",
				[]v1.Container{
					newContainer("burstable", getResourceList("5m", "10Mi"), getResourceList("", "")),
					newContainer("guaranteed", getResourceList("5m", "10Mi"), getResourceList("5m", "10Mi")),
				},
				getResourceRequirements(getResourceList("10m", "10Mi"), getResourceList("5m", "")),
			),
			expected:                 v1.PodQOSBurstable,
			podLevelResourcesEnabled: true,
		},
	}
	for id, testCase := range testCases {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, testCase.podLevelResourcesEnabled)
		if actual := ComputePodQOS(testCase.pod); testCase.expected != actual {
			t.Errorf("[%d]: invalid qos pod %s, expected: %s, actual: %s", id, testCase.pod.Name, testCase.expected, actual)
		}

		// Convert v1.Pod to core.Pod, and then check against `core.helper.ComputePodQOS`.
		pod := core.Pod{}
		corev1.Convert_v1_Pod_To_core_Pod(testCase.pod, &pod, nil)

		if actual := qos.ComputePodQOS(&pod); core.PodQOSClass(testCase.expected) != actual {
			t.Errorf("[%d]: conversion invalid qos pod %s, expected: %s, actual: %s", id, testCase.pod.Name, testCase.expected, actual)
		}
	}
}

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

func addResource(rName, value string, rl v1.ResourceList) v1.ResourceList {
	rl[v1.ResourceName(rName)] = resource.MustParse(value)
	return rl
}

func getResourceRequirements(requests, limits v1.ResourceList) *v1.ResourceRequirements {
	res := v1.ResourceRequirements{}
	res.Requests = requests
	res.Limits = limits
	return &res
}

func newContainer(name string, requests v1.ResourceList, limits v1.ResourceList) v1.Container {
	return v1.Container{
		Name:      name,
		Resources: *(getResourceRequirements(requests, limits)),
	}
}

func newPod(name string, containers []v1.Container) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: containers,
		},
	}
}

func newPodWithResources(name string, containers []v1.Container, podResources *v1.ResourceRequirements) *v1.Pod {
	pod := newPod(name, containers)
	if podResources != nil {
		pod.Spec.Resources = podResources
	}
	return pod
}

func newPodWithInitContainers(name string, containers []v1.Container, initContainers []v1.Container) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers:     containers,
			InitContainers: initContainers,
		},
	}
}
