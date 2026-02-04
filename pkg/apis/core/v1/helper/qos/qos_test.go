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
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
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
	guaranteedResources := v1.ResourceRequirements{
		// Ephemeral storage request should not affect QOS.
		Requests: v1.ResourceList{
			v1.ResourceCPU:              resource.MustParse("100m"),
			v1.ResourceMemory:           resource.MustParse("100Mi"),
			v1.ResourceEphemeralStorage: resource.MustParse("10Gi"),
		},
		Limits: getResourceList("100m", "100Mi"),
	}
	bestEffortResources := []v1.ResourceRequirements{
		{},
		{
			Requests: getResourceList("0", "0"),
		}, {
			Requests: v1.ResourceList{v1.ResourceEphemeralStorage: resource.MustParse("10Gi")},
		},
	}
	burstableResources := []v1.ResourceRequirements{
		{
			Requests: getResourceList("100m", "100Mi"),
			Limits:   getResourceList("200m", "200Mi"),
		}, {
			Requests: getResourceList("100m", "100Mi"),
			Limits:   getResourceList("100m", "200Mi"),
		}, {
			Requests: getResourceList("100m", "100Mi"),
			Limits:   getResourceList("200m", "100Mi"),
		}, {
			Requests: getResourceList("100m", "100Mi"),
		}, {
			Requests: getResourceList("100m", "100Mi"),
			Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
		},
	}

	type testCase struct {
		pod                      *v1.Pod
		expected                 v1.PodQOSClass
		podLevelResourcesEnabled bool
	}

	// Container level test cases
	singleContainerTests := []testCase{{
		pod: newPod("guaranteed", []v1.Container{
			{Resources: guaranteedResources},
		}),
		expected: v1.PodQOSGuaranteed,
	}}
	for i, res := range bestEffortResources {
		singleContainerTests = append(singleContainerTests, testCase{
			pod: newPod(fmt.Sprintf("best-effort-%d", i), []v1.Container{
				{Resources: res},
			}),
			expected: v1.PodQOSBestEffort,
		})
	}
	for i, res := range burstableResources {
		singleContainerTests = append(singleContainerTests, testCase{
			pod: newPod(fmt.Sprintf("burstable-%d", i), []v1.Container{
				{Resources: res},
			}),
			expected: v1.PodQOSBurstable,
		})
	}
	containerTests := slices.Clone(singleContainerTests)

	// 2 Container cases
	secondContainerTests := []testCase{{
		pod: newPod("2c-guaranteed", []v1.Container{
			{Resources: guaranteedResources},
			{Resources: guaranteedResources},
		}),
		expected: v1.PodQOSGuaranteed,
	}, {
		pod: newPod("2c-best-effort", []v1.Container{
			{Resources: bestEffortResources[0]},
			{Resources: bestEffortResources[1]},
		}),
		expected: v1.PodQOSBestEffort,
	}, {
		pod: newPod("2c-burstable", []v1.Container{
			{Resources: burstableResources[0]},
			{Resources: burstableResources[1]},
		}),
		expected: v1.PodQOSBurstable,
	}}
	// Mixed cases (all burstable)
	for i, res1 := range []v1.ResourceRequirements{guaranteedResources, bestEffortResources[0], burstableResources[0]} {
		for j, res2 := range []v1.ResourceRequirements{guaranteedResources, bestEffortResources[1], burstableResources[1]} {
			if i == j {
				continue
			}
			secondContainerTests = append(secondContainerTests, testCase{
				pod: newPod(fmt.Sprintf("2c-mixed-burstable-%d-%d", i, j), []v1.Container{
					{Resources: res1},
					{Resources: res2},
				}),
				expected: v1.PodQOSBurstable,
			})
		}
	}
	containerTests = append(containerTests, secondContainerTests...)

	// Add initContainer variant: doesn't matter if second container is an init container or not.
	for _, tc := range secondContainerTests {
		pod := tc.pod.DeepCopy()
		pod.Name += "-init"
		pod.Spec.InitContainers = pod.Spec.Containers[1:]
		pod.Spec.Containers = pod.Spec.Containers[:1]
		initTC := tc
		initTC.pod = pod
		containerTests = append(containerTests, initTC)
	}

	tests := slices.Clone(containerTests)

	// Enabling pod-level-resources without setting pod.spec.resources
	for _, tc := range containerTests {
		pod := tc.pod.DeepCopy()
		pod.Name += "-plr"
		plrTest := tc
		plrTest.pod = pod
		plrTest.podLevelResourcesEnabled = true
		tests = append(tests, plrTest)
	}

	// Pod level resources test cases
	for _, tc := range singleContainerTests {
		// variant 1: PLR enabled, guaranteed container with pod-level resources
		pod := tc.pod.DeepCopy()
		pod.Name = "pod-" + pod.Name + "-plr"
		pod.Spec.Resources = pod.Spec.Containers[0].Resources.DeepCopy()
		pod.Spec.Containers[0].Resources = guaranteedResources
		tests = append(tests, testCase{
			pod:                      pod,
			expected:                 tc.expected,
			podLevelResourcesEnabled: true,
		})

		// variant 2: PLR disabled, guaranteed pod
		pod = tc.pod.DeepCopy()
		pod.Name += "-pod-no-plr"
		pod.Spec.Resources = &guaranteedResources
		tests = append(tests, testCase{
			pod:                      pod,
			expected:                 tc.expected,
			podLevelResourcesEnabled: false,
		})
	}

	for _, testCase := range tests {
		t.Run(testCase.pod.Name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, testCase.podLevelResourcesEnabled)
			if actual := ComputePodQOS(testCase.pod); testCase.expected != actual {
				t.Errorf("ComputePodQOS error: expected: %s, actual: %s;\npod = %s", testCase.expected, actual, prettyPrintPod(testCase.pod))
			}

			// Convert v1.Pod to core.Pod, and then check against the internal version of `ComputePodQOS`.
			pod := core.Pod{}
			corev1.Convert_v1_Pod_To_core_Pod(testCase.pod, &pod, nil)

			if actual := v1.PodQOSClass(qos.ComputePodQOS(&pod)); testCase.expected != actual {
				t.Errorf("internal ComputePodQOS error: expected: %s, actual: %s;\npod = %s", testCase.expected, actual, prettyPrintPod(testCase.pod))
			}
		})
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

func prettyPrintPod(pod *v1.Pod) string {
	output, err := json.Marshal(pod)
	if err != nil {
		panic(err)
	}
	formatted := &bytes.Buffer{}
	if err := json.Indent(formatted, output, "", "  "); err != nil {
		panic(err)
	}
	return formatted.String()
}
