/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package limitranger

import (
	"strconv"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
)

func getResourceList(cpu, memory string) api.ResourceList {
	res := api.ResourceList{}
	if cpu != "" {
		res[api.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[api.ResourceMemory] = resource.MustParse(memory)
	}
	return res
}

func getResourceRequirements(limits, requests api.ResourceList) api.ResourceRequirements {
	res := api.ResourceRequirements{}
	res.Limits = limits
	res.Requests = requests
	return res
}

func validLimitRange() api.LimitRange {
	return api.LimitRange{
		ObjectMeta: api.ObjectMeta{
			Name: "abc",
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{
				{
					Type: api.LimitTypePod,
					Max:  getResourceList("200m", "4Gi"),
					Min:  getResourceList("50m", "2Mi"),
				},
				{
					Type:    api.LimitTypeContainer,
					Max:     getResourceList("100m", "2Gi"),
					Min:     getResourceList("25m", "1Mi"),
					Default: getResourceList("50m", "5Mi"),
				},
			},
		},
	}
}

func validPod(name string, numContainers int, resources api.ResourceRequirements) api.Pod {
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: name},
		Spec:       api.PodSpec{},
	}
	pod.Spec.Containers = make([]api.Container, 0, numContainers)
	for i := 0; i < numContainers; i++ {
		pod.Spec.Containers = append(pod.Spec.Containers, api.Container{
			Image:     "foo:V" + strconv.Itoa(i),
			Resources: resources,
		})
	}
	return pod
}

func TestDefaultContainerResourceRequirements(t *testing.T) {
	limitRange := validLimitRange()
	expected := api.ResourceRequirements{
		Limits:   getResourceList("50m", "5Mi"),
		Requests: api.ResourceList{},
	}

	actual := defaultContainerResourceRequirements(&limitRange)
	if !api.Semantic.DeepEqual(expected, actual) {
		t.Errorf("actual.Limits != expected.Limits; %v != %v", actual.Limits, expected.Limits)
		t.Errorf("actual.Requests != expected.Requests; %v != %v", actual.Requests, expected.Requests)
		t.Errorf("expected != actual; %v != %v", expected, actual)
	}
}

func TestMergePodResourceRequirements(t *testing.T) {
	limitRange := validLimitRange()

	// pod with no resources enumerated should get each resource from default
	expected := getResourceRequirements(getResourceList("", ""), getResourceList("", ""))
	pod := validPod("empty-resources", 1, expected)
	defaultRequirements := defaultContainerResourceRequirements(&limitRange)
	mergePodResourceRequirements(&pod, &defaultRequirements)
	for i := range pod.Spec.Containers {
		actual := pod.Spec.Containers[i].Resources
		if !api.Semantic.DeepEqual(expected, actual) {
			t.Errorf("pod %v, expected != actual; %v != %v", pod.Name, expected, actual)
		}
	}

	// pod with some resources enumerated should only merge empty
	input := getResourceRequirements(getResourceList("", "512Mi"), getResourceList("", ""))
	pod = validPod("limit-memory", 1, input)
	expected = api.ResourceRequirements{
		Limits: api.ResourceList{
			api.ResourceCPU:    defaultRequirements.Limits[api.ResourceCPU],
			api.ResourceMemory: resource.MustParse("512Mi"),
		},
		Requests: api.ResourceList{},
	}
	mergePodResourceRequirements(&pod, &defaultRequirements)
	for i := range pod.Spec.Containers {
		actual := pod.Spec.Containers[i].Resources
		if !api.Semantic.DeepEqual(expected, actual) {
			t.Errorf("pod %v, expected != actual; %v != %v", pod.Name, expected, actual)
		}
	}
}

func TestPodLimitFunc(t *testing.T) {
	limitRange := validLimitRange()
	successCases := []api.Pod{
		validPod("foo", 2, getResourceRequirements(getResourceList("100m", "2Gi"), getResourceList("", ""))),
		validPod("bar", 1, getResourceRequirements(getResourceList("100m", "2Gi"), getResourceList("", ""))),
	}

	errorCases := map[string]api.Pod{
		"min-container-cpu": validPod("foo", 1, getResourceRequirements(getResourceList("25m", "2Gi"), getResourceList("", ""))),
		"max-container-cpu": validPod("foo", 1, getResourceRequirements(getResourceList("110m", "1Gi"), getResourceList("", ""))),
		"min-container-mem": validPod("foo", 1, getResourceRequirements(getResourceList("30m", "0"), getResourceList("", ""))),
		"max-container-mem": validPod("foo", 1, getResourceRequirements(getResourceList("30m", "3Gi"), getResourceList("", ""))),
		"min-pod-cpu":       validPod("foo", 1, getResourceRequirements(getResourceList("40m", "2Gi"), getResourceList("", ""))),
		"max-pod-cpu":       validPod("foo", 4, getResourceRequirements(getResourceList("60m", "1Mi"), getResourceList("", ""))),
		"max-pod-memory":    validPod("foo", 3, getResourceRequirements(getResourceList("60m", "2Gi"), getResourceList("", ""))),
		"min-pod-memory":    validPod("foo", 3, getResourceRequirements(getResourceList("60m", "0"), getResourceList("", ""))),
	}

	for i := range successCases {
		err := PodLimitFunc(&limitRange, &successCases[i])
		if err != nil {
			t.Errorf("Unexpected error for valid pod: %v, %v", successCases[i].Name, err)
		}
	}

	for k, v := range errorCases {
		err := PodLimitFunc(&limitRange, &v)
		if err == nil {
			t.Errorf("Expected error for %s", k)
		}
	}
}

func TestPodLimitFuncApplyDefault(t *testing.T) {
	limitRange := validLimitRange()
	testPod := validPod("foo", 1, getResourceRequirements(api.ResourceList{}, api.ResourceList{}))
	err := PodLimitFunc(&limitRange, &testPod)
	if err != nil {
		t.Errorf("Unexpected error for valid pod: %v, %v", testPod.Name, err)
	}

	for i := range testPod.Spec.Containers {
		container := testPod.Spec.Containers[i]
		memory := testPod.Spec.Containers[i].Resources.Limits.Memory().String()
		cpu := testPod.Spec.Containers[i].Resources.Limits.Cpu().String()
		switch container.Image {
		case "boo:V1":
			if memory != "100Mi" {
				t.Errorf("Unexpected memory value %s", memory)
			}
			if cpu != "50m" {
				t.Errorf("Unexpected cpu value %s", cpu)
			}
		case "foo:V1":
			if memory != "2Gi" {
				t.Errorf("Unexpected memory value %s", memory)
			}
			if cpu != "100m" {
				t.Errorf("Unexpected cpu value %s", cpu)
			}
		}
	}
}
