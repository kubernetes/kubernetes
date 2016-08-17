/*
Copyright 2014 The Kubernetes Authors.

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
	"time"

	"github.com/hashicorp/golang-lru"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
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

func getResourceRequirements(requests, limits api.ResourceList) api.ResourceRequirements {
	res := api.ResourceRequirements{}
	res.Requests = requests
	res.Limits = limits
	return res
}

// createLimitRange creates a limit range with the specified data
func createLimitRange(limitType api.LimitType, min, max, defaultLimit, defaultRequest, maxLimitRequestRatio api.ResourceList) api.LimitRange {
	return api.LimitRange{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: "test",
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{
				{
					Type:                 limitType,
					Min:                  min,
					Max:                  max,
					Default:              defaultLimit,
					DefaultRequest:       defaultRequest,
					MaxLimitRequestRatio: maxLimitRequestRatio,
				},
			},
		},
	}
}

func validLimitRange() api.LimitRange {
	return api.LimitRange{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: "test",
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{
				{
					Type: api.LimitTypePod,
					Max:  getResourceList("200m", "4Gi"),
					Min:  getResourceList("50m", "2Mi"),
				},
				{
					Type:           api.LimitTypeContainer,
					Max:            getResourceList("100m", "2Gi"),
					Min:            getResourceList("25m", "1Mi"),
					Default:        getResourceList("75m", "10Mi"),
					DefaultRequest: getResourceList("50m", "5Mi"),
				},
			},
		},
	}
}

func validLimitRangeNoDefaults() api.LimitRange {
	return api.LimitRange{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: "test",
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{
				{
					Type: api.LimitTypePod,
					Max:  getResourceList("200m", "4Gi"),
					Min:  getResourceList("50m", "2Mi"),
				},
				{
					Type: api.LimitTypeContainer,
					Max:  getResourceList("100m", "2Gi"),
					Min:  getResourceList("25m", "1Mi"),
				},
			},
		},
	}
}

func validPod(name string, numContainers int, resources api.ResourceRequirements) api.Pod {
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: "test"},
		Spec:       api.PodSpec{},
	}
	pod.Spec.Containers = make([]api.Container, 0, numContainers)
	for i := 0; i < numContainers; i++ {
		pod.Spec.Containers = append(pod.Spec.Containers, api.Container{
			Image:     "foo:V" + strconv.Itoa(i),
			Resources: resources,
			Name:      "foo-" + strconv.Itoa(i),
		})
	}
	return pod
}

func validPodInit(pod api.Pod, resources ...api.ResourceRequirements) api.Pod {
	for i := 0; i < len(resources); i++ {
		pod.Spec.InitContainers = append(pod.Spec.InitContainers, api.Container{
			Image:     "foo:V" + strconv.Itoa(i),
			Resources: resources[i],
			Name:      "foo-" + strconv.Itoa(i),
		})
	}
	return pod
}

func TestDefaultContainerResourceRequirements(t *testing.T) {
	limitRange := validLimitRange()
	expected := api.ResourceRequirements{
		Requests: getResourceList("50m", "5Mi"),
		Limits:   getResourceList("75m", "10Mi"),
	}

	actual := defaultContainerResourceRequirements(&limitRange)
	if !api.Semantic.DeepEqual(expected, actual) {
		t.Errorf("actual.Limits != expected.Limits; %v != %v", actual.Limits, expected.Limits)
		t.Errorf("actual.Requests != expected.Requests; %v != %v", actual.Requests, expected.Requests)
		t.Errorf("expected != actual; %v != %v", expected, actual)
	}
}

func verifyAnnotation(t *testing.T, pod *api.Pod, expected string) {
	a, ok := pod.ObjectMeta.Annotations[limitRangerAnnotation]
	if !ok {
		t.Errorf("No annotation but expected %v", expected)
	}
	if a != expected {
		t.Errorf("Wrong annotation set by Limit Ranger: got %v, expected %v", a, expected)
	}
}

func expectNoAnnotation(t *testing.T, pod *api.Pod) {
	if a, ok := pod.ObjectMeta.Annotations[limitRangerAnnotation]; ok {
		t.Errorf("Expected no annotation but got %v", a)
	}
}

func TestMergePodResourceRequirements(t *testing.T) {
	limitRange := validLimitRange()

	// pod with no resources enumerated should get each resource from default request
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
	verifyAnnotation(t, &pod, "LimitRanger plugin set: cpu, memory request for container foo-0; cpu, memory limit for container foo-0")

	// pod with some resources enumerated should only merge empty
	input := getResourceRequirements(getResourceList("", "512Mi"), getResourceList("", ""))
	pod = validPodInit(validPod("limit-memory", 1, input), input)
	expected = api.ResourceRequirements{
		Requests: api.ResourceList{
			api.ResourceCPU:    defaultRequirements.Requests[api.ResourceCPU],
			api.ResourceMemory: resource.MustParse("512Mi"),
		},
		Limits: defaultRequirements.Limits,
	}
	mergePodResourceRequirements(&pod, &defaultRequirements)
	for i := range pod.Spec.Containers {
		actual := pod.Spec.Containers[i].Resources
		if !api.Semantic.DeepEqual(expected, actual) {
			t.Errorf("pod %v, expected != actual; %v != %v", pod.Name, expected, actual)
		}
	}
	for i := range pod.Spec.InitContainers {
		actual := pod.Spec.InitContainers[i].Resources
		if !api.Semantic.DeepEqual(expected, actual) {
			t.Errorf("pod %v, expected != actual; %v != %v", pod.Name, expected, actual)
		}
	}
	verifyAnnotation(t, &pod, "LimitRanger plugin set: cpu request for container foo-0; cpu, memory limit for container foo-0")

	// pod with all resources enumerated should not merge anything
	input = getResourceRequirements(getResourceList("100m", "512Mi"), getResourceList("200m", "1G"))
	initInputs := []api.ResourceRequirements{getResourceRequirements(getResourceList("200m", "1G"), getResourceList("400m", "2G"))}
	pod = validPodInit(validPod("limit-memory", 1, input), initInputs...)
	expected = input
	mergePodResourceRequirements(&pod, &defaultRequirements)
	for i := range pod.Spec.Containers {
		actual := pod.Spec.Containers[i].Resources
		if !api.Semantic.DeepEqual(expected, actual) {
			t.Errorf("pod %v, expected != actual; %v != %v", pod.Name, expected, actual)
		}
	}
	for i := range pod.Spec.InitContainers {
		actual := pod.Spec.InitContainers[i].Resources
		if !api.Semantic.DeepEqual(initInputs[i], actual) {
			t.Errorf("pod %v, expected != actual; %v != %v", pod.Name, initInputs[i], actual)
		}
	}
	expectNoAnnotation(t, &pod)
}

func TestPodLimitFunc(t *testing.T) {
	type testCase struct {
		pod        api.Pod
		limitRange api.LimitRange
	}

	successCases := []testCase{
		{
			pod:        validPod("ctr-min-cpu-request", 1, getResourceRequirements(getResourceList("100m", ""), getResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getResourceList("50m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-cpu-request-limit", 1, getResourceRequirements(getResourceList("100m", ""), getResourceList("200m", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getResourceList("50m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-memory-request", 1, getResourceRequirements(getResourceList("", "60Mi"), getResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getResourceList("", "50Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-memory-request-limit", 1, getResourceRequirements(getResourceList("", "60Mi"), getResourceList("", "100Mi"))),
			limitRange: createLimitRange(api.LimitTypeContainer, getResourceList("", "50Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-cpu-request-limit", 1, getResourceRequirements(getResourceList("500m", ""), getResourceList("1", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-cpu-limit", 1, getResourceRequirements(getResourceList("", ""), getResourceList("1", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-mem-request-limit", 1, getResourceRequirements(getResourceList("", "250Mi"), getResourceList("", "500Mi"))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-cpu-ratio", 1, getResourceRequirements(getResourceList("500m", ""), getResourceList("750m", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, getResourceList("1.5", "")),
		},
		{
			pod:        validPod("ctr-max-mem-limit", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", "500Mi"))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-cpu-request", 2, getResourceRequirements(getResourceList("75m", ""), getResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypePod, getResourceList("100m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-cpu-request-limit", 2, getResourceRequirements(getResourceList("75m", ""), getResourceList("200m", ""))),
			limitRange: createLimitRange(api.LimitTypePod, getResourceList("100m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-memory-request", 2, getResourceRequirements(getResourceList("", "60Mi"), getResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypePod, getResourceList("", "100Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-memory-request-limit", 2, getResourceRequirements(getResourceList("", "60Mi"), getResourceList("", "100Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, getResourceList("", "100Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod: validPodInit(
				validPod("pod-init-min-memory-request", 2, getResourceRequirements(getResourceList("", "60Mi"), getResourceList("", ""))),
				getResourceRequirements(getResourceList("", "100Mi"), getResourceList("", "")),
			),
			limitRange: createLimitRange(api.LimitTypePod, getResourceList("", "100Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod: validPodInit(
				validPod("pod-init-min-memory-request-limit", 2, getResourceRequirements(getResourceList("", "60Mi"), getResourceList("", "100Mi"))),
				getResourceRequirements(getResourceList("", "80Mi"), getResourceList("", "100Mi")),
			),
			limitRange: createLimitRange(api.LimitTypePod, getResourceList("", "100Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-cpu-request-limit", 2, getResourceRequirements(getResourceList("500m", ""), getResourceList("1", ""))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-cpu-limit", 2, getResourceRequirements(getResourceList("", ""), getResourceList("1", ""))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod: validPodInit(
				validPod("pod-init-max-cpu-request-limit", 2, getResourceRequirements(getResourceList("500m", ""), getResourceList("1", ""))),
				getResourceRequirements(getResourceList("1", ""), getResourceList("2", "")),
				getResourceRequirements(getResourceList("1", ""), getResourceList("1", "")),
			),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod: validPodInit(
				validPod("pod-init-max-cpu-limit", 2, getResourceRequirements(getResourceList("", ""), getResourceList("1", ""))),
				getResourceRequirements(getResourceList("", ""), getResourceList("2", "")),
				getResourceRequirements(getResourceList("", ""), getResourceList("2", "")),
			),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-mem-request-limit", 2, getResourceRequirements(getResourceList("", "250Mi"), getResourceList("", "500Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-mem-limit", 2, getResourceRequirements(getResourceList("", ""), getResourceList("", "500Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-mem-ratio", 3, getResourceRequirements(getResourceList("", "300Mi"), getResourceList("", "450Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getResourceList("", "2Gi"), api.ResourceList{}, api.ResourceList{}, getResourceList("", "1.5")),
		},
	}
	for i := range successCases {
		test := successCases[i]
		err := PodLimitFunc(&test.limitRange, &test.pod)
		if err != nil {
			t.Errorf("Unexpected error for pod: %s, %v", test.pod.Name, err)
		}
	}

	errorCases := []testCase{
		{
			pod:        validPod("ctr-min-cpu-request", 1, getResourceRequirements(getResourceList("40m", ""), getResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getResourceList("50m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-cpu-request-limit", 1, getResourceRequirements(getResourceList("40m", ""), getResourceList("200m", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getResourceList("50m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-cpu-no-request-limit", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getResourceList("50m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-memory-request", 1, getResourceRequirements(getResourceList("", "40Mi"), getResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getResourceList("", "50Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-memory-request-limit", 1, getResourceRequirements(getResourceList("", "40Mi"), getResourceList("", "100Mi"))),
			limitRange: createLimitRange(api.LimitTypeContainer, getResourceList("", "50Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-memory-no-request-limit", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getResourceList("", "50Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-cpu-request-limit", 1, getResourceRequirements(getResourceList("500m", ""), getResourceList("2500m", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-cpu-limit", 1, getResourceRequirements(getResourceList("", ""), getResourceList("2500m", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-cpu-no-request-limit", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-cpu-ratio", 1, getResourceRequirements(getResourceList("1250m", ""), getResourceList("2500m", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, getResourceList("1", "")),
		},
		{
			pod:        validPod("ctr-max-mem-request-limit", 1, getResourceRequirements(getResourceList("", "250Mi"), getResourceList("", "2Gi"))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-mem-limit", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", "2Gi"))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-mem-no-request-limit", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-cpu-request", 1, getResourceRequirements(getResourceList("75m", ""), getResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypePod, getResourceList("100m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-cpu-request-limit", 1, getResourceRequirements(getResourceList("75m", ""), getResourceList("200m", ""))),
			limitRange: createLimitRange(api.LimitTypePod, getResourceList("100m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-memory-request", 1, getResourceRequirements(getResourceList("", "60Mi"), getResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypePod, getResourceList("", "100Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-memory-request-limit", 1, getResourceRequirements(getResourceList("", "60Mi"), getResourceList("", "100Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, getResourceList("", "100Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-cpu-request-limit", 3, getResourceRequirements(getResourceList("500m", ""), getResourceList("1", ""))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-cpu-limit", 3, getResourceRequirements(getResourceList("", ""), getResourceList("1", ""))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-mem-request-limit", 3, getResourceRequirements(getResourceList("", "250Mi"), getResourceList("", "500Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-mem-limit", 3, getResourceRequirements(getResourceList("", ""), getResourceList("", "500Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod: validPodInit(
				validPod("pod-init-max-mem-limit", 1, getResourceRequirements(getResourceList("", ""), getResourceList("", "500Mi"))),
				getResourceRequirements(getResourceList("", ""), getResourceList("", "1.5Gi")),
			),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-mem-ratio", 3, getResourceRequirements(getResourceList("", "250Mi"), getResourceList("", "500Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getResourceList("", "2Gi"), api.ResourceList{}, api.ResourceList{}, getResourceList("", "1.5")),
		},
	}
	for i := range errorCases {
		test := errorCases[i]
		err := PodLimitFunc(&test.limitRange, &test.pod)
		if err == nil {
			t.Errorf("Expected error for pod: %s", test.pod.Name)
		}
	}
}

func TestPodLimitFuncApplyDefault(t *testing.T) {
	limitRange := validLimitRange()
	testPod := validPodInit(validPod("foo", 1, getResourceRequirements(api.ResourceList{}, api.ResourceList{})), getResourceRequirements(api.ResourceList{}, api.ResourceList{}))
	err := PodLimitFunc(&limitRange, &testPod)
	if err != nil {
		t.Errorf("Unexpected error for valid pod: %v, %v", testPod.Name, err)
	}

	for i := range testPod.Spec.Containers {
		container := testPod.Spec.Containers[i]
		limitMemory := container.Resources.Limits.Memory().String()
		limitCpu := container.Resources.Limits.Cpu().String()
		requestMemory := container.Resources.Requests.Memory().String()
		requestCpu := container.Resources.Requests.Cpu().String()

		if limitMemory != "10Mi" {
			t.Errorf("Unexpected memory value %s", limitMemory)
		}
		if limitCpu != "75m" {
			t.Errorf("Unexpected cpu value %s", limitCpu)
		}
		if requestMemory != "5Mi" {
			t.Errorf("Unexpected memory value %s", requestMemory)
		}
		if requestCpu != "50m" {
			t.Errorf("Unexpected cpu value %s", requestCpu)
		}
	}

	for i := range testPod.Spec.InitContainers {
		container := testPod.Spec.InitContainers[i]
		limitMemory := container.Resources.Limits.Memory().String()
		limitCpu := container.Resources.Limits.Cpu().String()
		requestMemory := container.Resources.Requests.Memory().String()
		requestCpu := container.Resources.Requests.Cpu().String()

		if limitMemory != "10Mi" {
			t.Errorf("Unexpected memory value %s", limitMemory)
		}
		if limitCpu != "75m" {
			t.Errorf("Unexpected cpu value %s", limitCpu)
		}
		if requestMemory != "5Mi" {
			t.Errorf("Unexpected memory value %s", requestMemory)
		}
		if requestCpu != "50m" {
			t.Errorf("Unexpected cpu value %s", requestCpu)
		}
	}
}

func TestLimitRangerIgnoresSubresource(t *testing.T) {
	client := fake.NewSimpleClientset()
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	handler := &limitRanger{
		Handler: admission.NewHandler(admission.Create, admission.Update),
		client:  client,
		actions: &DefaultLimitRangerActions{},
		indexer: indexer,
	}

	limitRange := validLimitRangeNoDefaults()
	testPod := validPod("testPod", 1, api.ResourceRequirements{})

	indexer.Add(&limitRange)
	err := handler.Admit(admission.NewAttributesRecord(&testPod, nil, api.Kind("Pod").WithVersion("version"), limitRange.Namespace, "testPod", api.Resource("pods").WithVersion("version"), "", admission.Update, nil))
	if err == nil {
		t.Errorf("Expected an error since the pod did not specify resource limits in its update call")
	}

	err = handler.Admit(admission.NewAttributesRecord(&testPod, nil, api.Kind("Pod").WithVersion("version"), limitRange.Namespace, "testPod", api.Resource("pods").WithVersion("version"), "status", admission.Update, nil))
	if err != nil {
		t.Errorf("Should have ignored calls to any subresource of pod %v", err)
	}

}

func TestLimitRangerCacheMisses(t *testing.T) {
	liveLookupCache, err := lru.New(10000)
	if err != nil {
		t.Fatal(err)
	}

	client := fake.NewSimpleClientset()
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	handler := &limitRanger{
		Handler:         admission.NewHandler(admission.Create, admission.Update),
		client:          client,
		actions:         &DefaultLimitRangerActions{},
		indexer:         indexer,
		liveLookupCache: liveLookupCache,
	}

	limitRange := validLimitRangeNoDefaults()
	testPod := validPod("testPod", 1, api.ResourceRequirements{})

	// add to the lru cache
	liveLookupCache.Add(limitRange.Namespace, liveLookupEntry{expiry: time.Now().Add(time.Duration(30 * time.Second)), items: []*api.LimitRange{&limitRange}})

	err = handler.Admit(admission.NewAttributesRecord(&testPod, nil, api.Kind("Pod").WithVersion("version"), limitRange.Namespace, "testPod", api.Resource("pods").WithVersion("version"), "", admission.Update, nil))
	if err == nil {
		t.Errorf("Expected an error since the pod did not specify resource limits in its update call")
	}

	err = handler.Admit(admission.NewAttributesRecord(&testPod, nil, api.Kind("Pod").WithVersion("version"), limitRange.Namespace, "testPod", api.Resource("pods").WithVersion("version"), "status", admission.Update, nil))
	if err != nil {
		t.Errorf("Should have ignored calls to any subresource of pod %v", err)
	}
}

func TestLimitRangerCacheAndLRUMisses(t *testing.T) {
	liveLookupCache, err := lru.New(10000)
	if err != nil {
		t.Fatal(err)
	}

	limitRange := validLimitRangeNoDefaults()
	client := fake.NewSimpleClientset(&limitRange)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	handler := &limitRanger{
		Handler:         admission.NewHandler(admission.Create, admission.Update),
		client:          client,
		actions:         &DefaultLimitRangerActions{},
		indexer:         indexer,
		liveLookupCache: liveLookupCache,
	}

	testPod := validPod("testPod", 1, api.ResourceRequirements{})

	err = handler.Admit(admission.NewAttributesRecord(&testPod, nil, api.Kind("Pod").WithVersion("version"), limitRange.Namespace, "testPod", api.Resource("pods").WithVersion("version"), "", admission.Update, nil))
	if err == nil {
		t.Errorf("Expected an error since the pod did not specify resource limits in its update call")
	}

	err = handler.Admit(admission.NewAttributesRecord(&testPod, nil, api.Kind("Pod").WithVersion("version"), limitRange.Namespace, "testPod", api.Resource("pods").WithVersion("version"), "status", admission.Update, nil))
	if err != nil {
		t.Errorf("Should have ignored calls to any subresource of pod %v", err)
	}
}

func TestLimitRangerCacheAndLRUExpiredMisses(t *testing.T) {
	liveLookupCache, err := lru.New(10000)
	if err != nil {
		t.Fatal(err)
	}

	limitRange := validLimitRangeNoDefaults()
	client := fake.NewSimpleClientset(&limitRange)
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	handler := &limitRanger{
		Handler:         admission.NewHandler(admission.Create, admission.Update),
		client:          client,
		actions:         &DefaultLimitRangerActions{},
		indexer:         indexer,
		liveLookupCache: liveLookupCache,
	}

	testPod := validPod("testPod", 1, api.ResourceRequirements{})

	// add to the lru cache
	liveLookupCache.Add(limitRange.Namespace, liveLookupEntry{expiry: time.Now().Add(time.Duration(-30 * time.Second)), items: []*api.LimitRange{}})

	err = handler.Admit(admission.NewAttributesRecord(&testPod, nil, api.Kind("Pod").WithVersion("version"), limitRange.Namespace, "testPod", api.Resource("pods").WithVersion("version"), "", admission.Update, nil))
	if err == nil {
		t.Errorf("Expected an error since the pod did not specify resource limits in its update call")
	}

	err = handler.Admit(admission.NewAttributesRecord(&testPod, nil, api.Kind("Pod").WithVersion("version"), limitRange.Namespace, "testPod", api.Resource("pods").WithVersion("version"), "status", admission.Update, nil))
	if err != nil {
		t.Errorf("Should have ignored calls to any subresource of pod %v", err)
	}
}
