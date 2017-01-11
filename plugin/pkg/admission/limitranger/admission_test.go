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
	"fmt"
	"strconv"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/controller/informers"
	kubeadmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
)

func getComputeResourceList(cpu, memory string) api.ResourceList {
	res := api.ResourceList{}
	if cpu != "" {
		res[api.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[api.ResourceMemory] = resource.MustParse(memory)
	}
	return res
}

func getStorageResourceList(storage string) api.ResourceList {
	res := api.ResourceList{}
	if storage != "" {
		res[api.ResourceStorage] = resource.MustParse(storage)
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
					Max:  getComputeResourceList("200m", "4Gi"),
					Min:  getComputeResourceList("50m", "2Mi"),
				},
				{
					Type:           api.LimitTypeContainer,
					Max:            getComputeResourceList("100m", "2Gi"),
					Min:            getComputeResourceList("25m", "1Mi"),
					Default:        getComputeResourceList("75m", "10Mi"),
					DefaultRequest: getComputeResourceList("50m", "5Mi"),
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
					Max:  getComputeResourceList("200m", "4Gi"),
					Min:  getComputeResourceList("50m", "2Mi"),
				},
				{
					Type: api.LimitTypeContainer,
					Max:  getComputeResourceList("100m", "2Gi"),
					Min:  getComputeResourceList("25m", "1Mi"),
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
		Requests: getComputeResourceList("50m", "5Mi"),
		Limits:   getComputeResourceList("75m", "10Mi"),
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
	expected := getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("", ""))
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
	input := getResourceRequirements(getComputeResourceList("", "512Mi"), getComputeResourceList("", ""))
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
	input = getResourceRequirements(getComputeResourceList("100m", "512Mi"), getComputeResourceList("200m", "1G"))
	initInputs := []api.ResourceRequirements{getResourceRequirements(getComputeResourceList("200m", "1G"), getComputeResourceList("400m", "2G"))}
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
			pod:        validPod("ctr-min-cpu-request", 1, getResourceRequirements(getComputeResourceList("100m", ""), getComputeResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getComputeResourceList("50m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-cpu-request-limit", 1, getResourceRequirements(getComputeResourceList("100m", ""), getComputeResourceList("200m", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getComputeResourceList("50m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-memory-request", 1, getResourceRequirements(getComputeResourceList("", "60Mi"), getComputeResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getComputeResourceList("", "50Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-memory-request-limit", 1, getResourceRequirements(getComputeResourceList("", "60Mi"), getComputeResourceList("", "100Mi"))),
			limitRange: createLimitRange(api.LimitTypeContainer, getComputeResourceList("", "50Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-cpu-request-limit", 1, getResourceRequirements(getComputeResourceList("500m", ""), getComputeResourceList("1", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getComputeResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-cpu-limit", 1, getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("1", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getComputeResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-mem-request-limit", 1, getResourceRequirements(getComputeResourceList("", "250Mi"), getComputeResourceList("", "500Mi"))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getComputeResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-cpu-ratio", 1, getResourceRequirements(getComputeResourceList("500m", ""), getComputeResourceList("750m", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, getComputeResourceList("1.5", "")),
		},
		{
			pod:        validPod("ctr-max-mem-limit", 1, getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("", "500Mi"))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getComputeResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-cpu-request", 2, getResourceRequirements(getComputeResourceList("75m", ""), getComputeResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypePod, getComputeResourceList("100m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-cpu-request-limit", 2, getResourceRequirements(getComputeResourceList("75m", ""), getComputeResourceList("200m", ""))),
			limitRange: createLimitRange(api.LimitTypePod, getComputeResourceList("100m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-memory-request", 2, getResourceRequirements(getComputeResourceList("", "60Mi"), getComputeResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypePod, getComputeResourceList("", "100Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-memory-request-limit", 2, getResourceRequirements(getComputeResourceList("", "60Mi"), getComputeResourceList("", "100Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, getComputeResourceList("", "100Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod: validPodInit(
				validPod("pod-init-min-memory-request", 2, getResourceRequirements(getComputeResourceList("", "60Mi"), getComputeResourceList("", ""))),
				getResourceRequirements(getComputeResourceList("", "100Mi"), getComputeResourceList("", "")),
			),
			limitRange: createLimitRange(api.LimitTypePod, getComputeResourceList("", "100Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod: validPodInit(
				validPod("pod-init-min-memory-request-limit", 2, getResourceRequirements(getComputeResourceList("", "60Mi"), getComputeResourceList("", "100Mi"))),
				getResourceRequirements(getComputeResourceList("", "80Mi"), getComputeResourceList("", "100Mi")),
			),
			limitRange: createLimitRange(api.LimitTypePod, getComputeResourceList("", "100Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-cpu-request-limit", 2, getResourceRequirements(getComputeResourceList("500m", ""), getComputeResourceList("1", ""))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getComputeResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-cpu-limit", 2, getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("1", ""))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getComputeResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod: validPodInit(
				validPod("pod-init-max-cpu-request-limit", 2, getResourceRequirements(getComputeResourceList("500m", ""), getComputeResourceList("1", ""))),
				getResourceRequirements(getComputeResourceList("1", ""), getComputeResourceList("2", "")),
				getResourceRequirements(getComputeResourceList("1", ""), getComputeResourceList("1", "")),
			),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getComputeResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod: validPodInit(
				validPod("pod-init-max-cpu-limit", 2, getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("1", ""))),
				getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("2", "")),
				getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("2", "")),
			),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getComputeResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-mem-request-limit", 2, getResourceRequirements(getComputeResourceList("", "250Mi"), getComputeResourceList("", "500Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getComputeResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-mem-limit", 2, getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("", "500Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getComputeResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-mem-ratio", 3, getResourceRequirements(getComputeResourceList("", "300Mi"), getComputeResourceList("", "450Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getComputeResourceList("", "2Gi"), api.ResourceList{}, api.ResourceList{}, getComputeResourceList("", "1.5")),
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
			pod:        validPod("ctr-min-cpu-request", 1, getResourceRequirements(getComputeResourceList("40m", ""), getComputeResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getComputeResourceList("50m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-cpu-request-limit", 1, getResourceRequirements(getComputeResourceList("40m", ""), getComputeResourceList("200m", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getComputeResourceList("50m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-cpu-no-request-limit", 1, getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getComputeResourceList("50m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-memory-request", 1, getResourceRequirements(getComputeResourceList("", "40Mi"), getComputeResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getComputeResourceList("", "50Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-memory-request-limit", 1, getResourceRequirements(getComputeResourceList("", "40Mi"), getComputeResourceList("", "100Mi"))),
			limitRange: createLimitRange(api.LimitTypeContainer, getComputeResourceList("", "50Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-min-memory-no-request-limit", 1, getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, getComputeResourceList("", "50Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-cpu-request-limit", 1, getResourceRequirements(getComputeResourceList("500m", ""), getComputeResourceList("2500m", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getComputeResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-cpu-limit", 1, getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("2500m", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getComputeResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-cpu-no-request-limit", 1, getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getComputeResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-cpu-ratio", 1, getResourceRequirements(getComputeResourceList("1250m", ""), getComputeResourceList("2500m", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, getComputeResourceList("1", "")),
		},
		{
			pod:        validPod("ctr-max-mem-request-limit", 1, getResourceRequirements(getComputeResourceList("", "250Mi"), getComputeResourceList("", "2Gi"))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getComputeResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-mem-limit", 1, getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("", "2Gi"))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getComputeResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("ctr-max-mem-no-request-limit", 1, getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypeContainer, api.ResourceList{}, getComputeResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-cpu-request", 1, getResourceRequirements(getComputeResourceList("75m", ""), getComputeResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypePod, getComputeResourceList("100m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-cpu-request-limit", 1, getResourceRequirements(getComputeResourceList("75m", ""), getComputeResourceList("200m", ""))),
			limitRange: createLimitRange(api.LimitTypePod, getComputeResourceList("100m", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-memory-request", 1, getResourceRequirements(getComputeResourceList("", "60Mi"), getComputeResourceList("", ""))),
			limitRange: createLimitRange(api.LimitTypePod, getComputeResourceList("", "100Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-min-memory-request-limit", 1, getResourceRequirements(getComputeResourceList("", "60Mi"), getComputeResourceList("", "100Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, getComputeResourceList("", "100Mi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-cpu-request-limit", 3, getResourceRequirements(getComputeResourceList("500m", ""), getComputeResourceList("1", ""))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getComputeResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-cpu-limit", 3, getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("1", ""))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getComputeResourceList("2", ""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-mem-request-limit", 3, getResourceRequirements(getComputeResourceList("", "250Mi"), getComputeResourceList("", "500Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getComputeResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-mem-limit", 3, getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("", "500Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getComputeResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod: validPodInit(
				validPod("pod-init-max-mem-limit", 1, getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("", "500Mi"))),
				getResourceRequirements(getComputeResourceList("", ""), getComputeResourceList("", "1.5Gi")),
			),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getComputeResourceList("", "1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pod:        validPod("pod-max-mem-ratio", 3, getResourceRequirements(getComputeResourceList("", "250Mi"), getComputeResourceList("", "500Mi"))),
			limitRange: createLimitRange(api.LimitTypePod, api.ResourceList{}, getComputeResourceList("", "2Gi"), api.ResourceList{}, api.ResourceList{}, getComputeResourceList("", "1.5")),
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
	limitRange := validLimitRangeNoDefaults()
	mockClient := newMockClientForTest([]api.LimitRange{limitRange})
	handler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	informerFactory.Start(wait.NeverStop)

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

func TestLimitRangerAdmitPod(t *testing.T) {
	limitRange := validLimitRangeNoDefaults()
	mockClient := newMockClientForTest([]api.LimitRange{limitRange})
	handler, informerFactory, err := newHandlerForTest(mockClient)
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	informerFactory.Start(wait.NeverStop)

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

// newMockClientForTest creates a mock client that returns a client configured for the specified list of limit ranges
func newMockClientForTest(limitRanges []api.LimitRange) *fake.Clientset {
	mockClient := &fake.Clientset{}
	mockClient.AddReactor("list", "limitranges", func(action core.Action) (bool, runtime.Object, error) {
		limitRangeList := &api.LimitRangeList{
			ListMeta: metav1.ListMeta{
				ResourceVersion: fmt.Sprintf("%d", len(limitRanges)),
			},
		}
		for index, value := range limitRanges {
			value.ResourceVersion = fmt.Sprintf("%d", index)
			limitRangeList.Items = append(limitRangeList.Items, value)
		}
		return true, limitRangeList, nil
	})
	return mockClient
}

// newHandlerForTest returns a handler configured for testing.
func newHandlerForTest(c clientset.Interface) (admission.Interface, informers.SharedInformerFactory, error) {
	f := informers.NewSharedInformerFactory(nil, c, 5*time.Minute)
	handler, err := NewLimitRanger(&DefaultLimitRangerActions{})
	if err != nil {
		return nil, f, err
	}
	pluginInitializer := kubeadmission.NewPluginInitializer(c, f, nil)
	pluginInitializer.Initialize(handler)
	err = admission.Validate(handler)
	return handler, f, err
}

func validPersistentVolumeClaim(name string, resources api.ResourceRequirements) api.PersistentVolumeClaim {
	pvc := api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: "test"},
		Spec: api.PersistentVolumeClaimSpec{
			Resources: resources,
		},
	}
	return pvc
}

func TestPersistentVolumeClaimLimitFunc(t *testing.T) {
	type testCase struct {
		pvc        api.PersistentVolumeClaim
		limitRange api.LimitRange
	}

	successCases := []testCase{
		{
			pvc:        validPersistentVolumeClaim("pvc-is-min-storage-request", getResourceRequirements(getStorageResourceList("1Gi"), getStorageResourceList(""))),
			limitRange: createLimitRange(api.LimitTypePersistentVolumeClaim, getStorageResourceList("1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pvc:        validPersistentVolumeClaim("pvc-is-max-storage-request", getResourceRequirements(getStorageResourceList("1Gi"), getStorageResourceList(""))),
			limitRange: createLimitRange(api.LimitTypePersistentVolumeClaim, api.ResourceList{}, getStorageResourceList("1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pvc:        validPersistentVolumeClaim("pvc-no-minmax-storage-request", getResourceRequirements(getStorageResourceList("100Gi"), getStorageResourceList(""))),
			limitRange: createLimitRange(api.LimitTypePersistentVolumeClaim, getStorageResourceList(""), getStorageResourceList(""), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pvc:        validPersistentVolumeClaim("pvc-within-minmax-storage-request", getResourceRequirements(getStorageResourceList("5Gi"), getStorageResourceList(""))),
			limitRange: createLimitRange(api.LimitTypePersistentVolumeClaim, getStorageResourceList("1Gi"), getStorageResourceList("10Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
	}
	for i := range successCases {
		test := successCases[i]
		err := PersistentVolumeClaimLimitFunc(&test.limitRange, &test.pvc)
		if err != nil {
			t.Errorf("Unexpected error for pvc: %s, %v", test.pvc.Name, err)
		}
	}

	errorCases := []testCase{
		{
			pvc:        validPersistentVolumeClaim("pvc-below-min-storage-request", getResourceRequirements(getStorageResourceList("500Mi"), getStorageResourceList(""))),
			limitRange: createLimitRange(api.LimitTypePersistentVolumeClaim, getStorageResourceList("1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
		{
			pvc:        validPersistentVolumeClaim("pvc-exceeds-max-storage-request", getResourceRequirements(getStorageResourceList("100Gi"), getStorageResourceList(""))),
			limitRange: createLimitRange(api.LimitTypePersistentVolumeClaim, getStorageResourceList("1Gi"), getStorageResourceList("1Gi"), api.ResourceList{}, api.ResourceList{}, api.ResourceList{}),
		},
	}
	for i := range errorCases {
		test := errorCases[i]
		err := PersistentVolumeClaimLimitFunc(&test.limitRange, &test.pvc)
		if err == nil {
			t.Errorf("Expected error for pvc: %s", test.pvc.Name)
		}
	}
}
