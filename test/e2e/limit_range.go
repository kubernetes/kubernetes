/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("LimitRange", func() {
	f := framework.NewDefaultFramework("limitrange")

	It("should create a LimitRange with defaults and ensure pod has those defaults applied.", func() {
		By("Creating a LimitRange")

		min := getResourceList("50m", "100Mi")
		max := getResourceList("500m", "500Mi")
		defaultLimit := getResourceList("500m", "500Mi")
		defaultRequest := getResourceList("100m", "200Mi")
		maxLimitRequestRatio := api.ResourceList{}
		limitRange := newLimitRange("limit-range", api.LimitTypeContainer,
			min, max,
			defaultLimit, defaultRequest,
			maxLimitRequestRatio)
		limitRange, err := f.Client.LimitRanges(f.Namespace.Name).Create(limitRange)
		Expect(err).NotTo(HaveOccurred())

		By("Fetching the LimitRange to ensure it has proper values")
		limitRange, err = f.Client.LimitRanges(f.Namespace.Name).Get(limitRange.Name)
		expected := api.ResourceRequirements{Requests: defaultRequest, Limits: defaultLimit}
		actual := api.ResourceRequirements{Requests: limitRange.Spec.Limits[0].DefaultRequest, Limits: limitRange.Spec.Limits[0].Default}
		err = equalResourceRequirement(expected, actual)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a Pod with no resource requirements")
		pod := newTestPod("pod-no-resources", api.ResourceList{}, api.ResourceList{})
		pod, err = f.Client.Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring Pod has resource requirements applied from LimitRange")
		pod, err = f.Client.Pods(f.Namespace.Name).Get(pod.Name)
		Expect(err).NotTo(HaveOccurred())
		for i := range pod.Spec.Containers {
			err = equalResourceRequirement(expected, pod.Spec.Containers[i].Resources)
			if err != nil {
				// Print the pod to help in debugging.
				framework.Logf("Pod %+v does not have the expected requirements", pod)
				Expect(err).NotTo(HaveOccurred())
			}
		}

		By("Creating a Pod with partial resource requirements")
		pod = newTestPod("pod-partial-resources", getResourceList("", "150Mi"), getResourceList("300m", ""))
		pod, err = f.Client.Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring Pod has merged resource requirements applied from LimitRange")
		pod, err = f.Client.Pods(f.Namespace.Name).Get(pod.Name)
		Expect(err).NotTo(HaveOccurred())
		// This is an interesting case, so it's worth a comment
		// If you specify a Limit, and no Request, the Limit will default to the Request
		// This means that the LimitRange.DefaultRequest will ONLY take affect if a container.resources.limit is not supplied
		expected = api.ResourceRequirements{Requests: getResourceList("300m", "150Mi"), Limits: getResourceList("300m", "500Mi")}
		for i := range pod.Spec.Containers {
			err = equalResourceRequirement(expected, pod.Spec.Containers[i].Resources)
			if err != nil {
				// Print the pod to help in debugging.
				framework.Logf("Pod %+v does not have the expected requirements", pod)
				Expect(err).NotTo(HaveOccurred())
			}
		}

		By("Failing to create a Pod with less than min resources")
		pod = newTestPod(podName, getResourceList("10m", "50Mi"), api.ResourceList{})
		pod, err = f.Client.Pods(f.Namespace.Name).Create(pod)
		Expect(err).To(HaveOccurred())

		By("Failing to create a Pod with more than max resources")
		pod = newTestPod(podName, getResourceList("600m", "600Mi"), api.ResourceList{})
		pod, err = f.Client.Pods(f.Namespace.Name).Create(pod)
		Expect(err).To(HaveOccurred())
	})

})

func equalResourceRequirement(expected api.ResourceRequirements, actual api.ResourceRequirements) error {
	framework.Logf("Verifying requests: expected %s with actual %s", expected.Requests, actual.Requests)
	err := equalResourceList(expected.Requests, actual.Requests)
	if err != nil {
		return err
	}
	framework.Logf("Verifying limits: expected %v with actual %v", expected.Limits, actual.Limits)
	err = equalResourceList(expected.Limits, actual.Limits)
	if err != nil {
		return err
	}
	return nil
}

func equalResourceList(expected api.ResourceList, actual api.ResourceList) error {
	for k, v := range expected {
		if actualValue, found := actual[k]; !found || (v.Cmp(actualValue) != 0) {
			return fmt.Errorf("resource %v expected %v actual %v", k, v.String(), actualValue.String())
		}
	}
	for k, v := range actual {
		if expectedValue, found := expected[k]; !found || (v.Cmp(expectedValue) != 0) {
			return fmt.Errorf("resource %v expected %v actual %v", k, expectedValue.String(), v.String())
		}
	}
	return nil
}

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

// newLimitRange returns a limit range with specified data
func newLimitRange(name string, limitType api.LimitType,
	min, max,
	defaultLimit, defaultRequest,
	maxLimitRequestRatio api.ResourceList) *api.LimitRange {
	return &api.LimitRange{
		ObjectMeta: api.ObjectMeta{
			Name: name,
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

// newTestPod returns a pod that has the specified requests and limits
func newTestPod(name string, requests api.ResourceList, limits api.ResourceList) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "nginx",
					Image: "gcr.io/google_containers/pause-amd64:3.0",
					Resources: api.ResourceRequirements{
						Requests: requests,
						Limits:   limits,
					},
				},
			},
		},
	}
}
