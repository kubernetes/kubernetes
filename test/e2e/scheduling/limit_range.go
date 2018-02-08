/*
Copyright 2015 The Kubernetes Authors.

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

package scheduling

import (
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	podName = "pfpod"
)

var _ = SIGDescribe("LimitRange", func() {
	f := framework.NewDefaultFramework("limitrange")

	It("should create a LimitRange with defaults and ensure pod has those defaults applied.", func() {
		By("Creating a LimitRange")

		min := getResourceList("50m", "100Mi")
		max := getResourceList("500m", "500Mi")
		defaultLimit := getResourceList("500m", "500Mi")
		defaultRequest := getResourceList("100m", "200Mi")
		maxLimitRequestRatio := v1.ResourceList{}
		limitRange := newLimitRange("limit-range", v1.LimitTypeContainer,
			min, max,
			defaultLimit, defaultRequest,
			maxLimitRequestRatio)

		By("Setting up watch")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"name": limitRange.Name}))
		options := metav1.ListOptions{LabelSelector: selector.String()}
		limitRanges, err := f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).List(options)
		Expect(err).NotTo(HaveOccurred(), "failed to query for limitRanges")
		Expect(len(limitRanges.Items)).To(Equal(0))
		options = metav1.ListOptions{
			LabelSelector:   selector.String(),
			ResourceVersion: limitRanges.ListMeta.ResourceVersion,
		}
		w, err := f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Watch(metav1.ListOptions{})
		Expect(err).NotTo(HaveOccurred(), "failed to set up watch")

		By("Submitting a LimitRange")
		limitRange, err = f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Create(limitRange)
		Expect(err).NotTo(HaveOccurred())

		By("Verifying LimitRange creation was observed")
		select {
		case event, _ := <-w.ResultChan():
			if event.Type != watch.Added {
				framework.Failf("Failed to observe pod creation: %v", event)
			}
		case <-time.After(framework.ServiceRespondingTimeout):
			framework.Failf("Timeout while waiting for LimitRange creation")
		}

		By("Fetching the LimitRange to ensure it has proper values")
		limitRange, err = f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Get(limitRange.Name, metav1.GetOptions{})
		expected := v1.ResourceRequirements{Requests: defaultRequest, Limits: defaultLimit}
		actual := v1.ResourceRequirements{Requests: limitRange.Spec.Limits[0].DefaultRequest, Limits: limitRange.Spec.Limits[0].Default}
		err = equalResourceRequirement(expected, actual)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a Pod with no resource requirements")
		pod := f.NewTestPod("pod-no-resources", v1.ResourceList{}, v1.ResourceList{})
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring Pod has resource requirements applied from LimitRange")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(pod.Name, metav1.GetOptions{})
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
		pod = f.NewTestPod("pod-partial-resources", getResourceList("", "150Mi"), getResourceList("300m", ""))
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring Pod has merged resource requirements applied from LimitRange")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		// This is an interesting case, so it's worth a comment
		// If you specify a Limit, and no Request, the Limit will default to the Request
		// This means that the LimitRange.DefaultRequest will ONLY take affect if a container.resources.limit is not supplied
		expected = v1.ResourceRequirements{Requests: getResourceList("300m", "150Mi"), Limits: getResourceList("300m", "500Mi")}
		for i := range pod.Spec.Containers {
			err = equalResourceRequirement(expected, pod.Spec.Containers[i].Resources)
			if err != nil {
				// Print the pod to help in debugging.
				framework.Logf("Pod %+v does not have the expected requirements", pod)
				Expect(err).NotTo(HaveOccurred())
			}
		}

		By("Failing to create a Pod with less than min resources")
		pod = f.NewTestPod(podName, getResourceList("10m", "50Mi"), v1.ResourceList{})
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
		Expect(err).To(HaveOccurred())

		By("Failing to create a Pod with more than max resources")
		pod = f.NewTestPod(podName, getResourceList("600m", "600Mi"), v1.ResourceList{})
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
		Expect(err).To(HaveOccurred())

		By("Updating a LimitRange")
		newMin := getResourceList("9m", "49Mi")
		limitRange.Spec.Limits[0].Min = newMin
		limitRange, err = f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Update(limitRange)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a Pod with less than former min resources")
		pod = f.NewTestPod(podName, getResourceList("10m", "50Mi"), v1.ResourceList{})
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		By("Failing to create a Pod with more than max resources")
		pod = f.NewTestPod(podName, getResourceList("600m", "600Mi"), v1.ResourceList{})
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
		Expect(err).To(HaveOccurred())

		By("Deleting a LimitRange")
		err = f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Delete(limitRange.Name, metav1.NewDeleteOptions(30))
		Expect(err).NotTo(HaveOccurred())

		By("Verifying the LimitRange was deleted")
		Expect(wait.Poll(time.Second*5, time.Second*30, func() (bool, error) {
			selector := labels.SelectorFromSet(labels.Set(map[string]string{"name": limitRange.Name}))
			options := metav1.ListOptions{LabelSelector: selector.String()}
			limitRanges, err := f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).List(options)

			if err != nil {
				framework.Logf("Unable to retrieve LimitRanges: %v", err)
				return false, nil
			}

			if len(limitRanges.Items) == 0 {
				framework.Logf("limitRange is already deleted")
				return true, nil
			}

			if len(limitRanges.Items) > 0 {
				if limitRanges.Items[0].ObjectMeta.DeletionTimestamp == nil {
					framework.Logf("deletion has not yet been observed")
					return false, nil
				}
				return true, nil
			}

			return false, nil

		})).NotTo(HaveOccurred(), "kubelet never observed the termination notice")

		By("Creating a Pod with more than former max resources")
		pod = f.NewTestPod(podName+"2", getResourceList("600m", "600Mi"), v1.ResourceList{})
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())
	})

})

var _ = framework.KubeDescribe("LimitRange", func() {
	f := framework.NewDefaultFramework("limitrange")

	BeforeEach(func() {
		// only run the tests when LocalStorageCapacityIsolation feature is enabled
		framework.SkipUnlessLocalEphemeralStorageEnabled()
	})

	It("should create a LimitRange with default ephemeral storage and ensure pod has the default applied.", func() {
		By("Creating a LimitRange")

		min := getEphemeralStorageResourceList("100Mi")
		max := getEphemeralStorageResourceList("500Mi")
		defaultLimit := getEphemeralStorageResourceList("500Mi")
		defaultRequest := getEphemeralStorageResourceList("200Mi")
		maxLimitRequestRatio := v1.ResourceList{}
		limitRange := newLimitRange("limit-range", v1.LimitTypeContainer,
			min, max,
			defaultLimit, defaultRequest,
			maxLimitRequestRatio)
		limitRange, err := f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Create(limitRange)
		Expect(err).NotTo(HaveOccurred())

		defer func() {
			By("Removing limitrange")
			err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(limitRange.Name, nil)
			Expect(err).NotTo(HaveOccurred())
		}()

		By("Fetching the LimitRange to ensure it has proper values")
		limitRange, err = f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Get(limitRange.Name, metav1.GetOptions{})
		expected := v1.ResourceRequirements{Requests: defaultRequest, Limits: defaultLimit}
		actual := v1.ResourceRequirements{Requests: limitRange.Spec.Limits[0].DefaultRequest, Limits: limitRange.Spec.Limits[0].Default}
		err = equalResourceRequirement(expected, actual)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a Pod with no resource requirements")
		pod := f.NewTestPod("pod-no-resources", v1.ResourceList{}, v1.ResourceList{})
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		defer func() {
			By("Removing pod")
			err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(pod.Name, nil)
			Expect(err).NotTo(HaveOccurred())
		}()

		By("Ensuring Pod has resource requirements applied from LimitRange")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		for i := range pod.Spec.Containers {
			err = equalResourceRequirement(expected, pod.Spec.Containers[i].Resources)
			if err != nil {
				// Print the pod to help in debugging.
				framework.Logf("Pod %+v does not have the expected requirements", pod)
				Expect(err).NotTo(HaveOccurred())
			}
		}

		By("Creating a Pod with request")
		pod = f.NewTestPod("pod-partial-resources", getEphemeralStorageResourceList("150m"), v1.ResourceList{})
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		defer func() {
			By("Removing pod")
			err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(pod.Name, nil)
			Expect(err).NotTo(HaveOccurred())
		}()

		By("Ensuring Pod has merged resource requirements applied from LimitRange")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		// If you specify a Request, and no Limit, the Limit will be set to default limit
		expected = v1.ResourceRequirements{Requests: getEphemeralStorageResourceList("150Mi"), Limits: defaultLimit}
		for i := range pod.Spec.Containers {
			err = equalResourceRequirement(expected, pod.Spec.Containers[i].Resources)
			if err != nil {
				// Print the pod to help in debugging.
				framework.Logf("Pod %+v does not have the expected requirements", pod)
				Expect(err).NotTo(HaveOccurred())
			}
		}

		By("Creating a Pod with limit")
		pod = f.NewTestPod("pod-partial-resources", v1.ResourceList{}, getEphemeralStorageResourceList("300m"))
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		defer func() {
			By("Removing pod")
			err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(pod.Name, nil)
			Expect(err).NotTo(HaveOccurred())
		}()

		By("Ensuring Pod has merged resource requirements applied from LimitRange")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		// If you specify a Limit, and no Request, the Limit will default to the Request
		// This means that the LimitRange.DefaultRequest will ONLY take affect if a container.resources.limit is not supplied
		expected = v1.ResourceRequirements{Requests: getEphemeralStorageResourceList("300Mi"), Limits: getEphemeralStorageResourceList("300Mi")}
		for i := range pod.Spec.Containers {
			err = equalResourceRequirement(expected, pod.Spec.Containers[i].Resources)
			if err != nil {
				// Print the pod to help in debugging.
				framework.Logf("Pod %+v does not have the expected requirements", pod)
				Expect(err).NotTo(HaveOccurred())
			}
		}

		By("Failing to create a Pod with less than min resources")
		pod = f.NewTestPod(podName, getEphemeralStorageResourceList("50Mi"), v1.ResourceList{})
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
		Expect(err).To(HaveOccurred())

		By("Failing to create a Pod with more than max resources")
		pod = f.NewTestPod(podName, getEphemeralStorageResourceList("600Mi"), v1.ResourceList{})
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
		Expect(err).To(HaveOccurred())
	})

})

func getEphemeralStorageResourceList(ephemeralStorage string) v1.ResourceList {
	res := v1.ResourceList{}
	if ephemeralStorage != "" {
		res[v1.ResourceEphemeralStorage] = resource.MustParse(ephemeralStorage)
	}
	return res
}

func equalResourceRequirement(expected v1.ResourceRequirements, actual v1.ResourceRequirements) error {
	framework.Logf("Verifying requests: expected %v with actual %v", expected.Requests, actual.Requests)
	err := equalResourceList(expected.Requests, actual.Requests)
	if err != nil {
		return err
	}
	framework.Logf("Verifying limits: expected %v with actual %v", expected.Limits, actual.Limits)
	err = equalResourceList(expected.Limits, actual.Limits)
	return err
}

func equalResourceList(expected v1.ResourceList, actual v1.ResourceList) error {
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

// newLimitRange returns a limit range with specified data
func newLimitRange(name string, limitType v1.LimitType,
	min, max,
	defaultLimit, defaultRequest,
	maxLimitRequestRatio v1.ResourceList) *v1.LimitRange {
	return &v1.LimitRange{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.LimitRangeSpec{
			Limits: []v1.LimitRangeItem{
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
