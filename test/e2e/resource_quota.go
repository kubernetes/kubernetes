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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// how long to wait for a resource quota update to occur
	resourceQuotaTimeout = 10 * time.Second
)

var _ = Describe("ResourceQuota [Conformance]", func() {
	f := NewFramework("resourcequota")

	It("should create a ResourceQuota and ensure its status is promptly calculated.", func() {
		By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		resourceQuota, err := createResourceQuota(f.Client, f.Namespace.Name, resourceQuota)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status is calculated")
		usedResources := api.ResourceList{}
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create a ResourceQuota and capture the life of a service.", func() {
		By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		resourceQuota, err := createResourceQuota(f.Client, f.Namespace.Name, resourceQuota)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a Service")
		service := newTestService("test-service")
		service, err = f.Client.Services(f.Namespace.Name).Create(service)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures service creation")
		usedResources := api.ResourceList{}
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		usedResources[api.ResourceServices] = resource.MustParse("1")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting a Service")
		err = f.Client.Services(f.Namespace.Name).Delete(service.Name)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released usage")
		usedResources[api.ResourceServices] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create a ResourceQuota and capture the life of a pod.", func() {
		By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		resourceQuota, err := createResourceQuota(f.Client, f.Namespace.Name, resourceQuota)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status is calculated")
		usedResources := api.ResourceList{}
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a Pod that fits quota")
		podName := "test-pod"
		requests := api.ResourceList{}
		requests[api.ResourceCPU] = resource.MustParse("500m")
		requests[api.ResourceMemory] = resource.MustParse("252Mi")
		pod := newTestPod(podName, requests, api.ResourceList{})
		pod, err = f.Client.Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures the pod usage")
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		usedResources[api.ResourcePods] = resource.MustParse("1")
		usedResources[api.ResourceCPU] = requests[api.ResourceCPU]
		usedResources[api.ResourceMemory] = requests[api.ResourceMemory]
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Not allowing a pod to be created that exceeds remaining quota")
		requests = api.ResourceList{}
		requests[api.ResourceCPU] = resource.MustParse("600m")
		requests[api.ResourceMemory] = resource.MustParse("100Mi")
		pod = newTestPod("fail-pod", requests, api.ResourceList{})
		pod, err = f.Client.Pods(f.Namespace.Name).Create(pod)
		Expect(err).To(HaveOccurred())

		By("Deleting the pod")
		err = f.Client.Pods(f.Namespace.Name).Delete(podName, api.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released the pod usage")
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		usedResources[api.ResourcePods] = resource.MustParse("0")
		usedResources[api.ResourceCPU] = resource.MustParse("0")
		usedResources[api.ResourceMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

})

// newTestResourceQuota returns a quota that enforces default constraints for testing
func newTestResourceQuota(name string) *api.ResourceQuota {
	hard := api.ResourceList{}
	hard[api.ResourcePods] = resource.MustParse("5")
	hard[api.ResourceServices] = resource.MustParse("10")
	hard[api.ResourceReplicationControllers] = resource.MustParse("10")
	hard[api.ResourceQuotas] = resource.MustParse("1")
	hard[api.ResourceCPU] = resource.MustParse("1")
	hard[api.ResourceMemory] = resource.MustParse("500Mi")
	return &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: name},
		Spec:       api.ResourceQuotaSpec{Hard: hard},
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
					Image: "gcr.io/google_containers/pause:2.0",
					Resources: api.ResourceRequirements{
						Requests: requests,
						Limits:   limits,
					},
				},
			},
		},
	}
}

// newTestService returns a simple service
func newTestService(name string) *api.Service {
	return &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{{
				Port:       80,
				TargetPort: intstr.FromInt(80),
			}},
		},
	}
}

// createResourceQuota in the specified namespace
func createResourceQuota(c *client.Client, namespace string, resourceQuota *api.ResourceQuota) (*api.ResourceQuota, error) {
	return c.ResourceQuotas(namespace).Create(resourceQuota)
}

// deleteResourceQuota with the specified name
func deleteResourceQuota(c *client.Client, namespace, name string) error {
	return c.ResourceQuotas(namespace).Delete(name)
}

// wait for resource quota status to show the expected used resources value
func waitForResourceQuota(c *client.Client, ns, quotaName string, used api.ResourceList) error {
	return wait.Poll(poll, resourceQuotaTimeout, func() (bool, error) {
		resourceQuota, err := c.ResourceQuotas(ns).Get(quotaName)
		if err != nil {
			return false, err
		}
		// used may not yet be calculated
		if resourceQuota.Status.Used == nil {
			return false, nil
		}
		// verify that the quota shows the expected used resource values
		for k, v := range used {
			if actualValue, found := resourceQuota.Status.Used[k]; !found || (actualValue.Cmp(v) != 0) {
				Logf("resource %s, expected %s, actual %s", k, v.String(), actualValue.String())
				return false, nil
			}
		}
		return true, nil
	})
}
