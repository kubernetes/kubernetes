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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// how long to wait for a resource quota update to occur
	resourceQuotaTimeout = 30 * time.Second
)

var _ = framework.KubeDescribe("ResourceQuota", func() {
	f := framework.NewDefaultFramework("resourcequota")

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

		By("Ensuring resource quota status is calculated")
		usedResources := api.ResourceList{}
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a Service")
		service := newTestServiceForQuota("test-service", api.ServiceTypeClusterIP)
		service, err = f.Client.Services(f.Namespace.Name).Create(service)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures service creation")
		usedResources = api.ResourceList{}
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

	It("should create a ResourceQuota and capture the life of a secret.", func() {
		By("Discovering how many secrets are in namespace by default")
		secrets, err := f.Client.Secrets(f.Namespace.Name).List(api.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		defaultSecrets := fmt.Sprintf("%d", len(secrets.Items))
		hardSecrets := fmt.Sprintf("%d", len(secrets.Items)+1)

		By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		resourceQuota.Spec.Hard[api.ResourceSecrets] = resource.MustParse(hardSecrets)
		resourceQuota, err = createResourceQuota(f.Client, f.Namespace.Name, resourceQuota)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status is calculated")
		usedResources := api.ResourceList{}
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		usedResources[api.ResourceSecrets] = resource.MustParse(defaultSecrets)
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a Secret")
		secret := newTestSecretForQuota("test-secret")
		secret, err = f.Client.Secrets(f.Namespace.Name).Create(secret)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures secret creation")
		usedResources = api.ResourceList{}
		usedResources[api.ResourceSecrets] = resource.MustParse(hardSecrets)
		// we expect there to be two secrets because each namespace will receive
		// a service account token secret by default
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting a secret")
		err = f.Client.Secrets(f.Namespace.Name).Delete(secret.Name)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released usage")
		usedResources[api.ResourceSecrets] = resource.MustParse(defaultSecrets)
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create a ResourceQuota and capture the life of a nodePort service.", func() {
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

		By("Creating a NodePort type Service")
		service := newTestServiceForQuota("test-service", api.ServiceTypeNodePort)
		service, err = f.Client.Services(f.Namespace.Name).Create(service)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures service creation")
		usedResources = api.ResourceList{}
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		usedResources[api.ResourceServices] = resource.MustParse("1")
		usedResources[api.ResourceServicesNodePorts] = resource.MustParse("1")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting a Service")
		err = f.Client.Services(f.Namespace.Name).Delete(service.Name)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released usage")
		usedResources[api.ResourceServices] = resource.MustParse("0")
		usedResources[api.ResourceServicesNodePorts] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create a ResourceQuota and capture the life of a nodePort service updated to clusterIP.", func() {
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

		By("Creating a NodePort type Service")
		service := newTestServiceForQuota("test-service", api.ServiceTypeNodePort)
		service, err = f.Client.Services(f.Namespace.Name).Create(service)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures service creation")
		usedResources = api.ResourceList{}
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		usedResources[api.ResourceServices] = resource.MustParse("1")
		usedResources[api.ResourceServicesNodePorts] = resource.MustParse("1")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Updating the service type to clusterIP")
		service.Spec.Type = api.ServiceTypeClusterIP
		service.Spec.Ports[0].NodePort = 0
		_, err = f.Client.Services(f.Namespace.Name).Update(service)
		Expect(err).NotTo(HaveOccurred())

		By("Checking resource quota status capture service update")
		usedResources = api.ResourceList{}
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		usedResources[api.ResourceServices] = resource.MustParse("1")
		usedResources[api.ResourceServicesNodePorts] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting a Service")
		err = f.Client.Services(f.Namespace.Name).Delete(service.Name)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released usage")
		usedResources[api.ResourceServices] = resource.MustParse("0")
		usedResources[api.ResourceServicesNodePorts] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create a ResourceQuota and capture the life of a loadBalancer service.", func() {
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

		By("Creating a loadBalancer type Service")
		service := newTestServiceForQuota("test-service", api.ServiceTypeLoadBalancer)
		service, err = f.Client.Services(f.Namespace.Name).Create(service)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures service creation")
		usedResources = api.ResourceList{}
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		usedResources[api.ResourceServices] = resource.MustParse("1")
		usedResources[api.ResourceServicesLoadBalancers] = resource.MustParse("1")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting a Service")
		err = f.Client.Services(f.Namespace.Name).Delete(service.Name)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released usage")
		usedResources[api.ResourceServices] = resource.MustParse("0")
		usedResources[api.ResourceServicesLoadBalancers] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create a ResourceQuota and capture the life of a nodePort service updated to loadBalancer.", func() {
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

		By("Creating a nodePort type Service")
		service := newTestServiceForQuota("test-service", api.ServiceTypeNodePort)
		service, err = f.Client.Services(f.Namespace.Name).Create(service)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures service creation")
		usedResources = api.ResourceList{}
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		usedResources[api.ResourceServices] = resource.MustParse("1")
		usedResources[api.ResourceServicesLoadBalancers] = resource.MustParse("0")
		usedResources[api.ResourceServicesNodePorts] = resource.MustParse("1")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Updating the service type to loadBalancer")
		service.Spec.Type = api.ServiceTypeLoadBalancer
		service.Spec.Ports[0].NodePort = 0
		_, err = f.Client.Services(f.Namespace.Name).Update(service)
		Expect(err).NotTo(HaveOccurred())

		By("Checking resource quota status capture service update")
		usedResources = api.ResourceList{}
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		usedResources[api.ResourceServices] = resource.MustParse("1")
		usedResources[api.ResourceServicesLoadBalancers] = resource.MustParse("1")
		usedResources[api.ResourceServicesNodePorts] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting a Service")
		err = f.Client.Services(f.Namespace.Name).Delete(service.Name)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released usage")
		usedResources[api.ResourceServices] = resource.MustParse("0")
		usedResources[api.ResourceServicesLoadBalancers] = resource.MustParse("0")
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
		pod := newTestPodForQuota(podName, requests, api.ResourceList{})
		pod, err = f.Client.Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())
		podToUpdate := pod

		By("Ensuring ResourceQuota status captures the pod usage")
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
		pod = newTestPodForQuota("fail-pod", requests, api.ResourceList{})
		pod, err = f.Client.Pods(f.Namespace.Name).Create(pod)
		Expect(err).To(HaveOccurred())

		By("Ensuring a pod cannot update its resource requirements")
		// a pod cannot dynamically update its resource requirements.
		requests = api.ResourceList{}
		requests[api.ResourceCPU] = resource.MustParse("100m")
		requests[api.ResourceMemory] = resource.MustParse("100Mi")
		podToUpdate.Spec.Containers[0].Resources.Requests = requests
		_, err = f.Client.Pods(f.Namespace.Name).Update(podToUpdate)
		Expect(err).To(HaveOccurred())

		By("Ensuring attempts to update pod resource requirements did not change quota usage")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

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

	It("should create a ResourceQuota and capture the life of a configMap.", func() {
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

		By("Creating a ConfigMap")
		configMap := newTestConfigMapForQuota("test-configmap")
		configMap, err = f.Client.ConfigMaps(f.Namespace.Name).Create(configMap)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures configMap creation")
		usedResources = api.ResourceList{}
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		usedResources[api.ResourceConfigMaps] = resource.MustParse("1")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting a ConfigMap")
		err = f.Client.ConfigMaps(f.Namespace.Name).Delete(configMap.Name)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released usage")
		usedResources[api.ResourceConfigMaps] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create a ResourceQuota and capture the life of a replication controller.", func() {
		By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		resourceQuota, err := createResourceQuota(f.Client, f.Namespace.Name, resourceQuota)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status is calculated")
		usedResources := api.ResourceList{}
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		usedResources[api.ResourceReplicationControllers] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a ReplicationController")
		replicationController := newTestReplicationControllerForQuota("test-rc", "nginx", 0)
		replicationController, err = f.Client.ReplicationControllers(f.Namespace.Name).Create(replicationController)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures replication controller creation")
		usedResources = api.ResourceList{}
		usedResources[api.ResourceReplicationControllers] = resource.MustParse("1")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting a ReplicationController")
		err = f.Client.ReplicationControllers(f.Namespace.Name).Delete(replicationController.Name)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released usage")
		usedResources[api.ResourceReplicationControllers] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create a ResourceQuota and capture the life of a persistent volume claim.", func() {
		By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		resourceQuota, err := createResourceQuota(f.Client, f.Namespace.Name, resourceQuota)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status is calculated")
		usedResources := api.ResourceList{}
		usedResources[api.ResourceQuotas] = resource.MustParse("1")
		usedResources[api.ResourcePersistentVolumeClaims] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a PersistentVolumeClaim")
		pvc := newTestPersistentVolumeClaimForQuota("test-claim")
		pvc, err = f.Client.PersistentVolumeClaims(f.Namespace.Name).Create(pvc)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures persistent volume claimcreation")
		usedResources = api.ResourceList{}
		usedResources[api.ResourcePersistentVolumeClaims] = resource.MustParse("1")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting a PersistentVolumeClaim")
		err = f.Client.PersistentVolumeClaims(f.Namespace.Name).Delete(pvc.Name)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released usage")
		usedResources[api.ResourcePersistentVolumeClaims] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should verify ResourceQuota with terminating scopes.", func() {
		By("Creating a ResourceQuota with terminating scope")
		quotaTerminatingName := "quota-terminating"
		resourceQuotaTerminating, err := createResourceQuota(f.Client, f.Namespace.Name, newTestResourceQuotaWithScope(quotaTerminatingName, api.ResourceQuotaScopeTerminating))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring ResourceQuota status is calculated")
		usedResources := api.ResourceList{}
		usedResources[api.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a ResourceQuota with not terminating scope")
		quotaNotTerminatingName := "quota-not-terminating"
		resourceQuotaNotTerminating, err := createResourceQuota(f.Client, f.Namespace.Name, newTestResourceQuotaWithScope(quotaNotTerminatingName, api.ResourceQuotaScopeNotTerminating))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring ResourceQuota status is calculated")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a long running pod")
		podName := "test-pod"
		requests := api.ResourceList{}
		requests[api.ResourceCPU] = resource.MustParse("500m")
		requests[api.ResourceMemory] = resource.MustParse("200Mi")
		limits := api.ResourceList{}
		limits[api.ResourceCPU] = resource.MustParse("1")
		limits[api.ResourceMemory] = resource.MustParse("400Mi")
		pod := newTestPodForQuota(podName, requests, limits)
		pod, err = f.Client.Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with not terminating scope captures the pod usage")
		usedResources[api.ResourcePods] = resource.MustParse("1")
		usedResources[api.ResourceRequestsCPU] = requests[api.ResourceCPU]
		usedResources[api.ResourceRequestsMemory] = requests[api.ResourceMemory]
		usedResources[api.ResourceLimitsCPU] = limits[api.ResourceCPU]
		usedResources[api.ResourceLimitsMemory] = limits[api.ResourceMemory]
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with terminating scope ignored the pod usage")
		usedResources[api.ResourcePods] = resource.MustParse("0")
		usedResources[api.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[api.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[api.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[api.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting the pod")
		err = f.Client.Pods(f.Namespace.Name).Delete(podName, api.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released the pod usage")
		usedResources[api.ResourcePods] = resource.MustParse("0")
		usedResources[api.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[api.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[api.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[api.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a terminating pod")
		podName = "terminating-pod"
		pod = newTestPodForQuota(podName, requests, limits)
		activeDeadlineSeconds := int64(3600)
		pod.Spec.ActiveDeadlineSeconds = &activeDeadlineSeconds
		pod, err = f.Client.Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with terminating scope captures the pod usage")
		usedResources[api.ResourcePods] = resource.MustParse("1")
		usedResources[api.ResourceRequestsCPU] = requests[api.ResourceCPU]
		usedResources[api.ResourceRequestsMemory] = requests[api.ResourceMemory]
		usedResources[api.ResourceLimitsCPU] = limits[api.ResourceCPU]
		usedResources[api.ResourceLimitsMemory] = limits[api.ResourceMemory]
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with not terminating scope ignored the pod usage")
		usedResources[api.ResourcePods] = resource.MustParse("0")
		usedResources[api.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[api.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[api.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[api.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting the pod")
		err = f.Client.Pods(f.Namespace.Name).Delete(podName, api.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released the pod usage")
		usedResources[api.ResourcePods] = resource.MustParse("0")
		usedResources[api.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[api.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[api.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[api.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should verify ResourceQuota with best effort scope.", func() {
		By("Creating a ResourceQuota with best effort scope")
		resourceQuotaBestEffort, err := createResourceQuota(f.Client, f.Namespace.Name, newTestResourceQuotaWithScope("quota-besteffort", api.ResourceQuotaScopeBestEffort))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring ResourceQuota status is calculated")
		usedResources := api.ResourceList{}
		usedResources[api.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a ResourceQuota with not best effort scope")
		resourceQuotaNotBestEffort, err := createResourceQuota(f.Client, f.Namespace.Name, newTestResourceQuotaWithScope("quota-not-besteffort", api.ResourceQuotaScopeNotBestEffort))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring ResourceQuota status is calculated")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a best-effort pod")
		pod := newTestPodForQuota(podName, api.ResourceList{}, api.ResourceList{})
		pod, err = f.Client.Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with best effort scope captures the pod usage")
		usedResources[api.ResourcePods] = resource.MustParse("1")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with not best effort ignored the pod usage")
		usedResources[api.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting the pod")
		err = f.Client.Pods(f.Namespace.Name).Delete(pod.Name, api.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released the pod usage")
		usedResources[api.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a not best-effort pod")
		requests := api.ResourceList{}
		requests[api.ResourceCPU] = resource.MustParse("500m")
		requests[api.ResourceMemory] = resource.MustParse("200Mi")
		limits := api.ResourceList{}
		limits[api.ResourceCPU] = resource.MustParse("1")
		limits[api.ResourceMemory] = resource.MustParse("400Mi")
		pod = newTestPodForQuota("burstable-pod", requests, limits)
		pod, err = f.Client.Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with not best effort scope captures the pod usage")
		usedResources[api.ResourcePods] = resource.MustParse("1")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with best effort scope ignored the pod usage")
		usedResources[api.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting the pod")
		err = f.Client.Pods(f.Namespace.Name).Delete(pod.Name, api.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released the pod usage")
		usedResources[api.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.Client, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})
})

// newTestResourceQuotaWithScope returns a quota that enforces default constraints for testing with scopes
func newTestResourceQuotaWithScope(name string, scope api.ResourceQuotaScope) *api.ResourceQuota {
	hard := api.ResourceList{}
	hard[api.ResourcePods] = resource.MustParse("5")
	switch scope {
	case api.ResourceQuotaScopeTerminating, api.ResourceQuotaScopeNotTerminating:
		hard[api.ResourceRequestsCPU] = resource.MustParse("1")
		hard[api.ResourceRequestsMemory] = resource.MustParse("500Mi")
		hard[api.ResourceLimitsCPU] = resource.MustParse("2")
		hard[api.ResourceLimitsMemory] = resource.MustParse("1Gi")
	}
	return &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: name},
		Spec:       api.ResourceQuotaSpec{Hard: hard, Scopes: []api.ResourceQuotaScope{scope}},
	}
}

// newTestResourceQuota returns a quota that enforces default constraints for testing
func newTestResourceQuota(name string) *api.ResourceQuota {
	hard := api.ResourceList{}
	hard[api.ResourcePods] = resource.MustParse("5")
	hard[api.ResourceServices] = resource.MustParse("10")
	hard[api.ResourceServicesNodePorts] = resource.MustParse("1")
	hard[api.ResourceServicesLoadBalancers] = resource.MustParse("1")
	hard[api.ResourceReplicationControllers] = resource.MustParse("10")
	hard[api.ResourceQuotas] = resource.MustParse("1")
	hard[api.ResourceCPU] = resource.MustParse("1")
	hard[api.ResourceMemory] = resource.MustParse("500Mi")
	hard[api.ResourceConfigMaps] = resource.MustParse("2")
	hard[api.ResourceSecrets] = resource.MustParse("10")
	hard[api.ResourcePersistentVolumeClaims] = resource.MustParse("10")
	return &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{Name: name},
		Spec:       api.ResourceQuotaSpec{Hard: hard},
	}
}

// newTestPodForQuota returns a pod that has the specified requests and limits
func newTestPodForQuota(name string, requests api.ResourceList, limits api.ResourceList) *api.Pod {
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

// newTestPersistentVolumeClaimForQuota returns a simple persistent volume claim
func newTestPersistentVolumeClaimForQuota(name string) *api.PersistentVolumeClaim {
	return &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
				api.ReadOnlyMany,
				api.ReadWriteMany,
			},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("1Gi"),
				},
			},
		},
	}
}

// newTestReplicationControllerForQuota returns a simple replication controller
func newTestReplicationControllerForQuota(name, image string, replicas int32) *api.ReplicationController {
	return &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
			Selector: map[string]string{
				"name": name,
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"name": name},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  name,
							Image: image,
						},
					},
				},
			},
		},
	}
}

// newTestServiceForQuota returns a simple service
func newTestServiceForQuota(name string, serviceType api.ServiceType) *api.Service {
	return &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ServiceSpec{
			Type: serviceType,
			Ports: []api.ServicePort{{
				Port:       80,
				TargetPort: intstr.FromInt(80),
			}},
		},
	}
}

func newTestConfigMapForQuota(name string) *api.ConfigMap {
	return &api.ConfigMap{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Data: map[string]string{
			"a": "b",
		},
	}
}

func newTestSecretForQuota(name string) *api.Secret {
	return &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Data: map[string][]byte{
			"data-1": []byte("value-1\n"),
			"data-2": []byte("value-2\n"),
			"data-3": []byte("value-3\n"),
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
	return wait.Poll(framework.Poll, resourceQuotaTimeout, func() (bool, error) {
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
				framework.Logf("resource %s, expected %s, actual %s", k, v.String(), actualValue.String())
				return false, nil
			}
		}
		return true, nil
	})
}
