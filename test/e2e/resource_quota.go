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

package e2e

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
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
		resourceQuota, err := createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create a ResourceQuota and capture the life of a service.", func() {
		By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		resourceQuota, err := createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a Service")
		service := newTestServiceForQuota("test-service", v1.ServiceTypeClusterIP)
		service, err = f.ClientSet.Core().Services(f.Namespace.Name).Create(service)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures service creation")
		usedResources = v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse("1")
		usedResources[v1.ResourceServices] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting a Service")
		err = f.ClientSet.Core().Services(f.Namespace.Name).Delete(service.Name, nil)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released usage")
		usedResources[v1.ResourceServices] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create a ResourceQuota and capture the life of a secret.", func() {
		By("Discovering how many secrets are in namespace by default")
		secrets, err := f.ClientSet.Core().Secrets(f.Namespace.Name).List(v1.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		defaultSecrets := fmt.Sprintf("%d", len(secrets.Items))
		hardSecrets := fmt.Sprintf("%d", len(secrets.Items)+1)

		By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		resourceQuota.Spec.Hard[v1.ResourceSecrets] = resource.MustParse(hardSecrets)
		resourceQuota, err = createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse("1")
		usedResources[v1.ResourceSecrets] = resource.MustParse(defaultSecrets)
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a Secret")
		secret := newTestSecretForQuota("test-secret")
		secret, err = f.ClientSet.Core().Secrets(f.Namespace.Name).Create(secret)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures secret creation")
		usedResources = v1.ResourceList{}
		usedResources[v1.ResourceSecrets] = resource.MustParse(hardSecrets)
		// we expect there to be two secrets because each namespace will receive
		// a service account token secret by default
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting a secret")
		err = f.ClientSet.Core().Secrets(f.Namespace.Name).Delete(secret.Name, nil)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released usage")
		usedResources[v1.ResourceSecrets] = resource.MustParse(defaultSecrets)
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create a ResourceQuota and capture the life of a pod.", func() {
		By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		resourceQuota, err := createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a Pod that fits quota")
		podName := "test-pod"
		requests := v1.ResourceList{}
		requests[v1.ResourceCPU] = resource.MustParse("500m")
		requests[v1.ResourceMemory] = resource.MustParse("252Mi")
		pod := newTestPodForQuota(f, podName, requests, v1.ResourceList{})
		pod, err = f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())
		podToUpdate := pod

		By("Ensuring ResourceQuota status captures the pod usage")
		usedResources[v1.ResourceQuotas] = resource.MustParse("1")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		usedResources[v1.ResourceCPU] = requests[v1.ResourceCPU]
		usedResources[v1.ResourceMemory] = requests[v1.ResourceMemory]
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Not allowing a pod to be created that exceeds remaining quota")
		requests = v1.ResourceList{}
		requests[v1.ResourceCPU] = resource.MustParse("600m")
		requests[v1.ResourceMemory] = resource.MustParse("100Mi")
		pod = newTestPodForQuota(f, "fail-pod", requests, v1.ResourceList{})
		pod, err = f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod)
		Expect(err).To(HaveOccurred())

		By("Ensuring a pod cannot update its resource requirements")
		// a pod cannot dynamically update its resource requirements.
		requests = v1.ResourceList{}
		requests[v1.ResourceCPU] = resource.MustParse("100m")
		requests[v1.ResourceMemory] = resource.MustParse("100Mi")
		podToUpdate.Spec.Containers[0].Resources.Requests = requests
		_, err = f.ClientSet.Core().Pods(f.Namespace.Name).Update(podToUpdate)
		Expect(err).To(HaveOccurred())

		By("Ensuring attempts to update pod resource requirements did not change quota usage")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting the pod")
		err = f.ClientSet.Core().Pods(f.Namespace.Name).Delete(podName, v1.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourceQuotas] = resource.MustParse("1")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceCPU] = resource.MustParse("0")
		usedResources[v1.ResourceMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create a ResourceQuota and capture the life of a configMap.", func() {
		By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		resourceQuota, err := createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a ConfigMap")
		configMap := newTestConfigMapForQuota("test-configmap")
		configMap, err = f.ClientSet.Core().ConfigMaps(f.Namespace.Name).Create(configMap)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures configMap creation")
		usedResources = v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse("1")
		usedResources[v1.ResourceConfigMaps] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting a ConfigMap")
		err = f.ClientSet.Core().ConfigMaps(f.Namespace.Name).Delete(configMap.Name, nil)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released usage")
		usedResources[v1.ResourceConfigMaps] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create a ResourceQuota and capture the life of a replication controller.", func() {
		By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		resourceQuota, err := createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse("1")
		usedResources[v1.ResourceReplicationControllers] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a ReplicationController")
		replicationController := newTestReplicationControllerForQuota("test-rc", "nginx", 0)
		replicationController, err = f.ClientSet.Core().ReplicationControllers(f.Namespace.Name).Create(replicationController)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures replication controller creation")
		usedResources = v1.ResourceList{}
		usedResources[v1.ResourceReplicationControllers] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting a ReplicationController")
		err = f.ClientSet.Core().ReplicationControllers(f.Namespace.Name).Delete(replicationController.Name, nil)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released usage")
		usedResources[v1.ResourceReplicationControllers] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create a ResourceQuota and capture the life of a persistent volume claim.", func() {
		By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		resourceQuota, err := createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse("1")
		usedResources[v1.ResourcePersistentVolumeClaims] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsStorage] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a PersistentVolumeClaim")
		pvc := newTestPersistentVolumeClaimForQuota("test-claim")
		pvc, err = f.ClientSet.Core().PersistentVolumeClaims(f.Namespace.Name).Create(pvc)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status captures persistent volume claimcreation")
		usedResources = v1.ResourceList{}
		usedResources[v1.ResourcePersistentVolumeClaims] = resource.MustParse("1")
		usedResources[v1.ResourceRequestsStorage] = resource.MustParse("1Gi")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting a PersistentVolumeClaim")
		err = f.ClientSet.Core().PersistentVolumeClaims(f.Namespace.Name).Delete(pvc.Name, nil)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released usage")
		usedResources[v1.ResourcePersistentVolumeClaims] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsStorage] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should verify ResourceQuota with terminating scopes.", func() {
		By("Creating a ResourceQuota with terminating scope")
		quotaTerminatingName := "quota-terminating"
		resourceQuotaTerminating, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScope(quotaTerminatingName, v1.ResourceQuotaScopeTerminating))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring ResourceQuota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a ResourceQuota with not terminating scope")
		quotaNotTerminatingName := "quota-not-terminating"
		resourceQuotaNotTerminating, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScope(quotaNotTerminatingName, v1.ResourceQuotaScopeNotTerminating))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring ResourceQuota status is calculated")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a long running pod")
		podName := "test-pod"
		requests := v1.ResourceList{}
		requests[v1.ResourceCPU] = resource.MustParse("500m")
		requests[v1.ResourceMemory] = resource.MustParse("200Mi")
		limits := v1.ResourceList{}
		limits[v1.ResourceCPU] = resource.MustParse("1")
		limits[v1.ResourceMemory] = resource.MustParse("400Mi")
		pod := newTestPodForQuota(f, podName, requests, limits)
		pod, err = f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with not terminating scope captures the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		usedResources[v1.ResourceRequestsCPU] = requests[v1.ResourceCPU]
		usedResources[v1.ResourceRequestsMemory] = requests[v1.ResourceMemory]
		usedResources[v1.ResourceLimitsCPU] = limits[v1.ResourceCPU]
		usedResources[v1.ResourceLimitsMemory] = limits[v1.ResourceMemory]
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with terminating scope ignored the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting the pod")
		err = f.ClientSet.Core().Pods(f.Namespace.Name).Delete(podName, v1.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a terminating pod")
		podName = "terminating-pod"
		pod = newTestPodForQuota(f, podName, requests, limits)
		activeDeadlineSeconds := int64(3600)
		pod.Spec.ActiveDeadlineSeconds = &activeDeadlineSeconds
		pod, err = f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with terminating scope captures the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		usedResources[v1.ResourceRequestsCPU] = requests[v1.ResourceCPU]
		usedResources[v1.ResourceRequestsMemory] = requests[v1.ResourceMemory]
		usedResources[v1.ResourceLimitsCPU] = limits[v1.ResourceCPU]
		usedResources[v1.ResourceLimitsMemory] = limits[v1.ResourceMemory]
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with not terminating scope ignored the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting the pod")
		err = f.ClientSet.Core().Pods(f.Namespace.Name).Delete(podName, v1.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should verify ResourceQuota with best effort scope.", func() {
		By("Creating a ResourceQuota with best effort scope")
		resourceQuotaBestEffort, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScope("quota-besteffort", v1.ResourceQuotaScopeBestEffort))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring ResourceQuota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a ResourceQuota with not best effort scope")
		resourceQuotaNotBestEffort, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScope("quota-not-besteffort", v1.ResourceQuotaScopeNotBestEffort))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring ResourceQuota status is calculated")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a best-effort pod")
		pod := newTestPodForQuota(f, podName, v1.ResourceList{}, v1.ResourceList{})
		pod, err = f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with best effort scope captures the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with not best effort ignored the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting the pod")
		err = f.ClientSet.Core().Pods(f.Namespace.Name).Delete(pod.Name, v1.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Creating a not best-effort pod")
		requests := v1.ResourceList{}
		requests[v1.ResourceCPU] = resource.MustParse("500m")
		requests[v1.ResourceMemory] = resource.MustParse("200Mi")
		limits := v1.ResourceList{}
		limits[v1.ResourceCPU] = resource.MustParse("1")
		limits[v1.ResourceMemory] = resource.MustParse("400Mi")
		pod = newTestPodForQuota(f, "burstable-pod", requests, limits)
		pod, err = f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with not best effort scope captures the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota with best effort scope ignored the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())

		By("Deleting the pod")
		err = f.ClientSet.Core().Pods(f.Namespace.Name).Delete(pod.Name, v1.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred())

		By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		Expect(err).NotTo(HaveOccurred())
	})
})

// newTestResourceQuotaWithScope returns a quota that enforces default constraints for testing with scopes
func newTestResourceQuotaWithScope(name string, scope v1.ResourceQuotaScope) *v1.ResourceQuota {
	hard := v1.ResourceList{}
	hard[v1.ResourcePods] = resource.MustParse("5")
	switch scope {
	case v1.ResourceQuotaScopeTerminating, v1.ResourceQuotaScopeNotTerminating:
		hard[v1.ResourceRequestsCPU] = resource.MustParse("1")
		hard[v1.ResourceRequestsMemory] = resource.MustParse("500Mi")
		hard[v1.ResourceLimitsCPU] = resource.MustParse("2")
		hard[v1.ResourceLimitsMemory] = resource.MustParse("1Gi")
	}
	return &v1.ResourceQuota{
		ObjectMeta: v1.ObjectMeta{Name: name},
		Spec:       v1.ResourceQuotaSpec{Hard: hard, Scopes: []v1.ResourceQuotaScope{scope}},
	}
}

// newTestResourceQuota returns a quota that enforces default constraints for testing
func newTestResourceQuota(name string) *v1.ResourceQuota {
	hard := v1.ResourceList{}
	hard[v1.ResourcePods] = resource.MustParse("5")
	hard[v1.ResourceServices] = resource.MustParse("10")
	hard[v1.ResourceServicesNodePorts] = resource.MustParse("1")
	hard[v1.ResourceServicesLoadBalancers] = resource.MustParse("1")
	hard[v1.ResourceReplicationControllers] = resource.MustParse("10")
	hard[v1.ResourceQuotas] = resource.MustParse("1")
	hard[v1.ResourceCPU] = resource.MustParse("1")
	hard[v1.ResourceMemory] = resource.MustParse("500Mi")
	hard[v1.ResourceConfigMaps] = resource.MustParse("2")
	hard[v1.ResourceSecrets] = resource.MustParse("10")
	hard[v1.ResourcePersistentVolumeClaims] = resource.MustParse("10")
	hard[v1.ResourceRequestsStorage] = resource.MustParse("10Gi")
	return &v1.ResourceQuota{
		ObjectMeta: v1.ObjectMeta{Name: name},
		Spec:       v1.ResourceQuotaSpec{Hard: hard},
	}
}

// newTestPodForQuota returns a pod that has the specified requests and limits
func newTestPodForQuota(f *framework.Framework, name string, requests v1.ResourceList, limits v1.ResourceList) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "pause",
					Image: framework.GetPauseImageName(f.ClientSet),
					Resources: v1.ResourceRequirements{
						Requests: requests,
						Limits:   limits,
					},
				},
			},
		},
	}
}

// newTestPersistentVolumeClaimForQuota returns a simple persistent volume claim
func newTestPersistentVolumeClaimForQuota(name string) *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		ObjectMeta: v1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadOnlyMany,
				v1.ReadWriteMany,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("1Gi"),
				},
			},
		},
	}
}

// newTestReplicationControllerForQuota returns a simple replication controller
func newTestReplicationControllerForQuota(name, image string, replicas int32) *v1.ReplicationController {
	return &v1.ReplicationController{
		ObjectMeta: v1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: func(i int32) *int32 { return &i }(replicas),
			Selector: map[string]string{
				"name": name,
			},
			Template: &v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
					Labels: map[string]string{"name": name},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
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
func newTestServiceForQuota(name string, serviceType v1.ServiceType) *v1.Service {
	return &v1.Service{
		ObjectMeta: v1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ServiceSpec{
			Type: serviceType,
			Ports: []v1.ServicePort{{
				Port:       80,
				TargetPort: intstr.FromInt(80),
			}},
		},
	}
}

func newTestConfigMapForQuota(name string) *v1.ConfigMap {
	return &v1.ConfigMap{
		ObjectMeta: v1.ObjectMeta{
			Name: name,
		},
		Data: map[string]string{
			"a": "b",
		},
	}
}

func newTestSecretForQuota(name string) *v1.Secret {
	return &v1.Secret{
		ObjectMeta: v1.ObjectMeta{
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
func createResourceQuota(c clientset.Interface, namespace string, resourceQuota *v1.ResourceQuota) (*v1.ResourceQuota, error) {
	return c.Core().ResourceQuotas(namespace).Create(resourceQuota)
}

// deleteResourceQuota with the specified name
func deleteResourceQuota(c clientset.Interface, namespace, name string) error {
	return c.Core().ResourceQuotas(namespace).Delete(name, nil)
}

// wait for resource quota status to show the expected used resources value
func waitForResourceQuota(c clientset.Interface, ns, quotaName string, used v1.ResourceList) error {
	return wait.Poll(framework.Poll, resourceQuotaTimeout, func() (bool, error) {
		resourceQuota, err := c.Core().ResourceQuotas(ns).Get(quotaName)
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
