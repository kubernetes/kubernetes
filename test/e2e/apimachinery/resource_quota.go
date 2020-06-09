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

package apimachinery

import (
	"context"
	"fmt"
	"strconv"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/quota/v1/evaluator/core"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/utils/crd"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

const (
	// how long to wait for a resource quota update to occur
	resourceQuotaTimeout = 30 * time.Second
	podName              = "pfpod"
)

var classGold = "gold"
var extendedResourceName = "example.com/dongle"

var _ = SIGDescribe("ResourceQuota", func() {
	f := framework.NewDefaultFramework("resourcequota")

	/*
		Release: v1.16
		Testname: ResourceQuota, object count quota, resourcequotas
		Description: Create a ResourceQuota. Creation MUST be successful and its ResourceQuotaStatus MUST match to expected used and total allowed resource quota count within namespace.
	*/
	framework.ConformanceIt("should create a ResourceQuota and ensure its status is promptly calculated.", func() {
		ginkgo.By("Counting existing ResourceQuota")
		c, err := countResourceQuota(f.ClientSet, f.Namespace.Name)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		_, err = createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse(strconv.Itoa(c + 1))
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.16
		Testname: ResourceQuota, object count quota, service
		Description: Create a ResourceQuota. Creation MUST be successful and its ResourceQuotaStatus MUST match to expected used and total allowed resource quota count within namespace.
		Create a Service. Its creation MUST be successful and resource usage count against the Service object and resourceQuota object MUST be captured in ResourceQuotaStatus of the ResourceQuota.
		Delete the Service. Deletion MUST succeed and resource usage count against the Service object MUST be released from ResourceQuotaStatus of the ResourceQuota.
	*/
	framework.ConformanceIt("should create a ResourceQuota and capture the life of a service.", func() {
		ginkgo.By("Counting existing ResourceQuota")
		c, err := countResourceQuota(f.ClientSet, f.Namespace.Name)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		_, err = createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse(strconv.Itoa(c + 1))
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a Service")
		service := newTestServiceForQuota("test-service", v1.ServiceTypeClusterIP)
		service, err = f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(context.TODO(), service, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status captures service creation")
		usedResources = v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse(strconv.Itoa(c + 1))
		usedResources[v1.ResourceServices] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting a Service")
		err = f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(context.TODO(), service.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released usage")
		usedResources[v1.ResourceServices] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.16
		Testname: ResourceQuota, object count quota, secret
		Description: Create a ResourceQuota. Creation MUST be successful and its ResourceQuotaStatus MUST match to expected used and total allowed resource quota count within namespace.
		Create a Secret. Its creation MUST be successful and resource usage count against the Secret object and resourceQuota object MUST be captured in ResourceQuotaStatus of the ResourceQuota.
		Delete the Secret. Deletion MUST succeed and resource usage count against the Secret object MUST be released from ResourceQuotaStatus of the ResourceQuota.
	*/
	framework.ConformanceIt("should create a ResourceQuota and capture the life of a secret.", func() {
		ginkgo.By("Discovering how many secrets are in namespace by default")
		found, unchanged := 0, 0
		// On contended servers the service account controller can slow down, leading to the count changing during a run.
		// Wait up to 5s for the count to stabilize, assuming that updates come at a consistent rate, and are not held indefinitely.
		wait.Poll(1*time.Second, 30*time.Second, func() (bool, error) {
			secrets, err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{})
			framework.ExpectNoError(err)
			if len(secrets.Items) == found {
				// loop until the number of secrets has stabilized for 5 seconds
				unchanged++
				return unchanged > 4, nil
			}
			unchanged = 0
			found = len(secrets.Items)
			return false, nil
		})
		defaultSecrets := fmt.Sprintf("%d", found)
		hardSecrets := fmt.Sprintf("%d", found+1)

		ginkgo.By("Counting existing ResourceQuota")
		c, err := countResourceQuota(f.ClientSet, f.Namespace.Name)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		resourceQuota.Spec.Hard[v1.ResourceSecrets] = resource.MustParse(hardSecrets)
		_, err = createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse(strconv.Itoa(c + 1))
		usedResources[v1.ResourceSecrets] = resource.MustParse(defaultSecrets)
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a Secret")
		secret := newTestSecretForQuota("test-secret")
		secret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(context.TODO(), secret, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status captures secret creation")
		usedResources = v1.ResourceList{}
		usedResources[v1.ResourceSecrets] = resource.MustParse(hardSecrets)
		// we expect there to be two secrets because each namespace will receive
		// a service account token secret by default
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting a secret")
		err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Delete(context.TODO(), secret.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released usage")
		usedResources[v1.ResourceSecrets] = resource.MustParse(defaultSecrets)
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.16
		Testname: ResourceQuota, object count quota, pod
		Description: Create a ResourceQuota. Creation MUST be successful and its ResourceQuotaStatus MUST match to expected used and total allowed resource quota count within namespace.
		Create a Pod with resource request count for CPU, Memory, EphemeralStorage and ExtendedResourceName. Pod creation MUST be successful and respective resource usage count MUST be captured in ResourceQuotaStatus of the ResourceQuota.
		Create another Pod with resource request exceeding remaining quota. Pod creation MUST fail as the request exceeds ResourceQuota limits.
		Update the successfully created pod's resource requests. Updation MUST fail as a Pod can not dynamically update its resource requirements.
		Delete the successfully created Pod. Pod Deletion MUST be scuccessful and it MUST release the allocated resource counts from ResourceQuotaStatus of the ResourceQuota.
	*/
	framework.ConformanceIt("should create a ResourceQuota and capture the life of a pod.", func() {
		ginkgo.By("Counting existing ResourceQuota")
		c, err := countResourceQuota(f.ClientSet, f.Namespace.Name)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		_, err = createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse(strconv.Itoa(c + 1))
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a Pod that fits quota")
		podName := "test-pod"
		requests := v1.ResourceList{}
		limits := v1.ResourceList{}
		requests[v1.ResourceCPU] = resource.MustParse("500m")
		requests[v1.ResourceMemory] = resource.MustParse("252Mi")
		requests[v1.ResourceEphemeralStorage] = resource.MustParse("30Gi")
		requests[v1.ResourceName(extendedResourceName)] = resource.MustParse("2")
		limits[v1.ResourceName(extendedResourceName)] = resource.MustParse("2")
		pod := newTestPodForQuota(f, podName, requests, limits)
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		podToUpdate := pod

		ginkgo.By("Ensuring ResourceQuota status captures the pod usage")
		usedResources[v1.ResourceQuotas] = resource.MustParse(strconv.Itoa(c + 1))
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		usedResources[v1.ResourceCPU] = requests[v1.ResourceCPU]
		usedResources[v1.ResourceMemory] = requests[v1.ResourceMemory]
		usedResources[v1.ResourceEphemeralStorage] = requests[v1.ResourceEphemeralStorage]
		usedResources[v1.ResourceName(v1.DefaultResourceRequestsPrefix+extendedResourceName)] = requests[v1.ResourceName(extendedResourceName)]
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Not allowing a pod to be created that exceeds remaining quota")
		requests = v1.ResourceList{}
		requests[v1.ResourceCPU] = resource.MustParse("600m")
		requests[v1.ResourceMemory] = resource.MustParse("100Mi")
		pod = newTestPodForQuota(f, "fail-pod", requests, v1.ResourceList{})
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectError(err)

		ginkgo.By("Not allowing a pod to be created that exceeds remaining quota(validation on extended resources)")
		requests = v1.ResourceList{}
		limits = v1.ResourceList{}
		requests[v1.ResourceCPU] = resource.MustParse("500m")
		requests[v1.ResourceMemory] = resource.MustParse("100Mi")
		requests[v1.ResourceEphemeralStorage] = resource.MustParse("30Gi")
		requests[v1.ResourceName(extendedResourceName)] = resource.MustParse("2")
		limits[v1.ResourceName(extendedResourceName)] = resource.MustParse("2")
		pod = newTestPodForQuota(f, "fail-pod-for-extended-resource", requests, limits)
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectError(err)

		ginkgo.By("Ensuring a pod cannot update its resource requirements")
		// a pod cannot dynamically update its resource requirements.
		requests = v1.ResourceList{}
		requests[v1.ResourceCPU] = resource.MustParse("100m")
		requests[v1.ResourceMemory] = resource.MustParse("100Mi")
		requests[v1.ResourceEphemeralStorage] = resource.MustParse("10Gi")
		podToUpdate.Spec.Containers[0].Resources.Requests = requests
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Update(context.TODO(), podToUpdate, metav1.UpdateOptions{})
		framework.ExpectError(err)

		ginkgo.By("Ensuring attempts to update pod resource requirements did not change quota usage")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting the pod")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), podName, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourceQuotas] = resource.MustParse(strconv.Itoa(c + 1))
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceCPU] = resource.MustParse("0")
		usedResources[v1.ResourceMemory] = resource.MustParse("0")
		usedResources[v1.ResourceEphemeralStorage] = resource.MustParse("0")
		usedResources[v1.ResourceName(v1.DefaultResourceRequestsPrefix+extendedResourceName)] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)
	})
	/*
		Release: v1.16
		Testname: ResourceQuota, object count quota, configmap
		Description: Create a ResourceQuota. Creation MUST be successful and its ResourceQuotaStatus MUST match to expected used and total allowed resource quota count within namespace.
		Create a ConfigMap. Its creation MUST be successful and resource usage count against the ConfigMap object MUST be captured in ResourceQuotaStatus of the ResourceQuota.
		Delete the ConfigMap. Deletion MUST succeed and resource usage count against the ConfigMap object MUST be released from ResourceQuotaStatus of the ResourceQuota.
	*/
	framework.ConformanceIt("should create a ResourceQuota and capture the life of a configMap.", func() {
		found, unchanged := 0, 0
		// On contended servers the service account controller can slow down, leading to the count changing during a run.
		// Wait up to 5s for the count to stabilize, assuming that updates come at a consistent rate, and are not held indefinitely.
		wait.Poll(1*time.Second, 30*time.Second, func() (bool, error) {
			configmaps, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{})
			framework.ExpectNoError(err)
			if len(configmaps.Items) == found {
				// loop until the number of configmaps has stabilized for 5 seconds
				unchanged++
				return unchanged > 4, nil
			}
			unchanged = 0
			found = len(configmaps.Items)
			return false, nil
		})
		defaultConfigMaps := fmt.Sprintf("%d", found)
		hardConfigMaps := fmt.Sprintf("%d", found+1)

		ginkgo.By("Counting existing ResourceQuota")
		c, err := countResourceQuota(f.ClientSet, f.Namespace.Name)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		_, err = createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse(strconv.Itoa(c + 1))
		usedResources[v1.ResourceConfigMaps] = resource.MustParse(defaultConfigMaps)
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ConfigMap")
		configMap := newTestConfigMapForQuota("test-configmap")
		configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status captures configMap creation")
		usedResources = v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse(strconv.Itoa(c + 1))
		// we expect there to be two configmaps because each namespace will receive
		// a ca.crt configmap by default.
		// ref:https://github.com/kubernetes/kubernetes/pull/68812
		usedResources[v1.ResourceConfigMaps] = resource.MustParse(hardConfigMaps)
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting a ConfigMap")
		err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), configMap.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released usage")
		usedResources[v1.ResourceConfigMaps] = resource.MustParse(defaultConfigMaps)
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.16
		Testname: ResourceQuota, object count quota, replicationController
		Description: Create a ResourceQuota. Creation MUST be successful and its ResourceQuotaStatus MUST match to expected used and total allowed resource quota count within namespace.
		Create a ReplicationController. Its creation MUST be successful and resource usage count against the ReplicationController object MUST be captured in ResourceQuotaStatus of the ResourceQuota.
		Delete the ReplicationController. Deletion MUST succeed and resource usage count against the ReplicationController object MUST be released from ResourceQuotaStatus of the ResourceQuota.
	*/
	framework.ConformanceIt("should create a ResourceQuota and capture the life of a replication controller.", func() {
		ginkgo.By("Counting existing ResourceQuota")
		c, err := countResourceQuota(f.ClientSet, f.Namespace.Name)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		_, err = createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse(strconv.Itoa(c + 1))
		usedResources[v1.ResourceReplicationControllers] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ReplicationController")
		replicationController := newTestReplicationControllerForQuota("test-rc", "nginx", 0)
		replicationController, err = f.ClientSet.CoreV1().ReplicationControllers(f.Namespace.Name).Create(context.TODO(), replicationController, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status captures replication controller creation")
		usedResources = v1.ResourceList{}
		usedResources[v1.ResourceReplicationControllers] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting a ReplicationController")
		// Without the delete options, the object isn't actually
		// removed until the GC verifies that all children have been
		// detached. ReplicationControllers default to "orphan", which
		// is different from most resources. (Why? To preserve a common
		// workflow from prior to the GC's introduction.)
		err = f.ClientSet.CoreV1().ReplicationControllers(f.Namespace.Name).Delete(context.TODO(), replicationController.Name, metav1.DeleteOptions{
			PropagationPolicy: func() *metav1.DeletionPropagation {
				p := metav1.DeletePropagationBackground
				return &p
			}(),
		})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released usage")
		usedResources[v1.ResourceReplicationControllers] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.16
		Testname: ResourceQuota, object count quota, replicaSet
		Description: Create a ResourceQuota. Creation MUST be successful and its ResourceQuotaStatus MUST match to expected used and total allowed resource quota count within namespace.
		Create a ReplicaSet. Its creation MUST be successful and resource usage count against the ReplicaSet object MUST be captured in ResourceQuotaStatus of the ResourceQuota.
		Delete the ReplicaSet. Deletion MUST succeed and resource usage count against the ReplicaSet object MUST be released from ResourceQuotaStatus of the ResourceQuota.
	*/
	framework.ConformanceIt("should create a ResourceQuota and capture the life of a replica set.", func() {
		ginkgo.By("Counting existing ResourceQuota")
		c, err := countResourceQuota(f.ClientSet, f.Namespace.Name)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		_, err = createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse(strconv.Itoa(c + 1))
		usedResources[v1.ResourceName("count/replicasets.apps")] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ReplicaSet")
		replicaSet := newTestReplicaSetForQuota("test-rs", "nginx", 0)
		replicaSet, err = f.ClientSet.AppsV1().ReplicaSets(f.Namespace.Name).Create(context.TODO(), replicaSet, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status captures replicaset creation")
		usedResources = v1.ResourceList{}
		usedResources[v1.ResourceName("count/replicasets.apps")] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting a ReplicaSet")
		err = f.ClientSet.AppsV1().ReplicaSets(f.Namespace.Name).Delete(context.TODO(), replicaSet.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released usage")
		usedResources[v1.ResourceName("count/replicasets.apps")] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.16
		Testname: ResourceQuota, object count quota, pvc
		Description: Create a ResourceQuota. Creation MUST be successful and its ResourceQuotaStatus MUST match to expected used and total allowed resource quota count within namespace.
		Create PersistentVolumeClaim (PVC) to request storage capacity of 1G. PVC creation MUST be successful and resource usage count against the PVC and storage object MUST be captured in ResourceQuotaStatus of the ResourceQuota.
		Delete the PVC. Deletion MUST succeed and resource usage count against its PVC and storage object MUST be released from ResourceQuotaStatus of the ResourceQuota.
		[NotConformancePromotable] as test suite do not have any e2e at this moment which are explicitly verifying PV and PVC behaviour.
	*/
	ginkgo.It("should create a ResourceQuota and capture the life of a persistent volume claim. [sig-storage]", func() {
		ginkgo.By("Counting existing ResourceQuota")
		c, err := countResourceQuota(f.ClientSet, f.Namespace.Name)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		_, err = createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse(strconv.Itoa(c + 1))
		usedResources[v1.ResourcePersistentVolumeClaims] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsStorage] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a PersistentVolumeClaim")
		pvc := newTestPersistentVolumeClaimForQuota("test-claim")
		pvc, err = f.ClientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Create(context.TODO(), pvc, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status captures persistent volume claim creation")
		usedResources = v1.ResourceList{}
		usedResources[v1.ResourcePersistentVolumeClaims] = resource.MustParse("1")
		usedResources[v1.ResourceRequestsStorage] = resource.MustParse("1Gi")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting a PersistentVolumeClaim")
		err = f.ClientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Delete(context.TODO(), pvc.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released usage")
		usedResources[v1.ResourcePersistentVolumeClaims] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsStorage] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.16
		Testname: ResourceQuota, object count quota, storageClass
		Description: Create a ResourceQuota. Creation MUST be successful and its ResourceQuotaStatus MUST match to expected used and total allowed resource quota count within namespace.
		Create PersistentVolumeClaim (PVC) with specified storageClass to request storage capacity of 1G. PVC creation MUST be successful and resource usage count against PVC, storageClass and storage object MUST be captured in ResourceQuotaStatus of the ResourceQuota.
		Delete the PVC. Deletion MUST succeed and resource usage count against  PVC, storageClass and storage object MUST be released from ResourceQuotaStatus of the ResourceQuota.
		[NotConformancePromotable] as test suite do not have any e2e at this moment which are explicitly verifying PV and PVC behaviour.
	*/
	ginkgo.It("should create a ResourceQuota and capture the life of a persistent volume claim with a storage class. [sig-storage]", func() {
		ginkgo.By("Counting existing ResourceQuota")
		c, err := countResourceQuota(f.ClientSet, f.Namespace.Name)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		_, err = createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse(strconv.Itoa(c + 1))
		usedResources[v1.ResourcePersistentVolumeClaims] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsStorage] = resource.MustParse("0")
		usedResources[core.V1ResourceByStorageClass(classGold, v1.ResourcePersistentVolumeClaims)] = resource.MustParse("0")
		usedResources[core.V1ResourceByStorageClass(classGold, v1.ResourceRequestsStorage)] = resource.MustParse("0")

		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a PersistentVolumeClaim with storage class")
		pvc := newTestPersistentVolumeClaimForQuota("test-claim")
		pvc.Spec.StorageClassName = &classGold
		pvc, err = f.ClientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Create(context.TODO(), pvc, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status captures persistent volume claim creation")
		usedResources = v1.ResourceList{}
		usedResources[v1.ResourcePersistentVolumeClaims] = resource.MustParse("1")
		usedResources[v1.ResourceRequestsStorage] = resource.MustParse("1Gi")
		usedResources[core.V1ResourceByStorageClass(classGold, v1.ResourcePersistentVolumeClaims)] = resource.MustParse("1")
		usedResources[core.V1ResourceByStorageClass(classGold, v1.ResourceRequestsStorage)] = resource.MustParse("1Gi")

		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting a PersistentVolumeClaim")
		err = f.ClientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Delete(context.TODO(), pvc.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released usage")
		usedResources[v1.ResourcePersistentVolumeClaims] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsStorage] = resource.MustParse("0")
		usedResources[core.V1ResourceByStorageClass(classGold, v1.ResourcePersistentVolumeClaims)] = resource.MustParse("0")
		usedResources[core.V1ResourceByStorageClass(classGold, v1.ResourceRequestsStorage)] = resource.MustParse("0")

		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)
	})

	ginkgo.It("should create a ResourceQuota and capture the life of a custom resource.", func() {
		ginkgo.By("Creating a Custom Resource Definition")
		testcrd, err := crd.CreateTestCRD(f)
		framework.ExpectNoError(err)
		defer testcrd.CleanUp()
		countResourceName := "count/" + testcrd.Crd.Spec.Names.Plural + "." + testcrd.Crd.Spec.Group
		// resourcequota controller needs to take 30 seconds at most to detect the new custom resource.
		// in order to make sure the resourcequota controller knows this resource, we create one test
		// resourcequota object, and triggering updates on it until the status is updated.
		quotaName := "quota-for-" + testcrd.Crd.Spec.Names.Plural
		_, err = createResourceQuota(f.ClientSet, f.Namespace.Name, &v1.ResourceQuota{
			ObjectMeta: metav1.ObjectMeta{Name: quotaName},
			Spec: v1.ResourceQuotaSpec{
				Hard: v1.ResourceList{
					v1.ResourceName(countResourceName): resource.MustParse("0"),
				},
			},
		})
		framework.ExpectNoError(err)
		err = updateResourceQuotaUntilUsageAppears(f.ClientSet, f.Namespace.Name, quotaName, v1.ResourceName(countResourceName))
		framework.ExpectNoError(err)
		err = f.ClientSet.CoreV1().ResourceQuotas(f.Namespace.Name).Delete(context.TODO(), quotaName, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Counting existing ResourceQuota")
		c, err := countResourceQuota(f.ClientSet, f.Namespace.Name)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ResourceQuota")
		quotaName = "test-quota"
		resourceQuota := newTestResourceQuota(quotaName)
		resourceQuota.Spec.Hard[v1.ResourceName(countResourceName)] = resource.MustParse("1")
		_, err = createResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuota)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourceQuotas] = resource.MustParse(strconv.Itoa(c + 1))
		usedResources[v1.ResourceName(countResourceName)] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a custom resource")
		resourceClient := testcrd.DynamicClients["v1"]
		testcr, err := instantiateCustomResource(&unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": testcrd.Crd.Spec.Group + "/" + testcrd.Crd.Spec.Versions[0].Name,
				"kind":       testcrd.Crd.Spec.Names.Kind,
				"metadata": map[string]interface{}{
					"name": "test-cr-1",
				},
			},
		}, resourceClient, testcrd.Crd)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status captures custom resource creation")
		usedResources = v1.ResourceList{}
		usedResources[v1.ResourceName(countResourceName)] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a second custom resource")
		_, err = instantiateCustomResource(&unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": testcrd.Crd.Spec.Group + "/" + testcrd.Crd.Spec.Versions[0].Name,
				"kind":       testcrd.Crd.Spec.Names.Kind,
				"metadata": map[string]interface{}{
					"name": "test-cr-2",
				},
			},
		}, resourceClient, testcrd.Crd)
		// since we only give one quota, this creation should fail.
		framework.ExpectError(err)

		ginkgo.By("Deleting a custom resource")
		err = deleteCustomResource(resourceClient, testcr.GetName())
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released usage")
		usedResources[v1.ResourceName(countResourceName)] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, quotaName, usedResources)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.16
		Testname: ResourceQuota, quota scope, Terminating and NotTerminating scope
		Description: Create two ResourceQuotas, one with 'Terminating' scope and another 'NotTerminating' scope. Request and the limit counts for CPU and Memory resources are set for the ResourceQuota. Creation MUST be successful and their ResourceQuotaStatus MUST match to expected used and total allowed resource quota count within namespace.
		Create a Pod with specified CPU and Memory ResourceRequirements fall within quota limits. Pod creation MUST be successful and usage count MUST be captured in ResourceQuotaStatus of 'NotTerminating' scoped ResourceQuota but MUST NOT in 'Terminating' scoped ResourceQuota.
		Delete the Pod. Pod deletion MUST succeed and Pod resource usage count MUST be released from ResourceQuotaStatus of 'NotTerminating' scoped ResourceQuota.
		Create a pod with specified activeDeadlineSeconds and resourceRequirements for CPU and Memory fall within quota limits. Pod creation MUST be successful and usage count MUST be captured in ResourceQuotaStatus of 'Terminating' scoped ResourceQuota but MUST NOT in 'NotTerminating' scoped ResourceQuota.
		Delete the Pod. Pod deletion MUST succeed and Pod resource usage count MUST be released from ResourceQuotaStatus of 'Terminating' scoped ResourceQuota.
	*/
	framework.ConformanceIt("should verify ResourceQuota with terminating scopes.", func() {
		ginkgo.By("Creating a ResourceQuota with terminating scope")
		quotaTerminatingName := "quota-terminating"
		resourceQuotaTerminating, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScope(quotaTerminatingName, v1.ResourceQuotaScopeTerminating))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring ResourceQuota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ResourceQuota with not terminating scope")
		quotaNotTerminatingName := "quota-not-terminating"
		resourceQuotaNotTerminating, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScope(quotaNotTerminatingName, v1.ResourceQuotaScopeNotTerminating))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring ResourceQuota status is calculated")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a long running pod")
		podName := "test-pod"
		requests := v1.ResourceList{}
		requests[v1.ResourceCPU] = resource.MustParse("500m")
		requests[v1.ResourceMemory] = resource.MustParse("200Mi")
		limits := v1.ResourceList{}
		limits[v1.ResourceCPU] = resource.MustParse("1")
		limits[v1.ResourceMemory] = resource.MustParse("400Mi")
		pod := newTestPodForQuota(f, podName, requests, limits)
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with not terminating scope captures the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		usedResources[v1.ResourceRequestsCPU] = requests[v1.ResourceCPU]
		usedResources[v1.ResourceRequestsMemory] = requests[v1.ResourceMemory]
		usedResources[v1.ResourceLimitsCPU] = limits[v1.ResourceCPU]
		usedResources[v1.ResourceLimitsMemory] = limits[v1.ResourceMemory]
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with terminating scope ignored the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting the pod")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), podName, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a terminating pod")
		podName = "terminating-pod"
		pod = newTestPodForQuota(f, podName, requests, limits)
		activeDeadlineSeconds := int64(3600)
		pod.Spec.ActiveDeadlineSeconds = &activeDeadlineSeconds
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with terminating scope captures the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		usedResources[v1.ResourceRequestsCPU] = requests[v1.ResourceCPU]
		usedResources[v1.ResourceRequestsMemory] = requests[v1.ResourceMemory]
		usedResources[v1.ResourceLimitsCPU] = limits[v1.ResourceCPU]
		usedResources[v1.ResourceLimitsMemory] = limits[v1.ResourceMemory]
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with not terminating scope ignored the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting the pod")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), podName, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.16
		Testname: ResourceQuota, quota scope, BestEffort and NotBestEffort scope
		Description: Create two ResourceQuotas, one with 'BestEffort' scope and another with 'NotBestEffort' scope. Creation MUST be successful and their ResourceQuotaStatus MUST match to expected used and total allowed resource quota count within namespace.
		Create a 'BestEffort' Pod by not explicitly specifying resource limits and requests. Pod creation MUST be successful and usage count MUST be captured in ResourceQuotaStatus of 'BestEffort' scoped ResourceQuota but MUST NOT in 'NotBestEffort' scoped ResourceQuota.
		Delete the Pod. Pod deletion MUST succeed and Pod resource usage count MUST be released from ResourceQuotaStatus of 'BestEffort' scoped ResourceQuota.
		Create a 'NotBestEffort' Pod by explicitly specifying resource limits and requests. Pod creation MUST be successful and usage count MUST be captured in ResourceQuotaStatus of 'NotBestEffort' scoped ResourceQuota but MUST NOT in 'BestEffort' scoped ResourceQuota.
		Delete the Pod. Pod deletion MUST succeed and Pod resource usage count MUST be released from ResourceQuotaStatus of 'NotBestEffort' scoped ResourceQuota.
	*/
	framework.ConformanceIt("should verify ResourceQuota with best effort scope.", func() {
		ginkgo.By("Creating a ResourceQuota with best effort scope")
		resourceQuotaBestEffort, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScope("quota-besteffort", v1.ResourceQuotaScopeBestEffort))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring ResourceQuota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ResourceQuota with not best effort scope")
		resourceQuotaNotBestEffort, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScope("quota-not-besteffort", v1.ResourceQuotaScopeNotBestEffort))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring ResourceQuota status is calculated")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a best-effort pod")
		pod := newTestPodForQuota(f, podName, v1.ResourceList{}, v1.ResourceList{})
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with best effort scope captures the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with not best effort ignored the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting the pod")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a not best-effort pod")
		requests := v1.ResourceList{}
		requests[v1.ResourceCPU] = resource.MustParse("500m")
		requests[v1.ResourceMemory] = resource.MustParse("200Mi")
		limits := v1.ResourceList{}
		limits[v1.ResourceCPU] = resource.MustParse("1")
		limits[v1.ResourceMemory] = resource.MustParse("400Mi")
		pod = newTestPodForQuota(f, "burstable-pod", requests, limits)
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with not best effort scope captures the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with best effort scope ignored the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting the pod")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.16
		Testname: ResourceQuota, update and delete
		Description: Create a ResourceQuota for CPU and Memory quota limits. Creation MUST be successful.
		When ResourceQuota is updated to modify CPU and Memory quota limits, update MUST succeed with updated values for CPU and Memory limits.
		When ResourceQuota is deleted, it MUST not be available in the namespace.
	*/
	framework.ConformanceIt("should be able to update and delete ResourceQuota.", func() {
		client := f.ClientSet
		ns := f.Namespace.Name

		ginkgo.By("Creating a ResourceQuota")
		quotaName := "test-quota"
		resourceQuota := &v1.ResourceQuota{
			Spec: v1.ResourceQuotaSpec{
				Hard: v1.ResourceList{},
			},
		}
		resourceQuota.ObjectMeta.Name = quotaName
		resourceQuota.Spec.Hard[v1.ResourceCPU] = resource.MustParse("1")
		resourceQuota.Spec.Hard[v1.ResourceMemory] = resource.MustParse("500Mi")
		_, err := createResourceQuota(client, ns, resourceQuota)
		framework.ExpectNoError(err)

		ginkgo.By("Getting a ResourceQuota")
		resourceQuotaResult, err := client.CoreV1().ResourceQuotas(ns).Get(context.TODO(), quotaName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(resourceQuotaResult.Spec.Hard[v1.ResourceCPU], resource.MustParse("1"))
		framework.ExpectEqual(resourceQuotaResult.Spec.Hard[v1.ResourceMemory], resource.MustParse("500Mi"))

		ginkgo.By("Updating a ResourceQuota")
		resourceQuota.Spec.Hard[v1.ResourceCPU] = resource.MustParse("2")
		resourceQuota.Spec.Hard[v1.ResourceMemory] = resource.MustParse("1Gi")
		resourceQuotaResult, err = client.CoreV1().ResourceQuotas(ns).Update(context.TODO(), resourceQuota, metav1.UpdateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(resourceQuotaResult.Spec.Hard[v1.ResourceCPU], resource.MustParse("2"))
		framework.ExpectEqual(resourceQuotaResult.Spec.Hard[v1.ResourceMemory], resource.MustParse("1Gi"))

		ginkgo.By("Verifying a ResourceQuota was modified")
		resourceQuotaResult, err = client.CoreV1().ResourceQuotas(ns).Get(context.TODO(), quotaName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(resourceQuotaResult.Spec.Hard[v1.ResourceCPU], resource.MustParse("2"))
		framework.ExpectEqual(resourceQuotaResult.Spec.Hard[v1.ResourceMemory], resource.MustParse("1Gi"))

		ginkgo.By("Deleting a ResourceQuota")
		err = deleteResourceQuota(client, ns, quotaName)
		framework.ExpectNoError(err)

		ginkgo.By("Verifying the deleted ResourceQuota")
		_, err = client.CoreV1().ResourceQuotas(ns).Get(context.TODO(), quotaName, metav1.GetOptions{})
		framework.ExpectEqual(apierrors.IsNotFound(err), true)
	})
})

var _ = SIGDescribe("ResourceQuota [Feature:ScopeSelectors]", func() {
	f := framework.NewDefaultFramework("scope-selectors")
	ginkgo.It("should verify ResourceQuota with best effort scope using scope-selectors.", func() {
		ginkgo.By("Creating a ResourceQuota with best effort scope")
		resourceQuotaBestEffort, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScopeSelector("quota-besteffort", v1.ResourceQuotaScopeBestEffort))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring ResourceQuota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ResourceQuota with not best effort scope")
		resourceQuotaNotBestEffort, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScopeSelector("quota-not-besteffort", v1.ResourceQuotaScopeNotBestEffort))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring ResourceQuota status is calculated")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a best-effort pod")
		pod := newTestPodForQuota(f, podName, v1.ResourceList{}, v1.ResourceList{})
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with best effort scope captures the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with not best effort ignored the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting the pod")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a not best-effort pod")
		requests := v1.ResourceList{}
		requests[v1.ResourceCPU] = resource.MustParse("500m")
		requests[v1.ResourceMemory] = resource.MustParse("200Mi")
		limits := v1.ResourceList{}
		limits[v1.ResourceCPU] = resource.MustParse("1")
		limits[v1.ResourceMemory] = resource.MustParse("400Mi")
		pod = newTestPodForQuota(f, "burstable-pod", requests, limits)
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with not best effort scope captures the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with best effort scope ignored the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaBestEffort.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting the pod")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotBestEffort.Name, usedResources)
		framework.ExpectNoError(err)
	})
	ginkgo.It("should verify ResourceQuota with terminating scopes through scope selectors.", func() {
		ginkgo.By("Creating a ResourceQuota with terminating scope")
		quotaTerminatingName := "quota-terminating"
		resourceQuotaTerminating, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScopeSelector(quotaTerminatingName, v1.ResourceQuotaScopeTerminating))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring ResourceQuota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a ResourceQuota with not terminating scope")
		quotaNotTerminatingName := "quota-not-terminating"
		resourceQuotaNotTerminating, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScopeSelector(quotaNotTerminatingName, v1.ResourceQuotaScopeNotTerminating))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring ResourceQuota status is calculated")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a long running pod")
		podName := "test-pod"
		requests := v1.ResourceList{}
		requests[v1.ResourceCPU] = resource.MustParse("500m")
		requests[v1.ResourceMemory] = resource.MustParse("200Mi")
		limits := v1.ResourceList{}
		limits[v1.ResourceCPU] = resource.MustParse("1")
		limits[v1.ResourceMemory] = resource.MustParse("400Mi")
		pod := newTestPodForQuota(f, podName, requests, limits)
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with not terminating scope captures the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		usedResources[v1.ResourceRequestsCPU] = requests[v1.ResourceCPU]
		usedResources[v1.ResourceRequestsMemory] = requests[v1.ResourceMemory]
		usedResources[v1.ResourceLimitsCPU] = limits[v1.ResourceCPU]
		usedResources[v1.ResourceLimitsMemory] = limits[v1.ResourceMemory]
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with terminating scope ignored the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting the pod")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), podName, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a terminating pod")
		podName = "terminating-pod"
		pod = newTestPodForQuota(f, podName, requests, limits)
		activeDeadlineSeconds := int64(3600)
		pod.Spec.ActiveDeadlineSeconds = &activeDeadlineSeconds
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with terminating scope captures the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		usedResources[v1.ResourceRequestsCPU] = requests[v1.ResourceCPU]
		usedResources[v1.ResourceRequestsMemory] = requests[v1.ResourceMemory]
		usedResources[v1.ResourceLimitsCPU] = limits[v1.ResourceCPU]
		usedResources[v1.ResourceLimitsMemory] = limits[v1.ResourceMemory]
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with not terminating scope ignored the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaNotTerminating.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting the pod")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), podName, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsMemory] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsMemory] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaTerminating.Name, usedResources)
		framework.ExpectNoError(err)
	})
})

var _ = SIGDescribe("ResourceQuota [Feature:PodPriority]", func() {
	f := framework.NewDefaultFramework("resourcequota-priorityclass")

	ginkgo.It("should verify ResourceQuota's priority class scope (quota set to pod count: 1) against a pod with same priority class.", func() {

		_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(context.TODO(), &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: "pclass1"}, Value: int32(1000)}, metav1.CreateOptions{})
		framework.ExpectEqual(err == nil || apierrors.IsAlreadyExists(err), true)

		hard := v1.ResourceList{}
		hard[v1.ResourcePods] = resource.MustParse("1")

		ginkgo.By("Creating a ResourceQuota with priority class scope")
		resourceQuotaPriorityClass, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScopeForPriorityClass("quota-priorityclass", hard, v1.ScopeSelectorOpIn, []string{"pclass1"}))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring ResourceQuota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a pod with priority class")
		podName := "testpod-pclass1"
		pod := newTestPodForQuotaWithPriority(f, podName, v1.ResourceList{}, v1.ResourceList{}, "pclass1")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with priority class scope captures the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting the pod")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)
	})

	ginkgo.It("should verify ResourceQuota's priority class scope (quota set to pod count: 1) against 2 pods with same priority class.", func() {

		_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(context.TODO(), &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: "pclass2"}, Value: int32(1000)}, metav1.CreateOptions{})
		framework.ExpectEqual(err == nil || apierrors.IsAlreadyExists(err), true)

		hard := v1.ResourceList{}
		hard[v1.ResourcePods] = resource.MustParse("1")

		ginkgo.By("Creating a ResourceQuota with priority class scope")
		resourceQuotaPriorityClass, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScopeForPriorityClass("quota-priorityclass", hard, v1.ScopeSelectorOpIn, []string{"pclass2"}))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring ResourceQuota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating first pod with priority class should pass")
		podName := "testpod-pclass2-1"
		pod := newTestPodForQuotaWithPriority(f, podName, v1.ResourceList{}, v1.ResourceList{}, "pclass2")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with priority class scope captures the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating 2nd pod with priority class should fail")
		podName2 := "testpod-pclass2-2"
		pod2 := newTestPodForQuotaWithPriority(f, podName2, v1.ResourceList{}, v1.ResourceList{}, "pclass2")
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod2, metav1.CreateOptions{})
		framework.ExpectError(err)

		ginkgo.By("Deleting first pod")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)
	})

	ginkgo.It("should verify ResourceQuota's priority class scope (quota set to pod count: 1) against 2 pods with different priority class.", func() {

		_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(context.TODO(), &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: "pclass3"}, Value: int32(1000)}, metav1.CreateOptions{})
		framework.ExpectEqual(err == nil || apierrors.IsAlreadyExists(err), true)

		hard := v1.ResourceList{}
		hard[v1.ResourcePods] = resource.MustParse("1")

		ginkgo.By("Creating a ResourceQuota with priority class scope")
		resourceQuotaPriorityClass, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScopeForPriorityClass("quota-priorityclass", hard, v1.ScopeSelectorOpIn, []string{"pclass4"}))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring ResourceQuota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a pod with priority class with pclass3")
		podName := "testpod-pclass3-1"
		pod := newTestPodForQuotaWithPriority(f, podName, v1.ResourceList{}, v1.ResourceList{}, "pclass3")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with priority class scope remains same")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a 2nd pod with priority class pclass3")
		podName2 := "testpod-pclass2-2"
		pod2 := newTestPodForQuotaWithPriority(f, podName2, v1.ResourceList{}, v1.ResourceList{}, "pclass3")
		pod2, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod2, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with priority class scope remains same")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting both pods")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod2.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)
	})

	ginkgo.It("should verify ResourceQuota's multiple priority class scope (quota set to pod count: 2) against 2 pods with same priority classes.", func() {
		_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(context.TODO(), &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: "pclass5"}, Value: int32(1000)}, metav1.CreateOptions{})
		framework.ExpectEqual(err == nil || apierrors.IsAlreadyExists(err), true)

		_, err = f.ClientSet.SchedulingV1().PriorityClasses().Create(context.TODO(), &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: "pclass6"}, Value: int32(1000)}, metav1.CreateOptions{})
		framework.ExpectEqual(err == nil || apierrors.IsAlreadyExists(err), true)

		hard := v1.ResourceList{}
		hard[v1.ResourcePods] = resource.MustParse("2")

		ginkgo.By("Creating a ResourceQuota with priority class scope")
		resourceQuotaPriorityClass, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScopeForPriorityClass("quota-priorityclass", hard, v1.ScopeSelectorOpIn, []string{"pclass5", "pclass6"}))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring ResourceQuota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a pod with priority class pclass5")
		podName := "testpod-pclass5"
		pod := newTestPodForQuotaWithPriority(f, podName, v1.ResourceList{}, v1.ResourceList{}, "pclass5")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with priority class is updated with the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating 2nd pod with priority class pclass6")
		podName2 := "testpod-pclass6"
		pod2 := newTestPodForQuotaWithPriority(f, podName2, v1.ResourceList{}, v1.ResourceList{}, "pclass6")
		pod2, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod2, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with priority class scope is updated with the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("2")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting both pods")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod2.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)
	})

	ginkgo.It("should verify ResourceQuota's priority class scope (quota set to pod count: 1) against a pod with different priority class (ScopeSelectorOpNotIn).", func() {

		_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(context.TODO(), &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: "pclass7"}, Value: int32(1000)}, metav1.CreateOptions{})
		framework.ExpectEqual(err == nil || apierrors.IsAlreadyExists(err), true)

		hard := v1.ResourceList{}
		hard[v1.ResourcePods] = resource.MustParse("1")

		ginkgo.By("Creating a ResourceQuota with priority class scope")
		resourceQuotaPriorityClass, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScopeForPriorityClass("quota-priorityclass", hard, v1.ScopeSelectorOpNotIn, []string{"pclass7"}))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring ResourceQuota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a pod with priority class pclass7")
		podName := "testpod-pclass7"
		pod := newTestPodForQuotaWithPriority(f, podName, v1.ResourceList{}, v1.ResourceList{}, "pclass7")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with priority class is not used")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting the pod")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)
	})

	ginkgo.It("should verify ResourceQuota's priority class scope (quota set to pod count: 1) against a pod with different priority class (ScopeSelectorOpExists).", func() {

		_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(context.TODO(), &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: "pclass8"}, Value: int32(1000)}, metav1.CreateOptions{})
		framework.ExpectEqual(err == nil || apierrors.IsAlreadyExists(err), true)

		hard := v1.ResourceList{}
		hard[v1.ResourcePods] = resource.MustParse("1")

		ginkgo.By("Creating a ResourceQuota with priority class scope")
		resourceQuotaPriorityClass, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScopeForPriorityClass("quota-priorityclass", hard, v1.ScopeSelectorOpExists, []string{}))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring ResourceQuota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a pod with priority class pclass8")
		podName := "testpod-pclass8"
		pod := newTestPodForQuotaWithPriority(f, podName, v1.ResourceList{}, v1.ResourceList{}, "pclass8")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with priority class is updated with the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting the pod")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)
	})

	ginkgo.It("should verify ResourceQuota's priority class scope (cpu, memory quota set) against a pod with same priority class.", func() {

		_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(context.TODO(), &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: "pclass9"}, Value: int32(1000)}, metav1.CreateOptions{})
		framework.ExpectEqual(err == nil || apierrors.IsAlreadyExists(err), true)

		hard := v1.ResourceList{}
		hard[v1.ResourcePods] = resource.MustParse("1")
		hard[v1.ResourceRequestsCPU] = resource.MustParse("1")
		hard[v1.ResourceRequestsMemory] = resource.MustParse("1Gi")
		hard[v1.ResourceLimitsCPU] = resource.MustParse("3")
		hard[v1.ResourceLimitsMemory] = resource.MustParse("3Gi")

		ginkgo.By("Creating a ResourceQuota with priority class scope")
		resourceQuotaPriorityClass, err := createResourceQuota(f.ClientSet, f.Namespace.Name, newTestResourceQuotaWithScopeForPriorityClass("quota-priorityclass", hard, v1.ScopeSelectorOpIn, []string{"pclass9"}))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring ResourceQuota status is calculated")
		usedResources := v1.ResourceList{}
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsMemory] = resource.MustParse("0Gi")
		usedResources[v1.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsMemory] = resource.MustParse("0Gi")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a pod with priority class")
		podName := "testpod-pclass9"
		request := v1.ResourceList{}
		request[v1.ResourceCPU] = resource.MustParse("1")
		request[v1.ResourceMemory] = resource.MustParse("1Gi")
		limit := v1.ResourceList{}
		limit[v1.ResourceCPU] = resource.MustParse("2")
		limit[v1.ResourceMemory] = resource.MustParse("2Gi")

		pod := newTestPodForQuotaWithPriority(f, podName, request, limit, "pclass9")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota with priority class scope captures the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("1")
		usedResources[v1.ResourceRequestsCPU] = resource.MustParse("1")
		usedResources[v1.ResourceRequestsMemory] = resource.MustParse("1Gi")
		usedResources[v1.ResourceLimitsCPU] = resource.MustParse("2")
		usedResources[v1.ResourceLimitsMemory] = resource.MustParse("2Gi")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)

		ginkgo.By("Deleting the pod")
		err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring resource quota status released the pod usage")
		usedResources[v1.ResourcePods] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceRequestsMemory] = resource.MustParse("0Gi")
		usedResources[v1.ResourceLimitsCPU] = resource.MustParse("0")
		usedResources[v1.ResourceLimitsMemory] = resource.MustParse("0Gi")
		err = waitForResourceQuota(f.ClientSet, f.Namespace.Name, resourceQuotaPriorityClass.Name, usedResources)
		framework.ExpectNoError(err)
	})

})

// newTestResourceQuotaWithScopeSelector returns a quota that enforces default constraints for testing with scopeSelectors
func newTestResourceQuotaWithScopeSelector(name string, scope v1.ResourceQuotaScope) *v1.ResourceQuota {
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
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.ResourceQuotaSpec{Hard: hard,
			ScopeSelector: &v1.ScopeSelector{
				MatchExpressions: []v1.ScopedResourceSelectorRequirement{
					{
						ScopeName: scope,
						Operator:  v1.ScopeSelectorOpExists},
				},
			},
		},
	}
}

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
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       v1.ResourceQuotaSpec{Hard: hard, Scopes: []v1.ResourceQuotaScope{scope}},
	}
}

// newTestResourceQuotaWithScopeForPriorityClass returns a quota
// that enforces default constraints for testing with ResourceQuotaScopePriorityClass scope
func newTestResourceQuotaWithScopeForPriorityClass(name string, hard v1.ResourceList, op v1.ScopeSelectorOperator, values []string) *v1.ResourceQuota {
	return &v1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.ResourceQuotaSpec{Hard: hard,
			ScopeSelector: &v1.ScopeSelector{
				MatchExpressions: []v1.ScopedResourceSelectorRequirement{
					{
						ScopeName: v1.ResourceQuotaScopePriorityClass,
						Operator:  op,
						Values:    values,
					},
				},
			},
		},
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
	hard[v1.ResourceEphemeralStorage] = resource.MustParse("50Gi")
	hard[core.V1ResourceByStorageClass(classGold, v1.ResourcePersistentVolumeClaims)] = resource.MustParse("10")
	hard[core.V1ResourceByStorageClass(classGold, v1.ResourceRequestsStorage)] = resource.MustParse("10Gi")
	// test quota on discovered resource type
	hard[v1.ResourceName("count/replicasets.apps")] = resource.MustParse("5")
	// test quota on extended resource
	hard[v1.ResourceName(v1.DefaultResourceRequestsPrefix+extendedResourceName)] = resource.MustParse("3")
	return &v1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       v1.ResourceQuotaSpec{Hard: hard},
	}
}

// newTestPodForQuota returns a pod that has the specified requests and limits
func newTestPodForQuota(f *framework.Framework, name string, requests v1.ResourceList, limits v1.ResourceList) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			// prevent disruption to other test workloads in parallel test runs by ensuring the quota
			// test pods don't get scheduled onto a node
			NodeSelector: map[string]string{
				"x-test.k8s.io/unsatisfiable": "not-schedulable",
			},
			Containers: []v1.Container{
				{
					Name:  "pause",
					Image: imageutils.GetPauseImageName(),
					Resources: v1.ResourceRequirements{
						Requests: requests,
						Limits:   limits,
					},
				},
			},
		},
	}
}

// newTestPodForQuotaWithPriority returns a pod that has the specified requests, limits and priority class
func newTestPodForQuotaWithPriority(f *framework.Framework, name string, requests v1.ResourceList, limits v1.ResourceList, pclass string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			// prevent disruption to other test workloads in parallel test runs by ensuring the quota
			// test pods don't get scheduled onto a node
			NodeSelector: map[string]string{
				"x-test.k8s.io/unsatisfiable": "not-schedulable",
			},
			Containers: []v1.Container{
				{
					Name:  "pause",
					Image: imageutils.GetPauseImageName(),
					Resources: v1.ResourceRequirements{
						Requests: requests,
						Limits:   limits,
					},
				},
			},
			PriorityClassName: pclass,
		},
	}
}

// newTestPersistentVolumeClaimForQuota returns a simple persistent volume claim
func newTestPersistentVolumeClaimForQuota(name string) *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
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
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: func(i int32) *int32 { return &i }(replicas),
			Selector: map[string]string{
				"name": name,
			},
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
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

// newTestReplicaSetForQuota returns a simple replica set
func newTestReplicaSetForQuota(name, image string, replicas int32) *appsv1.ReplicaSet {
	zero := int64(0)
	return &appsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: appsv1.ReplicaSetSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"name": name}},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"name": name},
				},
				Spec: v1.PodSpec{
					TerminationGracePeriodSeconds: &zero,
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
		ObjectMeta: metav1.ObjectMeta{
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
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Data: map[string]string{
			"a": "b",
		},
	}
}

func newTestSecretForQuota(name string) *v1.Secret {
	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
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
	return c.CoreV1().ResourceQuotas(namespace).Create(context.TODO(), resourceQuota, metav1.CreateOptions{})
}

// deleteResourceQuota with the specified name
func deleteResourceQuota(c clientset.Interface, namespace, name string) error {
	return c.CoreV1().ResourceQuotas(namespace).Delete(context.TODO(), name, metav1.DeleteOptions{})
}

// countResourceQuota counts the number of ResourceQuota in the specified namespace
// On contended servers the service account controller can slow down, leading to the count changing during a run.
// Wait up to 5s for the count to stabilize, assuming that updates come at a consistent rate, and are not held indefinitely.
func countResourceQuota(c clientset.Interface, namespace string) (int, error) {
	found, unchanged := 0, 0
	return found, wait.Poll(1*time.Second, 30*time.Second, func() (bool, error) {
		resourceQuotas, err := c.CoreV1().ResourceQuotas(namespace).List(context.TODO(), metav1.ListOptions{})
		framework.ExpectNoError(err)
		if len(resourceQuotas.Items) == found {
			// loop until the number of resource quotas has stabilized for 5 seconds
			unchanged++
			return unchanged > 4, nil
		}
		unchanged = 0
		found = len(resourceQuotas.Items)
		return false, nil
	})
}

// wait for resource quota status to show the expected used resources value
func waitForResourceQuota(c clientset.Interface, ns, quotaName string, used v1.ResourceList) error {
	return wait.Poll(framework.Poll, resourceQuotaTimeout, func() (bool, error) {
		resourceQuota, err := c.CoreV1().ResourceQuotas(ns).Get(context.TODO(), quotaName, metav1.GetOptions{})
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

// updateResourceQuotaUntilUsageAppears updates the resource quota object until the usage is populated
// for the specific resource name.
func updateResourceQuotaUntilUsageAppears(c clientset.Interface, ns, quotaName string, resourceName v1.ResourceName) error {
	return wait.Poll(framework.Poll, 1*time.Minute, func() (bool, error) {
		resourceQuota, err := c.CoreV1().ResourceQuotas(ns).Get(context.TODO(), quotaName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		// verify that the quota shows the expected used resource values
		_, ok := resourceQuota.Status.Used[resourceName]
		if ok {
			return true, nil
		}

		current := resourceQuota.Spec.Hard[resourceName]
		current.Add(resource.MustParse("1"))
		resourceQuota.Spec.Hard[resourceName] = current
		_, err = c.CoreV1().ResourceQuotas(ns).Update(context.TODO(), resourceQuota, metav1.UpdateOptions{})
		// ignoring conflicts since someone else may already updated it.
		if apierrors.IsConflict(err) {
			return false, nil
		}
		return false, err
	})
}
