/*
Copyright 2017 The Kubernetes Authors.

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

package openstack

import (
	"fmt"
	"os"
	"strconv"
	"sync"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	storageV1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	utils "k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("openstack cloud provider stress [Feature:openstack]", func() {
	f := framework.NewDefaultFramework("osp-stress")
	var (
		client        clientset.Interface
		namespace     string
		instances     int
		iterations    int
		err           error
		scNames       = []string{storageclass1, storageclass2, storageclass3, storageclass4}
		policyName    string
		datastoreName string
	)

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("openstack")
		client = f.ClientSet
		namespace = f.Namespace.Name

		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		Expect(nodeList.Items).NotTo(BeEmpty(), "Unable to find ready and schedulable Node")

		instancesStr := os.Getenv("OSP_STRESS_INSTANCES")
		Expect(instancesStr).NotTo(BeEmpty(), "ENV OSP_STRESS_INSTANCES is not set")
		instances, err = strconv.Atoi(instancesStr)
		Expect(err).NotTo(HaveOccurred(), "error parsing OSP-STRESS-INSTANCES: %v", err)
		Expect(instances <= volumesPerNode*len(nodeList.Items)).To(BeTrue(), fmt.Sprintf("number of Instances (%d) should be less or equal: %v", instances, volumesPerNode*len(nodeList.Items)))
		Expect(instances > len(scNames)).To(BeTrue(), "OSP_STRESS_INSTANCES should be less or equal: %v to utilize all 4 types of storage classes", volumesPerNode*len(nodeList.Items))

		iterationStr := os.Getenv("OSP_STRESS_ITERATIONS")
		Expect(instancesStr).NotTo(BeEmpty(), "ENV OSP_STRESS_ITERATIONS is not set")
		iterations, err = strconv.Atoi(iterationStr)
		Expect(err).NotTo(HaveOccurred(), "error parsing OSP_STRESS_ITERATIONS: %v", err)
		Expect(iterations > 0).To(BeTrue(), "OSP_STRESS_ITERATIONS should be greater than 0")

		policyName = os.Getenv("OPENSTACK_SPBM_POLICY_NAME")
		datastoreName = os.Getenv("OPENSTACK_DATASTORE")
		Expect(policyName).NotTo(BeEmpty(), "ENV OPENSTACK_SPBM_POLICY_NAME is not set")
		Expect(datastoreName).NotTo(BeEmpty(), "ENV OPENSTACK_DATASTORE is not set")
	})

	It("openstack stress tests", func() {
		scArrays := make([]*storageV1.StorageClass, len(scNames))
		for index, scname := range scNames {
			// Create openstack Storage Class
			By(fmt.Sprintf("Creating Storage Class : %v", scname))
			var sc *storageV1.StorageClass
			var err error
			switch scname {
			case storageclass1:
				sc, err = client.StorageV1().StorageClasses().Create(getOpenstackStorageClassSpec(storageclass1, nil))
			case storageclass2:
				var scOSanParameters map[string]string
				scOSanParameters = make(map[string]string)
				scOSanParameters[PolicyHostFailuresToTolerate] = "1"
				sc, err = client.StorageV1().StorageClasses().Create(getOpenstackStorageClassSpec(storageclass2, scOSanParameters))
			case storageclass3:
				var scSPBMPolicyParameters map[string]string
				scSPBMPolicyParameters = make(map[string]string)
				scSPBMPolicyParameters[SpbmStoragePolicy] = policyName
				sc, err = client.StorageV1().StorageClasses().Create(getOpenstackStorageClassSpec(storageclass3, scSPBMPolicyParameters))
			case storageclass4:
				var scWithDSParameters map[string]string
				scWithDSParameters = make(map[string]string)
				scWithDSParameters[Datastore] = datastoreName
				scWithDatastoreSpec := getOpenstackStorageClassSpec(storageclass4, scWithDSParameters)
				sc, err = client.StorageV1().StorageClasses().Create(scWithDatastoreSpec)
			}
			Expect(sc).NotTo(BeNil())
			Expect(err).NotTo(HaveOccurred())
			defer client.StorageV1().StorageClasses().Delete(scname, nil)
			scArrays[index] = sc
		}

		var wg sync.WaitGroup
		wg.Add(instances)
		for instanceCount := 0; instanceCount < instances; instanceCount++ {
			instanceID := fmt.Sprintf("Thread:%v", instanceCount+1)
			go PerformVolumeLifeCycleInParallel(f, client, namespace, instanceID, scArrays[instanceCount%len(scArrays)], iterations, &wg)
		}
		wg.Wait()
	})

})

// PerformVolumeLifeCycleInParallel performs volume lifecycle operations in parallel
func PerformVolumeLifeCycleInParallel(f *framework.Framework, client clientset.Interface, namespace string, instanceID string, sc *storageV1.StorageClass, iterations int, wg *sync.WaitGroup) {
	defer wg.Done()
	defer GinkgoRecover()
	osp, _, err := getOpenstack(f.ClientSet)
	Expect(err).NotTo(HaveOccurred())
	for iterationCount := 0; iterationCount < iterations; iterationCount++ {
		logPrefix := fmt.Sprintf("Instance: [%v], Iteration: [%v] :", instanceID, iterationCount+1)
		By(fmt.Sprintf("%v Creating PVC using the Storage Class: %v", logPrefix, sc.Name))
		pvclaim, err := framework.CreatePVC(client, namespace, getOpenstackClaimSpecWithStorageClassAnnotation(namespace, "1Gi", sc))
		Expect(err).NotTo(HaveOccurred())
		defer framework.DeletePersistentVolumeClaim(client, pvclaim.Name, namespace)

		var pvclaims []*v1.PersistentVolumeClaim
		pvclaims = append(pvclaims, pvclaim)
		By(fmt.Sprintf("%v Waiting for claim: %v to be in bound phase", logPrefix, pvclaim.Name))
		persistentvolumes, err := framework.WaitForPVClaimBoundPhase(client, pvclaims, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred())

		By(fmt.Sprintf("%v Creating Pod using the claim: %v", logPrefix, pvclaim.Name))
		// Create pod to attach Volume to Node
		pod, err := framework.CreatePod(client, namespace, nil, pvclaims, false, "")
		Expect(err).NotTo(HaveOccurred())

		By(fmt.Sprintf("%v Waiting for the Pod: %v to be in the running state", logPrefix, pod.Name))
		Expect(f.WaitForPodRunningSlow(pod.Name)).NotTo(HaveOccurred())

		// Get the copy of the Pod to know the assigned node name.
		pod, err = client.CoreV1().Pods(namespace).Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		By(fmt.Sprintf("%v Verifing the volume: %v is attached to the node VM: %v", logPrefix, persistentvolumes[0].Spec.Cinder.VolumeID, pod.Spec.NodeName))
		isVolumeAttached, verifyDiskAttachedError := verifyOpenstackDiskAttached(client, osp, instanceID, persistentvolumes[0].Spec.Cinder.VolumeID, types.NodeName(pod.Spec.NodeName))
		Expect(isVolumeAttached).To(BeTrue())
		Expect(verifyDiskAttachedError).NotTo(HaveOccurred())

		By(fmt.Sprintf("%v Verifing the volume: %v is accessible in the pod: %v", logPrefix, persistentvolumes[0].Spec.Cinder.VolumeID, pod.Name))
		verifyOpenstackVolumesAccessible(client, pod, persistentvolumes, instanceID, osp)

		By(fmt.Sprintf("%v Deleting pod: %v", logPrefix, pod.Name))
		err = framework.DeletePodWithWait(f, client, pod)
		Expect(err).NotTo(HaveOccurred())

		By(fmt.Sprintf("%v Waiting for volume: %v to be detached from the node: %v", logPrefix, persistentvolumes[0].Spec.Cinder.VolumeID, pod.Spec.NodeName))
		WaitForVolumeStatus(osp, persistentvolumes[0].Spec.Cinder.VolumeID, VolumeAvailableStatus)

		By(fmt.Sprintf("%v Deleting the Claim: %v", logPrefix, pvclaim.Name))
		Expect(framework.DeletePersistentVolumeClaim(client, pvclaim.Name, namespace)).NotTo(HaveOccurred())
	}
}
