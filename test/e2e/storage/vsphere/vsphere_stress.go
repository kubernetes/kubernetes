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

package vsphere

import (
	"fmt"
	"sync"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	storageV1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

/*
	Induce stress to create volumes in parallel with multiple threads based on user configurable values for number of threads and iterations per thread.
	The following actions will be performed as part of this test.

	1. Create Storage Classes of 4 Categories (Default, SC with Non Default Datastore, SC with SPBM Policy, SC with VSAN Storage Capalibilies.)
	2. READ VCP_STRESS_INSTANCES, VCP_STRESS_ITERATIONS, VSPHERE_SPBM_POLICY_NAME and VSPHERE_DATASTORE from System Environment.
	3. Launch goroutine for volume lifecycle operations.
	4. Each instance of routine iterates for n times, where n is read from system env - VCP_STRESS_ITERATIONS
	5. Each iteration creates 1 PVC, 1 POD using the provisioned PV, Verify disk is attached to the node, Verify pod can access the volume, delete the pod and finally delete the PVC.
*/
var _ = utils.SIGDescribe("vsphere cloud provider stress [Feature:vsphere]", func() {
	f := framework.NewDefaultFramework("vcp-stress")
	var (
		client        clientset.Interface
		namespace     string
		instances     int
		iterations    int
		policyName    string
		datastoreName string
		err           error
		scNames       = []string{storageclass1, storageclass2, storageclass3, storageclass4}
	)

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		client = f.ClientSet
		namespace = f.Namespace.Name

		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		Expect(nodeList.Items).NotTo(BeEmpty(), "Unable to find ready and schedulable Node")

		// if VCP_STRESS_INSTANCES = 12 and VCP_STRESS_ITERATIONS is 10. 12 threads will run in parallel for 10 times.
		// Resulting 120 Volumes and POD Creation. Volumes will be provisioned with each different types of Storage Class,
		// Each iteration creates PVC, verify PV is provisioned, then creates a pod, verify volume is attached to the node, and then delete the pod and delete pvc.
		instances = GetAndExpectIntEnvVar(VCPStressInstances)
		Expect(instances <= volumesPerNode*len(nodeList.Items)).To(BeTrue(), fmt.Sprintf("Number of Instances should be less or equal: %v", volumesPerNode*len(nodeList.Items)))
		Expect(instances > len(scNames)).To(BeTrue(), "VCP_STRESS_INSTANCES should be greater than 3 to utilize all 4 types of storage classes")

		iterations = GetAndExpectIntEnvVar(VCPStressIterations)
		Expect(err).NotTo(HaveOccurred(), "Error Parsing VCP_STRESS_ITERATIONS")
		Expect(iterations > 0).To(BeTrue(), "VCP_STRESS_ITERATIONS should be greater than 0")

		policyName = GetAndExpectStringEnvVar(SPBMPolicyName)
		datastoreName = GetAndExpectStringEnvVar(StorageClassDatastoreName)
	})

	It("vsphere stress tests", func() {
		scArrays := make([]*storageV1.StorageClass, len(scNames))
		for index, scname := range scNames {
			// Create vSphere Storage Class
			By(fmt.Sprintf("Creating Storage Class : %v", scname))
			var sc *storageV1.StorageClass
			var err error
			switch scname {
			case storageclass1:
				sc, err = client.StorageV1().StorageClasses().Create(getVSphereStorageClassSpec(storageclass1, nil))
			case storageclass2:
				var scVSanParameters map[string]string
				scVSanParameters = make(map[string]string)
				scVSanParameters[Policy_HostFailuresToTolerate] = "1"
				sc, err = client.StorageV1().StorageClasses().Create(getVSphereStorageClassSpec(storageclass2, scVSanParameters))
			case storageclass3:
				var scSPBMPolicyParameters map[string]string
				scSPBMPolicyParameters = make(map[string]string)
				scSPBMPolicyParameters[SpbmStoragePolicy] = policyName
				sc, err = client.StorageV1().StorageClasses().Create(getVSphereStorageClassSpec(storageclass3, scSPBMPolicyParameters))
			case storageclass4:
				var scWithDSParameters map[string]string
				scWithDSParameters = make(map[string]string)
				scWithDSParameters[Datastore] = datastoreName
				scWithDatastoreSpec := getVSphereStorageClassSpec(storageclass4, scWithDSParameters)
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
			instanceId := fmt.Sprintf("Thread:%v", instanceCount+1)
			go PerformVolumeLifeCycleInParallel(f, client, namespace, instanceId, scArrays[instanceCount%len(scArrays)], iterations, &wg)
		}
		wg.Wait()
	})

})

// goroutine to perform volume lifecycle operations in parallel
func PerformVolumeLifeCycleInParallel(f *framework.Framework, client clientset.Interface, namespace string, instanceId string, sc *storageV1.StorageClass, iterations int, wg *sync.WaitGroup) {
	defer wg.Done()
	defer GinkgoRecover()

	for iterationCount := 0; iterationCount < iterations; iterationCount++ {
		logPrefix := fmt.Sprintf("Instance: [%v], Iteration: [%v] :", instanceId, iterationCount+1)
		By(fmt.Sprintf("%v Creating PVC using the Storage Class: %v", logPrefix, sc.Name))
		pvclaim, err := framework.CreatePVC(client, namespace, getVSphereClaimSpecWithStorageClass(namespace, "1Gi", sc))
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

		By(fmt.Sprintf("%v Verifing the volume: %v is attached to the node VM: %v", logPrefix, persistentvolumes[0].Spec.VsphereVolume.VolumePath, pod.Spec.NodeName))
		isVolumeAttached, verifyDiskAttachedError := diskIsAttached(persistentvolumes[0].Spec.VsphereVolume.VolumePath, pod.Spec.NodeName)
		Expect(isVolumeAttached).To(BeTrue())
		Expect(verifyDiskAttachedError).NotTo(HaveOccurred())

		By(fmt.Sprintf("%v Verifing the volume: %v is accessible in the pod: %v", logPrefix, persistentvolumes[0].Spec.VsphereVolume.VolumePath, pod.Name))
		verifyVSphereVolumesAccessible(client, pod, persistentvolumes)

		By(fmt.Sprintf("%v Deleting pod: %v", logPrefix, pod.Name))
		err = framework.DeletePodWithWait(f, client, pod)
		Expect(err).NotTo(HaveOccurred())

		By(fmt.Sprintf("%v Waiting for volume: %v to be detached from the node: %v", logPrefix, persistentvolumes[0].Spec.VsphereVolume.VolumePath, pod.Spec.NodeName))
		err = waitForVSphereDiskToDetach(persistentvolumes[0].Spec.VsphereVolume.VolumePath, pod.Spec.NodeName)
		Expect(err).NotTo(HaveOccurred())

		By(fmt.Sprintf("%v Deleting the Claim: %v", logPrefix, pvclaim.Name))
		Expect(framework.DeletePersistentVolumeClaim(client, pvclaim.Name, namespace)).NotTo(HaveOccurred())
	}
}
