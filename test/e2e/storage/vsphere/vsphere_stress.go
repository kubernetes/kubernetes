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
	"context"
	"fmt"
	"sync"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

/*
	Induce stress to create volumes in parallel with multiple threads based on user configurable values for number of threads and iterations per thread.
	The following actions will be performed as part of this test.

	1. Create Storage Classes of 4 Categories (Default, SC with Non Default Datastore, SC with SPBM Policy, SC with VSAN Storage Capabilities.)
	2. READ VCP_STRESS_INSTANCES, VCP_STRESS_ITERATIONS, VSPHERE_SPBM_POLICY_NAME and VSPHERE_DATASTORE from System Environment.
	3. Launch goroutine for volume lifecycle operations.
	4. Each instance of routine iterates for n times, where n is read from system env - VCP_STRESS_ITERATIONS
	5. Each iteration creates 1 PVC, 1 POD using the provisioned PV, Verify disk is attached to the node, Verify pod can access the volume, delete the pod and finally delete the PVC.
*/
var _ = utils.SIGDescribe("vsphere cloud provider stress [Feature:vsphere]", func() {
	f := framework.NewDefaultFramework("vcp-stress")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	var (
		client        clientset.Interface
		namespace     string
		instances     int
		iterations    int
		policyName    string
		datastoreName string
		scNames       = []string{storageclass1, storageclass2, storageclass3, storageclass4}
	)

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("vsphere")
		Bootstrap(f)
		client = f.ClientSet
		namespace = f.Namespace.Name

		nodeList, err := e2enode.GetReadySchedulableNodes(f.ClientSet)
		framework.ExpectNoError(err)

		// if VCP_STRESS_INSTANCES = 12 and VCP_STRESS_ITERATIONS is 10. 12 threads will run in parallel for 10 times.
		// Resulting 120 Volumes and POD Creation. Volumes will be provisioned with each different types of Storage Class,
		// Each iteration creates PVC, verify PV is provisioned, then creates a pod, verify volume is attached to the node, and then delete the pod and delete pvc.
		instances = GetAndExpectIntEnvVar(VCPStressInstances)
		framework.ExpectEqual(instances <= volumesPerNode*len(nodeList.Items), true, fmt.Sprintf("Number of Instances should be less or equal: %v", volumesPerNode*len(nodeList.Items)))
		framework.ExpectEqual(instances > len(scNames), true, "VCP_STRESS_INSTANCES should be greater than 3 to utilize all 4 types of storage classes")

		iterations = GetAndExpectIntEnvVar(VCPStressIterations)
		framework.ExpectEqual(iterations > 0, true, "VCP_STRESS_ITERATIONS should be greater than 0")

		policyName = GetAndExpectStringEnvVar(SPBMPolicyName)
		datastoreName = GetAndExpectStringEnvVar(StorageClassDatastoreName)
	})

	ginkgo.It("vsphere stress tests", func() {
		scArrays := make([]*storagev1.StorageClass, len(scNames))
		for index, scname := range scNames {
			// Create vSphere Storage Class
			ginkgo.By(fmt.Sprintf("Creating Storage Class : %v", scname))
			var sc *storagev1.StorageClass
			var err error
			switch scname {
			case storageclass1:
				sc, err = client.StorageV1().StorageClasses().Create(context.TODO(), getVSphereStorageClassSpec(storageclass1, nil, nil, ""), metav1.CreateOptions{})
			case storageclass2:
				var scVSanParameters map[string]string
				scVSanParameters = make(map[string]string)
				scVSanParameters[PolicyHostFailuresToTolerate] = "1"
				sc, err = client.StorageV1().StorageClasses().Create(context.TODO(), getVSphereStorageClassSpec(storageclass2, scVSanParameters, nil, ""), metav1.CreateOptions{})
			case storageclass3:
				var scSPBMPolicyParameters map[string]string
				scSPBMPolicyParameters = make(map[string]string)
				scSPBMPolicyParameters[SpbmStoragePolicy] = policyName
				sc, err = client.StorageV1().StorageClasses().Create(context.TODO(), getVSphereStorageClassSpec(storageclass3, scSPBMPolicyParameters, nil, ""), metav1.CreateOptions{})
			case storageclass4:
				var scWithDSParameters map[string]string
				scWithDSParameters = make(map[string]string)
				scWithDSParameters[Datastore] = datastoreName
				scWithDatastoreSpec := getVSphereStorageClassSpec(storageclass4, scWithDSParameters, nil, "")
				sc, err = client.StorageV1().StorageClasses().Create(context.TODO(), scWithDatastoreSpec, metav1.CreateOptions{})
			}
			gomega.Expect(sc).NotTo(gomega.BeNil())
			framework.ExpectNoError(err)
			defer client.StorageV1().StorageClasses().Delete(context.TODO(), scname, metav1.DeleteOptions{})
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

// PerformVolumeLifeCycleInParallel performs volume lifecycle operations
// Called as a go routine to perform operations in parallel
func PerformVolumeLifeCycleInParallel(f *framework.Framework, client clientset.Interface, namespace string, instanceID string, sc *storagev1.StorageClass, iterations int, wg *sync.WaitGroup) {
	defer wg.Done()
	defer ginkgo.GinkgoRecover()

	for iterationCount := 0; iterationCount < iterations; iterationCount++ {
		logPrefix := fmt.Sprintf("Instance: [%v], Iteration: [%v] :", instanceID, iterationCount+1)
		ginkgo.By(fmt.Sprintf("%v Creating PVC using the Storage Class: %v", logPrefix, sc.Name))
		pvclaim, err := e2epv.CreatePVC(client, namespace, getVSphereClaimSpecWithStorageClass(namespace, "1Gi", sc))
		framework.ExpectNoError(err)
		defer e2epv.DeletePersistentVolumeClaim(client, pvclaim.Name, namespace)

		var pvclaims []*v1.PersistentVolumeClaim
		pvclaims = append(pvclaims, pvclaim)
		ginkgo.By(fmt.Sprintf("%v Waiting for claim: %v to be in bound phase", logPrefix, pvclaim.Name))
		persistentvolumes, err := e2epv.WaitForPVClaimBoundPhase(client, pvclaims, f.Timeouts.ClaimProvision)
		framework.ExpectNoError(err)

		ginkgo.By(fmt.Sprintf("%v Creating Pod using the claim: %v", logPrefix, pvclaim.Name))
		// Create pod to attach Volume to Node
		pod, err := e2epod.CreatePod(client, namespace, nil, pvclaims, false, "")
		framework.ExpectNoError(err)

		ginkgo.By(fmt.Sprintf("%v Waiting for the Pod: %v to be in the running state", logPrefix, pod.Name))
		err = e2epod.WaitTimeoutForPodRunningInNamespace(f.ClientSet, pod.Name, f.Namespace.Name, f.Timeouts.PodStartSlow)
		framework.ExpectNoError(err)

		// Get the copy of the Pod to know the assigned node name.
		pod, err = client.CoreV1().Pods(namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		ginkgo.By(fmt.Sprintf("%v Verifying the volume: %v is attached to the node VM: %v", logPrefix, persistentvolumes[0].Spec.VsphereVolume.VolumePath, pod.Spec.NodeName))
		isVolumeAttached, verifyDiskAttachedError := diskIsAttached(persistentvolumes[0].Spec.VsphereVolume.VolumePath, pod.Spec.NodeName)
		framework.ExpectEqual(isVolumeAttached, true)
		framework.ExpectNoError(verifyDiskAttachedError)

		ginkgo.By(fmt.Sprintf("%v Verifying the volume: %v is accessible in the pod: %v", logPrefix, persistentvolumes[0].Spec.VsphereVolume.VolumePath, pod.Name))
		verifyVSphereVolumesAccessible(client, pod, persistentvolumes)

		ginkgo.By(fmt.Sprintf("%v Deleting pod: %v", logPrefix, pod.Name))
		err = e2epod.DeletePodWithWait(client, pod)
		framework.ExpectNoError(err)

		ginkgo.By(fmt.Sprintf("%v Waiting for volume: %v to be detached from the node: %v", logPrefix, persistentvolumes[0].Spec.VsphereVolume.VolumePath, pod.Spec.NodeName))
		err = waitForVSphereDiskToDetach(persistentvolumes[0].Spec.VsphereVolume.VolumePath, pod.Spec.NodeName)
		framework.ExpectNoError(err)

		ginkgo.By(fmt.Sprintf("%v Deleting the Claim: %v", logPrefix, pvclaim.Name))
		err = e2epv.DeletePersistentVolumeClaim(client, pvclaim.Name, namespace)
		framework.ExpectNoError(err)
	}
}
