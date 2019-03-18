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
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	storageV1 "k8s.io/api/storage/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

/* This test calculates latency numbers for volume lifecycle operations
1. Create 4 type of storage classes
2. Read the total number of volumes to be created and volumes per pod
3. Create total PVCs (number of volumes)
4. Create Pods with attached volumes per pod
5. Verify access to the volumes
6. Delete pods and wait for volumes to detach
7. Delete the PVCs
*/
const (
	SCSIUnitsAvailablePerNode = 55
	CreateOp                  = "CreateOp"
	AttachOp                  = "AttachOp"
	DetachOp                  = "DetachOp"
	DeleteOp                  = "DeleteOp"
)

var _ = utils.SIGDescribe("vcp-performance [Feature:vsphere]", func() {
	f := framework.NewDefaultFramework("vcp-performance")

	var (
		client           clientset.Interface
		namespace        string
		nodeSelectorList []*NodeSelector
		policyName       string
		datastoreName    string
		volumeCount      int
		volumesPerPod    int
		iterations       int
	)

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		Bootstrap(f)
		client = f.ClientSet
		namespace = f.Namespace.Name

		// Read the environment variables
		volumeCount = GetAndExpectIntEnvVar(VCPPerfVolumeCount)
		volumesPerPod = GetAndExpectIntEnvVar(VCPPerfVolumesPerPod)
		iterations = GetAndExpectIntEnvVar(VCPPerfIterations)

		policyName = GetAndExpectStringEnvVar(SPBMPolicyName)
		datastoreName = GetAndExpectStringEnvVar(StorageClassDatastoreName)

		nodes := framework.GetReadySchedulableNodesOrDie(client)
		Expect(len(nodes.Items)).To(BeNumerically(">=", 1), "Requires at least %d nodes (not %d)", 2, len(nodes.Items))

		msg := fmt.Sprintf("Cannot attach %d volumes to %d nodes. Maximum volumes that can be attached on %d nodes is %d", volumeCount, len(nodes.Items), len(nodes.Items), SCSIUnitsAvailablePerNode*len(nodes.Items))
		Expect(volumeCount).To(BeNumerically("<=", SCSIUnitsAvailablePerNode*len(nodes.Items)), msg)

		msg = fmt.Sprintf("Cannot attach %d volumes per pod. Maximum volumes that can be attached per pod is %d", volumesPerPod, SCSIUnitsAvailablePerNode)
		Expect(volumesPerPod).To(BeNumerically("<=", SCSIUnitsAvailablePerNode), msg)

		nodeSelectorList = createNodeLabels(client, namespace, nodes)
	})

	It("vcp performance tests", func() {
		scList := getTestStorageClasses(client, policyName, datastoreName)
		defer func(scList []*storageV1.StorageClass) {
			for _, sc := range scList {
				client.StorageV1().StorageClasses().Delete(sc.Name, nil)
			}
		}(scList)

		sumLatency := make(map[string]float64)
		for i := 0; i < iterations; i++ {
			latency := invokeVolumeLifeCyclePerformance(f, client, namespace, scList, volumesPerPod, volumeCount, nodeSelectorList)
			for key, val := range latency {
				sumLatency[key] += val
			}
		}

		iterations64 := float64(iterations)
		framework.Logf("Average latency for below operations")
		framework.Logf("Creating %d PVCs and waiting for bound phase: %v seconds", volumeCount, sumLatency[CreateOp]/iterations64)
		framework.Logf("Creating %v Pod: %v seconds", volumeCount/volumesPerPod, sumLatency[AttachOp]/iterations64)
		framework.Logf("Deleting %v Pod and waiting for disk to be detached: %v seconds", volumeCount/volumesPerPod, sumLatency[DetachOp]/iterations64)
		framework.Logf("Deleting %v PVCs: %v seconds", volumeCount, sumLatency[DeleteOp]/iterations64)

	})
})

func getTestStorageClasses(client clientset.Interface, policyName, datastoreName string) []*storageV1.StorageClass {
	const (
		storageclass1 = "sc-default"
		storageclass2 = "sc-vsan"
		storageclass3 = "sc-spbm"
		storageclass4 = "sc-user-specified-ds"
	)
	scNames := []string{storageclass1, storageclass2, storageclass3, storageclass4}
	scArrays := make([]*storageV1.StorageClass, len(scNames))
	for index, scname := range scNames {
		// Create vSphere Storage Class
		By(fmt.Sprintf("Creating Storage Class : %v", scname))
		var sc *storageV1.StorageClass
		var err error
		switch scname {
		case storageclass1:
			sc, err = client.StorageV1().StorageClasses().Create(getVSphereStorageClassSpec(storageclass1, nil, nil))
		case storageclass2:
			var scVSanParameters map[string]string
			scVSanParameters = make(map[string]string)
			scVSanParameters[Policy_HostFailuresToTolerate] = "1"
			sc, err = client.StorageV1().StorageClasses().Create(getVSphereStorageClassSpec(storageclass2, scVSanParameters, nil))
		case storageclass3:
			var scSPBMPolicyParameters map[string]string
			scSPBMPolicyParameters = make(map[string]string)
			scSPBMPolicyParameters[SpbmStoragePolicy] = policyName
			sc, err = client.StorageV1().StorageClasses().Create(getVSphereStorageClassSpec(storageclass3, scSPBMPolicyParameters, nil))
		case storageclass4:
			var scWithDSParameters map[string]string
			scWithDSParameters = make(map[string]string)
			scWithDSParameters[Datastore] = datastoreName
			scWithDatastoreSpec := getVSphereStorageClassSpec(storageclass4, scWithDSParameters, nil)
			sc, err = client.StorageV1().StorageClasses().Create(scWithDatastoreSpec)
		}
		Expect(sc).NotTo(BeNil())
		Expect(err).NotTo(HaveOccurred())
		scArrays[index] = sc
	}
	return scArrays
}

// invokeVolumeLifeCyclePerformance peforms full volume life cycle management and records latency for each operation
func invokeVolumeLifeCyclePerformance(f *framework.Framework, client clientset.Interface, namespace string, sc []*storageV1.StorageClass, volumesPerPod int, volumeCount int, nodeSelectorList []*NodeSelector) (latency map[string]float64) {
	var (
		totalpvclaims [][]*v1.PersistentVolumeClaim
		totalpvs      [][]*v1.PersistentVolume
		totalpods     []*v1.Pod
	)
	nodeVolumeMap := make(map[string][]string)
	latency = make(map[string]float64)
	numPods := volumeCount / volumesPerPod

	By(fmt.Sprintf("Creating %d PVCs", volumeCount))
	start := time.Now()
	for i := 0; i < numPods; i++ {
		var pvclaims []*v1.PersistentVolumeClaim
		for j := 0; j < volumesPerPod; j++ {
			currsc := sc[((i*numPods)+j)%len(sc)]
			pvclaim, err := framework.CreatePVC(client, namespace, getVSphereClaimSpecWithStorageClass(namespace, "2Gi", currsc))
			Expect(err).NotTo(HaveOccurred())
			pvclaims = append(pvclaims, pvclaim)
		}
		totalpvclaims = append(totalpvclaims, pvclaims)
	}
	for _, pvclaims := range totalpvclaims {
		persistentvolumes, err := framework.WaitForPVClaimBoundPhase(client, pvclaims, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred())
		totalpvs = append(totalpvs, persistentvolumes)
	}
	elapsed := time.Since(start)
	latency[CreateOp] = elapsed.Seconds()

	By("Creating pod to attach PVs to the node")
	start = time.Now()
	for i, pvclaims := range totalpvclaims {
		nodeSelector := nodeSelectorList[i%len(nodeSelectorList)]
		pod, err := framework.CreatePod(client, namespace, map[string]string{nodeSelector.labelKey: nodeSelector.labelValue}, pvclaims, false, "")
		Expect(err).NotTo(HaveOccurred())
		totalpods = append(totalpods, pod)

		defer framework.DeletePodWithWait(f, client, pod)
	}
	elapsed = time.Since(start)
	latency[AttachOp] = elapsed.Seconds()

	for i, pod := range totalpods {
		verifyVSphereVolumesAccessible(client, pod, totalpvs[i])
	}

	By("Deleting pods")
	start = time.Now()
	for _, pod := range totalpods {
		err := framework.DeletePodWithWait(f, client, pod)
		Expect(err).NotTo(HaveOccurred())
	}
	elapsed = time.Since(start)
	latency[DetachOp] = elapsed.Seconds()

	for i, pod := range totalpods {
		for _, pv := range totalpvs[i] {
			nodeVolumeMap[pod.Spec.NodeName] = append(nodeVolumeMap[pod.Spec.NodeName], pv.Spec.VsphereVolume.VolumePath)
		}
	}

	err := waitForVSphereDisksToDetach(nodeVolumeMap)
	Expect(err).NotTo(HaveOccurred())

	By("Deleting the PVCs")
	start = time.Now()
	for _, pvclaims := range totalpvclaims {
		for _, pvc := range pvclaims {
			err = framework.DeletePersistentVolumeClaim(client, pvc.Name, namespace)
			Expect(err).NotTo(HaveOccurred())
		}
	}
	elapsed = time.Since(start)
	latency[DeleteOp] = elapsed.Seconds()

	return latency
}
