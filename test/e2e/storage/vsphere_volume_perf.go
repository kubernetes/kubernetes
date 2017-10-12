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

package storage

import (
	"fmt"
	"os"
	"strconv"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	storageV1 "k8s.io/api/storage/v1"
	k8stype "k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	vsphere "k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = SIGDescribe("vcp-performance", func() {
	f := framework.NewDefaultFramework("vcp-performance")

	var (
		client    clientset.Interface
		namespace string
		//iterations int
	)

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		client = f.ClientSet
		namespace = f.Namespace.Name

		Expect(os.Getenv("VSPHERE_SPBM_GOLD_POLICY")).NotTo(BeEmpty(), "ENV VSPHERE_SPBM_GOLD_POLICY is not set")
		Expect(os.Getenv("VSPHERE_DATASTORE")).NotTo(BeEmpty(), "ENV VSPHERE_DATASTORE is not set")
		Expect(os.Getenv("VCP_PERF_VOLUME_PER_POD")).NotTo(BeEmpty(), "ENV VCP_PERF_VOLUME_PER_POD is not set")

		nodes := framework.GetReadySchedulableNodesOrDie(client)
		if len(nodes.Items) < 2 {
			framework.Skipf("Requires at least %d nodes (not %d)", 2, len(nodes.Items))
		}
	})

	It("vcp performance tests", func() {
		// Volumes will be provisioned with each different types of Storage Class
		scArrays := make([]*storageV1.StorageClass, 4)
		// Create default vSphere Storage Class
		By("Creating Storage Class : sc-default")
		scDefaultSpec := getVSphereStorageClassSpec("sc-default", nil)
		scDefault, err := client.StorageV1().StorageClasses().Create(scDefaultSpec)
		Expect(err).NotTo(HaveOccurred())
		defer client.StorageV1().StorageClasses().Delete(scDefault.Name, nil)
		scArrays[0] = scDefault

		// Create Storage Class with vsan storage capabilities
		By("Creating Storage Class : sc-vsan")
		var scvsanParameters map[string]string
		scvsanParameters = make(map[string]string)
		scvsanParameters[Policy_HostFailuresToTolerate] = "1"
		scvsanSpec := getVSphereStorageClassSpec("sc-vsan", scvsanParameters)
		scvsan, err := client.StorageV1().StorageClasses().Create(scvsanSpec)
		Expect(err).NotTo(HaveOccurred())
		defer client.StorageV1().StorageClasses().Delete(scvsan.Name, nil)
		scArrays[1] = scvsan

		// Create Storage Class with SPBM Policy
		By("Creating Storage Class : sc-spbm")
		var scSpbmPolicyParameters map[string]string
		scSpbmPolicyParameters = make(map[string]string)
		goldPolicy := os.Getenv("VSPHERE_SPBM_GOLD_POLICY")
		Expect(goldPolicy).NotTo(BeEmpty())
		scSpbmPolicyParameters[SpbmStoragePolicy] = goldPolicy
		scSpbmPolicySpec := getVSphereStorageClassSpec("sc-spbm", scSpbmPolicyParameters)
		scSpbmPolicy, err := client.StorageV1().StorageClasses().Create(scSpbmPolicySpec)
		Expect(err).NotTo(HaveOccurred())
		defer client.StorageV1().StorageClasses().Delete(scSpbmPolicy.Name, nil)
		scArrays[2] = scSpbmPolicy

		// Create Storage Class with User Specified Datastore.
		By("Creating Storage Class : sc-user-specified-datastore")
		var scWithDatastoreParameters map[string]string
		scWithDatastoreParameters = make(map[string]string)
		datastore := os.Getenv("VSPHERE_DATASTORE")
		Expect(goldPolicy).NotTo(BeEmpty())
		scWithDatastoreParameters[Datastore] = datastore
		scWithDatastoreSpec := getVSphereStorageClassSpec("sc-user-specified-ds", scWithDatastoreParameters)
		scWithDatastore, err := client.StorageV1().StorageClasses().Create(scWithDatastoreSpec)
		Expect(err).NotTo(HaveOccurred())
		defer client.StorageV1().StorageClasses().Delete(scWithDatastore.Name, nil)
		scArrays[3] = scWithDatastore

		volumesPerPod, err := strconv.Atoi(os.Getenv("VCP_PERF_VOLUME_PER_POD"))
		Expect(err).NotTo(HaveOccurred(), "Error Parsing VCP_PERF_VOLUME_PER_POD")

		for i := 0; i < 3; i++ {
			latency := PerformVolumeLifeCycleAtScale(f, client, namespace, scArrays, volumesPerPod)
			framework.Logf("Performance numbers for iteration %d", i)
			framework.Logf("Creating PVCs and waiting for bound phase: %v microseconds", latency[0])
			framework.Logf("Creating Pod and verifying attached status: %v microseconds", latency[1])
			framework.Logf("Deleting Pod and waiting for disk to be detached: %v microseconds", latency[2])
			framework.Logf("Deleting the PVCs: %v microseconds", latency[3])
		}
	})
})

// PerformVolumeLifeCycleAtScale peforms full volume life cycle management at scale
func PerformVolumeLifeCycleAtScale(f *framework.Framework, client clientset.Interface, namespace string, sc []*storageV1.StorageClass, volumesPerPod int) []int64 {
	var latency []int64

	By(fmt.Sprintf("Creating %v PVCs per pod ", volumesPerPod))
	var pvclaims []*v1.PersistentVolumeClaim
	start := time.Now()
	for i := 0; i < volumesPerPod; i++ {
		pvclaim, err := framework.CreatePVC(client, namespace, getVSphereClaimSpecWithStorageClassAnnotation(namespace, sc[i%len(sc)]))
		Expect(err).NotTo(HaveOccurred())
		pvclaims = append(pvclaims, pvclaim)
	}

	persistentvolumes, err := framework.WaitForPVClaimBoundPhase(client, pvclaims, framework.ClaimProvisionTimeout)
	Expect(err).NotTo(HaveOccurred())
	elapsed := time.Since(start)
	latency = append(latency, elapsed.Nanoseconds()/1000)

	By("Creating pod to attach PVs to the node and verifying access")
	start = time.Now()
	pod, err := framework.CreatePod(client, namespace, pvclaims, false, "")
	Expect(err).NotTo(HaveOccurred())

	vsp, err := vsphere.GetVSphere()
	Expect(err).NotTo(HaveOccurred())

	verifyVSphereVolumesAccessible(pod, persistentvolumes, vsp)
	elapsed = time.Since(start)
	latency = append(latency, elapsed.Nanoseconds()/1000)

	By("Deleting pod and waiting for volumes to be detached from the node")
	start = time.Now()
	err = framework.DeletePodWithWait(f, client, pod)
	Expect(err).NotTo(HaveOccurred())

	for _, pv := range persistentvolumes {
		err = waitForVSphereDiskToDetach(vsp, pv.Spec.VsphereVolume.VolumePath, k8stype.NodeName(pod.Spec.NodeName))
		Expect(err).NotTo(HaveOccurred())
	}
	elapsed = time.Since(start)
	latency = append(latency, elapsed.Nanoseconds()/1000)

	By("Deleting the PVCs")
	start = time.Now()
	for _, pvc := range pvclaims {
		err = framework.DeletePersistentVolumeClaim(client, pvc.Name, namespace)
		Expect(err).NotTo(HaveOccurred())
	}
	elapsed = time.Since(start)
	latency = append(latency, elapsed.Nanoseconds()/1000)

	return latency
}
