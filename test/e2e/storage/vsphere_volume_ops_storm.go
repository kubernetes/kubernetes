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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	k8stype "k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	storage "k8s.io/kubernetes/pkg/apis/storage/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	vsphere "k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/test/e2e/framework"
)

/*
	Test to perform Disk Ops storm.

	Steps
    	1. Create storage class for thin Provisioning.
    	2. Create 30 PVCs using above storage class in annotation, requesting 2 GB files.
    	3. Wait until all disks are ready and all PVs and PVCs get bind. (CreateVolume storm)
    	4. Create pod to mount volumes using PVCs created in step 2. (AttachDisk storm)
    	5. Wait for pod status to be running.
    	6. Verify all volumes accessible and available in the pod.
    	7. Delete pod.
    	8. wait until volumes gets detached. (DetachDisk storm)
    	9. Delete all PVCs. This should delete all Disks. (DeleteVolume storm)
		10. Delete storage class.
*/

var _ = framework.KubeDescribe("vsphere volume operations storm [Volume]", func() {
	f := framework.NewDefaultFramework("volume-ops-storm")
	const DEFAULT_VOLUME_OPS_SCALE = 30
	var (
		client            clientset.Interface
		namespace         string
		storageclass      *storage.StorageClass
		pvclaims          []*v1.PersistentVolumeClaim
		persistentvolumes []*v1.PersistentVolume
		err               error
		volume_ops_scale  int
		vsp               *vsphere.VSphere
	)
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		client = f.ClientSet
		namespace = f.Namespace.Name
		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		if len(nodeList.Items) == 0 {
			framework.Failf("Unable to find ready and schedulable Node")
		}
		if os.Getenv("VOLUME_OPS_SCALE") != "" {
			volume_ops_scale, err = strconv.Atoi(os.Getenv("VOLUME_OPS_SCALE"))
			Expect(err).NotTo(HaveOccurred())
		} else {
			volume_ops_scale = DEFAULT_VOLUME_OPS_SCALE
		}
		pvclaims = make([]*v1.PersistentVolumeClaim, volume_ops_scale)
		vsp, err = vsphere.GetVSphere()
		Expect(err).NotTo(HaveOccurred())
	})
	AfterEach(func() {
		By("Deleting PVCs")
		for _, claim := range pvclaims {
			framework.DeletePersistentVolumeClaim(client, claim.Name, namespace)
		}
		By("Deleting StorageClass")
		err = client.StorageV1().StorageClasses().Delete(storageclass.Name, nil)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should create pod with many volumes and verify no attach call fails", func() {
		By(fmt.Sprintf("Running test with VOLUME_OPS_SCALE: %v", volume_ops_scale))
		By("Creating Storage Class")
		scParameters := make(map[string]string)
		scParameters["diskformat"] = "thin"
		storageclass, err = client.StorageV1().StorageClasses().Create(getVSphereStorageClassSpec("thinsc", scParameters))
		Expect(err).NotTo(HaveOccurred())

		By("Creating PVCs using the Storage Class")
		count := 0
		for count < volume_ops_scale {
			pvclaims[count], err = framework.CreatePVC(client, namespace, getVSphereClaimSpecWithStorageClassAnnotation(namespace, storageclass))
			Expect(err).NotTo(HaveOccurred())
			count++
		}

		By("Waiting for all claims to be in bound phase")
		persistentvolumes, err = framework.WaitForPVClaimBoundPhase(client, pvclaims)
		Expect(err).NotTo(HaveOccurred())

		By("Creating pod to attach PVs to the node")
		pod, err := framework.CreatePod(client, namespace, pvclaims, false, "")
		Expect(err).NotTo(HaveOccurred())

		By("Verify all volumes are accessible and available in the pod")
		verifyVSphereVolumesAccessible(pod, persistentvolumes, vsp)

		By("Deleting pod")
		framework.ExpectNoError(framework.DeletePodWithWait(f, client, pod))

		By("Waiting for volumes to be detached from the node")
		for _, pv := range persistentvolumes {
			waitForVSphereDiskToDetach(vsp, pv.Spec.VsphereVolume.VolumePath, k8stype.NodeName(pod.Spec.NodeName))
		}
	})
})
