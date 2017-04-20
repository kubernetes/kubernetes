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
	"strconv"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	vsphere "k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = framework.KubeDescribe("PersistentVolumes [Feature:ReclaimPolicy]", func() {
	f := framework.NewDefaultFramework("persistentvolumereclaim")
	var (
		c          clientset.Interface
		ns         string
		volumePath string
		pv         *v1.PersistentVolume
		pvc        *v1.PersistentVolumeClaim
	)

	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))
	})

	framework.KubeDescribe("persistentvolumereclaim:vsphere", func() {
		BeforeEach(func() {
			framework.SkipUnlessProviderIs("vsphere")
			pv = nil
			pvc = nil
			volumePath = ""
		})

		AfterEach(func() {
			vsp, err := vsphere.GetVSphere()
			Expect(err).NotTo(HaveOccurred())
			testCleanupVSpherePersistentVolumeReclaim(vsp, c, ns, volumePath, pv, pvc)
		})

		/*
			This test verifies persistent volume should be deleted when reclaimPolicy on the PV is set to delete and
			associated claim is deleted

			Test Steps:
			1. Create vmdk
			2. Create PV Spec with volume path set to VMDK file created in Step-1, and PersistentVolumeReclaimPolicy is set to Delete
			3. Create PVC with the storage request set to PV's storage capacity.
			4. Wait for PV and PVC to bound.
			5. Delete PVC
			6. Verify PV is deleted automatically.
		*/
		It("should delete persistent volume when reclaimPolicy set to delete and associated claim is deleted", func() {
			vsp, err := vsphere.GetVSphere()
			Expect(err).NotTo(HaveOccurred())

			volumePath, pv, pvc, err = testSetupVSpherePersistentVolumeReclaim(vsp, c, ns, v1.PersistentVolumeReclaimDelete)
			Expect(err).NotTo(HaveOccurred())

			deletePVCAfterBind(c, ns, pvc, pv)
			pvc = nil

			By("verify pv is deleted")
			err = framework.WaitForPersistentVolumeDeleted(c, pv.Name, 3*time.Second, 300*time.Second)
			Expect(err).NotTo(HaveOccurred())

			pv = nil
			volumePath = ""
		})

		/*
			This test Verify persistent volume should be retained when reclaimPolicy on the PV is set to retain
			and associated claim is deleted

			Test Steps:
			1. Create vmdk
			2. Create PV Spec with volume path set to VMDK file created in Step-1, and PersistentVolumeReclaimPolicy is set to Retain
			3. Create PVC with the storage request set to PV's storage capacity.
			4. Wait for PV and PVC to bound.
			5. Write some content in the volume.
			6. Delete PVC
			7. Verify PV is retained.
			8. Delete retained PV.
			9. Create PV Spec with the same volume path used in step 2.
			10. Create PVC with the storage request set to PV's storage capacity.
			11. Created POD using PVC created in Step 10 and verify volume content is matching.
		*/

		It("should retain persistent volume when reclaimPolicy set to retain when associated claim is deleted", func() {
			var volumeFileContent = "hello from vsphere cloud provider, Random Content is :" + strconv.FormatInt(time.Now().UnixNano(), 10)
			vsp, err := vsphere.GetVSphere()
			Expect(err).NotTo(HaveOccurred())

			volumePath, pv, pvc, err = testSetupVSpherePersistentVolumeReclaim(vsp, c, ns, v1.PersistentVolumeReclaimRetain)
			Expect(err).NotTo(HaveOccurred())

			writeContentToVSpherePV(c, pvc, volumeFileContent)

			By("Delete PVC")
			framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvc.Name, ns), "Failed to delete PVC ", pvc.Name)
			pvc = nil

			By("Verify PV is retained")
			framework.Logf("Waiting for PV %v to become Released", pv.Name)
			err = framework.WaitForPersistentVolumePhase(v1.VolumeReleased, c, pv.Name, 3*time.Second, 300*time.Second)
			Expect(err).NotTo(HaveOccurred())
			framework.ExpectNoError(framework.DeletePersistentVolume(c, pv.Name), "Failed to delete PV ", pv.Name)

			By("Creating the PV for same volume path")
			pv = getVSpherePersistentVolumeSpec(volumePath, v1.PersistentVolumeReclaimRetain, nil)
			pv, err = c.CoreV1().PersistentVolumes().Create(pv)
			Expect(err).NotTo(HaveOccurred())

			By("creating the pvc")
			pvc = getVSpherePersistentVolumeClaimSpec(ns, nil)
			pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Create(pvc)
			Expect(err).NotTo(HaveOccurred())

			By("wait for the pv and pvc to bind")
			framework.ExpectNoError(framework.WaitOnPVandPVC(c, ns, pv, pvc))
			verifyContentOfVSpherePV(c, pvc, volumeFileContent)

		})
	})
})

// Test Setup for persistentvolumereclaim tests for vSphere Provider
func testSetupVSpherePersistentVolumeReclaim(vsp *vsphere.VSphere, c clientset.Interface, ns string, persistentVolumeReclaimPolicy v1.PersistentVolumeReclaimPolicy) (volumePath string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim, err error) {
	By("running testSetupVSpherePersistentVolumeReclaim")
	By("creating vmdk")
	volumePath, err = createVSphereVolume(vsp, nil)
	if err != nil {
		return
	}
	By("creating the pv")
	pv = getVSpherePersistentVolumeSpec(volumePath, persistentVolumeReclaimPolicy, nil)
	pv, err = c.CoreV1().PersistentVolumes().Create(pv)
	if err != nil {
		return
	}
	By("creating the pvc")
	pvc = getVSpherePersistentVolumeClaimSpec(ns, nil)
	pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Create(pvc)
	return
}

// Test Cleanup for persistentvolumereclaim tests for vSphere Provider
func testCleanupVSpherePersistentVolumeReclaim(vsp *vsphere.VSphere, c clientset.Interface, ns string, volumePath string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) {
	By("running testCleanupVSpherePersistentVolumeReclaim")
	if len(volumePath) > 0 {
		vsp.DeleteVolume(volumePath)
	}
	if pv != nil {
		framework.ExpectNoError(framework.DeletePersistentVolume(c, pv.Name), "Failed to delete PV ", pv.Name)
	}
	if pvc != nil {
		framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvc.Name, ns), "Failed to delete PVC ", pvc.Name)
	}
}

// func to wait until PV and PVC bind and once bind completes, delete the PVC
func deletePVCAfterBind(c clientset.Interface, ns string, pvc *v1.PersistentVolumeClaim, pv *v1.PersistentVolume) {
	var err error

	By("wait for the pv and pvc to bind")
	framework.ExpectNoError(framework.WaitOnPVandPVC(c, ns, pv, pvc))

	By("delete pvc")
	framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvc.Name, ns), "Failed to delete PVC ", pvc.Name)
	pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Get(pvc.Name, metav1.GetOptions{})
	if !apierrs.IsNotFound(err) {
		Expect(err).NotTo(HaveOccurred())
	}
}
