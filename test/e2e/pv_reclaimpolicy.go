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

package e2e

import (
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	vsphere "k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = framework.KubeDescribe("persistentvolumereclaim", func() {
	f := framework.NewDefaultFramework("persistentvolumereclaim")

	var (
		c          clientset.Interface
		ns         string
		volumePath string
		pv         *v1.PersistentVolume
		pvc        *v1.PersistentVolumeClaim
	)

	framework.KubeDescribe("persistentvolumereclaim:vsphere", func() {

		BeforeEach(func() {
			framework.SkipUnlessProviderIs("vsphere")
			c = f.ClientSet
			ns = f.Namespace.Name
			framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))
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
			framework.SkipUnlessProviderIs("vsphere")
			vsp, err := vsphere.GetVSphere()
			Expect(err).NotTo(HaveOccurred())

			volumePath, pv, pvc, err = testSetupVSpherePersistentVolumeReclaim(vsp, c, ns, v1.PersistentVolumeReclaimDelete)
			Expect(err).NotTo(HaveOccurred())

			By("verify pv is deleted")
			err = framework.WaitForPersistentVolumeDeleted(c, pv.Name, 3*time.Second, 300*time.Second)
			Expect(err).NotTo(HaveOccurred())
		})

		/*
			This test Verify persistent volume should be retained when reclaimPolicy on the PV is set to retain
			and associated claim is deleted

			Test Steps:
			1. Create vmdk
			2. Create PV Spec with volume path set to VMDK file created in Step-1, and PersistentVolumeReclaimPolicy is set to Retain
			3. Create PVC with the storage request set to PV's storage capacity.
			4. Wait for PV and PVC to bound.
			5. Delete PVC
			6. Verify PV is retained.
			7. Delete PV.
		*/

		It("should retain persistent volume when reclaimPolicy set to retain when associated claim is deleted", func() {
			framework.SkipUnlessProviderIs("vsphere")
			vsp, err := vsphere.GetVSphere()
			Expect(err).NotTo(HaveOccurred())

			volumePath, pv, pvc, err = testSetupVSpherePersistentVolumeReclaim(vsp, c, ns, v1.PersistentVolumeReclaimRetain)
			Expect(err).NotTo(HaveOccurred())

			By("verify pv is retained")
			framework.Logf("Waiting for PV %v to become Released", pv.Name)
			err = framework.WaitForPersistentVolumePhase(v1.VolumeReleased, c, pv.Name, 3*time.Second, 300*time.Second)
			Expect(err).NotTo(HaveOccurred())

			By("delete pv")
			deletePersistentVolume(c, pv.Name)
			err = framework.WaitForPersistentVolumeDeleted(c, pv.Name, 3*time.Second, 300*time.Second)
			Expect(err).NotTo(HaveOccurred())

			By("delete volume")
			err = vsp.DeleteVolume(volumePath)
			Expect(err).NotTo(HaveOccurred())
		})
	})
})

// Test Setup for persistentvolumereclaim tests for vSphere Provider
func testSetupVSpherePersistentVolumeReclaim(vsp *vsphere.VSphere, c clientset.Interface, ns string, persistentVolumeReclaimPolicy v1.PersistentVolumeReclaimPolicy) (volumePath string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim, err error) {
	By("running testSetupVSpherePersistentVolumeReclaim")
	By("creating vmdk")
	volumePath, err = createVSphereVolume(vsp, nil)
	if err != nil {
		return "", nil, nil, err
	}

	By("creating the pv")
	pv = getVSpherePersistentVolumeSpec(volumePath, persistentVolumeReclaimPolicy, nil)
	pv, err = c.CoreV1().PersistentVolumes().Create(pv)
	if err != nil {
		return volumePath, nil, nil, err
	}

	By("creating the pvc")
	pvc = getVSpherePersistentVolumeClaimSpec(ns, nil)
	pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Create(pvc)
	if err != nil {
		return volumePath, pv, nil, err
	}

	By("wait for the pv and pvc")
	waitOnPVandPVC(c, ns, pv, pvc)

	By("delete pvc")
	deletePersistentVolumeClaim(c, pvc.Name, ns)
	return volumePath, pv, pvc, nil
}

// Test Cleanup for persistentvolumereclaim tests for vSphere Provider
func testCleanupVSpherePersistentVolumeReclaim(vsp *vsphere.VSphere, c clientset.Interface, ns string, volumePath string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) {
	By("running testCleanupVSpherePersistentVolumeReclaim")
	if len(volumePath) > 0 {
		vsp.DeleteVolume(volumePath)
	}
	pvPvcCleanup(c, ns, pv, pvc)
}
