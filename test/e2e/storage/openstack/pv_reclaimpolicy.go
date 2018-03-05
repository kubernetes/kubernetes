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
	"strconv"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	openstack "k8s.io/kubernetes/pkg/cloudprovider/providers/openstack"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("PersistentVolumes [Feature:ReclaimPolicy]", func() {
	f := framework.NewDefaultFramework("persistentvolumereclaim")
	var (
		c        clientset.Interface
		ns       string
		volumeID string
		pv       *v1.PersistentVolume
		pvc      *v1.PersistentVolumeClaim
	)

	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))
	})

	utils.SIGDescribe("persistentvolumereclaim:openstack", func() {
		BeforeEach(func() {
			framework.SkipUnlessProviderIs("openstack")
			pv = nil
			pvc = nil
			volumeID = ""
		})

		AfterEach(func() {
			osp, _, err := getOpenstack(c)
			Expect(err).NotTo(HaveOccurred())
			testCleanupOpenstackPersistentVolumeReclaim(osp, c, ns, volumeID, pv, pvc)
		})

		It("should delete persistent volume when reclaimPolicy set to delete and associated claim is deleted", func() {
			osp, _, err := getOpenstack(c)
			Expect(err).NotTo(HaveOccurred())

			volumeID, pv, pvc, err = testSetupOpenstackPersistentVolumeReclaim(osp, c, ns, v1.PersistentVolumeReclaimDelete)
			Expect(err).NotTo(HaveOccurred())

			deletePVCAfterBind(c, ns, pvc, pv)
			pvc = nil

			By("verify pv is deleted")
			err = framework.WaitForPersistentVolumeDeleted(c, pv.Name, 3*time.Second, 300*time.Second)
			Expect(err).NotTo(HaveOccurred())

			pv = nil
			volumeID = ""
		})

		It("should not detach and unmount PV when associated pvc with delete as reclaimPolicy is deleted when it is in use by the pod", func() {
			osp, id, err := getOpenstack(c)
			Expect(err).NotTo(HaveOccurred())

			volumeID, pv, pvc, err = testSetupOpenstackPersistentVolumeReclaim(osp, c, ns, v1.PersistentVolumeReclaimDelete)
			Expect(err).NotTo(HaveOccurred())
			// Wait for PV and PVC to Bind
			framework.ExpectNoError(framework.WaitOnPVandPVC(c, ns, pv, pvc))

			By("Creating the Pod")
			pod, err := framework.CreateClientPod(c, ns, pvc)
			Expect(err).NotTo(HaveOccurred())
			node := types.NodeName(pod.Spec.NodeName)

			By("Deleting the Claim")
			framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvc.Name, ns), "Failed to delete PVC ", pvc.Name)
			pvc = nil

			// Verify PV is Present, after PVC is deleted and PV status should be Failed.
			pv, err := c.CoreV1().PersistentVolumes().Get(pv.Name, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())
			Expect(framework.WaitForPersistentVolumePhase(v1.VolumeFailed, c, pv.Name, 1*time.Second, 60*time.Second)).NotTo(HaveOccurred())

			By("Verify the volume is attached to the node")
			isVolumeAttached, verifyDiskAttachedError := verifyOpenstackDiskAttached(c, osp, id, pv.Spec.Cinder.VolumeID, node)
			Expect(verifyDiskAttachedError).NotTo(HaveOccurred())
			Expect(isVolumeAttached).To(BeTrue())

			By("Verify the volume is accessible and available in the pod")
			verifyOpenstackVolumesAccessible(c, pod, []*v1.PersistentVolume{pv}, id, osp)
			framework.Logf("Verified that Volume is accessible in the POD after deleting PV claim")

			By("Deleting the Pod")
			framework.ExpectNoError(framework.DeletePodWithWait(f, c, pod), "Failed to delete pod ", pod.Name)

			By("Verify PV is detached from the node after Pod is deleted")
			Expect(waitForOpenstackDiskToDetach(c, id, osp, pv.Spec.Cinder.VolumeID, types.NodeName(pod.Spec.NodeName))).NotTo(HaveOccurred())

			By("Verify PV should be deleted automatically")
			framework.ExpectNoError(framework.WaitForPersistentVolumeDeleted(c, pv.Name, 1*time.Second, 30*time.Second))
			pv = nil
			volumeID = ""
		})

		It("should retain persistent volume when reclaimPolicy set to retain when associated claim is deleted", func() {
			var volumeFileContent = "hello from openstack cloud provider, Random Content is :" + strconv.FormatInt(time.Now().UnixNano(), 10)
			osp, _, err := getOpenstack(c)
			Expect(err).NotTo(HaveOccurred())

			volumeID, pv, pvc, err = testSetupOpenstackPersistentVolumeReclaim(osp, c, ns, v1.PersistentVolumeReclaimRetain)
			Expect(err).NotTo(HaveOccurred())

			writeContentToOpenstackPV(c, pvc, volumeFileContent)

			By("Delete PVC")
			framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvc.Name, ns), "Failed to delete PVC ", pvc.Name)
			pvc = nil

			By("Verify PV is retained")
			framework.Logf("Waiting for PV %v to become Released", pv.Name)
			err = framework.WaitForPersistentVolumePhase(v1.VolumeReleased, c, pv.Name, 3*time.Second, 300*time.Second)
			Expect(err).NotTo(HaveOccurred())
			framework.ExpectNoError(framework.DeletePersistentVolume(c, pv.Name), "Failed to delete PV ", pv.Name)

			By("Creating the PV for same volume path")
			pv = getOpenstackPersistentVolumeSpec(volumeID, v1.PersistentVolumeReclaimRetain, nil)
			pv, err = c.CoreV1().PersistentVolumes().Create(pv)
			Expect(err).NotTo(HaveOccurred())

			By("creating the pvc")
			pvc = getOpenstackPersistentVolumeClaimSpec(ns, nil)
			pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Create(pvc)
			Expect(err).NotTo(HaveOccurred())

			By("wait for the pv and pvc to bind")
			framework.ExpectNoError(framework.WaitOnPVandPVC(c, ns, pv, pvc))
			verifyContentOfOpenstackPV(c, pvc, volumeFileContent)

		})
	})
})

// Test Setup for persistentvolumereclaim tests for openstack Provider
func testSetupOpenstackPersistentVolumeReclaim(osp *openstack.OpenStack, c clientset.Interface, ns string, persistentVolumeReclaimPolicy v1.PersistentVolumeReclaimPolicy) (volumeID string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim, err error) {
	By("running testSetupOpenstackPersistentVolumeReclaim")
	By("creating vmdk")
	volumeID, err = createOpenstackVolume(osp)
	if err != nil {
		return
	}
	By("creating the pv")
	pv = getOpenstackPersistentVolumeSpec(volumeID, persistentVolumeReclaimPolicy, nil)
	pv, err = c.CoreV1().PersistentVolumes().Create(pv)
	if err != nil {
		return
	}
	By("creating the pvc")
	pvc = getOpenstackPersistentVolumeClaimSpec(ns, nil)
	pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Create(pvc)
	return
}

// Test Cleanup for persistentvolumereclaim tests for openstack Provider
func testCleanupOpenstackPersistentVolumeReclaim(osp *openstack.OpenStack, c clientset.Interface, ns string, volumeID string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) {
	By("running testCleanupOpenstackPersistentVolumeReclaim")
	if len(volumeID) > 0 {
		osp.DeleteVolume(volumeID)
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
