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
	"strconv"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("PersistentVolumes [Feature:vsphere][Feature:ReclaimPolicy]", func() {
	f := framework.NewDefaultFramework("persistentvolumereclaim")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	var (
		c          clientset.Interface
		ns         string
		volumePath string
		pv         *v1.PersistentVolume
		pvc        *v1.PersistentVolumeClaim
		nodeInfo   *NodeInfo
	)

	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))
	})

	ginkgo.Describe("persistentvolumereclaim:vsphere [Feature:vsphere]", func() {
		ginkgo.BeforeEach(func() {
			e2eskipper.SkipUnlessProviderIs("vsphere")
			Bootstrap(f)
			nodeInfo = GetReadySchedulableRandomNodeInfo()
			pv = nil
			pvc = nil
			volumePath = ""
		})

		ginkgo.AfterEach(func() {
			testCleanupVSpherePersistentVolumeReclaim(c, nodeInfo, ns, volumePath, pv, pvc)
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
		ginkgo.It("should delete persistent volume when reclaimPolicy set to delete and associated claim is deleted", func() {
			var err error
			volumePath, pv, pvc, err = testSetupVSpherePersistentVolumeReclaim(c, nodeInfo, ns, v1.PersistentVolumeReclaimDelete)
			framework.ExpectNoError(err)

			deletePVCAfterBind(c, ns, pvc, pv)
			pvc = nil

			ginkgo.By("verify pv is deleted")
			err = e2epv.WaitForPersistentVolumeDeleted(c, pv.Name, 3*time.Second, 300*time.Second)
			framework.ExpectNoError(err)

			pv = nil
			volumePath = ""
		})

		/*
			Test Steps:
			1. Create vmdk
			2. Create PV Spec with volume path set to VMDK file created in Step-1, and PersistentVolumeReclaimPolicy is set to Delete
			3. Create PVC with the storage request set to PV's storage capacity.
			4. Wait for PV and PVC to bound.
			5. Delete PVC.
			6. Verify volume is attached to the node and volume is accessible in the pod.
			7. Verify PV status should be failed.
			8. Delete the pod.
			9. Verify PV should be detached from the node and automatically deleted.
		*/
		ginkgo.It("should not detach and unmount PV when associated pvc with delete as reclaimPolicy is deleted when it is in use by the pod", func() {
			var err error

			volumePath, pv, pvc, err = testSetupVSpherePersistentVolumeReclaim(c, nodeInfo, ns, v1.PersistentVolumeReclaimDelete)
			framework.ExpectNoError(err)
			// Wait for PV and PVC to Bind
			framework.ExpectNoError(e2epv.WaitOnPVandPVC(c, f.Timeouts, ns, pv, pvc))

			ginkgo.By("Creating the Pod")
			pod, err := e2epod.CreateClientPod(c, ns, pvc)
			framework.ExpectNoError(err)

			ginkgo.By("Deleting the Claim")
			framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(c, pvc.Name, ns), "Failed to delete PVC ", pvc.Name)
			pvc = nil

			// Verify PV is Present, after PVC is deleted and PV status should be Failed.
			pv, err := c.CoreV1().PersistentVolumes().Get(context.TODO(), pv.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			err = e2epv.WaitForPersistentVolumePhase(v1.VolumeFailed, c, pv.Name, 1*time.Second, 60*time.Second)
			framework.ExpectNoError(err)

			ginkgo.By("Verify the volume is attached to the node")
			isVolumeAttached, verifyDiskAttachedError := diskIsAttached(pv.Spec.VsphereVolume.VolumePath, pod.Spec.NodeName)
			framework.ExpectNoError(verifyDiskAttachedError)
			if !isVolumeAttached {
				framework.Failf("Disk %s is not attached with the node %s", pv.Spec.VsphereVolume.VolumePath, pod.Spec.NodeName)
			}

			ginkgo.By("Verify the volume is accessible and available in the pod")
			verifyVSphereVolumesAccessible(c, pod, []*v1.PersistentVolume{pv})
			framework.Logf("Verified that Volume is accessible in the POD after deleting PV claim")

			ginkgo.By("Deleting the Pod")
			framework.ExpectNoError(e2epod.DeletePodWithWait(c, pod), "Failed to delete pod ", pod.Name)

			ginkgo.By("Verify PV is detached from the node after Pod is deleted")
			err = waitForVSphereDiskToDetach(pv.Spec.VsphereVolume.VolumePath, pod.Spec.NodeName)
			framework.ExpectNoError(err)

			ginkgo.By("Verify PV should be deleted automatically")
			framework.ExpectNoError(e2epv.WaitForPersistentVolumeDeleted(c, pv.Name, 1*time.Second, 30*time.Second))
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

		ginkgo.It("should retain persistent volume when reclaimPolicy set to retain when associated claim is deleted", func() {
			var err error
			var volumeFileContent = "hello from vsphere cloud provider, Random Content is :" + strconv.FormatInt(time.Now().UnixNano(), 10)

			volumePath, pv, pvc, err = testSetupVSpherePersistentVolumeReclaim(c, nodeInfo, ns, v1.PersistentVolumeReclaimRetain)
			framework.ExpectNoError(err)

			writeContentToVSpherePV(c, f.Timeouts, pvc, volumeFileContent)

			ginkgo.By("Delete PVC")
			framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(c, pvc.Name, ns), "Failed to delete PVC ", pvc.Name)
			pvc = nil

			ginkgo.By("Verify PV is retained")
			framework.Logf("Waiting for PV %v to become Released", pv.Name)
			err = e2epv.WaitForPersistentVolumePhase(v1.VolumeReleased, c, pv.Name, 3*time.Second, 300*time.Second)
			framework.ExpectNoError(err)
			framework.ExpectNoError(e2epv.DeletePersistentVolume(c, pv.Name), "Failed to delete PV ", pv.Name)

			ginkgo.By("Creating the PV for same volume path")
			pv = getVSpherePersistentVolumeSpec(volumePath, v1.PersistentVolumeReclaimRetain, nil)
			pv, err = c.CoreV1().PersistentVolumes().Create(context.TODO(), pv, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("creating the pvc")
			pvc = getVSpherePersistentVolumeClaimSpec(ns, nil)
			pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Create(context.TODO(), pvc, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("wait for the pv and pvc to bind")
			framework.ExpectNoError(e2epv.WaitOnPVandPVC(c, f.Timeouts, ns, pv, pvc))
			verifyContentOfVSpherePV(c, f.Timeouts, pvc, volumeFileContent)

		})
	})
})

// Test Setup for persistentvolumereclaim tests for vSphere Provider
func testSetupVSpherePersistentVolumeReclaim(c clientset.Interface, nodeInfo *NodeInfo, ns string, persistentVolumeReclaimPolicy v1.PersistentVolumeReclaimPolicy) (volumePath string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim, err error) {
	ginkgo.By("running testSetupVSpherePersistentVolumeReclaim")
	ginkgo.By("creating vmdk")
	volumePath, err = nodeInfo.VSphere.CreateVolume(&VolumeOptions{}, nodeInfo.DataCenterRef)
	if err != nil {
		return
	}
	ginkgo.By("creating the pv")
	pv = getVSpherePersistentVolumeSpec(volumePath, persistentVolumeReclaimPolicy, nil)
	pv, err = c.CoreV1().PersistentVolumes().Create(context.TODO(), pv, metav1.CreateOptions{})
	if err != nil {
		return
	}
	ginkgo.By("creating the pvc")
	pvc = getVSpherePersistentVolumeClaimSpec(ns, nil)
	pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Create(context.TODO(), pvc, metav1.CreateOptions{})
	return
}

// Test Cleanup for persistentvolumereclaim tests for vSphere Provider
func testCleanupVSpherePersistentVolumeReclaim(c clientset.Interface, nodeInfo *NodeInfo, ns string, volumePath string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) {
	ginkgo.By("running testCleanupVSpherePersistentVolumeReclaim")
	if len(volumePath) > 0 {
		err := nodeInfo.VSphere.DeleteVolume(volumePath, nodeInfo.DataCenterRef)
		framework.ExpectNoError(err)
	}
	if pv != nil {
		framework.ExpectNoError(e2epv.DeletePersistentVolume(c, pv.Name), "Failed to delete PV ", pv.Name)
	}
	if pvc != nil {
		framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(c, pvc.Name, ns), "Failed to delete PVC ", pvc.Name)
	}
}

// func to wait until PV and PVC bind and once bind completes, delete the PVC
func deletePVCAfterBind(c clientset.Interface, ns string, pvc *v1.PersistentVolumeClaim, pv *v1.PersistentVolume) {
	var err error

	ginkgo.By("wait for the pv and pvc to bind")
	framework.ExpectNoError(e2epv.WaitOnPVandPVC(c, f.Timeouts, ns, pv, pvc))

	ginkgo.By("delete pvc")
	framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(c, pvc.Name, ns), "Failed to delete PVC ", pvc.Name)
	_, err = c.CoreV1().PersistentVolumeClaims(ns).Get(context.TODO(), pvc.Name, metav1.GetOptions{})
	if !apierrors.IsNotFound(err) {
		framework.ExpectNoError(err)
	}
}
