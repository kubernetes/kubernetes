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
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// Testing configurations of single a PV/PVC pair attached to a vSphere Disk
var _ = utils.SIGDescribe("PersistentVolumes:vsphere [Feature:vsphere]", func() {
	var (
		c          clientset.Interface
		ns         string
		volumePath string
		pv         *v1.PersistentVolume
		pvc        *v1.PersistentVolumeClaim
		clientPod  *v1.Pod
		pvConfig   e2epv.PersistentVolumeConfig
		pvcConfig  e2epv.PersistentVolumeClaimConfig
		err        error
		node       string
		volLabel   labels.Set
		selector   *metav1.LabelSelector
		nodeInfo   *NodeInfo
	)

	f := framework.NewDefaultFramework("pv")
	/*
		Test Setup

		1. Create volume (vmdk)
		2. Create PV with volume path for the vmdk.
		3. Create PVC to bind with PV.
		4. Create a POD using the PVC.
		5. Verify Disk and Attached to the node.
	*/
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("vsphere")
		Bootstrap(f)
		c = f.ClientSet
		ns = f.Namespace.Name
		clientPod = nil
		pvc = nil
		pv = nil
		nodeInfo = GetReadySchedulableRandomNodeInfo()

		volLabel = labels.Set{e2epv.VolumeSelectorKey: ns}
		selector = metav1.SetAsLabelSelector(volLabel)

		if volumePath == "" {
			volumePath, err = nodeInfo.VSphere.CreateVolume(&VolumeOptions{}, nodeInfo.DataCenterRef)
			framework.ExpectNoError(err)
			pvConfig = e2epv.PersistentVolumeConfig{
				NamePrefix: "vspherepv-",
				Labels:     volLabel,
				PVSource: v1.PersistentVolumeSource{
					VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
						VolumePath: volumePath,
						FSType:     "ext4",
					},
				},
				Prebind: nil,
			}
			emptyStorageClass := ""
			pvcConfig = e2epv.PersistentVolumeClaimConfig{
				Selector:         selector,
				StorageClassName: &emptyStorageClass,
			}
		}
		ginkgo.By("Creating the PV and PVC")
		pv, pvc, err = e2epv.CreatePVPVC(c, pvConfig, pvcConfig, ns, false)
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epv.WaitOnPVandPVC(c, ns, pv, pvc))

		ginkgo.By("Creating the Client Pod")
		clientPod, err = e2epod.CreateClientPod(c, ns, pvc)
		framework.ExpectNoError(err)
		node = clientPod.Spec.NodeName

		ginkgo.By("Verify disk should be attached to the node")
		isAttached, err := diskIsAttached(volumePath, node)
		framework.ExpectNoError(err)
		framework.ExpectEqual(isAttached, true, "disk is not attached with the node")
	})

	ginkgo.AfterEach(func() {
		framework.Logf("AfterEach: Cleaning up test resources")
		if c != nil {
			framework.ExpectNoError(e2epod.DeletePodWithWait(c, clientPod), "AfterEach: failed to delete pod ", clientPod.Name)

			if pv != nil {
				framework.ExpectNoError(e2epv.DeletePersistentVolume(c, pv.Name), "AfterEach: failed to delete PV ", pv.Name)
			}
			if pvc != nil {
				framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(c, pvc.Name, ns), "AfterEach: failed to delete PVC ", pvc.Name)
			}
		}
	})
	/*
		Clean up

		1. Wait and verify volume is detached from the node
		2. Delete PV
		3. Delete Volume (vmdk)
	*/
	framework.AddCleanupAction(func() {
		// Cleanup actions will be called even when the tests are skipped and leaves namespace unset.
		if len(ns) > 0 && len(volumePath) > 0 {
			framework.ExpectNoError(waitForVSphereDiskToDetach(volumePath, node))
			nodeInfo.VSphere.DeleteVolume(volumePath, nodeInfo.DataCenterRef)
		}
	})

	/*
		Delete the PVC and then the pod.  Expect the pod to succeed in unmounting and detaching PD on delete.

		Test Steps:
		1. Delete PVC.
		2. Delete POD, POD deletion should succeed.
	*/

	ginkgo.It("should test that deleting a PVC before the pod does not cause pod deletion to fail on vsphere volume detach", func() {
		ginkgo.By("Deleting the Claim")
		framework.ExpectNoError(e2epv.DeletePersistentVolumeClaim(c, pvc.Name, ns), "Failed to delete PVC ", pvc.Name)
		pvc = nil

		ginkgo.By("Deleting the Pod")
		framework.ExpectNoError(e2epod.DeletePodWithWait(c, clientPod), "Failed to delete pod ", clientPod.Name)
	})

	/*
		Delete the PV and then the pod.  Expect the pod to succeed in unmounting and detaching PD on delete.

		Test Steps:
		1. Delete PV.
		2. Delete POD, POD deletion should succeed.
	*/
	ginkgo.It("should test that deleting the PV before the pod does not cause pod deletion to fail on vspehre volume detach", func() {
		ginkgo.By("Deleting the Persistent Volume")
		framework.ExpectNoError(e2epv.DeletePersistentVolume(c, pv.Name), "Failed to delete PV ", pv.Name)
		pv = nil

		ginkgo.By("Deleting the pod")
		framework.ExpectNoError(e2epod.DeletePodWithWait(c, clientPod), "Failed to delete pod ", clientPod.Name)
	})
	/*
		This test verifies that a volume mounted to a pod remains mounted after a kubelet restarts.
		Steps:
		1. Write to the volume
		2. Restart kubelet
		3. Verify that written file is accessible after kubelet restart
	*/
	ginkgo.It("should test that a file written to the vspehre volume mount before kubelet restart can be read after restart [Disruptive]", func() {
		e2eskipper.SkipUnlessSSHKeyPresent()
		utils.TestKubeletRestartsAndRestoresMount(c, f, clientPod)
	})

	/*
		This test verifies that a volume mounted to a pod that is deleted while the kubelet is down
		unmounts volume when the kubelet returns.

		Steps:
		1. Verify volume is mounted on the node.
		2. Stop kubelet.
		3. Delete pod.
		4. Start kubelet.
		5. Verify that volume mount not to be found.
	*/
	ginkgo.It("should test that a vspehre volume mounted to a pod that is deleted while the kubelet is down unmounts when the kubelet returns [Disruptive]", func() {
		e2eskipper.SkipUnlessSSHKeyPresent()
		utils.TestVolumeUnmountsFromDeletedPod(c, f, clientPod)
	})

	/*
		This test verifies that deleting the Namespace of a PVC and Pod causes the successful detach of Persistent Disk

		Steps:
		1. Delete Namespace.
		2. Wait for namespace to get deleted. (Namespace deletion should trigger deletion of belonging pods)
		3. Verify volume should be detached from the node.
	*/
	ginkgo.It("should test that deleting the Namespace of a PVC and Pod causes the successful detach of vsphere volume", func() {
		ginkgo.By("Deleting the Namespace")
		err := c.CoreV1().Namespaces().Delete(context.TODO(), ns, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		err = framework.WaitForNamespacesDeleted(c, []string{ns}, 3*time.Minute)
		framework.ExpectNoError(err)

		ginkgo.By("Verifying Persistent Disk detaches")
		waitForVSphereDiskToDetach(volumePath, node)
	})
})
