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
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	vsphere "k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Testing configurations of single a PV/PVC pair attached to a vSphere Disk
var _ = framework.KubeDescribe("PersistentVolumes:vsphere", func() {
	var (
		c          clientset.Interface
		ns         string
		volumePath string
		pv         *v1.PersistentVolume
		pvc        *v1.PersistentVolumeClaim
		clientPod  *v1.Pod
		pvConfig   framework.PersistentVolumeConfig
		vsp        *vsphere.VSphere
		err        error
		node       types.NodeName
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
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		c = f.ClientSet
		ns = f.Namespace.Name
		clientPod = nil
		pvc = nil
		pv = nil

		if vsp == nil {
			vsp, err = vsphere.GetVSphere()
			Expect(err).NotTo(HaveOccurred())
		}
		if volumePath == "" {
			volumePath, err = createVSphereVolume(vsp, nil)
			Expect(err).NotTo(HaveOccurred())
			pvConfig = framework.PersistentVolumeConfig{
				NamePrefix: "vspherepv-",
				PVSource: v1.PersistentVolumeSource{
					VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
						VolumePath: volumePath,
						FSType:     "ext4",
					},
				},
				Prebind: nil,
			}
		}
		By("Creating the PV and PVC")
		pv, pvc = framework.CreatePVPVC(c, pvConfig, ns, false)
		framework.WaitOnPVandPVC(c, ns, pv, pvc)

		By("Creating the Client Pod")
		clientPod = framework.CreateClientPod(c, ns, pvc)
		node := types.NodeName(clientPod.Spec.NodeName)

		By("Verify disk should be attached to the node")
		isAttached, err := verifyVSphereDiskAttached(vsp, volumePath, node)
		Expect(err).NotTo(HaveOccurred())
		Expect(isAttached).To(BeTrue(), "disk is not attached with the node")
	})

	AfterEach(func() {
		framework.Logf("AfterEach: Cleaning up test resources")
		if c != nil {
			if clientPod != nil {
				clientPod, err = c.CoreV1().Pods(ns).Get(clientPod.Name, metav1.GetOptions{})
				if !apierrs.IsNotFound(err) {
					framework.DeletePodWithWait(f, c, clientPod)
				}
			}

			if pv != nil {
				framework.DeletePersistentVolume(c, pv.Name)
			}
			if pvc != nil {
				framework.DeletePersistentVolumeClaim(c, pvc.Name, ns)
			}
		}
	})
	/*
		Clean up

		1. Wait and verify volume is detached from the node
		2. Delete PV
		3. Delete Volume (vmdk)
	*/
	AddCleanupAction(func() {
		if len(volumePath) > 0 {
			waitForVSphereDiskToDetach(vsp, volumePath, node)
			vsp.DeleteVolume(volumePath)
		}
	})

	/*
		Delete the PVC and then the pod.  Expect the pod to succeed in unmounting and detaching PD on delete.

		Test Steps:
		1. Delete PVC.
		2. Delete POD, POD deletion should succeed.
	*/

	It("should test that deleting a PVC before the pod does not cause pod deletion to fail on PD detach", func() {
		By("Deleting the Claim")
		framework.DeletePersistentVolumeClaim(c, pvc.Name, ns)

		pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Get(pvc.Name, metav1.GetOptions{})
		if !apierrs.IsNotFound(err) {
			Expect(err).NotTo(HaveOccurred())
		}
		pvc = nil
		By("Deleting the Pod")
		framework.DeletePodWithWait(f, c, clientPod)

	})

	/*
		Delete the PV and then the pod.  Expect the pod to succeed in unmounting and detaching PD on delete.

		Test Steps:
		1. Delete PV.
		2. Delete POD, POD deletion should succeed.
	*/
	It("should test that deleting the PV before the pod does not cause pod deletion to fail on PD detach", func() {
		By("Deleting the Persistent Volume")
		framework.DeletePersistentVolume(c, pv.Name)
		pv, err = c.CoreV1().PersistentVolumes().Get(pv.Name, metav1.GetOptions{})
		if !apierrs.IsNotFound(err) {
			Expect(err).NotTo(HaveOccurred())
		}
		pv = nil
		By("Deleting the pod")
		framework.DeletePodWithWait(f, c, clientPod)
	})
})
