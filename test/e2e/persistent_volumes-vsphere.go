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
		pvConfig   persistentVolumeConfig
		vsp        *vsphere.VSphere
		err        error
	)

	f := framework.NewDefaultFramework("pv")

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		c = f.ClientSet
		ns = f.Namespace.Name

		if vsp == nil {
			vsp, err = vsphere.GetVSphere()
			Expect(err).NotTo(HaveOccurred())
		}
		if volumePath == "" {
			volumePath, err = createVSphereVolume(vsp, nil)
			Expect(err).NotTo(HaveOccurred())
			pvConfig = persistentVolumeConfig{
				namePrefix: "vspherepv-",
				pvSource: v1.PersistentVolumeSource{
					VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
						VolumePath: volumePath,
						FSType:     "ext4",
					},
				},
				prebind: nil,
			}
		}
	})

	AfterEach(func() {
		framework.Logf("AfterEach: Cleaning up test resources")
		if c != nil {
			deletePodWithWait(f, c, clientPod)
			pvPvcCleanup(c, ns, pv, pvc)
			clientPod = nil
			pvc = nil
			pv = nil
		}
	})

	AddCleanupAction(func() {
		if len(volumePath) > 0 {
			vsp.DeleteVolume(volumePath)
		}
	})

	/*
		Delete the PVC and then the pod.  Expect the pod to succeed in unmounting and detaching PD on delete.

		Test Steps:
		1. Create volume (vmdk)
		2. Create PV with volume path for the vmdk.
		3. Create PVC to bind with PV.
		4. Create a POD using the PVC.
		5. Delete PVC.
		6. Verify Disk is attached to the node.
		7. Delete POD, POD deletion should succeed.
		8. Wait and Verify Disk is Detached from the node.

		Clean up.
		1. Delete PV
		2. Delete Volume (vmdk)
	*/

	It("should test that deleting a PVC before the pod does not cause pod deletion to fail on PD detach", func() {
		By("Creating the PV and PVC")
		pv, pvc = createPVPVC(c, pvConfig, ns, false)
		waitOnPVandPVC(c, ns, pv, pvc)

		By("Creating the Client Pod")
		clientPod = createClientPod(c, ns, pvc)
		node := types.NodeName(clientPod.Spec.NodeName)

		By("Verify disk should be attached to the node")
		isAttached, err := verifyVSphereDiskAttached(vsp, volumePath, node)
		Expect(err).NotTo(HaveOccurred())
		Expect(isAttached).To(BeTrue(), "disk is not attached with the node")

		By("Deleting the Claim")
		deletePersistentVolumeClaim(c, pvc.Name, ns)

		By("Deleting the Pod")
		deletePodWithWait(f, c, clientPod)

		By("Verifying Persistent disk is detached")
		waitForVSphereDiskToDetach(vsp, volumePath, node)
	})

	/*
		Delete the PV and then the pod.  Expect the pod to succeed in unmounting and detaching PD on delete.

		Test Steps:
		1. Create volume (vmdk)
		2. Create PV with volume path for the vmdk.
		3. Create PVC to bind with PV.
		4. Create a POD using the PVC.
		5. Delete PV.
		6. Verify Disk is attached to the node.
		7. Delete POD, POD deletion should succeed.
		8. Wait and Verify Disk is Detached from the node.

		Clean up.
		1. Delete PVC
		2. Delete Volume (vmdk)
	*/
	It("should test that deleting the PV before the pod does not cause pod deletion to fail on PD detach", func() {
		By("Creating the PV and PVC")
		pv, pvc = createPVPVC(c, pvConfig, ns, false)
		waitOnPVandPVC(c, ns, pv, pvc)

		By("Creating the Client Pod")
		clientPod = createClientPod(c, ns, pvc)
		node := types.NodeName(clientPod.Spec.NodeName)

		By("Verify disk should be attached to the node")
		isAttached, err := verifyVSphereDiskAttached(vsp, volumePath, node)
		Expect(err).NotTo(HaveOccurred())
		Expect(isAttached).To(BeTrue(), "disk is not attached with the node")

		By("Deleting the Persistent Volume")
		deletePersistentVolume(c, pv.Name)

		By("Deleting the client pod")
		deletePodWithWait(f, c, clientPod)

		By("Verifying Persistent disk is detached")
		waitForVSphereDiskToDetach(vsp, volumePath, node)
	})
})
