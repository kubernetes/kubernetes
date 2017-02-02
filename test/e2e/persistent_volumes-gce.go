/*
Copyright 2015 The Kubernetes Authors.

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
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"
)

// Sanity check for GCE testing.  Verify the persistent disk attached to the node.
func verifyGCEDiskAttached(diskName string, nodeName types.NodeName) bool {
	gceCloud, err := getGCECloud()
	Expect(err).NotTo(HaveOccurred())
	isAttached, err := gceCloud.DiskIsAttached(diskName, nodeName)
	Expect(err).NotTo(HaveOccurred())
	return isAttached
}

// Testing configurations of single a PV/PVC pair attached to a GCE PD
var _ = framework.KubeDescribe("PersistentVolumes:GCEPD", func() {

	var (
		c         clientset.Interface
		diskName  string
		ns        string
		err       error
		pv        *v1.PersistentVolume
		pvc       *v1.PersistentVolumeClaim
		clientPod *v1.Pod
		pvConfig  persistentVolumeConfig
	)

	f := framework.NewDefaultFramework("pv")

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce")

		c = f.ClientSet
		ns = f.Namespace.Name

		if diskName == "" {
			diskName, err = createPDWithRetry()
			Expect(err).NotTo(HaveOccurred())
			pvConfig = persistentVolumeConfig{
				namePrefix: "gce-",
				pvSource: v1.PersistentVolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
						PDName:   diskName,
						FSType:   "ext3",
						ReadOnly: false,
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
		if len(diskName) > 0 {
			deletePDWithRetry(diskName)
		}
	})

	// Attach a persistent disk to a pod using a PVC.
	// Delete the PVC and then the pod.  Expect the pod to succeed in unmounting and detaching PD on delete.
	It("should test that deleting a PVC before the pod does not cause pod deletion to fail on PD detach", func() {
		By("Creating the PV and PVC")
		pv, pvc = createPVPVC(c, pvConfig, ns, false)
		waitOnPVandPVC(c, ns, pv, pvc)

		By("Creating the Client Pod")
		clientPod = createClientPod(c, ns, pvc)
		node := types.NodeName(clientPod.Spec.NodeName)

		By("Deleting the Claim")
		deletePersistentVolumeClaim(c, pvc.Name, ns)
		verifyGCEDiskAttached(diskName, node)

		By("Deleting the Pod")
		deletePodWithWait(f, c, clientPod)

		By("Verifying Persistent Disk detach")
		err = waitForPDDetach(diskName, node)
		Expect(err).NotTo(HaveOccurred())
	})

	// Attach a persistent disk to a pod using a PVC.
	// Delete the PV and then the pod.  Expect the pod to succeed in unmounting and detaching PD on delete.
	It("should test that deleting the PV before the pod does not cause pod deletion to fail on PD detach", func() {
		By("Creating the PV and PVC")
		pv, pvc = createPVPVC(c, pvConfig, ns, false)
		waitOnPVandPVC(c, ns, pv, pvc)

		By("Creating the Client Pod")
		clientPod = createClientPod(c, ns, pvc)
		node := types.NodeName(clientPod.Spec.NodeName)

		By("Deleting the Persistent Volume")
		deletePersistentVolume(c, pv.Name)
		verifyGCEDiskAttached(diskName, node)

		By("Deleting the client pod")
		deletePodWithWait(f, c, clientPod)

		By("Verifying Persistent Disk detaches")
		err = waitForPDDetach(diskName, node)
		Expect(err).NotTo(HaveOccurred())
	})
})
