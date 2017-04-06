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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"
)

// verifyGCEDiskAttached performs a sanity check to verify the PD attached to the node
func verifyGCEDiskAttached(diskName string, nodeName types.NodeName) bool {
	gceCloud, err := framework.GetGCECloud()
	Expect(err).NotTo(HaveOccurred())
	isAttached, err := gceCloud.DiskIsAttached(diskName, nodeName)
	Expect(err).NotTo(HaveOccurred())
	return isAttached
}

// initializeGCETestSpec creates a PV, PVC, and ClientPod that will run until killed by test or clean up.
func initializeGCETestSpec(c clientset.Interface, ns string, pvConfig framework.PersistentVolumeConfig, pvcConfig framework.PersistentVolumeClaimConfig, isPrebound bool) (*v1.Pod, *v1.PersistentVolume, *v1.PersistentVolumeClaim) {
	By("Creating the PV and PVC")
	pv, pvc := framework.CreatePVPVC(c, pvConfig, pvcConfig, ns, isPrebound)
	framework.WaitOnPVandPVC(c, ns, pv, pvc)

	By("Creating the Client Pod")
	clientPod := framework.CreateClientPod(c, ns, pvc)
	return clientPod, pv, pvc
}

// Testing configurations of single a PV/PVC pair attached to a GCE PD
var _ = framework.KubeDescribe("PersistentVolumes:GCEPD [Volume]", func() {
	var (
		c         clientset.Interface
		diskName  string
		ns        string
		err       error
		pv        *v1.PersistentVolume
		pvc       *v1.PersistentVolumeClaim
		clientPod *v1.Pod
		pvConfig  framework.PersistentVolumeConfig
		pvcConfig framework.PersistentVolumeClaimConfig
		volLabel  labels.Set
		selector  *metav1.LabelSelector
		node      types.NodeName
	)

	f := framework.NewDefaultFramework("pv")
	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name

		// Enforce binding only within test space via selector labels
		volLabel = labels.Set{framework.VolumeSelectorKey: ns}
		selector = metav1.SetAsLabelSelector(volLabel)

		framework.SkipUnlessProviderIs("gce", "gke")
		By("Initializing Test Spec")
		diskName, err = framework.CreatePDWithRetry()
		Expect(err).NotTo(HaveOccurred())
		pvConfig = framework.PersistentVolumeConfig{
			NamePrefix: "gce-",
			Labels:     volLabel,
			PVSource: v1.PersistentVolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
					PDName:   diskName,
					FSType:   "ext3",
					ReadOnly: false,
				},
			},
			Prebind: nil,
		}
		pvcConfig = framework.PersistentVolumeClaimConfig{
			Annotations: map[string]string{
				v1.BetaStorageClassAnnotation: "",
			},
			Selector: selector,
		}
		clientPod, pv, pvc = initializeGCETestSpec(c, ns, pvConfig, pvcConfig, false)
		node = types.NodeName(clientPod.Spec.NodeName)
	})

	AfterEach(func() {
		framework.Logf("AfterEach: Cleaning up test resources")
		if c != nil {
			framework.DeletePodWithWait(f, c, clientPod)
			framework.PVPVCCleanup(c, ns, pv, pvc)
			clientPod, pv, pvc, node = nil, nil, nil, ""
			if diskName != "" {
				framework.DeletePDWithRetry(diskName)
			}
		}
	})

	// Attach a persistent disk to a pod using a PVC.
	// Delete the PVC and then the pod.  Expect the pod to succeed in unmounting and detaching PD on delete.
	It("should test that deleting a PVC before the pod does not cause pod deletion to fail on PD detach", func() {

		By("Deleting the Claim")
		framework.DeletePersistentVolumeClaim(c, pvc.Name, ns)
		verifyGCEDiskAttached(diskName, node)

		By("Deleting the Pod")
		framework.DeletePodWithWait(f, c, clientPod)

		By("Verifying Persistent Disk detach")
		err = waitForPDDetach(diskName, node)
		Expect(err).NotTo(HaveOccurred())
	})

	// Attach a persistent disk to a pod using a PVC.
	// Delete the PV and then the pod.  Expect the pod to succeed in unmounting and detaching PD on delete.
	It("should test that deleting the PV before the pod does not cause pod deletion to fail on PD detach", func() {

		By("Deleting the Persistent Volume")
		framework.DeletePersistentVolume(c, pv.Name)
		verifyGCEDiskAttached(diskName, node)

		By("Deleting the client pod")
		framework.DeletePodWithWait(f, c, clientPod)

		By("Verifying Persistent Disk detaches")
		err = waitForPDDetach(diskName, node)
		Expect(err).NotTo(HaveOccurred())
	})

	// Test that a Pod and PVC attached to a GCEPD successfully unmounts and detaches when the encompassing Namespace is deleted.
	It("should test that deleting the Namespace of a PVC and Pod causes the successful detach of Persistent Disk", func() {

		By("Deleting the Namespace")
		err := c.CoreV1().Namespaces().Delete(ns, nil)
		Expect(err).NotTo(HaveOccurred())

		err = framework.WaitForNamespacesDeleted(c, []string{ns}, 3*time.Minute)
		Expect(err).NotTo(HaveOccurred())

		By("Verifying Persistent Disk detaches")
		err = waitForPDDetach(diskName, node)
		Expect(err).NotTo(HaveOccurred())
	})
})
