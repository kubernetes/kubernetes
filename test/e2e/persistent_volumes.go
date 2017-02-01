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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"
)

// Validate PV/PVC, create and verify writer pod, delete the PVC, and validate the PV's
// phase. Note: the PV is deleted in the AfterEach, not here.
func completeTest(f *framework.Framework, c clientset.Interface, ns string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) {

	// 1. verify that the PV and PVC have binded correctly
	By("Validating the PV-PVC binding")
	waitOnPVandPVC(c, ns, pv, pvc)

	// 2. create the nfs writer pod, test if the write was successful,
	//    then delete the pod and verify that it was deleted
	By("Checking pod has write access to PersistentVolume")
	createWaitAndDeletePod(f, c, ns, pvc.Name)

	// 3. delete the PVC, wait for PV to become "Available"
	By("Deleting the PVC to invoke the recycler")
	deletePVCandValidatePV(c, ns, pvc, pv, v1.VolumeAvailable)
}

// Validate pairs of PVs and PVCs, create and verify writer pod, delete PVC and validate
// PV. Ensure each step succeeds.
// Note: the PV is deleted in the AfterEach, not here.
// Note: this func is serialized, we wait for each pod to be deleted before creating the
//   next pod. Adding concurrency is a TODO item.
// Note: this func is called recursively when there are more claims than pvs.
func completeMultiTest(f *framework.Framework, c clientset.Interface, ns string, pvols pvmap, claims pvcmap) {

	// 1. verify each PV permits write access to a client pod
	By("Checking pod has write access to PersistentVolumes")
	for pvcKey := range claims {
		pvc, err := c.Core().PersistentVolumeClaims(pvcKey.Namespace).Get(pvcKey.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		if len(pvc.Spec.VolumeName) == 0 {
			continue // claim is not bound
		}
		// sanity test to ensure our maps are in sync
		_, found := pvols[pvc.Spec.VolumeName]
		Expect(found).To(BeTrue())
		// TODO: currently a serialized test of each PV
		createWaitAndDeletePod(f, c, pvcKey.Namespace, pvcKey.Name)
	}

	// 2. delete each PVC, wait for its bound PV to become "Available"
	By("Deleting PVCs to invoke recycler")
	deletePVCandValidatePVGroup(c, ns, pvols, claims)
}

var _ = framework.KubeDescribe("PersistentVolumes [Volume][Serial]", func() {

	// global vars for the Context()s and It()'s below
	f := framework.NewDefaultFramework("pv")
	var c clientset.Interface
	var ns string

	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	///////////////////////////////////////////////////////////////////////
	//				NFS
	///////////////////////////////////////////////////////////////////////
	// Testing configurations of a single a PV/PVC pair, multiple evenly paired PVs/PVCs,
	// and multiple unevenly paired PV/PVCs
	framework.KubeDescribe("PersistentVolumes:NFS", func() {

		var (
			NFSconfig    VolumeTestConfig
			nfsServerPod *v1.Pod
			serverIP     string
			pvConfig     persistentVolumeConfig
		)

		// config for the nfs-server pod in the default namespace
		NFSconfig = VolumeTestConfig{
			namespace:   metav1.NamespaceDefault,
			prefix:      "nfs",
			serverImage: NfsServerImage,
			serverPorts: []int{2049},
			serverArgs:  []string{"-G", "777", "/exports"},
		}

		BeforeEach(func() {
			// If it doesn't exist, create the nfs server pod in the "default" ns.
			// The "default" ns is used so that individual tests can delete their
			// ns without impacting the nfs-server pod.
			if nfsServerPod == nil {
				nfsServerPod = startVolumeServer(c, NFSconfig)
				serverIP = nfsServerPod.Status.PodIP
				framework.Logf("NFS server IP address: %v", serverIP)
			}
			pvConfig = persistentVolumeConfig{
				namePrefix: "nfs-",
				pvSource: v1.PersistentVolumeSource{
					NFS: &v1.NFSVolumeSource{
						Server:   serverIP,
						Path:     "/exports",
						ReadOnly: false,
					},
				},
			}
		})

		// Execute after *all* the tests have run
		AddCleanupAction(func() {
			if nfsServerPod != nil && c != nil {
				framework.Logf("AfterSuite: nfs-server pod %v is non-nil, deleting pod", nfsServerPod.Name)
				nfsServerPodCleanup(c, NFSconfig)
				nfsServerPod = nil
			}
		})

		Context("with Single PV - PVC pairs", func() {

			var pv *v1.PersistentVolume
			var pvc *v1.PersistentVolumeClaim

			// Note: this is the only code where the pv is deleted.
			AfterEach(func() {
				framework.Logf("AfterEach: Cleaning up test resources.")
				pvPvcCleanup(c, ns, pv, pvc)
			})

			// Individual tests follow:
			//
			// Create an nfs PV, then a claim that matches the PV, and a pod that
			// contains the claim. Verify that the PV and PVC bind correctly, and
			// that the pod can write to the nfs volume.
			It("should create a non-pre-bound PV and PVC: test write access ", func() {
				pv, pvc = createPVPVC(c, pvConfig, ns, false)
				completeTest(f, c, ns, pv, pvc)
			})

			// Create a claim first, then a nfs PV that matches the claim, and a
			// pod that contains the claim. Verify that the PV and PVC bind
			// correctly, and that the pod can write to the nfs volume.
			It("create a PVC and non-pre-bound PV: test write access", func() {
				pv, pvc = createPVCPV(c, pvConfig, ns, false)
				completeTest(f, c, ns, pv, pvc)
			})

			// Create a claim first, then a pre-bound nfs PV that matches the claim,
			// and a pod that contains the claim. Verify that the PV and PVC bind
			// correctly, and that the pod can write to the nfs volume.
			It("create a PVC and a pre-bound PV: test write access", func() {
				pv, pvc = createPVCPV(c, pvConfig, ns, true)
				completeTest(f, c, ns, pv, pvc)
			})

			// Create a nfs PV first, then a pre-bound PVC that matches the PV,
			// and a pod that contains the claim. Verify that the PV and PVC bind
			// correctly, and that the pod can write to the nfs volume.
			It("create a PV and a pre-bound PVC: test write access", func() {
				pv, pvc = createPVPVC(c, pvConfig, ns, true)
				completeTest(f, c, ns, pv, pvc)
			})
		})

		// Create multiple pvs and pvcs, all in the same namespace. The PVs-PVCs are
		// verified to bind, though it's not known in advanced which PV will bind to
		// which claim. For each pv-pvc pair create a pod that writes to the nfs mount.
		// Note: when the number of PVs exceeds the number of PVCs the max binding wait
		//   time will occur for each PV in excess. This is expected but the delta
		//   should be kept small so that the tests aren't unnecessarily slow.
		// Note: future tests may wish to incorporate the following:
		//   a) pre-binding, b) create pvcs before pvs, c) create pvcs and pods
		//   in different namespaces.
		Context("with multiple PVs and PVCs all in same ns", func() {

			// define the maximum number of PVs and PVCs supported by these tests
			const maxNumPVs = 10
			const maxNumPVCs = 10
			// create the pv and pvc maps to be reused in the It blocks
			pvols := make(pvmap, maxNumPVs)
			claims := make(pvcmap, maxNumPVCs)

			AfterEach(func() {
				framework.Logf("AfterEach: deleting %v PVCs and %v PVs...", len(claims), len(pvols))
				pvPvcMapCleanup(c, ns, pvols, claims)
			})

			// Create 2 PVs and 4 PVCs.
			// Note: PVs are created before claims and no pre-binding
			It("should create 2 PVs and 4 PVCs: test write access", func() {
				numPVs, numPVCs := 2, 4
				pvols, claims = createPVsPVCs(numPVs, numPVCs, c, ns, pvConfig)
				waitAndVerifyBinds(c, ns, pvols, claims, true)
				completeMultiTest(f, c, ns, pvols, claims)
			})

			// Create 3 PVs and 3 PVCs.
			// Note: PVs are created before claims and no pre-binding
			It("should create 3 PVs and 3 PVCs: test write access", func() {
				numPVs, numPVCs := 3, 3
				pvols, claims = createPVsPVCs(numPVs, numPVCs, c, ns, pvConfig)
				waitAndVerifyBinds(c, ns, pvols, claims, true)
				completeMultiTest(f, c, ns, pvols, claims)
			})

			// Create 4 PVs and 2 PVCs.
			// Note: PVs are created before claims and no pre-binding.
			It("should create 4 PVs and 2 PVCs: test write access", func() {
				numPVs, numPVCs := 4, 2
				pvols, claims = createPVsPVCs(numPVs, numPVCs, c, ns, pvConfig)
				waitAndVerifyBinds(c, ns, pvols, claims, true)
				completeMultiTest(f, c, ns, pvols, claims)
			})
		})
	})
	///////////////////////////////////////////////////////////////////////
	//				GCE PD
	///////////////////////////////////////////////////////////////////////
	// Testing configurations of single a PV/PVC pair attached to a GCE PD
	framework.KubeDescribe("PersistentVolumes:GCEPD", func() {

		var (
			diskName  string
			err       error
			pv        *v1.PersistentVolume
			pvc       *v1.PersistentVolumeClaim
			clientPod *v1.Pod
			pvConfig  persistentVolumeConfig
		)

		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce")
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
})
