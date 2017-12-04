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
	"fmt"
	"strings"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
)

// Validate PV/PVC, create and verify writer pod, delete the PVC, and validate the PV's
// phase. Note: the PV is deleted in the AfterEach, not here.
func completeISCSITest(f *framework.Framework, c clientset.Interface, ns string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) {
	// 1. verify that the PV and PVC have bound correctly
	By("Validating the PV-PVC binding")
	framework.ExpectNoError(framework.WaitOnPVandPVC(c, ns, pv, pvc))

	// 2. create the iscsi writer pod, test if the write was successful,
	//    then delete the pod and verify that it was deleted
	By("Checking pod has write access to PersistentVolume")
	framework.ExpectNoError(framework.CreateWaitAndDeletePod(f, c, ns, pvc))

	// 3. delete the PVC, wait for PV to become "Released"
	By("Deleting the PVC to invoke the reclaim policy.")
	framework.ExpectNoError(framework.DeletePVCandValidatePV(c, ns, pvc, pv, v1.VolumeReleased))
}

// Validate pairs of PVs and PVCs, create and verify writer pod, delete PVC and validate
// PV. Ensure each step succeeds.
// Note: the PV is deleted in the AfterEach, not here.
// Note: this func is serialized, we wait for each pod to be deleted before creating the
//   next pod. Adding concurrency is a TODO item.
func completeISCSIMultiTest(f *framework.Framework, c clientset.Interface, ns string, pvols framework.PVMap, claims framework.PVCMap, expectPhase v1.PersistentVolumePhase) error {
	var err error

	// 1. verify each PV permits write access to a client pod
	By("Checking pod has write access to PersistentVolumes")
	for pvcKey := range claims {
		pvc, err := c.CoreV1().PersistentVolumeClaims(pvcKey.Namespace).Get(pvcKey.Name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("error getting pvc %q: %v", pvcKey.Name, err)
		}
		if len(pvc.Spec.VolumeName) == 0 {
			continue // claim is not bound
		}
		// sanity test to ensure our maps are in sync
		_, found := pvols[pvc.Spec.VolumeName]
		if !found {
			return fmt.Errorf("internal: pvols map is missing volume %q", pvc.Spec.VolumeName)
		}
		// TODO: currently a serialized test of each PV
		if err = framework.CreateWaitAndDeletePod(f, c, pvcKey.Namespace, pvc); err != nil {
			return err
		}
	}

	// 2. delete each PVC, wait for its bound PV to reach `expectedPhase`
	By("Deleting PVCs to invoke reclaim policy")
	if err = framework.DeletePVCandValidatePVGroup(c, ns, pvols, claims, expectPhase); err != nil {
		return err
	}
	return nil
}

var _ = SIGDescribe("PersistentVolumes iSCSI", func() {

	// global vars for the Context()s and It()'s below
	f := framework.NewDefaultFramework("pv")
	var (
		c         clientset.Interface
		ns        string
		pvConfig  framework.PersistentVolumeConfig
		pvcConfig framework.PersistentVolumeClaimConfig
		volLabel  labels.Set
		selector  *metav1.LabelSelector
		pv        *v1.PersistentVolume
		pvc       *v1.PersistentVolumeClaim
		err       error
	)

	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		// Enforce binding only within test space via selector labels
		volLabel = labels.Set{framework.VolumeSelectorKey: ns}
		selector = metav1.SetAsLabelSelector(volLabel)
	})

	// Testing configurations of a single a PV/PVC pair, multiple evenly paired PVs/PVCs,
	// and multiple unevenly paired PV/PVCs
	Describe("iSCSI", func() {

		var (
			iscsiServerPod *v1.Pod
			serverIP       string
		)

		BeforeEach(func() {
			_, iscsiServerPod, serverIP = framework.NewISCSIServer(c, ns)
			pvConfig = framework.PersistentVolumeConfig{
				NamePrefix: "iscsi-",
				Labels:     volLabel,
				PVSource: v1.PersistentVolumeSource{
					ISCSI: &v1.ISCSIPersistentVolumeSource{
						TargetPortal: serverIP + ":3260",
						// from test/images/volumes-tester/iscsi/initiatorname.iscsi
						IQN:    "iqn.2003-01.org.linux-iscsi.f21.x8664:sn.4b0aae584f7c",
						Lun:    0,
						FSType: "ext2",
					},
				},
			}
			pvcConfig = framework.PersistentVolumeClaimConfig{
				Annotations: map[string]string{
					v1.BetaStorageClassAnnotation: "",
				},
				Selector: selector,
			}
		})

		AfterEach(func() {
			framework.ExpectNoError(framework.DeletePodWithWait(f, c, iscsiServerPod), "AfterEach: Failed to delete pod ", iscsiServerPod.Name)
			pv, pvc = nil, nil
			pvConfig, pvcConfig = framework.PersistentVolumeConfig{}, framework.PersistentVolumeClaimConfig{}
		})

		Context("with Single PV - PVC pairs", func() {
			// Note: this is the only code where the pv is deleted.
			AfterEach(func() {
				framework.Logf("AfterEach: Cleaning up test resources.")
				if errs := framework.PVPVCCleanup(c, ns, pv, pvc); len(errs) > 0 {
					framework.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
				}
			})

			// Individual tests follow:
			//
			// Create an iscsi PV, then a claim that matches the PV, and a pod that
			// contains the claim. Verify that the PV and PVC bind correctly, and
			// that the pod can write to the iscsi volume.
			It("should create a non-pre-bound PV and PVC: test write access ", func() {
				pv, pvc, err = framework.CreatePVPVC(c, pvConfig, pvcConfig, ns, false)
				Expect(err).NotTo(HaveOccurred())
				completeISCSITest(f, c, ns, pv, pvc)
			})

			// Create a claim first, then a iscsi PV that matches the claim, and a
			// pod that contains the claim. Verify that the PV and PVC bind
			// correctly, and that the pod can write to the iscsi volume.
			It("create a PVC and non-pre-bound PV: test write access", func() {
				pv, pvc, err = framework.CreatePVCPV(c, pvConfig, pvcConfig, ns, false)
				Expect(err).NotTo(HaveOccurred())
				completeISCSITest(f, c, ns, pv, pvc)
			})

			// Create a claim first, then a pre-bound iscsi PV that matches the claim,
			// and a pod that contains the claim. Verify that the PV and PVC bind
			// correctly, and that the pod can write to the iscsi volume.
			It("create a PVC and a pre-bound PV: test write access", func() {
				pv, pvc, err = framework.CreatePVCPV(c, pvConfig, pvcConfig, ns, true)
				Expect(err).NotTo(HaveOccurred())
				completeISCSITest(f, c, ns, pv, pvc)
			})

			// Create a iscsi PV first, then a pre-bound PVC that matches the PV,
			// and a pod that contains the claim. Verify that the PV and PVC bind
			// correctly, and that the pod can write to the iscsi volume.
			It("create a PV and a pre-bound PVC: test write access", func() {
				pv, pvc, err = framework.CreatePVPVC(c, pvConfig, pvcConfig, ns, true)
				Expect(err).NotTo(HaveOccurred())
				completeISCSITest(f, c, ns, pv, pvc)
			})
		})

		// Create multiple pvs and pvcs, all in the same namespace. The PVs-PVCs are
		// verified to bind, though it's not known in advanced which PV will bind to
		// which claim. For each pv-pvc pair create a pod that writes to the iscsi mount.
		// Note: when the number of PVs exceeds the number of PVCs the max binding wait
		//   time will occur for each PV in excess. This is expected but the delta
		//   should be kept small so that the tests aren't unnecessarily slow.
		// Note: future tests may wish to incorporate the following:
		//   a) pre-binding, b) create pvcs before pvs, c) create pvcs and pods
		//   in different namespaces.
		Context("with multiple PVs and PVCs all in same ns", func() {

			// scope the pv and pvc maps to be available in the AfterEach
			// note: these maps are created fresh in CreatePVsPVCs()
			var pvols framework.PVMap
			var claims framework.PVCMap

			AfterEach(func() {
				framework.Logf("AfterEach: deleting %v PVCs and %v PVs...", len(claims), len(pvols))
				errs := framework.PVPVCMapCleanup(c, ns, pvols, claims)
				if len(errs) > 0 {
					errmsg := []string{}
					for _, e := range errs {
						errmsg = append(errmsg, e.Error())
					}
					framework.Failf("AfterEach: Failed to delete 1 or more PVs/PVCs. Errors: %v", strings.Join(errmsg, "; "))
				}
			})

			// Create 2 PVs and 4 PVCs.
			// Note: PVs are created before claims and no pre-binding
			It("should create 2 PVs and 4 PVCs: test write access", func() {
				numPVs, numPVCs := 2, 4
				pvols, claims, err = framework.CreatePVsPVCs(numPVs, numPVCs, c, ns, pvConfig, pvcConfig)
				Expect(err).NotTo(HaveOccurred())
				framework.ExpectNoError(framework.WaitAndVerifyBinds(c, ns, pvols, claims, true))
				framework.ExpectNoError(completeISCSIMultiTest(f, c, ns, pvols, claims, v1.VolumeReleased))
			})

			// Create 3 PVs and 3 PVCs.
			// Note: PVs are created before claims and no pre-binding
			It("should create 3 PVs and 3 PVCs: test write access", func() {
				numPVs, numPVCs := 3, 3
				pvols, claims, err = framework.CreatePVsPVCs(numPVs, numPVCs, c, ns, pvConfig, pvcConfig)
				Expect(err).NotTo(HaveOccurred())
				framework.ExpectNoError(framework.WaitAndVerifyBinds(c, ns, pvols, claims, true))
				framework.ExpectNoError(completeISCSIMultiTest(f, c, ns, pvols, claims, v1.VolumeReleased))
			})

			// Create 4 PVs and 2 PVCs.
			// Note: PVs are created before claims and no pre-binding.
			It("should create 4 PVs and 2 PVCs: test write access [Slow]", func() {
				numPVs, numPVCs := 4, 2
				pvols, claims, err = framework.CreatePVsPVCs(numPVs, numPVCs, c, ns, pvConfig, pvcConfig)
				Expect(err).NotTo(HaveOccurred())
				framework.ExpectNoError(framework.WaitAndVerifyBinds(c, ns, pvols, claims, true))
				framework.ExpectNoError(completeISCSIMultiTest(f, c, ns, pvols, claims, v1.VolumeReleased))
			})
		})
	})
})
