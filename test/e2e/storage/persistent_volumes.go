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

package storage

import (
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"
)

// Validate PV/PVC, create and verify writer pod, delete the PVC, and validate the PV's
// phase. Note: the PV is deleted in the AfterEach, not here.
func completeTest(f *framework.Framework, c clientset.Interface, ns string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) {
	// 1. verify that the PV and PVC have bound correctly
	By("Validating the PV-PVC binding")
	framework.ExpectNoError(framework.WaitOnPVandPVC(c, ns, pv, pvc))

	// 2. create the nfs writer pod, test if the write was successful,
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
func completeMultiTest(f *framework.Framework, c clientset.Interface, ns string, pvols framework.PVMap, claims framework.PVCMap, expectPhase v1.PersistentVolumePhase) error {
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

// initNFSserverPod wraps volumes.go's startVolumeServer to return a running nfs host pod
// commonly used by persistent volume testing
func initNFSserverPod(c clientset.Interface, ns string) *v1.Pod {
	return framework.StartVolumeServer(c, framework.VolumeTestConfig{
		Namespace:   ns,
		Prefix:      "nfs",
		ServerImage: framework.NfsServerImage,
		ServerPorts: []int{2049},
		ServerArgs:  []string{"-G", "777", "/exports"},
	})
}

var _ = framework.KubeDescribe("PersistentVolumes", func() {

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
	framework.KubeDescribe("PersistentVolumes:NFS", func() {

		var (
			nfsServerPod *v1.Pod
			serverIP     string
		)

		BeforeEach(func() {
			framework.Logf("[BeforeEach] Creating NFS Server Pod")
			nfsServerPod = initNFSserverPod(c, ns)
			serverIP = nfsServerPod.Status.PodIP
			framework.Logf("[BeforeEach] Configuring PersistentVolume")
			pvConfig = framework.PersistentVolumeConfig{
				NamePrefix: "nfs-",
				Labels:     volLabel,
				PVSource: v1.PersistentVolumeSource{
					NFS: &v1.NFSVolumeSource{
						Server:   serverIP,
						Path:     "/exports",
						ReadOnly: false,
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
			framework.ExpectNoError(framework.DeletePodWithWait(f, c, nfsServerPod), "AfterEach: Failed to delete pod ", nfsServerPod.Name)
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
			// Create an nfs PV, then a claim that matches the PV, and a pod that
			// contains the claim. Verify that the PV and PVC bind correctly, and
			// that the pod can write to the nfs volume.
			It("should create a non-pre-bound PV and PVC: test write access ", func() {
				pv, pvc, err = framework.CreatePVPVC(c, pvConfig, pvcConfig, ns, false)
				Expect(err).NotTo(HaveOccurred())
				completeTest(f, c, ns, pv, pvc)
			})

			// Create a claim first, then a nfs PV that matches the claim, and a
			// pod that contains the claim. Verify that the PV and PVC bind
			// correctly, and that the pod can write to the nfs volume.
			It("create a PVC and non-pre-bound PV: test write access", func() {
				pv, pvc, err = framework.CreatePVCPV(c, pvConfig, pvcConfig, ns, false)
				Expect(err).NotTo(HaveOccurred())
				completeTest(f, c, ns, pv, pvc)
			})

			// Create a claim first, then a pre-bound nfs PV that matches the claim,
			// and a pod that contains the claim. Verify that the PV and PVC bind
			// correctly, and that the pod can write to the nfs volume.
			It("create a PVC and a pre-bound PV: test write access", func() {
				pv, pvc, err = framework.CreatePVCPV(c, pvConfig, pvcConfig, ns, true)
				Expect(err).NotTo(HaveOccurred())
				completeTest(f, c, ns, pv, pvc)
			})

			// Create a nfs PV first, then a pre-bound PVC that matches the PV,
			// and a pod that contains the claim. Verify that the PV and PVC bind
			// correctly, and that the pod can write to the nfs volume.
			It("create a PV and a pre-bound PVC: test write access", func() {
				pv, pvc, err = framework.CreatePVPVC(c, pvConfig, pvcConfig, ns, true)
				Expect(err).NotTo(HaveOccurred())
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
				framework.ExpectNoError(completeMultiTest(f, c, ns, pvols, claims, v1.VolumeReleased))
			})

			// Create 3 PVs and 3 PVCs.
			// Note: PVs are created before claims and no pre-binding
			It("should create 3 PVs and 3 PVCs: test write access", func() {
				numPVs, numPVCs := 3, 3
				pvols, claims, err = framework.CreatePVsPVCs(numPVs, numPVCs, c, ns, pvConfig, pvcConfig)
				Expect(err).NotTo(HaveOccurred())
				framework.ExpectNoError(framework.WaitAndVerifyBinds(c, ns, pvols, claims, true))
				framework.ExpectNoError(completeMultiTest(f, c, ns, pvols, claims, v1.VolumeReleased))
			})

			// Create 4 PVs and 2 PVCs.
			// Note: PVs are created before claims and no pre-binding.
			It("should create 4 PVs and 2 PVCs: test write access [Slow]", func() {
				numPVs, numPVCs := 4, 2
				pvols, claims, err = framework.CreatePVsPVCs(numPVs, numPVCs, c, ns, pvConfig, pvcConfig)
				Expect(err).NotTo(HaveOccurred())
				framework.ExpectNoError(framework.WaitAndVerifyBinds(c, ns, pvols, claims, true))
				framework.ExpectNoError(completeMultiTest(f, c, ns, pvols, claims, v1.VolumeReleased))
			})
		})

		// This Context isolates and tests the "Recycle" reclaim behavior.  On deprecation of the
		// Recycler, this entire context can be removed without affecting the test suite or leaving behind
		// dead code.
		Context("when invoking the Recycle reclaim policy", func() {
			BeforeEach(func() {
				pvConfig.ReclaimPolicy = v1.PersistentVolumeReclaimRecycle
				pv, pvc, err = framework.CreatePVPVC(c, pvConfig, pvcConfig, ns, false)
				Expect(err).NotTo(HaveOccurred(), "BeforeEach: Failed to create PV/PVC")
				framework.ExpectNoError(framework.WaitOnPVandPVC(c, ns, pv, pvc), "BeforeEach: WaitOnPVandPVC failed")
			})

			AfterEach(func() {
				framework.Logf("AfterEach: Cleaning up test resources.")
				if errs := framework.PVPVCCleanup(c, ns, pv, pvc); len(errs) > 0 {
					framework.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
				}
			})

			// This It() tests a scenario where a PV is written to by a Pod, recycled, then the volume checked
			// for files. If files are found, the checking Pod fails, failing the test.  Otherwise, the pod
			// (and test) succeed.
			It("should test that a PV becomes Available and is clean after the PVC is deleted. [Volume]", func() {
				By("Writing to the volume.")
				pod := framework.MakeWritePod(ns, pvc)
				pod, err = c.CoreV1().Pods(ns).Create(pod)
				Expect(err).NotTo(HaveOccurred())
				framework.ExpectNoError(framework.WaitForPodSuccessInNamespace(c, pod.Name, ns))

				By("Deleting the claim")
				framework.ExpectNoError(framework.DeletePVCandValidatePV(c, ns, pvc, pv, v1.VolumeAvailable))

				By("Re-mounting the volume.")
				pvc = framework.MakePersistentVolumeClaim(pvcConfig, ns)
				pvc, err = framework.CreatePVC(c, ns, pvc)
				Expect(err).NotTo(HaveOccurred())
				framework.ExpectNoError(framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, c, ns, pvc.Name, 2*time.Second, 60*time.Second), "Failed to reach 'Bound' for PVC ", pvc.Name)

				// If a file is detected in /mnt, fail the pod and do not restart it.
				By("Verifying the mount has been cleaned.")
				mount := pod.Spec.Containers[0].VolumeMounts[0].MountPath
				pod = framework.MakePod(ns, []*v1.PersistentVolumeClaim{pvc}, true, fmt.Sprintf("[ $(ls -A %s | wc -l) -eq 0 ] && exit 0 || exit 1", mount))
				pod, err = c.CoreV1().Pods(ns).Create(pod)
				Expect(err).NotTo(HaveOccurred())
				framework.ExpectNoError(framework.WaitForPodSuccessInNamespace(c, pod.Name, ns))
				framework.Logf("Pod exited without failure; the volume has been recycled.")
			})
		})
	})
})
