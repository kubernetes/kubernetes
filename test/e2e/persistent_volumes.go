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
	"fmt"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
	"k8s.io/kubernetes/test/e2e/framework"
)

// Data type to track PVs as keys and their bound PVCs as values.  All PVs should be stored in the map regardless of their status.
// Unbound PVs keys have nil values. PVCs are mutually exclusive between the pvclist and the bindmap. One or the other
// will track a PVC, but never both at once.  Once a PV is identified as bound, the respective PVC should be assigned to that
// PV's value and set to nil in the pvclist object
// NOTE: When working with PV keys, it is safe to delete them from the map using the delete() built-in.  However, you cannot
// update the key itself.  For example, if you Get() a new instance of a PV, you must delete original PV key while adding the
// new version to the map.  WARNING: It's unsafe to add keys to a map in a loop.  Their insertion in the map is unpredictable
// and can result in the same pv.Name being iterated over again.
type bindmap map[*api.PersistentVolume]*api.PersistentVolumeClaim

// Data type to store PVCs prior to testing their status.  As noted above, when a PV status becomes "Bound" and the bound PVC identified,
// assign the PVC as the value of the PV and set to nil in the pvclist.  It is important to maintain the mutual exclusivity of each PVC
// between the bindmap and the pvclist be.  When removing a PVC, the element should be nil-ed.
type pvclist []*api.PersistentVolumeClaim

// Delete the nfs-server pod.
func nfsServerPodCleanup(c *client.Client, config VolumeTestConfig) {
	defer GinkgoRecover()

	podClient := c.Pods(config.namespace)

	if config.serverImage != "" {
		podName := config.prefix + "-server"
		err := podClient.Delete(podName, nil)
		Expect(err).NotTo(HaveOccurred())
	}
}

// Delete the PV.
func deletePersistentVolume(c *client.Client, pv *api.PersistentVolume) {

	framework.Logf("Deleting PersistentVolume %v", pv.Name)
	err := c.PersistentVolumes().Delete(pv.Name)
	Expect(err).NotTo(HaveOccurred())

	// Wait for PersistentVolume to delete
	deleteDuration := 90 * time.Second
	err = framework.WaitForPersistentVolumeDeleted(c, pv.Name, 3*time.Second, deleteDuration)
	Expect(err).NotTo(HaveOccurred())

	return
}

// Delete the PVC and wait for the PV to become Available again. Validate that
// the PV has recycled (assumption here about reclaimPolicy).
func deletePVCandValidatePV(c *client.Client, ns string, pvc *api.PersistentVolumeClaim, pv *api.PersistentVolume) {

	framework.Logf("Deleting PersistentVolumeClaim %v to trigger PV Recycling", pvc.Name)
	err := c.PersistentVolumeClaims(ns).Delete(pvc.Name)
	Expect(err).NotTo(HaveOccurred())

	// Check that the PVC is really deleted.
	pvc, err = c.PersistentVolumeClaims(ns).Get(pvc.Name)
	Expect(err).To(HaveOccurred())

	// Wait for the PV's phase to return to Available
	framework.Logf("Waiting for recycling process to complete.")
	err = framework.WaitForPersistentVolumePhase(api.VolumeAvailable, c, pv.Name, 3*time.Second, 300*time.Second)
	Expect(err).NotTo(HaveOccurred())

	// Examine the pv.ClaimRef and UID. Expect nil values.
	pv, err = c.PersistentVolumes().Get(pv.Name)
	Expect(err).NotTo(HaveOccurred())
	if cr := pv.Spec.ClaimRef; cr != nil {
		Expect(cr.UID).To(BeEmpty())
	}
	framework.Logf("PV %v now in %v phase", pv.Name, pv.Status.Phase)
	return
}

// Takes a map of PV:PVC bound pairs. Wraps deletePVCandValidatePV by calling the function in a loop over
// the supplied map. Each call deletes a PVC and validates the paired PV for phase "Available".
func deletePVCandValidatePVGroup(c *client.Client, ns string, pvPvcBindMap bindmap) {

	var pairCountIn int = len(pvPvcBindMap)
	var pairCountOut int

	for pv, pvc := range pvPvcBindMap {
		if pvc != nil {
			framework.Logf("Deleting PVC %v", pvc.Name)
			deletePVCandValidatePV(c, ns, pvc, pv)
			pvPvcBindMap[pv] = nil
			pairCountOut++
		}
	}
	Expect(pairCountIn).To(Equal(pairCountOut))
	return
}

// create the PV resource. Fails test on error.
func createPV(c *client.Client, pv *api.PersistentVolume) *api.PersistentVolume {

	pv, err := c.PersistentVolumes().Create(pv)
	Expect(err).NotTo(HaveOccurred())
	return pv
}

// create the PVC resource. Fails test on error.
func createPVC(c *client.Client, ns string, pvc *api.PersistentVolumeClaim) *api.PersistentVolumeClaim {

	pvc, err := c.PersistentVolumeClaims(ns).Create(pvc)
	Expect(err).NotTo(HaveOccurred())
	return pvc
}

// Create a PVC followed by the PV based on the passed in nfs-server ip and
// namespace. If the "preBind" bool is true then pre-bind the PV to the PVC
// via the PV's ClaimRef. Return the pv and pvc to reflect the created objects.
// Note: in the pre-bind case the real PVC name, which is generated, is not
//   known until after the PVC is instantiated. This is why the pvc is created
//   before the pv.
func createPVCPV(c *client.Client, serverIP, ns string, preBind bool) (*api.PersistentVolume, *api.PersistentVolumeClaim) {

	var bindTo *api.PersistentVolumeClaim
	var preBindMsg string

	// make the pvc definition first
	pvc := makePersistentVolumeClaim(ns)
	if preBind {
		preBindMsg = " pre-bound"
		bindTo = pvc
	}
	// make the pv spec
	pv := makePersistentVolume(serverIP, bindTo)

	By(fmt.Sprintf("Creating a PVC followed by a%s PV", preBindMsg))
	// instantiate the pvc
	pvc = createPVC(c, ns, pvc)

	// instantiate the pv, handle pre-binding by ClaimRef if needed
	if preBind {
		pv.Spec.ClaimRef.Name = pvc.Name
	}
	pv = createPV(c, pv)

	return pv, pvc
}

// Create a PV followed by the PVC based on the passed in nfs-server ip and
// namespace. If the "preBind" bool is true then pre-bind the PVC to the PV
// via the PVC's VolumeName. Return the pv and pvc to reflect the created
// objects.
// Note: the pv and pvc are returned back to the It() caller so that the
//   AfterEach func can delete these objects if they are not nil.
// Note: in the pre-bind case the real PV name, which is generated, is not
//   known until after the PV is instantiated. This is why the pv is created
//   before the pvc.
func createPVPVC(c *client.Client, serverIP, ns string, preBind bool) (*api.PersistentVolume, *api.PersistentVolumeClaim) {

	preBindMsg := ""
	if preBind {
		preBindMsg = " pre-bound"
	}

	By(fmt.Sprintf("Creating a PV followed by a%s PVC", preBindMsg))

	// make the pv and pvc definitions
	pv := makePersistentVolume(serverIP, nil)
	pvc := makePersistentVolumeClaim(ns)

	// instantiate the pv
	pv = createPV(c, pv)
	// instantiate the pvc, handle pre-binding by VolumeName if needed
	if preBind {
		pvc.Spec.VolumeName = pv.Name
	}
	pvc = createPVC(c, ns, pvc)

	return pv, pvc
}

// Wait for the pv and pvc to bind to each other.
func waitOnPVandPVC(c *client.Client, ns string, pv *api.PersistentVolume, pvc *api.PersistentVolumeClaim) {

	// Wait for newly created PVC to bind to the PV
	framework.Logf("Waiting for PV %v to bind to PVC %v", pv.Name, pvc.Name)
	err := framework.WaitForPersistentVolumeClaimPhase(api.ClaimBound, c, ns, pvc.Name, 3*time.Second, 300*time.Second)
	Expect(err).NotTo(HaveOccurred())

	// Wait for PersistentVolume.Status.Phase to be Bound, which it should be
	// since the PVC is already bound.
	err = framework.WaitForPersistentVolumePhase(api.VolumeBound, c, pv.Name, 3*time.Second, 300*time.Second)
	Expect(err).NotTo(HaveOccurred())

	// Re-get the pv and pvc objects
	pv, err = c.PersistentVolumes().Get(pv.Name)
	Expect(err).NotTo(HaveOccurred())

	// Re-get the pvc and
	pvc, err = c.PersistentVolumeClaims(ns).Get(pvc.Name)
	Expect(err).NotTo(HaveOccurred())

	// The pv and pvc are both bound, but to each other?
	// Check that the PersistentVolume.ClaimRef matches the PVC
	Expect(pv.Spec.ClaimRef).NotTo(BeNil())
	Expect(pv.Spec.ClaimRef.Name).To(Equal(pvc.Name))
	Expect(pvc.Spec.VolumeName).To(Equal(pv.Name))
	Expect(pv.Spec.ClaimRef.UID).To(Equal(pvc.UID))

	return
}

// Generate a bindmap according the new state of binds.  When a PV key has a nil PVC value, find
// the bound PVC in the pvcList and assign it to the PV value. Then remove it from the list to preserve
// mutual exclusivity of the PVC.  If testExpected == true, compare the actual count of bound PVs and
// bound PVCs against their respective expected values.
// NOTE:  Each iteration waits for a maximum of 3 minutes per PV and, if the PV is bound, up to 3 minute limit
//        for the PVC.  When the number of PVs != number of PVCs, this can lead to situations where the
//        maximum wait times are repeatedly reached, extending test time. Thus, it is recommended to keep
//        the delta between PVs and PVCs low.
func waitAndAggregatePairs(c *client.Client, ns string, pvPvcBindMap bindmap, pvcList pvclist, testExpected bool, expectedPVsBound, expectedPVCsBound int) {

	var err error
	var actualPVsBound, actualPVCsBound int

	for pv, pvc := range pvPvcBindMap {
		// Wait for the PV Phase to establish
		err = framework.WaitForPersistentVolumePhase(api.VolumeBound, c, pv.Name, 3*time.Second, 180*time.Second)
		Expect(err).NotTo(HaveOccurred())
		actualPVsBound++

		// Get fresh PV state and claimRef
		newpv, err := c.PersistentVolumes().Get(pv.Name)
		Expect(err).NotTo(HaveOccurred())

		// Examine bound PVs; update their paired PVC.  If unpaired, search the PVC list.
		if cr := newpv.Spec.ClaimRef; cr != nil && len(cr.Name) > 0 {
			// Where is the PVC?
			if pvc == nil {
				// Delete pvc from pvcList. This slice is a list of unbound pvcs.
				err = framework.WaitForPersistentVolumeClaimPhase(api.ClaimBound, c, ns, cr.Name, 3*time.Second, 180*time.Second)
				Expect(err).NotTo(HaveOccurred())
				for i, pvcTarget := range pvcList {
					if pvcTarget != nil && cr.Name == pvcTarget.Name && cr.Namespace == pvcTarget.Namespace {
						pvPvcBindMap[pv] = pvcTarget
						pvcList[i] = nil
						actualPVCsBound++
						break
					}
				}
			} else { // check that the claim is bound
				err = framework.WaitForPersistentVolumeClaimPhase(api.ClaimBound, c, ns, pvc.Name, 3*time.Second, 180*time.Second)
				Expect(err).NotTo(HaveOccurred())
				actualPVCsBound++
			}
		}
	}

	if testExpected {
		Expect(actualPVsBound).To(Equal(expectedPVsBound))
		Expect(actualPVCsBound).To(Equal(expectedPVCsBound))
	}

	return
}

// Test the pod's exit code to be zero.
func testPodSuccessOrFail(f *framework.Framework, c *client.Client, ns string, pod *api.Pod) {

	By("Pod should terminate with exitcode 0 (success)")
	err := framework.WaitForPodSuccessInNamespace(c, pod.Name, pod.Spec.Containers[0].Name, ns)
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Pod %v succeeded ", pod.Name)

	return
}

// Delete the passed in pod.
func deletePod(f *framework.Framework, c *client.Client, ns string, pod *api.Pod) {

	framework.Logf("Deleting pod %v", pod.Name)
	err := c.Pods(ns).Delete(pod.Name, nil)
	Expect(err).NotTo(HaveOccurred())

	// Wait for pod to terminate.  Expect apierr NotFound
	err = f.WaitForPodTerminated(pod.Name, "")
	Expect(err).To(HaveOccurred())
	Expect(apierrs.IsNotFound(err)).To(BeTrue())

	framework.Logf("Ignore \"not found\" error above. Pod %v successfully deleted", pod.Name)
	return
}

// Create the test pod, wait for (hopefully) success, and then delete the pod.
func createWaitAndDeletePod(f *framework.Framework, c *client.Client, ns string, claimName string) {

	framework.Logf("Creating nfs test pod")

	// Make pod spec
	pod := makeWritePod(ns, claimName)

	// Instantiate pod (Create)
	runPod, err := c.Pods(ns).Create(pod)
	Expect(err).NotTo(HaveOccurred())
	Expect(runPod).NotTo(BeNil())

	defer deletePod(f, c, ns, runPod)

	// Wait for the test pod to complete its lifecycle
	testPodSuccessOrFail(f, c, ns, runPod)

	return
}

// Validate PV/PVC, create and verify writer pod, delete PVC and PV. Ensure that
// all of these steps were successful.
// Note: the pv and pvc are returned back to the It() caller so that the
//   AfterEach func can delete these objects if they are not nil.
func completeTest(f *framework.Framework, c *client.Client, ns string, pv *api.PersistentVolume, pvc *api.PersistentVolumeClaim) {
	// 1. verify that the PV and PVC have binded correctly
	By("Validating the PV-PVC binding")
	waitOnPVandPVC(c, ns, pv, pvc)

	// 2. create the nfs writer pod, test if the write was successful,
	//    then delete the pod and verify that it was deleted
	By("Checking pod has write access to PersistentVolume")
	createWaitAndDeletePod(f, c, ns, pvc.Name)

	// 3. delete the PVC before deleting PV, wait for PV to be "Available"
	By("Deleting the PVC to invoke the recycler")
	deletePVCandValidatePV(c, ns, pvc, pv)

	return
}

// Validate pairs of PVs and PVCs, create and verify writer pod, delete PVC and validate PV.
// Ensure each step succeeds.
func completeMultiTest(f *framework.Framework, c *client.Client, ns string, pvPvcBindMap bindmap) {

	// 1. Verify each PV permits write access to a client pod
	By("Checking pod has write access to PersistentVolumes")
	for _, pvc := range pvPvcBindMap {
		if pvc != nil {
			// TODO Currently a serialized test of each PV.  Consider goroutine + channel
			createWaitAndDeletePod(f, c, ns, pvc.Name)
		}
	}

	// 2.  Delete each PVC, wait for its bound PV to become "Available"
	By("Deleting PVCs to invoke recycler")
	deletePVCandValidatePVGroup(c, ns, pvPvcBindMap)

	return
}

var _ = framework.KubeDescribe("PersistentVolumes", func() {

	// global vars for the Context()s and It()'s below
	f := framework.NewDefaultFramework("pv")
	var c *client.Client
	var ns string
	var NFSconfig VolumeTestConfig
	var serverIP string
	var nfsServerPod *api.Pod

	// config for the nfs-server pod in the default namespace
	NFSconfig = VolumeTestConfig{
		namespace:   api.NamespaceDefault,
		prefix:      "nfs",
		serverImage: "gcr.io/google_containers/volume-nfs:0.7",
		serverPorts: []int{2049},
		serverArgs:  []string{"-G", "777", "/exports"},
	}

	BeforeEach(func() {
		c = f.Client
		ns = f.Namespace.Name

		// If it doesn't exist, create the nfs server pod in "default" ns
		// The "default" ns is used so that individual tests can delete
		// their ns without impacting the nfs-server pod.
		if nfsServerPod == nil {
			nfsServerPod = startVolumeServer(c, NFSconfig)
			serverIP = nfsServerPod.Status.PodIP
			framework.Logf("NFS server IP address: %v", serverIP)
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

		var pv *api.PersistentVolume
		var pvc *api.PersistentVolumeClaim

		AfterEach(func() {
			if c != nil && len(ns) > 0 { // still have client and namespace
				if pvc != nil && len(pvc.Name) > 0 {
					// Delete the PersistentVolumeClaim
					framework.Logf("AfterEach: Deleting remaining PVCs")
					if _, errmsg := c.PersistentVolumeClaims(ns).Get(pvc.Name); !apierrs.IsNotFound(errmsg) {
						err := c.PersistentVolumeClaims(ns).Delete(pvc.Name)
						Expect(err).NotTo(HaveOccurred())
						framework.Logf("Deleted PersistentVolumeClaim: %v", pvc.Name)
					}

					pvc = nil
				}
				if pv != nil && len(pv.Name) > 0 {
					// Delete the PersistentVolume
					framework.Logf("AfterEach: Deleting remaining PVs")
					if _, errmsg := c.PersistentVolumes().Get(pv.Name); !apierrs.IsNotFound(errmsg) {
						err := c.PersistentVolumes().Delete(pv.Name)
						Expect(err).NotTo(HaveOccurred())
						framework.Logf("Deleted PersistentVolume: %v", pv.Name)
					}
					pv = nil
				}
			}
		})

		// Individual tests follow:
		//
		// Create an nfs PV, then a claim that matches the PV, and a pod that
		// contains the claim. Verify that the PV and PVC bind correctly, and
		// that the pod can write to the nfs volume.
		It("should create a non-pre-bound PV and PVC: test write access [Flaky]", func() {

			pv, pvc = createPVPVC(c, serverIP, ns, false)

			// validate PV-PVC, create and verify writer pod, delete PVC
			// and PV
			completeTest(f, c, ns, pv, pvc)
		})

		// Create a claim first, then a nfs PV that matches the claim, and a
		// pod that contains the claim. Verify that the PV and PVC bind
		// correctly, and that the pod can write to the nfs volume.
		It("create a PVC and non-pre-bound PV: test write access [Flaky]", func() {

			pv, pvc = createPVCPV(c, serverIP, ns, false)

			// validate PV-PVC, create and verify writer pod, delete PVC
			// and PV
			completeTest(f, c, ns, pv, pvc)
		})

		// Create a claim first, then a pre-bound nfs PV that matches the claim,
		// and a pod that contains the claim. Verify that the PV and PVC bind
		// correctly, and that the pod can write to the nfs volume.
		It("create a PVC and a pre-bound PV: test write access [Flaky]", func() {

			pv, pvc = createPVCPV(c, serverIP, ns, true)

			// validate PV-PVC, create and verify writer pod, delete PVC
			// and PV
			completeTest(f, c, ns, pv, pvc)
		})

		// Create a nfs PV first, then a pre-bound PVC that matches the PV,
		// and a pod that contains the claim. Verify that the PV and PVC bind
		// correctly, and that the pod can write to the nfs volume.
		It("create a PV and a pre-bound PVC: test write access [Flaky]", func() {

			pv, pvc = createPVPVC(c, serverIP, ns, true)

			// validate PV-PVC, create and verify writer pod, delete PVC
			// and PV
			completeTest(f, c, ns, pv, pvc)
		})
	})

	// NOTE:  Though designed to instantiate multple PVs and PVCs, this context supports only 1 namespace, defined by the
	//        PVC spec in makePersistentVolumeClaim.
	// NOTE:  waitAndAggregate waits for a maximum of 20 seconds per PV and, if the PV is bound, up to 20 seconds for the PVC.
	//        When the number of PVs != number of PVCs, this can lead to situations where the maximum wait times are repeatedly
	//        reached, extending test time.  Thus, it is recommended to keep the delta between PVs and PVCs low.
	Context("with multiple PVs and PVCs", func() {

		var (
			pvcList      pvclist
			pvPvcBindMap bindmap
		)

		AfterEach(func() {
			if c != nil && len(ns) > 0 {

				if len(pvcList) > 0 {
					framework.Logf("Deleting remaing PVCs from pvclist")
					for _, pvc := range pvcList {
						if pvc != nil && len(pvc.Name) > 0 {
							if _, errmsg := c.PersistentVolumeClaims(ns).Get(pvc.Name); !apierrs.IsNotFound(errmsg) {
								err := c.PersistentVolumeClaims(ns).Delete(pvc.Name)
								Expect(err).NotTo(HaveOccurred())
								framework.Logf("Deleted PersistentVolumeClaim: %v", pvc.Name)
							}
						}
					}
					pvcList = nil
				}

				// Delete PVs/PVCs that were remove from their lists and added to the map
				if len(pvPvcBindMap) > 0 {
					framework.Logf("Deleting remaining PVs & PVCs in the bindMap")
					for pv, pvc := range pvPvcBindMap {
						if pv != nil {
							if _, errmsg := c.PersistentVolumes().Get(pv.Name); !apierrs.IsNotFound(errmsg) {
								err := c.PersistentVolumes().Delete(pv.Name)
								Expect(err).NotTo(HaveOccurred())
								framework.Logf("Deleted PersistentVolume: %v", pv.Name)
							}
						}
						if pvc != nil {
							if _, errmsg := c.PersistentVolumeClaims(ns).Get(pvc.Name); !apierrs.IsNotFound(errmsg) {
								err := c.PersistentVolumeClaims(ns).Delete(pvc.Name)
								Expect(err).NotTo(HaveOccurred())
								framework.Logf("Deleted PersistentVolumeClaim: %v", pvc.Name)
							}
						}
					}
					pvPvcBindMap = nil
				}
			}
		})

		// Create a group of nfs PVs first, then a group of PVCs
		// and a pod that mounts one claim at a time. Verify that the PVs and PVCs bind
		// correctly, and that the pod can write to the nfs volume.
		It("should create 3 PVs and 3 PVCs: test write access[Flaky]", func() {

			pairs := 3
			pvPvcBindMap = make(bindmap, pairs)
			pvcList = make(pvclist, pairs)

			for i := 0; i < pairs; i++ {
				pv := createAndAddPV(c, serverIP, nil)
				pvPvcBindMap[pv] = nil
				pvcList[i] = createAndAddPVC(c, ns)
			}

			// Then aggregate the pairs
			waitAndAggregatePairs(c, ns, pvPvcBindMap, pvcList, true, pairs, pairs)
			completeMultiTest(f, c, ns, pvPvcBindMap)
		})
	})
})

// Returns a PV definition based on the nfs server IP. If the PVC is not nil
// then the PV is defined with a ClaimRef which includes the PVC's namespace.
// If the PVC is nil then the PV is not defined with a ClaimRef.
// Note: the passed-in claim does not have a name until it is created
//   (instantiated) and thus the PV's ClaimRef cannot be completely filled-in in
//   this func. Therefore, the ClaimRef's name is added later in
//   createPVCPV.
func makePersistentVolume(serverIP string, pvc *api.PersistentVolumeClaim) *api.PersistentVolume {
	// Specs are expected to match this test's PersistentVolumeClaim

	var claimRef *api.ObjectReference

	if pvc != nil {
		claimRef = &api.ObjectReference{
			Name:      pvc.Name,
			Namespace: pvc.Namespace,
		}
	}

	return &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "nfs-",
			Annotations: map[string]string{
				volumehelper.VolumeGidAnnotationKey: "777",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimRecycle,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("2Gi"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				NFS: &api.NFSVolumeSource{
					Server:   serverIP,
					Path:     "/exports",
					ReadOnly: false,
				},
			},
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
				api.ReadOnlyMany,
				api.ReadWriteMany,
			},
			ClaimRef: claimRef,
		},
	}
}

// Returns a PVC definition based on the namespace.
// Note: if this PVC is intended to be pre-bound to a PV, whose name is not
//   known until the PV is instantiated, then the func createPVPVC will add
//   pvc.Spec.VolumeName to this claim.
func makePersistentVolumeClaim(ns string) *api.PersistentVolumeClaim {
	// Specs are expected to match this test's PersistentVolume

	return &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pvc-",
			Namespace:    ns,
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
				api.ReadOnlyMany,
				api.ReadWriteMany,
			},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("1Gi"),
				},
			},
		},
	}
}

// Returns a pod definition based on the namespace. The pod references the PVC's
// name.
func makeWritePod(ns string, pvcName string) *api.Pod {
	// Prepare pod that mounts the NFS volume again and
	// checks that /mnt/index.html was scrubbed there

	var isPrivileged bool = true
	return &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: registered.GroupOrDie(api.GroupName).GroupVersion.String(),
		},
		ObjectMeta: api.ObjectMeta{
			GenerateName: "write-pod-",
			Namespace:    ns,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:    "write-pod",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", "touch /mnt/SUCCESS && (id -G | grep -E '\\b777\\b')"},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "nfs-pvc",
							MountPath: "/mnt",
						},
					},
					SecurityContext: &api.SecurityContext{
						Privileged: &isPrivileged,
					},
				},
			},
			RestartPolicy: api.RestartPolicyNever,
			Volumes: []api.Volume{
				{
					Name: "nfs-pvc",
					VolumeSource: api.VolumeSource{
						PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
							ClaimName: pvcName,
						},
					},
				},
			},
		},
	}
}

// Defines and instantiates a single PV and then adds it to the bindmap.
func createAndAddPV(c *client.Client, serverIP string, preBoundPVC *api.PersistentVolumeClaim) *api.PersistentVolume {

	pv := makePersistentVolume(serverIP, preBoundPVC)
	pv = createPV(c, pv)
	return pv
}

// Defines and instantiates a single PVC then adds it to the pvcList.
// Takes a slice of PVCs.  If nil, creates a new slice
func createAndAddPVC(c *client.Client, ns string) *api.PersistentVolumeClaim {

	pvc := makePersistentVolumeClaim(ns)
	pvc = createPVC(c, ns, pvc)
	return pvc
}
