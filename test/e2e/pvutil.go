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
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
	"k8s.io/kubernetes/test/e2e/framework"
)

// Map of all PVs used in the multi pv-pvc tests. The key is the PV's name, which is
// guaranteed to be unique. The value is {} (empty struct) since we're only interested
// in the PV's name and if it is present. We must always Get the pv object before
// referencing any of its values, eg its ClaimRef.
type pvval struct{}
type pvmap map[string]pvval

// Map of all PVCs used in the multi pv-pvc tests. The key is "namespace/pvc.Name". The
// value is {} (empty struct) since we're only interested in the PVC's name and if it is
// present. We must always Get the pvc object before referencing any of its values, eg.
// its VolumeName.
// Note: It's unsafe to add keys to a map in a loop. Their insertion in the map is
//   unpredictable and can result in the same key being iterated over again.
type pvcval struct{}
type pvcmap map[types.NamespacedName]pvcval

// Configuration for a persistent volume.  To create PVs for varying storage options (NFS, ceph, glusterFS, etc.)
// define the pvSource as below.  prebind holds a pre-bound PVC if there is one.
// pvSource: api.PersistentVolumeSource{
// 	NFS: &api.NFSVolumeSource{
// 		...
// 	},
// }
type persistentVolumeConfig struct {
	pvSource   v1.PersistentVolumeSource
	prebind    *v1.PersistentVolumeClaim
	namePrefix string
}

// Delete the nfs-server pod. Only done once per KubeDescription().
func nfsServerPodCleanup(c clientset.Interface, config VolumeTestConfig) {
	defer GinkgoRecover()

	podClient := c.Core().Pods(config.namespace)

	if config.serverImage != "" {
		podName := config.prefix + "-server"
		err := podClient.Delete(podName, nil)
		Expect(err).NotTo(HaveOccurred())
	}
}

// Clean up a pv and pvc in a single pv/pvc test case.
func pvPvcCleanup(c clientset.Interface, ns string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) {
	deletePersistentVolumeClaim(c, pvc.Name, ns)
	deletePersistentVolume(c, pv.Name)
}

// Clean up pvs and pvcs in multi-pv-pvc test cases. All entries found in the pv and
// claims maps are deleted.
func pvPvcMapCleanup(c clientset.Interface, ns string, pvols pvmap, claims pvcmap) {
	for pvcKey := range claims {
		deletePersistentVolumeClaim(c, pvcKey.Name, ns)
		delete(claims, pvcKey)
	}

	for pvKey := range pvols {
		deletePersistentVolume(c, pvKey)
		delete(pvols, pvKey)
	}
}

// Delete the PV.
func deletePersistentVolume(c clientset.Interface, pvName string) {
	if c != nil && len(pvName) > 0 {
		framework.Logf("Deleting PersistentVolume %v", pvName)
		err := c.Core().PersistentVolumes().Delete(pvName, nil)
		if err != nil && !apierrs.IsNotFound(err) {
			Expect(err).NotTo(HaveOccurred())
		}
	}
}

// Delete the Claim
func deletePersistentVolumeClaim(c clientset.Interface, pvcName string, ns string) {
	if c != nil && len(pvcName) > 0 {
		framework.Logf("Deleting PersistentVolumeClaim %v", pvcName)
		err := c.Core().PersistentVolumeClaims(ns).Delete(pvcName, nil)
		if err != nil && !apierrs.IsNotFound(err) {
			Expect(err).NotTo(HaveOccurred())
		}
	}
}

// Delete the PVC and wait for the PV to become Available again. Validate that the PV
// has recycled (assumption here about reclaimPolicy). Caller tells this func which
// phase value to expect for the pv bound to the to-be-deleted claim.
func deletePVCandValidatePV(c clientset.Interface, ns string, pvc *v1.PersistentVolumeClaim, pv *v1.PersistentVolume, expctPVPhase v1.PersistentVolumePhase) {

	pvname := pvc.Spec.VolumeName
	framework.Logf("Deleting PVC %v to trigger recycling of PV %v", pvc.Name, pvname)
	deletePersistentVolumeClaim(c, pvc.Name, ns)

	// Check that the PVC is really deleted.
	pvc, err := c.Core().PersistentVolumeClaims(ns).Get(pvc.Name, metav1.GetOptions{})
	Expect(apierrs.IsNotFound(err)).To(BeTrue())

	// Wait for the PV's phase to return to Available
	framework.Logf("Waiting for recycling process to complete.")
	err = framework.WaitForPersistentVolumePhase(expctPVPhase, c, pv.Name, 1*time.Second, 300*time.Second)
	Expect(err).NotTo(HaveOccurred())

	// examine the pv's ClaimRef and UID and compare to expected values
	pv, err = c.Core().PersistentVolumes().Get(pv.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	cr := pv.Spec.ClaimRef
	if expctPVPhase == v1.VolumeAvailable {
		if cr != nil { // may be ok if cr != nil
			Expect(len(cr.UID)).To(BeZero())
		}
	} else if expctPVPhase == v1.VolumeBound {
		Expect(cr).NotTo(BeNil())
		Expect(len(cr.UID)).NotTo(BeZero())
	}

	framework.Logf("PV %v now in %q phase", pv.Name, expctPVPhase)
}

// Wraps deletePVCandValidatePV() by calling the function in a loop over the PV map. Only
// bound PVs are deleted. Validates that the claim was deleted and the PV is Available.
// Note: if there are more claims than pvs then some of the remaining claims will bind to
//   the just-made-available pvs.
func deletePVCandValidatePVGroup(c clientset.Interface, ns string, pvols pvmap, claims pvcmap) {

	var boundPVs, deletedPVCs int
	var expctPVPhase v1.PersistentVolumePhase

	for pvName := range pvols {
		pv, err := c.Core().PersistentVolumes().Get(pvName, metav1.GetOptions{})
		Expect(apierrs.IsNotFound(err)).To(BeFalse())
		cr := pv.Spec.ClaimRef
		// if pv is bound then delete the pvc it is bound to
		if cr != nil && len(cr.Name) > 0 {
			boundPVs++
			// Assert bound PVC is tracked in this test. Failing this might
			// indicate external PVCs interfering with the test.
			pvcKey := makePvcKey(ns, cr.Name)
			_, found := claims[pvcKey]
			Expect(found).To(BeTrue())
			pvc, err := c.Core().PersistentVolumeClaims(ns).Get(cr.Name, metav1.GetOptions{})
			Expect(apierrs.IsNotFound(err)).To(BeFalse())

			// what Phase do we expect the PV that was bound to the claim to
			// be in after that claim is deleted?
			expctPVPhase = v1.VolumeAvailable
			if len(claims) > len(pvols) {
				// there are excess pvcs so expect the previously bound
				// PV to become bound again
				expctPVPhase = v1.VolumeBound
			}

			deletePVCandValidatePV(c, ns, pvc, pv, expctPVPhase)
			delete(claims, pvcKey)
			deletedPVCs++
		}
	}
	Expect(boundPVs).To(Equal(deletedPVCs))
}

// create the PV resource. Fails test on error.
func createPV(c clientset.Interface, pv *v1.PersistentVolume) *v1.PersistentVolume {

	pv, err := c.Core().PersistentVolumes().Create(pv)
	Expect(err).NotTo(HaveOccurred())
	return pv
}

// create the PVC resource. Fails test on error.
func createPVC(c clientset.Interface, ns string, pvc *v1.PersistentVolumeClaim) *v1.PersistentVolumeClaim {

	pvc, err := c.Core().PersistentVolumeClaims(ns).Create(pvc)
	Expect(err).NotTo(HaveOccurred())
	return pvc
}

// Create a PVC followed by the PV based on the passed in nfs-server ip and
// namespace. If the "preBind" bool is true then pre-bind the PV to the PVC
// via the PV's ClaimRef. Return the pv and pvc to reflect the created objects.
// Note: in the pre-bind case the real PVC name, which is generated, is not
//   known until after the PVC is instantiated. This is why the pvc is created
//   before the pv.
func createPVCPV(c clientset.Interface, pvConfig persistentVolumeConfig, ns string, preBind bool) (*v1.PersistentVolume, *v1.PersistentVolumeClaim) {

	var preBindMsg string

	// make the pvc definition first
	pvc := makePersistentVolumeClaim(ns)
	if preBind {
		preBindMsg = " pre-bound"
		pvConfig.prebind = pvc
	}
	// make the pv spec
	pv := makePersistentVolume(pvConfig)

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
// Note: in the pre-bind case the real PV name, which is generated, is not
//   known until after the PV is instantiated. This is why the pv is created
//   before the pvc.
func createPVPVC(c clientset.Interface, pvConfig persistentVolumeConfig, ns string, preBind bool) (*v1.PersistentVolume, *v1.PersistentVolumeClaim) {

	preBindMsg := ""
	if preBind {
		preBindMsg = " pre-bound"
	}
	framework.Logf("Creating a PV followed by a%s PVC", preBindMsg)

	// make the pv and pvc definitions
	pv := makePersistentVolume(pvConfig)
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

// Create the desired number of PVs and PVCs and return them in separate maps. If the
// number of PVs != the number of PVCs then the min of those two counts is the number of
// PVs expected to bind.
func createPVsPVCs(numpvs, numpvcs int, c clientset.Interface, ns string, pvConfig persistentVolumeConfig) (pvmap, pvcmap) {

	var i int
	var pv *v1.PersistentVolume
	var pvc *v1.PersistentVolumeClaim
	pvMap := make(pvmap, numpvs)
	pvcMap := make(pvcmap, numpvcs)

	var extraPVs, extraPVCs int
	extraPVs = numpvs - numpvcs
	if extraPVs < 0 {
		extraPVCs = -extraPVs
		extraPVs = 0
	}
	pvsToCreate := numpvs - extraPVs // want the min(numpvs, numpvcs)

	// create pvs and pvcs
	for i = 0; i < pvsToCreate; i++ {
		pv, pvc = createPVPVC(c, pvConfig, ns, false)
		pvMap[pv.Name] = pvval{}
		pvcMap[makePvcKey(ns, pvc.Name)] = pvcval{}
	}

	// create extra pvs or pvcs as needed
	for i = 0; i < extraPVs; i++ {
		pv = makePersistentVolume(pvConfig)
		pv = createPV(c, pv)
		pvMap[pv.Name] = pvval{}
	}
	for i = 0; i < extraPVCs; i++ {
		pvc = makePersistentVolumeClaim(ns)
		pvc = createPVC(c, ns, pvc)
		pvcMap[makePvcKey(ns, pvc.Name)] = pvcval{}
	}

	return pvMap, pvcMap
}

// Wait for the pv and pvc to bind to each other.
func waitOnPVandPVC(c clientset.Interface, ns string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) {

	// Wait for newly created PVC to bind to the PV
	framework.Logf("Waiting for PV %v to bind to PVC %v", pv.Name, pvc.Name)
	err := framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, c, ns, pvc.Name, 3*time.Second, 300*time.Second)
	Expect(err).NotTo(HaveOccurred())

	// Wait for PersistentVolume.Status.Phase to be Bound, which it should be
	// since the PVC is already bound.
	err = framework.WaitForPersistentVolumePhase(v1.VolumeBound, c, pv.Name, 3*time.Second, 300*time.Second)
	Expect(err).NotTo(HaveOccurred())

	// Re-get the pv and pvc objects
	pv, err = c.Core().PersistentVolumes().Get(pv.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	// Re-get the pvc and
	pvc, err = c.Core().PersistentVolumeClaims(ns).Get(pvc.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	// The pv and pvc are both bound, but to each other?
	// Check that the PersistentVolume.ClaimRef matches the PVC
	Expect(pv.Spec.ClaimRef).NotTo(BeNil())
	Expect(pv.Spec.ClaimRef.Name).To(Equal(pvc.Name))
	Expect(pvc.Spec.VolumeName).To(Equal(pv.Name))
	Expect(pv.Spec.ClaimRef.UID).To(Equal(pvc.UID))
}

// Search for bound PVs and PVCs by examining pvols for non-nil claimRefs.
// NOTE: Each iteration waits for a maximum of 3 minutes per PV and, if the PV is bound,
//   up to 3 minutes for the PVC. When the number of PVs != number of PVCs, this can lead
//   to situations where the maximum wait times are reached several times in succession,
//   extending test time. Thus, it is recommended to keep the delta between PVs and PVCs
//   small.
func waitAndVerifyBinds(c clientset.Interface, ns string, pvols pvmap, claims pvcmap, testExpected bool) {

	var actualBinds int
	expectedBinds := len(pvols)
	if expectedBinds > len(claims) { // want the min of # pvs or #pvcs
		expectedBinds = len(claims)
	}

	for pvName := range pvols {
		err := framework.WaitForPersistentVolumePhase(v1.VolumeBound, c, pvName, 3*time.Second, 180*time.Second)
		if err != nil && len(pvols) > len(claims) {
			framework.Logf("WARN: pv %v is not bound after max wait", pvName)
			framework.Logf("      This may be ok since there are more pvs than pvcs")
			continue
		}
		Expect(err).NotTo(HaveOccurred())

		pv, err := c.Core().PersistentVolumes().Get(pvName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		if cr := pv.Spec.ClaimRef; cr != nil && len(cr.Name) > 0 {
			// Assert bound pvc is a test resource. Failing assertion could
			// indicate non-test PVC interference or a bug in the test
			pvcKey := makePvcKey(ns, cr.Name)
			_, found := claims[pvcKey]
			Expect(found).To(BeTrue())

			err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, c, ns, cr.Name, 3*time.Second, 180*time.Second)
			Expect(err).NotTo(HaveOccurred())
			actualBinds++
		}
	}

	if testExpected {
		Expect(actualBinds).To(Equal(expectedBinds))
	}
}

// Test the pod's exit code to be zero.
func testPodSuccessOrFail(c clientset.Interface, ns string, pod *v1.Pod) {

	By("Pod should terminate with exitcode 0 (success)")
	err := framework.WaitForPodSuccessInNamespace(c, pod.Name, ns)
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Pod %v succeeded ", pod.Name)
}

// Deletes the passed-in pod and waits for the pod to be terminated. Resilient to the pod
// not existing.
func deletePodWithWait(f *framework.Framework, c clientset.Interface, pod *v1.Pod) {

	if pod == nil {
		return
	}
	framework.Logf("Deleting pod %v", pod.Name)
	err := c.Core().Pods(pod.Namespace).Delete(pod.Name, nil)
	if err != nil {
		if apierrs.IsNotFound(err) {
			return // assume pod was deleted already
		}
		Expect(err).NotTo(HaveOccurred())
	}

	// wait for pod to terminate. Expect apierr NotFound
	err = f.WaitForPodTerminated(pod.Name, "")
	Expect(err).To(HaveOccurred())
	if !apierrs.IsNotFound(err) {
		framework.Logf("Error! Expected IsNotFound error deleting pod %q, instead got: %v", pod.Name, err)
		Expect(apierrs.IsNotFound(err)).To(BeTrue())
	}
	framework.Logf("Ignore \"not found\" error above. Pod %v successfully deleted", pod.Name)
}

// Create the test pod, wait for (hopefully) success, and then delete the pod.
func createWaitAndDeletePod(f *framework.Framework, c clientset.Interface, ns string, claimName string) {

	framework.Logf("Creating nfs test pod")

	// Make pod spec
	pod := makeWritePod(ns, claimName)

	// Instantiate pod (Create)
	runPod, err := c.Core().Pods(ns).Create(pod)
	Expect(err).NotTo(HaveOccurred())
	Expect(runPod).NotTo(BeNil())

	defer deletePodWithWait(f, c, runPod)

	// Wait for the test pod to complete its lifecycle
	testPodSuccessOrFail(c, ns, runPod)
}

// Sanity check for GCE testing.  Verify the persistent disk attached to the node.
func verifyGCEDiskAttached(diskName string, nodeName types.NodeName) bool {
	gceCloud, err := getGCECloud()
	Expect(err).NotTo(HaveOccurred())
	isAttached, err := gceCloud.DiskIsAttached(diskName, nodeName)
	Expect(err).NotTo(HaveOccurred())
	return isAttached
}

// Return a pvckey struct.
func makePvcKey(ns, name string) types.NamespacedName {
	return types.NamespacedName{Namespace: ns, Name: name}
}

// Returns a PV definition based on the nfs server IP. If the PVC is not nil
// then the PV is defined with a ClaimRef which includes the PVC's namespace.
// If the PVC is nil then the PV is not defined with a ClaimRef.
// Note: the passed-in claim does not have a name until it is created
//   (instantiated) and thus the PV's ClaimRef cannot be completely filled-in in
//   this func. Therefore, the ClaimRef's name is added later in
//   createPVCPV.

func makePersistentVolume(pvConfig persistentVolumeConfig) *v1.PersistentVolume {
	// Specs are expected to match this test's PersistentVolumeClaim

	var claimRef *v1.ObjectReference

	if pvConfig.prebind != nil {
		claimRef = &v1.ObjectReference{
			Name:      pvConfig.prebind.Name,
			Namespace: pvConfig.prebind.Namespace,
		}
	}

	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: pvConfig.namePrefix,
			Annotations: map[string]string{
				volumehelper.VolumeGidAnnotationKey: "777",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimRecycle,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("2Gi"),
			},
			PersistentVolumeSource: pvConfig.pvSource,
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadOnlyMany,
				v1.ReadWriteMany,
			},
			ClaimRef: claimRef,
		},
	}
}

// Returns a PVC definition based on the namespace.
// Note: if this PVC is intended to be pre-bound to a PV, whose name is not
//   known until the PV is instantiated, then the func createPVPVC will add
//   pvc.Spec.VolumeName to this claim.
func makePersistentVolumeClaim(ns string) *v1.PersistentVolumeClaim {
	// Specs are expected to match this test's PersistentVolume

	return &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-",
			Namespace:    ns,
			Annotations: map[string]string{
				"volume.beta.kubernetes.io/storage-class": "",
			},
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
				v1.ReadOnlyMany,
				v1.ReadWriteMany,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("1Gi"),
				},
			},
		},
	}
}

// Returns a pod definition based on the namespace. The pod references the PVC's
// name.
func makeWritePod(ns string, pvcName string) *v1.Pod {
	return makePod(ns, pvcName, "touch /mnt/SUCCESS && (id -G | grep -E '\\b777\\b')")
}

// Returns a pod definition based on the namespace. The pod references the PVC's
// name.  A slice of BASH commands can be supplied as args to be run by the pod
func makePod(ns string, pvcName string, command ...string) *v1.Pod {

	if len(command) == 0 {
		command = []string{"while true; do sleep 1; done"}
	}
	var isPrivileged bool = true
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "client-",
			Namespace:    ns,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "write-pod",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"/bin/sh", "-c"},
					Args:    command,
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      pvcName,
							MountPath: "/mnt",
						},
					},
					SecurityContext: &v1.SecurityContext{
						Privileged: &isPrivileged,
					},
				},
			},
			RestartPolicy: v1.RestartPolicyOnFailure,
			Volumes: []v1.Volume{
				{
					Name: pvcName,
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: pvcName,
						},
					},
				},
			},
		},
	}
}

// Define and create a pod with a mounted PV.  Pod runs infinite loop until killed.
func createClientPod(c clientset.Interface, ns string, pvc *v1.PersistentVolumeClaim) *v1.Pod {
	clientPod := makePod(ns, pvc.Name)
	clientPod, err := c.Core().Pods(ns).Create(clientPod)
	Expect(err).NotTo(HaveOccurred())

	// Verify the pod is running before returning it
	err = framework.WaitForPodRunningInNamespace(c, clientPod)
	Expect(err).NotTo(HaveOccurred())
	clientPod, err = c.Core().Pods(ns).Get(clientPod.Name, metav1.GetOptions{})
	Expect(apierrs.IsNotFound(err)).To(BeFalse())
	return clientPod
}
