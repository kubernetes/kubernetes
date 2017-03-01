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

package framework

import (
	"fmt"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ec2"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"google.golang.org/api/googleapi"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	awscloud "k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

const (
	PDRetryTimeout  = 5 * time.Minute
	PDRetryPollTime = 5 * time.Second
)

// Map of all PVs used in the multi pv-pvc tests. The key is the PV's name, which is
// guaranteed to be unique. The value is {} (empty struct) since we're only interested
// in the PV's name and if it is present. We must always Get the pv object before
// referencing any of its values, eg its ClaimRef.
type pvval struct{}
type PVMap map[string]pvval

// Map of all PVCs used in the multi pv-pvc tests. The key is "namespace/pvc.Name". The
// value is {} (empty struct) since we're only interested in the PVC's name and if it is
// present. We must always Get the pvc object before referencing any of its values, eg.
// its VolumeName.
// Note: It's unsafe to add keys to a map in a loop. Their insertion in the map is
//   unpredictable and can result in the same key being iterated over again.
type pvcval struct{}
type PVCMap map[types.NamespacedName]pvcval

// Configuration for a persistent volume.  To create PVs for varying storage options (NFS, ceph, glusterFS, etc.)
// define the pvSource as below.  prebind holds a pre-bound PVC if there is one.
// pvSource: api.PersistentVolumeSource{
// 	NFS: &api.NFSVolumeSource{
// 		...
// 	},
// }
type PersistentVolumeConfig struct {
	PVSource      v1.PersistentVolumeSource
	Prebind       *v1.PersistentVolumeClaim
	ReclaimPolicy v1.PersistentVolumeReclaimPolicy
	NamePrefix    string
}

// Clean up a pv and pvc in a single pv/pvc test case.
func PVPVCCleanup(c clientset.Interface, ns string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) {
	DeletePersistentVolumeClaim(c, pvc.Name, ns)
	DeletePersistentVolume(c, pv.Name)
}

// Clean up pvs and pvcs in multi-pv-pvc test cases. All entries found in the pv and
// claims maps are deleted.
func PVPVCMapCleanup(c clientset.Interface, ns string, pvols PVMap, claims PVCMap) {
	for pvcKey := range claims {
		DeletePersistentVolumeClaim(c, pvcKey.Name, ns)
		delete(claims, pvcKey)
	}

	for pvKey := range pvols {
		DeletePersistentVolume(c, pvKey)
		delete(pvols, pvKey)
	}
}

// Delete the PV.
func DeletePersistentVolume(c clientset.Interface, pvName string) {
	if c != nil && len(pvName) > 0 {
		Logf("Deleting PersistentVolume %v", pvName)
		err := c.Core().PersistentVolumes().Delete(pvName, nil)
		if err != nil && !apierrs.IsNotFound(err) {
			Expect(err).NotTo(HaveOccurred())
		}
	}
}

// Delete the Claim
func DeletePersistentVolumeClaim(c clientset.Interface, pvcName string, ns string) {
	if c != nil && len(pvcName) > 0 {
		Logf("Deleting PersistentVolumeClaim %v", pvcName)
		err := c.Core().PersistentVolumeClaims(ns).Delete(pvcName, nil)
		if err != nil && !apierrs.IsNotFound(err) {
			Expect(err).NotTo(HaveOccurred())
		}
	}
}

// Delete the PVC and wait for the PV to enter its expected phase. Validate that the PV
// has been reclaimed (assumption here about reclaimPolicy). Caller tells this func which
// phase value to expect for the pv bound to the to-be-deleted claim.
func DeletePVCandValidatePV(c clientset.Interface, ns string, pvc *v1.PersistentVolumeClaim, pv *v1.PersistentVolume, expectPVPhase v1.PersistentVolumePhase) {

	pvname := pvc.Spec.VolumeName
	Logf("Deleting PVC %v to trigger reclamation of PV %v", pvc.Name, pvname)
	DeletePersistentVolumeClaim(c, pvc.Name, ns)

	// Check that the PVC is really deleted.
	pvc, err := c.Core().PersistentVolumeClaims(ns).Get(pvc.Name, metav1.GetOptions{})
	Expect(apierrs.IsNotFound(err)).To(BeTrue())

	// Wait for the PV's phase to return to be `expectPVPhase`
	Logf("Waiting for reclaim process to complete.")
	err = WaitForPersistentVolumePhase(expectPVPhase, c, pv.Name, 1*time.Second, 300*time.Second)
	Expect(err).NotTo(HaveOccurred())

	// examine the pv's ClaimRef and UID and compare to expected values
	pv, err = c.Core().PersistentVolumes().Get(pv.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	cr := pv.Spec.ClaimRef
	if expectPVPhase == v1.VolumeAvailable {
		if cr != nil { // may be ok if cr != nil
			Expect(cr.UID).To(BeEmpty())
		}
	} else if expectPVPhase == v1.VolumeBound {
		Expect(cr).NotTo(BeNil())
		Expect(cr.UID).NotTo(BeEmpty())
	}

	Logf("PV %v now in %q phase", pv.Name, expectPVPhase)
}

// Wraps deletePVCandValidatePV() by calling the function in a loop over the PV map. Only
// bound PVs are deleted. Validates that the claim was deleted and the PV is in the relevant Phase (Released, Available,
// Bound).
// Note: if there are more claims than pvs then some of the remaining claims will bind to
//   the just-made-available pvs.
func DeletePVCandValidatePVGroup(c clientset.Interface, ns string, pvols PVMap, claims PVCMap, expectPVPhase v1.PersistentVolumePhase) {

	var boundPVs, deletedPVCs int

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
			DeletePVCandValidatePV(c, ns, pvc, pv, expectPVPhase)
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
func CreatePVC(c clientset.Interface, ns string, pvc *v1.PersistentVolumeClaim) *v1.PersistentVolumeClaim {

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
func CreatePVCPV(c clientset.Interface, pvConfig PersistentVolumeConfig, ns string, preBind bool) (*v1.PersistentVolume, *v1.PersistentVolumeClaim) {

	var preBindMsg string

	// make the pvc definition first
	pvc := MakePersistentVolumeClaim(ns)
	if preBind {
		preBindMsg = " pre-bound"
		pvConfig.Prebind = pvc
	}
	// make the pv spec
	pv := makePersistentVolume(pvConfig)

	By(fmt.Sprintf("Creating a PVC followed by a%s PV", preBindMsg))
	// instantiate the pvc
	pvc = CreatePVC(c, ns, pvc)

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
func CreatePVPVC(c clientset.Interface, pvConfig PersistentVolumeConfig, ns string, preBind bool) (*v1.PersistentVolume, *v1.PersistentVolumeClaim) {

	preBindMsg := ""
	if preBind {
		preBindMsg = " pre-bound"
	}
	Logf("Creating a PV followed by a%s PVC", preBindMsg)

	// make the pv and pvc definitions
	pv := makePersistentVolume(pvConfig)
	pvc := MakePersistentVolumeClaim(ns)

	// instantiate the pv
	pv = createPV(c, pv)
	// instantiate the pvc, handle pre-binding by VolumeName if needed
	if preBind {
		pvc.Spec.VolumeName = pv.Name
	}
	pvc = CreatePVC(c, ns, pvc)

	return pv, pvc
}

// Create the desired number of PVs and PVCs and return them in separate maps. If the
// number of PVs != the number of PVCs then the min of those two counts is the number of
// PVs expected to bind.
func CreatePVsPVCs(numpvs, numpvcs int, c clientset.Interface, ns string, pvConfig PersistentVolumeConfig) (PVMap, PVCMap) {

	var i int
	var pv *v1.PersistentVolume
	var pvc *v1.PersistentVolumeClaim
	pvMap := make(PVMap, numpvs)
	pvcMap := make(PVCMap, numpvcs)

	var extraPVs, extraPVCs int
	extraPVs = numpvs - numpvcs
	if extraPVs < 0 {
		extraPVCs = -extraPVs
		extraPVs = 0
	}
	pvsToCreate := numpvs - extraPVs // want the min(numpvs, numpvcs)

	// create pvs and pvcs
	for i = 0; i < pvsToCreate; i++ {
		pv, pvc = CreatePVPVC(c, pvConfig, ns, false)
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
		pvc = MakePersistentVolumeClaim(ns)
		pvc = CreatePVC(c, ns, pvc)
		pvcMap[makePvcKey(ns, pvc.Name)] = pvcval{}
	}

	return pvMap, pvcMap
}

// Wait for the pv and pvc to bind to each other.
func WaitOnPVandPVC(c clientset.Interface, ns string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) {

	// Wait for newly created PVC to bind to the PV
	Logf("Waiting for PV %v to bind to PVC %v", pv.Name, pvc.Name)
	err := WaitForPersistentVolumeClaimPhase(v1.ClaimBound, c, ns, pvc.Name, 3*time.Second, 300*time.Second)
	Expect(err).NotTo(HaveOccurred())

	// Wait for PersistentVolume.Status.Phase to be Bound, which it should be
	// since the PVC is already bound.
	err = WaitForPersistentVolumePhase(v1.VolumeBound, c, pv.Name, 3*time.Second, 300*time.Second)
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
func WaitAndVerifyBinds(c clientset.Interface, ns string, pvols PVMap, claims PVCMap, testExpected bool) {

	var actualBinds int
	expectedBinds := len(pvols)
	if expectedBinds > len(claims) { // want the min of # pvs or #pvcs
		expectedBinds = len(claims)
	}

	for pvName := range pvols {
		err := WaitForPersistentVolumePhase(v1.VolumeBound, c, pvName, 3*time.Second, 180*time.Second)
		if err != nil && len(pvols) > len(claims) {
			Logf("WARN: pv %v is not bound after max wait", pvName)
			Logf("      This may be ok since there are more pvs than pvcs")
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

			err = WaitForPersistentVolumeClaimPhase(v1.ClaimBound, c, ns, cr.Name, 3*time.Second, 180*time.Second)
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
	err := WaitForPodSuccessInNamespace(c, pod.Name, ns)
	Expect(err).NotTo(HaveOccurred())
	Logf("Pod %v succeeded ", pod.Name)
}

// Deletes the passed-in pod and waits for the pod to be terminated. Resilient to the pod
// not existing.
func DeletePodWithWait(f *Framework, c clientset.Interface, pod *v1.Pod) {

	if pod == nil {
		return
	}
	Logf("Deleting pod %v", pod.Name)
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
		Logf("Error! Expected IsNotFound error deleting pod %q, instead got: %v", pod.Name, err)
		Expect(apierrs.IsNotFound(err)).To(BeTrue())
	}
	Logf("Ignore \"not found\" error above. Pod %v successfully deleted", pod.Name)
}

// Create the test pod, wait for (hopefully) success, and then delete the pod.
func CreateWaitAndDeletePod(f *Framework, c clientset.Interface, ns string, claimName string) {

	Logf("Creating nfs test pod")

	// Make pod spec
	pod := MakeWritePod(ns, claimName)

	// Instantiate pod (Create)
	runPod, err := c.Core().Pods(ns).Create(pod)
	Expect(err).NotTo(HaveOccurred())
	Expect(runPod).NotTo(BeNil())

	defer DeletePodWithWait(f, c, runPod)

	// Wait for the test pod to complete its lifecycle
	testPodSuccessOrFail(c, ns, runPod)
}

// Sanity check for GCE testing.  Verify the persistent disk attached to the node.
func VerifyGCEDiskAttached(diskName string, nodeName types.NodeName) bool {
	gceCloud, err := GetGCECloud()
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
// If the PVC is nil then the PV is not defined with a ClaimRef.  If no reclaimPolicy
// is assigned, assumes "Retain".
// Note: the passed-in claim does not have a name until it is created
//   (instantiated) and thus the PV's ClaimRef cannot be completely filled-in in
//   this func. Therefore, the ClaimRef's name is added later in
//   createPVCPV.
func makePersistentVolume(pvConfig PersistentVolumeConfig) *v1.PersistentVolume {
	// Specs are expected to match this test's PersistentVolumeClaim

	var claimRef *v1.ObjectReference
	// If the reclaimPolicy is not provided, assume Retain
	if pvConfig.ReclaimPolicy == "" {
		pvConfig.ReclaimPolicy = v1.PersistentVolumeReclaimRetain
	}
	if pvConfig.Prebind != nil {
		claimRef = &v1.ObjectReference{
			Name:      pvConfig.Prebind.Name,
			Namespace: pvConfig.Prebind.Namespace,
		}
	}
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: pvConfig.NamePrefix,
			Annotations: map[string]string{
				volumehelper.VolumeGidAnnotationKey: "777",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: pvConfig.ReclaimPolicy,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("2Gi"),
			},
			PersistentVolumeSource: pvConfig.PVSource,
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
//   known until the PV is instantiated, then the func CreatePVPVC will add
//   pvc.Spec.VolumeName to this claim.
func MakePersistentVolumeClaim(ns string) *v1.PersistentVolumeClaim {
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
func MakeWritePod(ns string, pvcName string) *v1.Pod {
	return MakePod(ns, pvcName, true, "touch /mnt/SUCCESS && (id -G | grep -E '\\b777\\b')")
}

// Returns a pod definition based on the namespace. The pod references the PVC's
// name.  A slice of BASH commands can be supplied as args to be run by the pod
func MakePod(ns string, pvcName string, isPrivileged bool, command string) *v1.Pod {

	if len(command) == 0 {
		command = "while true; do sleep 1; done"
	}
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
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", command},
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
func CreateClientPod(c clientset.Interface, ns string, pvc *v1.PersistentVolumeClaim) *v1.Pod {
	clientPod := MakePod(ns, pvc.Name, true, "")
	clientPod, err := c.Core().Pods(ns).Create(clientPod)
	Expect(err).NotTo(HaveOccurred())

	// Verify the pod is running before returning it
	err = WaitForPodRunningInNamespace(c, clientPod)
	Expect(err).NotTo(HaveOccurred())
	clientPod, err = c.Core().Pods(ns).Get(clientPod.Name, metav1.GetOptions{})
	Expect(apierrs.IsNotFound(err)).To(BeFalse())
	return clientPod
}

func CreatePDWithRetry() (string, error) {
	newDiskName := ""
	var err error
	for start := time.Now(); time.Since(start) < PDRetryTimeout; time.Sleep(PDRetryPollTime) {
		if newDiskName, err = createPD(); err != nil {
			Logf("Couldn't create a new PD. Sleeping 5 seconds (%v)", err)
			continue
		}
		Logf("Successfully created a new PD: %q.", newDiskName)
		break
	}
	return newDiskName, err
}

func DeletePDWithRetry(diskName string) {
	var err error
	for start := time.Now(); time.Since(start) < PDRetryTimeout; time.Sleep(PDRetryPollTime) {
		if err = deletePD(diskName); err != nil {
			Logf("Couldn't delete PD %q. Sleeping 5 seconds (%v)", diskName, err)
			continue
		}
		Logf("Successfully deleted PD %q.", diskName)
		break
	}
	ExpectNoError(err, "Error deleting PD")
}

func createPD() (string, error) {
	if TestContext.Provider == "gce" || TestContext.Provider == "gke" {
		pdName := fmt.Sprintf("%s-%s", TestContext.Prefix, string(uuid.NewUUID()))

		gceCloud, err := GetGCECloud()
		if err != nil {
			return "", err
		}

		tags := map[string]string{}
		err = gceCloud.CreateDisk(pdName, gcecloud.DiskTypeSSD, TestContext.CloudConfig.Zone, 10 /* sizeGb */, tags)
		if err != nil {
			return "", err
		}
		return pdName, nil
	} else if TestContext.Provider == "aws" {
		client := ec2.New(session.New())

		request := &ec2.CreateVolumeInput{}
		request.AvailabilityZone = aws.String(TestContext.CloudConfig.Zone)
		request.Size = aws.Int64(10)
		request.VolumeType = aws.String(awscloud.DefaultVolumeType)
		response, err := client.CreateVolume(request)
		if err != nil {
			return "", err
		}

		az := aws.StringValue(response.AvailabilityZone)
		awsID := aws.StringValue(response.VolumeId)

		volumeName := "aws://" + az + "/" + awsID
		return volumeName, nil
	} else {
		return "", fmt.Errorf("Provider does not support volume creation")
	}
}

func deletePD(pdName string) error {
	if TestContext.Provider == "gce" || TestContext.Provider == "gke" {
		gceCloud, err := GetGCECloud()
		if err != nil {
			return err
		}

		err = gceCloud.DeleteDisk(pdName)

		if err != nil {
			if gerr, ok := err.(*googleapi.Error); ok && len(gerr.Errors) > 0 && gerr.Errors[0].Reason == "notFound" {
				// PD already exists, ignore error.
				return nil
			}

			Logf("Error deleting PD %q: %v", pdName, err)
		}
		return err
	} else if TestContext.Provider == "aws" {
		client := ec2.New(session.New())

		tokens := strings.Split(pdName, "/")
		awsVolumeID := tokens[len(tokens)-1]

		request := &ec2.DeleteVolumeInput{VolumeId: aws.String(awsVolumeID)}
		_, err := client.DeleteVolume(request)
		if err != nil {
			if awsError, ok := err.(awserr.Error); ok && awsError.Code() == "InvalidVolume.NotFound" {
				Logf("Volume deletion implicitly succeeded because volume %q does not exist.", pdName)
			} else {
				return fmt.Errorf("error deleting EBS volumes: %v", err)
			}
		}
		return nil
	} else {
		return fmt.Errorf("Provider does not support volume deletion")
	}
}
