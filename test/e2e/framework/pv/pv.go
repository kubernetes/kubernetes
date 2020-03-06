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
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	storageutil "k8s.io/kubernetes/pkg/apis/storage/v1/util"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
)

const (
	pdRetryTimeout  = 5 * time.Minute
	pdRetryPollTime = 5 * time.Second

	// PVBindingTimeout is how long PVs have to become bound.
	PVBindingTimeout = 3 * time.Minute

	// ClaimBindingTimeout is how long claims have to become bound.
	ClaimBindingTimeout = 3 * time.Minute

	// PVReclaimingTimeout is how long PVs have to beome reclaimed.
	PVReclaimingTimeout = 3 * time.Minute

	// PVDeletingTimeout is how long PVs have to become deleted.
	PVDeletingTimeout = 3 * time.Minute

	// VolumeSelectorKey is the key for volume selector.
	VolumeSelectorKey = "e2e-pv-pool"
)

var (
	// SELinuxLabel is common selinux labels.
	SELinuxLabel = &v1.SELinuxOptions{
		Level: "s0:c0,c1"}
)

type pvval struct{}

// PVMap is a map of all PVs used in the multi pv-pvc tests. The key is the PV's name, which is
// guaranteed to be unique. The value is {} (empty struct) since we're only interested
// in the PV's name and if it is present. We must always Get the pv object before
// referencing any of its values, eg its ClaimRef.
type PVMap map[string]pvval

type pvcval struct{}

// PVCMap is a map of all PVCs used in the multi pv-pvc tests. The key is "namespace/pvc.Name". The
// value is {} (empty struct) since we're only interested in the PVC's name and if it is
// present. We must always Get the pvc object before referencing any of its values, eg.
// its VolumeName.
// Note: It's unsafe to add keys to a map in a loop. Their insertion in the map is
//   unpredictable and can result in the same key being iterated over again.
type PVCMap map[types.NamespacedName]pvcval

// PersistentVolumeConfig is consumed by MakePersistentVolume() to generate a PV object
// for varying storage options (NFS, ceph, glusterFS, etc.).
// (+optional) prebind holds a pre-bound PVC
// Example pvSource:
//	pvSource: api.PersistentVolumeSource{
//		NFS: &api.NFSVolumeSource{
//	 		...
//	 	},
//	 }
type PersistentVolumeConfig struct {
	// [Optional] NamePrefix defaults to "pv-" if unset
	NamePrefix string
	// [Optional] Labels contains information used to organize and categorize
	// objects
	Labels labels.Set
	// PVSource contains the details of the underlying volume and must be set
	PVSource v1.PersistentVolumeSource
	// [Optional] Prebind lets you specify a PVC to bind this PV to before
	// creation
	Prebind *v1.PersistentVolumeClaim
	// [Optiona] ReclaimPolicy defaults to "Reclaim" if unset
	ReclaimPolicy    v1.PersistentVolumeReclaimPolicy
	StorageClassName string
	// [Optional] NodeAffinity defines constraints that limit what nodes this
	// volume can be accessed from.
	NodeAffinity *v1.VolumeNodeAffinity
	// [Optional] VolumeMode defaults to "Filesystem" if unset
	VolumeMode *v1.PersistentVolumeMode
	// [Optional] AccessModes defaults to RWO if unset
	AccessModes []v1.PersistentVolumeAccessMode
	// [Optional] Capacity is the storage capacity in Quantity format. Defaults
	// to "2Gi" if unset
	Capacity string
}

// PersistentVolumeClaimConfig is consumed by MakePersistentVolumeClaim() to
// generate a PVC object.
type PersistentVolumeClaimConfig struct {
	// NamePrefix defaults to "pvc-" if unspecified
	NamePrefix string
	// ClaimSize must be specified in the Quantity format. Defaults to 2Gi if
	// unspecified
	ClaimSize string
	// AccessModes defaults to RWO if unspecified
	AccessModes      []v1.PersistentVolumeAccessMode
	Annotations      map[string]string
	Selector         *metav1.LabelSelector
	StorageClassName *string
	// VolumeMode defaults to nil if unspecified or specified as the empty
	// string
	VolumeMode *v1.PersistentVolumeMode
}

// PVPVCCleanup cleans up a pv and pvc in a single pv/pvc test case.
// Note: delete errors are appended to []error so that we can attempt to delete both the pvc and pv.
func PVPVCCleanup(c clientset.Interface, ns string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) []error {
	var errs []error

	if pvc != nil {
		err := DeletePersistentVolumeClaim(c, pvc.Name, ns)
		if err != nil {
			errs = append(errs, fmt.Errorf("failed to delete PVC %q: %v", pvc.Name, err))
		}
	} else {
		framework.Logf("pvc is nil")
	}
	if pv != nil {
		err := DeletePersistentVolume(c, pv.Name)
		if err != nil {
			errs = append(errs, fmt.Errorf("failed to delete PV %q: %v", pv.Name, err))
		}
	} else {
		framework.Logf("pv is nil")
	}
	return errs
}

// PVPVCMapCleanup Cleans up pvs and pvcs in multi-pv-pvc test cases. Entries found in the pv and claim maps are
// deleted as long as the Delete api call succeeds.
// Note: delete errors are appended to []error so that as many pvcs and pvs as possible are deleted.
func PVPVCMapCleanup(c clientset.Interface, ns string, pvols PVMap, claims PVCMap) []error {
	var errs []error

	for pvcKey := range claims {
		err := DeletePersistentVolumeClaim(c, pvcKey.Name, ns)
		if err != nil {
			errs = append(errs, fmt.Errorf("failed to delete PVC %q: %v", pvcKey.Name, err))
		} else {
			delete(claims, pvcKey)
		}
	}

	for pvKey := range pvols {
		err := DeletePersistentVolume(c, pvKey)
		if err != nil {
			errs = append(errs, fmt.Errorf("failed to delete PV %q: %v", pvKey, err))
		} else {
			delete(pvols, pvKey)
		}
	}
	return errs
}

// DeletePersistentVolume deletes the PV.
func DeletePersistentVolume(c clientset.Interface, pvName string) error {
	if c != nil && len(pvName) > 0 {
		framework.Logf("Deleting PersistentVolume %q", pvName)
		err := c.CoreV1().PersistentVolumes().Delete(context.TODO(), pvName, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("PV Delete API error: %v", err)
		}
	}
	return nil
}

// DeletePersistentVolumeClaim deletes the Claim.
func DeletePersistentVolumeClaim(c clientset.Interface, pvcName string, ns string) error {
	if c != nil && len(pvcName) > 0 {
		framework.Logf("Deleting PersistentVolumeClaim %q", pvcName)
		err := c.CoreV1().PersistentVolumeClaims(ns).Delete(context.TODO(), pvcName, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("PVC Delete API error: %v", err)
		}
	}
	return nil
}

// DeletePVCandValidatePV deletes the PVC and waits for the PV to enter its expected phase. Validate that the PV
// has been reclaimed (assumption here about reclaimPolicy). Caller tells this func which
// phase value to expect for the pv bound to the to-be-deleted claim.
func DeletePVCandValidatePV(c clientset.Interface, ns string, pvc *v1.PersistentVolumeClaim, pv *v1.PersistentVolume, expectPVPhase v1.PersistentVolumePhase) error {
	pvname := pvc.Spec.VolumeName
	framework.Logf("Deleting PVC %v to trigger reclamation of PV %v", pvc.Name, pvname)
	err := DeletePersistentVolumeClaim(c, pvc.Name, ns)
	if err != nil {
		return err
	}

	// Wait for the PV's phase to return to be `expectPVPhase`
	framework.Logf("Waiting for reclaim process to complete.")
	err = WaitForPersistentVolumePhase(expectPVPhase, c, pv.Name, framework.Poll, PVReclaimingTimeout)
	if err != nil {
		return fmt.Errorf("pv %q phase did not become %v: %v", pv.Name, expectPVPhase, err)
	}

	// examine the pv's ClaimRef and UID and compare to expected values
	pv, err = c.CoreV1().PersistentVolumes().Get(context.TODO(), pv.Name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("PV Get API error: %v", err)
	}
	cr := pv.Spec.ClaimRef
	if expectPVPhase == v1.VolumeAvailable {
		if cr != nil && len(cr.UID) > 0 {
			return fmt.Errorf("PV is 'Available' but ClaimRef.UID is not empty")
		}
	} else if expectPVPhase == v1.VolumeBound {
		if cr == nil {
			return fmt.Errorf("PV is 'Bound' but ClaimRef is nil")
		}
		if len(cr.UID) == 0 {
			return fmt.Errorf("PV is 'Bound' but ClaimRef.UID is empty")
		}
	}

	framework.Logf("PV %v now in %q phase", pv.Name, expectPVPhase)
	return nil
}

// DeletePVCandValidatePVGroup wraps deletePVCandValidatePV() by calling the function in a loop over the PV map. Only bound PVs
// are deleted. Validates that the claim was deleted and the PV is in the expected Phase (Released,
// Available, Bound).
// Note: if there are more claims than pvs then some of the remaining claims may bind to just made
//   available pvs.
func DeletePVCandValidatePVGroup(c clientset.Interface, ns string, pvols PVMap, claims PVCMap, expectPVPhase v1.PersistentVolumePhase) error {
	var boundPVs, deletedPVCs int

	for pvName := range pvols {
		pv, err := c.CoreV1().PersistentVolumes().Get(context.TODO(), pvName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("PV Get API error: %v", err)
		}
		cr := pv.Spec.ClaimRef
		// if pv is bound then delete the pvc it is bound to
		if cr != nil && len(cr.Name) > 0 {
			boundPVs++
			// Assert bound PVC is tracked in this test. Failing this might
			// indicate external PVCs interfering with the test.
			pvcKey := makePvcKey(ns, cr.Name)
			if _, found := claims[pvcKey]; !found {
				return fmt.Errorf("internal: claims map is missing pvc %q", pvcKey)
			}
			// get the pvc for the delete call below
			pvc, err := c.CoreV1().PersistentVolumeClaims(ns).Get(context.TODO(), cr.Name, metav1.GetOptions{})
			if err == nil {
				if err = DeletePVCandValidatePV(c, ns, pvc, pv, expectPVPhase); err != nil {
					return err
				}
			} else if !apierrors.IsNotFound(err) {
				return fmt.Errorf("PVC Get API error: %v", err)
			}
			// delete pvckey from map even if apierrors.IsNotFound above is true and thus the
			// claim was not actually deleted here
			delete(claims, pvcKey)
			deletedPVCs++
		}
	}
	if boundPVs != deletedPVCs {
		return fmt.Errorf("expect number of bound PVs (%v) to equal number of deleted PVCs (%v)", boundPVs, deletedPVCs)
	}
	return nil
}

// create the PV resource. Fails test on error.
func createPV(c clientset.Interface, pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	pv, err := c.CoreV1().PersistentVolumes().Create(context.TODO(), pv, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("PV Create API error: %v", err)
	}
	return pv, nil
}

// CreatePV creates the PV resource. Fails test on error.
func CreatePV(c clientset.Interface, pv *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	return createPV(c, pv)
}

// CreatePVC creates the PVC resource. Fails test on error.
func CreatePVC(c clientset.Interface, ns string, pvc *v1.PersistentVolumeClaim) (*v1.PersistentVolumeClaim, error) {
	pvc, err := c.CoreV1().PersistentVolumeClaims(ns).Create(context.TODO(), pvc, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("PVC Create API error: %v", err)
	}
	return pvc, nil
}

// CreatePVCPV creates a PVC followed by the PV based on the passed in nfs-server ip and
// namespace. If the "preBind" bool is true then pre-bind the PV to the PVC
// via the PV's ClaimRef. Return the pv and pvc to reflect the created objects.
// Note: in the pre-bind case the real PVC name, which is generated, is not
//   known until after the PVC is instantiated. This is why the pvc is created
//   before the pv.
func CreatePVCPV(c clientset.Interface, pvConfig PersistentVolumeConfig, pvcConfig PersistentVolumeClaimConfig, ns string, preBind bool) (*v1.PersistentVolume, *v1.PersistentVolumeClaim, error) {
	// make the pvc spec
	pvc := MakePersistentVolumeClaim(pvcConfig, ns)
	preBindMsg := ""
	if preBind {
		preBindMsg = " pre-bound"
		pvConfig.Prebind = pvc
	}
	// make the pv spec
	pv := MakePersistentVolume(pvConfig)

	ginkgo.By(fmt.Sprintf("Creating a PVC followed by a%s PV", preBindMsg))
	pvc, err := CreatePVC(c, ns, pvc)
	if err != nil {
		return nil, nil, err
	}

	// instantiate the pv, handle pre-binding by ClaimRef if needed
	if preBind {
		pv.Spec.ClaimRef.Name = pvc.Name
	}
	pv, err = createPV(c, pv)
	if err != nil {
		return nil, pvc, err
	}
	return pv, pvc, nil
}

// CreatePVPVC creates a PV followed by the PVC based on the passed in nfs-server ip and
// namespace. If the "preBind" bool is true then pre-bind the PVC to the PV
// via the PVC's VolumeName. Return the pv and pvc to reflect the created
// objects.
// Note: in the pre-bind case the real PV name, which is generated, is not
//   known until after the PV is instantiated. This is why the pv is created
//   before the pvc.
func CreatePVPVC(c clientset.Interface, pvConfig PersistentVolumeConfig, pvcConfig PersistentVolumeClaimConfig, ns string, preBind bool) (*v1.PersistentVolume, *v1.PersistentVolumeClaim, error) {
	preBindMsg := ""
	if preBind {
		preBindMsg = " pre-bound"
	}
	framework.Logf("Creating a PV followed by a%s PVC", preBindMsg)

	// make the pv and pvc definitions
	pv := MakePersistentVolume(pvConfig)
	pvc := MakePersistentVolumeClaim(pvcConfig, ns)

	// instantiate the pv
	pv, err := createPV(c, pv)
	if err != nil {
		return nil, nil, err
	}
	// instantiate the pvc, handle pre-binding by VolumeName if needed
	if preBind {
		pvc.Spec.VolumeName = pv.Name
	}
	pvc, err = CreatePVC(c, ns, pvc)
	if err != nil {
		return pv, nil, err
	}
	return pv, pvc, nil
}

// CreatePVsPVCs creates the desired number of PVs and PVCs and returns them in separate maps. If the
// number of PVs != the number of PVCs then the min of those two counts is the number of
// PVs expected to bind. If a Create error occurs, the returned maps may contain pv and pvc
// entries for the resources that were successfully created. In other words, when the caller
// sees an error returned, it needs to decide what to do about entries in the maps.
// Note: when the test suite deletes the namespace orphaned pvcs and pods are deleted. However,
//   orphaned pvs are not deleted and will remain after the suite completes.
func CreatePVsPVCs(numpvs, numpvcs int, c clientset.Interface, ns string, pvConfig PersistentVolumeConfig, pvcConfig PersistentVolumeClaimConfig) (PVMap, PVCMap, error) {
	pvMap := make(PVMap, numpvs)
	pvcMap := make(PVCMap, numpvcs)
	extraPVCs := 0
	extraPVs := numpvs - numpvcs
	if extraPVs < 0 {
		extraPVCs = -extraPVs
		extraPVs = 0
	}
	pvsToCreate := numpvs - extraPVs // want the min(numpvs, numpvcs)

	// create pvs and pvcs
	for i := 0; i < pvsToCreate; i++ {
		pv, pvc, err := CreatePVPVC(c, pvConfig, pvcConfig, ns, false)
		if err != nil {
			return pvMap, pvcMap, err
		}
		pvMap[pv.Name] = pvval{}
		pvcMap[makePvcKey(ns, pvc.Name)] = pvcval{}
	}

	// create extra pvs or pvcs as needed
	for i := 0; i < extraPVs; i++ {
		pv := MakePersistentVolume(pvConfig)
		pv, err := createPV(c, pv)
		if err != nil {
			return pvMap, pvcMap, err
		}
		pvMap[pv.Name] = pvval{}
	}
	for i := 0; i < extraPVCs; i++ {
		pvc := MakePersistentVolumeClaim(pvcConfig, ns)
		pvc, err := CreatePVC(c, ns, pvc)
		if err != nil {
			return pvMap, pvcMap, err
		}
		pvcMap[makePvcKey(ns, pvc.Name)] = pvcval{}
	}
	return pvMap, pvcMap, nil
}

// WaitOnPVandPVC waits for the pv and pvc to bind to each other.
func WaitOnPVandPVC(c clientset.Interface, ns string, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) error {
	// Wait for newly created PVC to bind to the PV
	framework.Logf("Waiting for PV %v to bind to PVC %v", pv.Name, pvc.Name)
	err := WaitForPersistentVolumeClaimPhase(v1.ClaimBound, c, ns, pvc.Name, framework.Poll, ClaimBindingTimeout)
	if err != nil {
		return fmt.Errorf("PVC %q did not become Bound: %v", pvc.Name, err)
	}

	// Wait for PersistentVolume.Status.Phase to be Bound, which it should be
	// since the PVC is already bound.
	err = WaitForPersistentVolumePhase(v1.VolumeBound, c, pv.Name, framework.Poll, PVBindingTimeout)
	if err != nil {
		return fmt.Errorf("PV %q did not become Bound: %v", pv.Name, err)
	}

	// Re-get the pv and pvc objects
	pv, err = c.CoreV1().PersistentVolumes().Get(context.TODO(), pv.Name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("PV Get API error: %v", err)
	}
	pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Get(context.TODO(), pvc.Name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("PVC Get API error: %v", err)
	}

	// The pv and pvc are both bound, but to each other?
	// Check that the PersistentVolume.ClaimRef matches the PVC
	if pv.Spec.ClaimRef == nil {
		return fmt.Errorf("PV %q ClaimRef is nil", pv.Name)
	}
	if pv.Spec.ClaimRef.Name != pvc.Name {
		return fmt.Errorf("PV %q ClaimRef's name (%q) should be %q", pv.Name, pv.Spec.ClaimRef.Name, pvc.Name)
	}
	if pvc.Spec.VolumeName != pv.Name {
		return fmt.Errorf("PVC %q VolumeName (%q) should be %q", pvc.Name, pvc.Spec.VolumeName, pv.Name)
	}
	if pv.Spec.ClaimRef.UID != pvc.UID {
		return fmt.Errorf("PV %q ClaimRef's UID (%q) should be %q", pv.Name, pv.Spec.ClaimRef.UID, pvc.UID)
	}
	return nil
}

// WaitAndVerifyBinds searches for bound PVs and PVCs by examining pvols for non-nil claimRefs.
// NOTE: Each iteration waits for a maximum of 3 minutes per PV and, if the PV is bound,
//   up to 3 minutes for the PVC. When the number of PVs != number of PVCs, this can lead
//   to situations where the maximum wait times are reached several times in succession,
//   extending test time. Thus, it is recommended to keep the delta between PVs and PVCs
//   small.
func WaitAndVerifyBinds(c clientset.Interface, ns string, pvols PVMap, claims PVCMap, testExpected bool) error {
	var actualBinds int
	expectedBinds := len(pvols)
	if expectedBinds > len(claims) { // want the min of # pvs or #pvcs
		expectedBinds = len(claims)
	}

	for pvName := range pvols {
		err := WaitForPersistentVolumePhase(v1.VolumeBound, c, pvName, framework.Poll, PVBindingTimeout)
		if err != nil && len(pvols) > len(claims) {
			framework.Logf("WARN: pv %v is not bound after max wait", pvName)
			framework.Logf("      This may be ok since there are more pvs than pvcs")
			continue
		}
		if err != nil {
			return fmt.Errorf("PV %q did not become Bound: %v", pvName, err)
		}

		pv, err := c.CoreV1().PersistentVolumes().Get(context.TODO(), pvName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("PV Get API error: %v", err)
		}
		cr := pv.Spec.ClaimRef
		if cr != nil && len(cr.Name) > 0 {
			// Assert bound pvc is a test resource. Failing assertion could
			// indicate non-test PVC interference or a bug in the test
			pvcKey := makePvcKey(ns, cr.Name)
			if _, found := claims[pvcKey]; !found {
				return fmt.Errorf("internal: claims map is missing pvc %q", pvcKey)
			}

			err := WaitForPersistentVolumeClaimPhase(v1.ClaimBound, c, ns, cr.Name, framework.Poll, ClaimBindingTimeout)
			if err != nil {
				return fmt.Errorf("PVC %q did not become Bound: %v", cr.Name, err)
			}
			actualBinds++
		}
	}

	if testExpected && actualBinds != expectedBinds {
		return fmt.Errorf("expect number of bound PVs (%v) to equal number of claims (%v)", actualBinds, expectedBinds)
	}
	return nil
}

// Return a pvckey struct.
func makePvcKey(ns, name string) types.NamespacedName {
	return types.NamespacedName{Namespace: ns, Name: name}
}

// MakePersistentVolume returns a PV definition based on the nfs server IP. If the PVC is not nil
// then the PV is defined with a ClaimRef which includes the PVC's namespace.
// If the PVC is nil then the PV is not defined with a ClaimRef.  If no reclaimPolicy
// is assigned, assumes "Retain". Specs are expected to match the test's PVC.
// Note: the passed-in claim does not have a name until it is created and thus the PV's
//   ClaimRef cannot be completely filled-in in this func. Therefore, the ClaimRef's name
//   is added later in CreatePVCPV.
func MakePersistentVolume(pvConfig PersistentVolumeConfig) *v1.PersistentVolume {
	var claimRef *v1.ObjectReference

	if len(pvConfig.AccessModes) == 0 {
		pvConfig.AccessModes = append(pvConfig.AccessModes, v1.ReadWriteOnce)
	}

	if len(pvConfig.NamePrefix) == 0 {
		pvConfig.NamePrefix = "pv-"
	}

	if pvConfig.ReclaimPolicy == "" {
		pvConfig.ReclaimPolicy = v1.PersistentVolumeReclaimRetain
	}

	if len(pvConfig.Capacity) == 0 {
		pvConfig.Capacity = "2Gi"
	}

	if pvConfig.Prebind != nil {
		claimRef = &v1.ObjectReference{
			Kind:       "PersistentVolumeClaim",
			APIVersion: "v1",
			Name:       pvConfig.Prebind.Name,
			Namespace:  pvConfig.Prebind.Namespace,
			UID:        pvConfig.Prebind.UID,
		}
	}

	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: pvConfig.NamePrefix,
			Labels:       pvConfig.Labels,
			Annotations: map[string]string{
				util.VolumeGidAnnotationKey: "777",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: pvConfig.ReclaimPolicy,
			Capacity: v1.ResourceList{
				v1.ResourceStorage: resource.MustParse(pvConfig.Capacity),
			},
			PersistentVolumeSource: pvConfig.PVSource,
			AccessModes:            pvConfig.AccessModes,
			ClaimRef:               claimRef,
			StorageClassName:       pvConfig.StorageClassName,
			NodeAffinity:           pvConfig.NodeAffinity,
			VolumeMode:             pvConfig.VolumeMode,
		},
	}
}

// MakePersistentVolumeClaim returns a PVC API Object based on the PersistentVolumeClaimConfig.
func MakePersistentVolumeClaim(cfg PersistentVolumeClaimConfig, ns string) *v1.PersistentVolumeClaim {

	if len(cfg.AccessModes) == 0 {
		cfg.AccessModes = append(cfg.AccessModes, v1.ReadWriteOnce)
	}

	if len(cfg.ClaimSize) == 0 {
		cfg.ClaimSize = "2Gi"
	}

	if len(cfg.NamePrefix) == 0 {
		cfg.NamePrefix = "pvc-"
	}

	if cfg.VolumeMode != nil && *cfg.VolumeMode == "" {
		framework.Logf("Warning: Making PVC: VolumeMode specified as invalid empty string, treating as nil")
		cfg.VolumeMode = nil
	}

	return &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: cfg.NamePrefix,
			Namespace:    ns,
			Annotations:  cfg.Annotations,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			Selector:    cfg.Selector,
			AccessModes: cfg.AccessModes,
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceStorage: resource.MustParse(cfg.ClaimSize),
				},
			},
			StorageClassName: cfg.StorageClassName,
			VolumeMode:       cfg.VolumeMode,
		},
	}
}

func createPDWithRetry(zone string) (string, error) {
	var err error
	var newDiskName string
	for start := time.Now(); time.Since(start) < pdRetryTimeout; time.Sleep(pdRetryPollTime) {
		newDiskName, err = createPD(zone)
		if err != nil {
			framework.Logf("Couldn't create a new PD, sleeping 5 seconds: %v", err)
			continue
		}
		framework.Logf("Successfully created a new PD: %q.", newDiskName)
		return newDiskName, nil
	}
	return "", err
}

// CreatePDWithRetry creates PD with retry.
func CreatePDWithRetry() (string, error) {
	return createPDWithRetry("")
}

// CreatePDWithRetryAndZone creates PD on zone with retry.
func CreatePDWithRetryAndZone(zone string) (string, error) {
	return createPDWithRetry(zone)
}

// DeletePDWithRetry deletes PD with retry.
func DeletePDWithRetry(diskName string) error {
	var err error
	for start := time.Now(); time.Since(start) < pdRetryTimeout; time.Sleep(pdRetryPollTime) {
		err = deletePD(diskName)
		if err != nil {
			framework.Logf("Couldn't delete PD %q, sleeping %v: %v", diskName, pdRetryPollTime, err)
			continue
		}
		framework.Logf("Successfully deleted PD %q.", diskName)
		return nil
	}
	return fmt.Errorf("unable to delete PD %q: %v", diskName, err)
}

func createPD(zone string) (string, error) {
	if zone == "" {
		zone = framework.TestContext.CloudConfig.Zone
	}
	return framework.TestContext.CloudConfig.Provider.CreatePD(zone)
}

func deletePD(pdName string) error {
	return framework.TestContext.CloudConfig.Provider.DeletePD(pdName)
}

// WaitForPVClaimBoundPhase waits until all pvcs phase set to bound
func WaitForPVClaimBoundPhase(client clientset.Interface, pvclaims []*v1.PersistentVolumeClaim, timeout time.Duration) ([]*v1.PersistentVolume, error) {
	persistentvolumes := make([]*v1.PersistentVolume, len(pvclaims))

	for index, claim := range pvclaims {
		err := WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, claim.Namespace, claim.Name, framework.Poll, timeout)
		if err != nil {
			return persistentvolumes, err
		}
		// Get new copy of the claim
		claim, err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(context.TODO(), claim.Name, metav1.GetOptions{})
		if err != nil {
			return persistentvolumes, fmt.Errorf("PVC Get API error: %v", err)
		}
		// Get the bounded PV
		persistentvolumes[index], err = client.CoreV1().PersistentVolumes().Get(context.TODO(), claim.Spec.VolumeName, metav1.GetOptions{})
		if err != nil {
			return persistentvolumes, fmt.Errorf("PV Get API error: %v", err)
		}
	}
	return persistentvolumes, nil
}

// WaitForPersistentVolumePhase waits for a PersistentVolume to be in a specific phase or until timeout occurs, whichever comes first.
func WaitForPersistentVolumePhase(phase v1.PersistentVolumePhase, c clientset.Interface, pvName string, Poll, timeout time.Duration) error {
	framework.Logf("Waiting up to %v for PersistentVolume %s to have phase %s", timeout, pvName, phase)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		pv, err := c.CoreV1().PersistentVolumes().Get(context.TODO(), pvName, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Get persistent volume %s in failed, ignoring for %v: %v", pvName, Poll, err)
			continue
		}
		if pv.Status.Phase == phase {
			framework.Logf("PersistentVolume %s found and phase=%s (%v)", pvName, phase, time.Since(start))
			return nil
		}
		framework.Logf("PersistentVolume %s found but phase is %s instead of %s.", pvName, pv.Status.Phase, phase)
	}
	return fmt.Errorf("PersistentVolume %s not in phase %s within %v", pvName, phase, timeout)
}

// WaitForPersistentVolumeClaimPhase waits for a PersistentVolumeClaim to be in a specific phase or until timeout occurs, whichever comes first.
func WaitForPersistentVolumeClaimPhase(phase v1.PersistentVolumeClaimPhase, c clientset.Interface, ns string, pvcName string, Poll, timeout time.Duration) error {
	return WaitForPersistentVolumeClaimsPhase(phase, c, ns, []string{pvcName}, Poll, timeout, true)
}

// WaitForPersistentVolumeClaimsPhase waits for any (if matchAny is true) or all (if matchAny is false) PersistentVolumeClaims
// to be in a specific phase or until timeout occurs, whichever comes first.
func WaitForPersistentVolumeClaimsPhase(phase v1.PersistentVolumeClaimPhase, c clientset.Interface, ns string, pvcNames []string, Poll, timeout time.Duration, matchAny bool) error {
	if len(pvcNames) == 0 {
		return fmt.Errorf("Incorrect parameter: Need at least one PVC to track. Found 0")
	}
	framework.Logf("Waiting up to %v for PersistentVolumeClaims %v to have phase %s", timeout, pvcNames, phase)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		phaseFoundInAllClaims := true
		for _, pvcName := range pvcNames {
			pvc, err := c.CoreV1().PersistentVolumeClaims(ns).Get(context.TODO(), pvcName, metav1.GetOptions{})
			if err != nil {
				framework.Logf("Failed to get claim %q, retrying in %v. Error: %v", pvcName, Poll, err)
				continue
			}
			if pvc.Status.Phase == phase {
				framework.Logf("PersistentVolumeClaim %s found and phase=%s (%v)", pvcName, phase, time.Since(start))
				if matchAny {
					return nil
				}
			} else {
				framework.Logf("PersistentVolumeClaim %s found but phase is %s instead of %s.", pvcName, pvc.Status.Phase, phase)
				phaseFoundInAllClaims = false
			}
		}
		if phaseFoundInAllClaims {
			return nil
		}
	}
	return fmt.Errorf("PersistentVolumeClaims %v not all in phase %s within %v", pvcNames, phase, timeout)
}

// CreatePVSource creates a PV source.
func CreatePVSource(zone string) (*v1.PersistentVolumeSource, error) {
	diskName, err := CreatePDWithRetryAndZone(zone)
	if err != nil {
		return nil, err
	}
	return framework.TestContext.CloudConfig.Provider.CreatePVSource(zone, diskName)
}

// DeletePVSource deletes a PV source.
func DeletePVSource(pvSource *v1.PersistentVolumeSource) error {
	return framework.TestContext.CloudConfig.Provider.DeletePVSource(pvSource)
}

// GetDefaultStorageClassName returns default storageClass or return error
func GetDefaultStorageClassName(c clientset.Interface) (string, error) {
	list, err := c.StorageV1().StorageClasses().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return "", fmt.Errorf("Error listing storage classes: %v", err)
	}
	var scName string
	for _, sc := range list.Items {
		if storageutil.IsDefaultAnnotation(sc.ObjectMeta) {
			if len(scName) != 0 {
				return "", fmt.Errorf("Multiple default storage classes found: %q and %q", scName, sc.Name)
			}
			scName = sc.Name
		}
	}
	if len(scName) == 0 {
		return "", fmt.Errorf("No default storage class found")
	}
	framework.Logf("Default storage class: %q", scName)
	return scName, nil
}

// SkipIfNoDefaultStorageClass skips tests if no default SC can be found.
func SkipIfNoDefaultStorageClass(c clientset.Interface) {
	_, err := GetDefaultStorageClassName(c)
	if err != nil {
		e2eskipper.Skipf("error finding default storageClass : %v", err)
	}
}
