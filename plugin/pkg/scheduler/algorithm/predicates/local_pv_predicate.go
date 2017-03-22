/*
Copyright 2014 The Kubernetes Authors.

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

package predicates

import (
	"fmt"
	"reflect"

	"github.com/golang/glog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

type VolumeNodeChecker struct {
	pvInfo  PersistentVolumeInfo
	pvcInfo PersistentVolumeClaimInfo
	client  clientset.Interface
}

// annBoundByController annotation applies to PVs and PVCs.  It indicates that
// the binding (PV->PVC or PVC->PV) was installed by the controller.  The
// absence of this annotation means the binding was done by the user (i.e.
// pre-bound). Value of this annotation does not matter.
const annBoundByController = "pv.kubernetes.io/bound-by-controller"

// annBindCompleted annotation applies to PVCs. It indicates that the lifecycle
// of the PVC has passed through the initial setup. This information changes how
// we interpret some observations of the state of the objects. Value of this
// annotation does not matter.
const annBindCompleted = "pv.kubernetes.io/bind-completed"

// VolumeNodeChecker evaluates if a pod can fit due to the volumes it requests, given
// that some volumes have node scheduling constraints, particularly when using LocalStorage PVs.
// The requirement is that any pod that uses a PVC that is bound to a LocalStorage PV must be scheduled to the
// LocalStorage PV's node
func NewVolumeNodePredicate(pvInfo PersistentVolumeInfo, pvcInfo PersistentVolumeClaimInfo, client clientset.Interface) algorithm.FitPredicate {
	c := &VolumeNodeChecker{
		pvInfo:  pvInfo,
		pvcInfo: pvcInfo,
		client:  client,
	}
	return c.predicate
}

func (c *VolumeNodeChecker) getAllLocalVolumes(node *v1.Node) ([]*v1.PersistentVolume, error) {
	volumes, err := c.pvInfo.ListVolumes(labels.Everything())
	if err != nil {
		return nil, fmt.Errorf("volume node predicate cannot get a list of persistent volumes: %v", err)
	}
	var ret []*v1.PersistentVolume
	for _, v := range volumes {
		if v.Spec.LocalStorage != nil {
			if v.Spec.LocalStorage.NodeName == node.Name {
				ret = append(ret, v)
			}
		}
	}
	return ret, nil
}

func (c *VolumeNodeChecker) predicate(pod *v1.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	// If a pod doesn't have any volume attached to it, the predicate will always be true.
	// Thus we make a fast path for it, to avoid unnecessary computations in this case.
	if len(pod.Spec.Volumes) == 0 {
		return true, nil, nil
	}

	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}

	if err := c.handleUnboundClaims(pod, node); err != nil {
		return false, nil, err
	}

	namespace := pod.Namespace
	manifest := &(pod.Spec)
	for i := range manifest.Volumes {
		volume := &manifest.Volumes[i]
		if volume.PersistentVolumeClaim == nil {
			continue
		}
		pvcName := volume.PersistentVolumeClaim.ClaimName
		if pvcName == "" {
			return false, nil, fmt.Errorf("PersistentVolumeClaim had no name")
		}
		pvc, err := c.pvcInfo.GetPersistentVolumeClaimInfo(namespace, pvcName)
		if err != nil {
			return false, nil, err
		}

		if pvc == nil {
			return false, nil, fmt.Errorf("PersistentVolumeClaim was not found: %q", pvcName)
		}
		pvName := pvc.Spec.VolumeName
		if pvName == "" {
			return false, nil, fmt.Errorf("PersistentVolumeClaim is not bound: %q", pvcName)
		}

		pv, err := c.pvInfo.GetPersistentVolumeInfo(pvName)
		if err != nil {
			return false, nil, err
		}

		if pv == nil {
			return false, nil, fmt.Errorf("PersistentVolume not found: %q", pvName)
		}

		// Check specifically for LocalStorage PV. TODO: generalize this
		localSpec := pv.Spec.PersistentVolumeSource.LocalStorage
		if localSpec != nil {
			if localSpec.NodeName == node.Name {
				glog.V(2).Infof("VolumeNode predicate allows node %q for pod %q due to volume %q", node.Name, pod.Name, pvName)
			} else {
				glog.V(2).Infof("Won't schedule pod %q onto node %q due to volume %q node mismatch", pod.Name, node.Name, pvName)
				return false, []algorithm.PredicateFailureReason{ErrVolumeNodeConflict}, nil
			}
		}

	}
	return true, nil, nil
}

func (c *VolumeNodeChecker) handleUnboundClaims(pod *v1.Pod, node *v1.Node) error {
	var unboundClaims []*v1.PersistentVolumeClaim

	namespace := pod.Namespace
	manifest := &(pod.Spec)
	for i := range manifest.Volumes {
		volume := &manifest.Volumes[i]
		if volume.PersistentVolumeClaim == nil {
			continue
		}
		pvcName := volume.PersistentVolumeClaim.ClaimName
		if pvcName == "" {
			return fmt.Errorf("PersistentVolumeClaim had no name")
		}
		pvc, err := c.pvcInfo.GetPersistentVolumeClaimInfo(namespace, pvcName)
		if err != nil {
			return err
		}

		if pvc == nil {
			return fmt.Errorf("PersistentVolumeClaim was not found: %q", pvcName)
		}
		// Handle binding for local PVs here.
		if pvc.Spec.VolumeType != nil && *pvc.Spec.VolumeType == v1.SemiPersistentLocalStorage {
			unboundClaims = append(unboundClaims, pvc)
		}
	}
	if len(unboundClaims) == 0 {
		return nil
	}
	availableVolumes, err := c.getAllLocalVolumes(node)
	if err != nil {
		return err
	}

	type claimToVolume struct {
		claim  *v1.PersistentVolumeClaim
		volume *v1.PersistentVolume
	}
	var boundClaims []claimToVolume
	for _, claim := range unboundClaims {
		allocatedVolume, err := c.findBestVolumeForClaim(claim, availableVolumes)
		if err != nil {
			return err
		}
		if allocatedVolume == nil {
			return fmt.Errorf("failed to find %d local persistent volumes on node %q", len(unboundClaims), node.Name)
		}
		boundClaims = append(boundClaims, claimToVolume{claim, allocatedVolume})
		for idx, vol := range availableVolumes {
			if vol.Name == allocatedVolume.Name {
				availableVolumes = append(availableVolumes[:idx], availableVolumes[idx+1:]...)
			}
		}
	}
	for _, boundClaim := range boundClaims {
		err := c.bind(boundClaim.volume, boundClaim.claim)
		if err != nil {
			glog.Fatalf("<vishh> Handle this error!")
		}
	}
	return nil
}

// isVolumeBoundToClaim returns true, if given volume is pre-bound or bound
// to specific claim. Both claim.Name and claim.Namespace must be equal.
// If claim.UID is present in volume.Spec.ClaimRef, it must be equal too.
func isVolumeBoundToClaim(volume *v1.PersistentVolume, claim *v1.PersistentVolumeClaim) bool {
	if volume.Spec.ClaimRef == nil {
		return false
	}
	if claim.Name != volume.Spec.ClaimRef.Name || claim.Namespace != volume.Spec.ClaimRef.Namespace {
		return false
	}
	if volume.Spec.ClaimRef.UID != "" && claim.UID != volume.Spec.ClaimRef.UID {
		return false
	}
	return true
}

func (c *VolumeNodeChecker) findBestVolumeForClaim(claim *v1.PersistentVolumeClaim, volumes []*v1.PersistentVolume) (*v1.PersistentVolume, error) {
	var smallestVolume *v1.PersistentVolume
	var smallestVolumeSize int64
	requestedQty := claim.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	requestedSize := requestedQty.Value()
	requestedClass := v1.GetPersistentVolumeClaimClass(claim)
	var selector labels.Selector
	if claim.Spec.Selector != nil {
		internalSelector, err := metav1.LabelSelectorAsSelector(claim.Spec.Selector)
		if err != nil {
			// should be unreachable code due to validation
			return nil, fmt.Errorf("error creating internal label selector for claim: %v: %v", claimToClaimKey(claim), err)
		}
		selector = internalSelector
	}

	// Go through all available volumes with two goals:
	// - find a volume that is either pre-bound by user or dynamically
	//   provisioned for this claim. Because of this we need to loop through
	//   all volumes.
	// - find the smallest matching one if there is no volume pre-bound to
	//   the claim.
	for _, volume := range volumes {
		if isVolumeBoundToClaim(volume, claim) {
			// this claim and volume are pre-bound; return
			// the volume if the size request is satisfied,
			// otherwise continue searching for a match
			volumeQty := volume.Spec.Capacity[v1.ResourceStorage]
			volumeSize := volumeQty.Value()
			if volumeSize < requestedSize {
				continue
			}
			return volume, nil
		}

		// In Alpha dynamic provisioning, we do now want not match claims
		// with existing PVs, findByClaim must find only PVs that are
		// pre-bound to the claim (by dynamic provisioning). TODO: remove in
		// 1.5
		if metav1.HasAnnotation(claim.ObjectMeta, v1.AlphaStorageClassAnnotation) {
			continue
		}

		// filter out:
		// - volumes bound to another claim
		// - volumes whose labels don't match the claim's selector, if specified
		// - volumes in Class that is not requested
		if volume.Spec.ClaimRef != nil {
			continue
		} else if selector != nil && !selector.Matches(labels.Set(volume.Labels)) {
			continue
		}
		if v1.GetPersistentVolumeClass(volume) != requestedClass {
			continue
		}

		volumeQty := volume.Spec.Capacity[v1.ResourceStorage]
		volumeSize := volumeQty.Value()
		if volumeSize >= requestedSize {
			if smallestVolume == nil || smallestVolumeSize > volumeSize {
				smallestVolume = volume
				smallestVolumeSize = volumeSize
			}
		}
	}

	if smallestVolume != nil {
		// Found a matching volume
		return smallestVolume, nil
	}
	return nil, nil
}

// bindVolumeToClaim modifes given volume to be bound to a claim and saves it to
// API server. The claim is not modified in this method!
func (c *VolumeNodeChecker) bindVolumeToClaim(volume *v1.PersistentVolume, claim *v1.PersistentVolumeClaim) (*v1.PersistentVolume, error) {
	glog.V(4).Infof("updating PersistentVolume[%s]: binding to %q", volume.Name, claimToClaimKey(claim))

	dirty := false

	// Check if the volume was already bound (either by user or by controller)
	shouldSetBoundByController := false
	if !isVolumeBoundToClaim(volume, claim) {
		shouldSetBoundByController = true
	}

	// The volume from method args can be pointing to watcher cache. We must not
	// modify these, therefore create a copy.
	clone, err := api.Scheme.DeepCopy(volume)
	if err != nil {
		return nil, fmt.Errorf("Error cloning pv: %v", err)
	}
	volumeClone, ok := clone.(*v1.PersistentVolume)
	if !ok {
		return nil, fmt.Errorf("Unexpected volume cast error : %v", volumeClone)
	}

	// Bind the volume to the claim if it is not bound yet
	if volume.Spec.ClaimRef == nil ||
		volume.Spec.ClaimRef.Name != claim.Name ||
		volume.Spec.ClaimRef.Namespace != claim.Namespace ||
		volume.Spec.ClaimRef.UID != claim.UID {

		claimRef, err := v1.GetReference(api.Scheme, claim)
		if err != nil {
			return nil, fmt.Errorf("Unexpected error getting claim reference: %v", err)
		}
		volumeClone.Spec.ClaimRef = claimRef
		dirty = true
	}

	// Set annBoundByController if it is not set yet
	if shouldSetBoundByController && !metav1.HasAnnotation(volumeClone.ObjectMeta, annBoundByController) {
		metav1.SetMetaDataAnnotation(&volumeClone.ObjectMeta, annBoundByController, "yes")
		dirty = true
	}

	// Save the volume only if something was changed
	if dirty {
		glog.V(2).Infof("claim %q bound to volume %q", claimToClaimKey(claim), volume.Name)
		newVol, err := c.client.Core().PersistentVolumes().Update(volumeClone)
		if err != nil {
			glog.V(4).Infof("updating PersistentVolume[%s]: binding to %q failed: %v", volume.Name, claimToClaimKey(claim), err)
			return newVol, err
		}
		glog.V(4).Infof("updating PersistentVolume[%s]: bound to %q", newVol.Name, claimToClaimKey(claim))
		return newVol, nil
	}

	glog.V(4).Infof("updating PersistentVolume[%s]: already bound to %q", volume.Name, claimToClaimKey(claim))
	return volume, nil
}

// bindClaimToVolume modifies the given claim to be bound to a volume and
// saves it to API server. The volume is not modified in this method!
func (c *VolumeNodeChecker) bindClaimToVolume(claim *v1.PersistentVolumeClaim, volume *v1.PersistentVolume) (*v1.PersistentVolumeClaim, error) {
	glog.V(4).Infof("updating PersistentVolumeClaim[%s]: binding to %q", claimToClaimKey(claim), volume.Name)

	dirty := false

	// Check if the claim was already bound (either by controller or by user)
	shouldSetBoundByController := false
	if volume.Name != claim.Spec.VolumeName {
		shouldSetBoundByController = true
	}

	// The claim from method args can be pointing to watcher cache. We must not
	// modify these, therefore create a copy.
	clone, err := api.Scheme.DeepCopy(claim)
	if err != nil {
		return nil, fmt.Errorf("Error cloning claim: %v", err)
	}
	claimClone, ok := clone.(*v1.PersistentVolumeClaim)
	if !ok {
		return nil, fmt.Errorf("Unexpected claim cast error : %v", claimClone)
	}

	// Bind the claim to the volume if it is not bound yet
	if claimClone.Spec.VolumeName != volume.Name {
		claimClone.Spec.VolumeName = volume.Name
		dirty = true
	}

	// Set annBoundByController if it is not set yet
	if shouldSetBoundByController && !metav1.HasAnnotation(claimClone.ObjectMeta, annBoundByController) {
		metav1.SetMetaDataAnnotation(&claimClone.ObjectMeta, annBoundByController, "yes")
		dirty = true
	}

	// Set annBindCompleted if it is not set yet
	if !metav1.HasAnnotation(claimClone.ObjectMeta, annBindCompleted) {
		metav1.SetMetaDataAnnotation(&claimClone.ObjectMeta, annBindCompleted, "yes")
		dirty = true
	}

	if dirty {
		glog.V(2).Infof("volume %q bound to claim %q", volume.Name, claimToClaimKey(claim))
		newClaim, err := c.client.Core().PersistentVolumeClaims(claim.Namespace).Update(claimClone)
		if err != nil {
			glog.V(4).Infof("updating PersistentVolumeClaim[%s]: binding to %q failed: %v", claimToClaimKey(claim), volume.Name, err)
			return newClaim, err
		}
		glog.V(4).Infof("updating PersistentVolumeClaim[%s]: bound to %q", claimToClaimKey(claim), volume.Name)
		return newClaim, nil
	}

	glog.V(4).Infof("updating PersistentVolumeClaim[%s]: already bound to %q", claimToClaimKey(claim), volume.Name)
	return claim, nil
}

// updateVolumePhase saves new volume phase to API server.
func (c *VolumeNodeChecker) updateVolumePhase(volume *v1.PersistentVolume, phase v1.PersistentVolumePhase, message string) (*v1.PersistentVolume, error) {
	glog.V(4).Infof("updating PersistentVolume[%s]: set phase %s", volume.Name, phase)
	if volume.Status.Phase == phase {
		// Nothing to do.
		glog.V(4).Infof("updating PersistentVolume[%s]: phase %s already set", volume.Name, phase)
		return volume, nil
	}

	clone, err := api.Scheme.DeepCopy(volume)
	if err != nil {
		return nil, fmt.Errorf("Error cloning claim: %v", err)
	}
	volumeClone, ok := clone.(*v1.PersistentVolume)
	if !ok {
		return nil, fmt.Errorf("Unexpected volume cast error : %v", volumeClone)
	}

	volumeClone.Status.Phase = phase
	volumeClone.Status.Message = message

	newVol, err := c.client.Core().PersistentVolumes().UpdateStatus(volumeClone)
	if err != nil {
		glog.V(4).Infof("updating PersistentVolume[%s]: set phase %s failed: %v", volume.Name, phase, err)
		return newVol, err
	}
	glog.V(2).Infof("volume %q entered phase %q", volume.Name, phase)
	return newVol, err
}

// updateClaimStatus saves new claim.Status to API server.
// Parameters:
//  claim - claim to update
//  phasephase - phase to set
//  volume - volume which Capacity is set into claim.Status.Capacity
func (c *VolumeNodeChecker) updateClaimStatus(claim *v1.PersistentVolumeClaim, phase v1.PersistentVolumeClaimPhase, volume *v1.PersistentVolume) (*v1.PersistentVolumeClaim, error) {
	glog.V(4).Infof("updating PersistentVolumeClaim[%s] status: set phase %s", claimToClaimKey(claim), phase)

	dirty := false

	clone, err := api.Scheme.DeepCopy(claim)
	if err != nil {
		return nil, fmt.Errorf("Error cloning claim: %v", err)
	}
	claimClone, ok := clone.(*v1.PersistentVolumeClaim)
	if !ok {
		return nil, fmt.Errorf("Unexpected claim cast error : %v", claimClone)
	}

	if claim.Status.Phase != phase {
		claimClone.Status.Phase = phase
		dirty = true
	}

	if volume == nil {
		// Need to reset AccessModes and Capacity
		if claim.Status.AccessModes != nil {
			claimClone.Status.AccessModes = nil
			dirty = true
		}
		if claim.Status.Capacity != nil {
			claimClone.Status.Capacity = nil
			dirty = true
		}
	} else {
		// Need to update AccessModes and Capacity
		if !reflect.DeepEqual(claim.Status.AccessModes, volume.Spec.AccessModes) {
			claimClone.Status.AccessModes = volume.Spec.AccessModes
			dirty = true
		}

		volumeCap, ok := volume.Spec.Capacity[v1.ResourceStorage]
		if !ok {
			return nil, fmt.Errorf("PersistentVolume %q is without a storage capacity", volume.Name)
		}
		claimCap, ok := claim.Status.Capacity[v1.ResourceStorage]
		if !ok || volumeCap.Cmp(claimCap) != 0 {
			claimClone.Status.Capacity = volume.Spec.Capacity
			dirty = true
		}
	}

	if !dirty {
		// Nothing to do.
		glog.V(4).Infof("updating PersistentVolumeClaim[%s] status: phase %s already set", claimToClaimKey(claim), phase)
		return claim, nil
	}

	newClaim, err := c.client.Core().PersistentVolumeClaims(claimClone.Namespace).UpdateStatus(claimClone)
	if err != nil {
		glog.V(4).Infof("updating PersistentVolumeClaim[%s] status: set phase %s failed: %v", claimToClaimKey(claim), phase, err)
		return newClaim, err
	}
	glog.V(2).Infof("claim %q entered phase %q", claimToClaimKey(claim), phase)
	return newClaim, nil
}

func claimToClaimKey(claim *v1.PersistentVolumeClaim) string {
	return fmt.Sprintf("%s/%s", claim.Namespace, claim.Name)
}

func claimrefToClaimKey(claimref *v1.ObjectReference) string {
	return fmt.Sprintf("%s/%s", claimref.Namespace, claimref.Name)
}

// bind saves binding information both to the volume and the claim and marks
// both objects as Bound. Volume is saved first.
// It returns on first error, it's up to the caller to implement some retry
// mechanism.
func (c *VolumeNodeChecker) bind(volume *v1.PersistentVolume, claim *v1.PersistentVolumeClaim) error {
	var err error
	// use updateClaim/updatedVolume to keep the original claim/volume for
	// logging in error cases.
	var updatedClaim *v1.PersistentVolumeClaim
	var updatedVolume *v1.PersistentVolume

	glog.V(4).Infof("binding volume %q to claim %q", volume.Name, claimToClaimKey(claim))

	if updatedVolume, err = c.bindVolumeToClaim(volume, claim); err != nil {
		glog.V(3).Infof("error binding volume %q to claim %q: failed saving the volume: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}
	volume = updatedVolume

	if updatedVolume, err = c.updateVolumePhase(volume, v1.VolumeBound, ""); err != nil {
		glog.V(3).Infof("error binding volume %q to claim %q: failed saving the volume status: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}
	volume = updatedVolume

	if updatedClaim, err = c.bindClaimToVolume(claim, volume); err != nil {
		glog.V(3).Infof("error binding volume %q to claim %q: failed saving the claim: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}
	claim = updatedClaim

	if updatedClaim, err = c.updateClaimStatus(claim, v1.ClaimBound, volume); err != nil {
		glog.V(3).Infof("error binding volume %q to claim %q: failed saving the claim status: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}
	claim = updatedClaim

	glog.V(4).Infof("volume %q bound to claim %q", volume.Name, claimToClaimKey(claim))
	return nil
}
