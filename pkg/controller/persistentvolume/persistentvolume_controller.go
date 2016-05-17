/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package persistentvolume

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	vol "k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// Design:
//
// The fundamental key to this design is the bi-directional "pointer" between
// PersistentVolumes (PVs) and PersistentVolumeClaims (PVCs), which is
// represented here as pvc.Spec.VolumeName and pv.Spec.ClaimRef. The bi-directionality
// is complicated to manage in a transactionless system, but without it we
// can't ensure sane behavior in the face of different forms of trouble.  For
// example, a rogue HA controller instance could end up racing and making
// multiple bindings that are indistinguishable, resulting in potential data
// loss.
//
// This controller is designed to work in active-passive high availability mode.
// It *could* work also in active-active HA mode, all the object transitions are
// designed to cope with this, however performance could be lower as these two
// active controllers will step on each other toes frequently.
//
// This controller supports pre-bound (by the creator) objects in both
// directions: a PVC that wants a specific PV or a PV that is reserved for a
// specific PVC.
//
// The binding is two-step process. PV.Spec.ClaimRef is modified first and
// PVC.Spec.VolumeName second. At any point of this transaction, the PV or PVC
// can be modified by user or other controller or completelly deleted. Also, two
// (or more) controllers may try to bind different volumes to different claims
// at the same time. The controller must recover from any conflicts that may
// arise from these conditions.

// annBindCompleted annotation applies to PVCs. It indicates that the lifecycle
// of the PVC has passed through the initial setup. This information changes how
// we interpret some observations of the state of the objects. Value of this
// annotation does not matter.
const annBindCompleted = "pv.kubernetes.io/bind-completed"

// annBoundByController annotation applies to PVs and PVCs.  It indicates that
// the binding (PV->PVC or PVC->PV) was installed by the controller.  The
// absence of this annotation means the binding was done by the user (i.e.
// pre-bound). Value of this annotation does not matter.
const annBoundByController = "pv.kubernetes.io/bound-by-controller"

// annClass annotation represents a new field which instructs dynamic
// provisioning to choose a particular storage class (aka profile).
// Value of this annotation should be empty.
const annClass = "volume.alpha.kubernetes.io/storage-class"

// PersistentVolumeController is a controller that synchronizes
// PersistentVolumeClaims and PersistentVolumes. It starts two
// framework.Controllers that watch PerstentVolume and PersistentVolumeClaim
// changes.
type PersistentVolumeController struct {
	volumes                persistentVolumeOrderedIndex
	volumeController       *framework.Controller
	volumeControllerStopCh chan struct{}
	claims                 cache.Store
	claimController        *framework.Controller
	claimControllerStopCh  chan struct{}
	kubeClient             clientset.Interface
}

// NewPersistentVolumeController creates a new PersistentVolumeController
func NewPersistentVolumeController(
	kubeClient clientset.Interface,
	syncPeriod time.Duration,
	provisioner vol.ProvisionableVolumePlugin,
	recyclers []vol.VolumePlugin,
	cloud cloudprovider.Interface) *PersistentVolumeController {

	controller := &PersistentVolumeController{
		kubeClient: kubeClient,
	}

	volumeSource := &cache.ListWatch{
		ListFunc: func(options api.ListOptions) (runtime.Object, error) {
			return kubeClient.Core().PersistentVolumes().List(options)
		},
		WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
			return kubeClient.Core().PersistentVolumes().Watch(options)
		},
	}

	claimSource := &cache.ListWatch{
		ListFunc: func(options api.ListOptions) (runtime.Object, error) {
			return kubeClient.Core().PersistentVolumeClaims(api.NamespaceAll).List(options)
		},
		WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
			return kubeClient.Core().PersistentVolumeClaims(api.NamespaceAll).Watch(options)
		},
	}

	controller.initializeController(syncPeriod, volumeSource, claimSource)

	return controller
}

// initializeController prepares watching for PersistentVolume and
// PersistentVolumeClaim events from given sources. This should be used to
// initialize the controller for real operation (with real event sources) and
// also during testing (with fake ones).
func (ctrl *PersistentVolumeController) initializeController(syncPeriod time.Duration, volumeSource, claimSource cache.ListerWatcher) {
	glog.V(4).Infof("initializing PersistentVolumeController, sync every %s", syncPeriod.String())
	ctrl.volumes.store, ctrl.volumeController = framework.NewIndexerInformer(
		volumeSource,
		&api.PersistentVolume{},
		syncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    ctrl.addVolume,
			UpdateFunc: ctrl.updateVolume,
			DeleteFunc: ctrl.deleteVolume,
		},
		cache.Indexers{"accessmodes": accessModesIndexFunc},
	)
	ctrl.claims, ctrl.claimController = framework.NewInformer(
		claimSource,
		&api.PersistentVolumeClaim{},
		syncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    ctrl.addClaim,
			UpdateFunc: ctrl.updateClaim,
			DeleteFunc: ctrl.deleteClaim,
		},
	)
}

// addVolume is callback from framework.Controller watching PersistentVolume
// events.
func (ctrl *PersistentVolumeController) addVolume(obj interface{}) {
	if !ctrl.isFullySynced() {
		return
	}

	pv, ok := obj.(*api.PersistentVolume)
	if !ok {
		glog.Errorf("expected PersistentVolume but handler received %+v", obj)
		return
	}
	if err := ctrl.syncVolume(pv); err != nil {
		glog.Errorf("PersistentVolumeController could not add volume %q: %+v", pv.Name, err)
	}
}

// updateVolume is callback from framework.Controller watching PersistentVolume
// events.
func (ctrl *PersistentVolumeController) updateVolume(oldObj, newObj interface{}) {
	if !ctrl.isFullySynced() {
		return
	}

	newVolume, ok := newObj.(*api.PersistentVolume)
	if !ok {
		glog.Errorf("Expected PersistentVolume but handler received %+v", newObj)
		return
	}
	if err := ctrl.syncVolume(newVolume); err != nil {
		glog.Errorf("PersistentVolumeController could not update volume %q: %+v", newVolume.Name, err)
	}
}

// deleteVolume is callback from framework.Controller watching PersistentVolume
// events.
func (ctrl *PersistentVolumeController) deleteVolume(obj interface{}) {
	// Intentionally left blank - we do not react on deleted volumes
}

// addClaim is callback from framework.Controller watching PersistentVolumeClaim
// events.
func (ctrl *PersistentVolumeController) addClaim(obj interface{}) {
	if !ctrl.isFullySynced() {
		return
	}

	claim, ok := obj.(*api.PersistentVolumeClaim)
	if !ok {
		glog.Errorf("Expected PersistentVolumeClaim but addClaim received %+v", obj)
		return
	}
	if err := ctrl.syncClaim(claim); err != nil {
		glog.Errorf("PersistentVolumeController could not add claim %q: %+v", claimToClaimKey(claim), err)
	}
}

// updateClaim is callback from framework.Controller watching PersistentVolumeClaim
// events.
func (ctrl *PersistentVolumeController) updateClaim(oldObj, newObj interface{}) {
	if !ctrl.isFullySynced() {
		return
	}

	newClaim, ok := newObj.(*api.PersistentVolumeClaim)
	if !ok {
		glog.Errorf("Expected PersistentVolumeClaim but updateClaim received %+v", newObj)
		return
	}
	if err := ctrl.syncClaim(newClaim); err != nil {
		glog.Errorf("PersistentVolumeController could not update claim %q: %+v", claimToClaimKey(newClaim), err)
	}
}

// deleteClaim is callback from framework.Controller watching PersistentVolumeClaim
// events.
func (ctrl *PersistentVolumeController) deleteClaim(obj interface{}) {
	if !ctrl.isFullySynced() {
		return
	}

	var volume *api.PersistentVolume
	var claim *api.PersistentVolumeClaim
	var ok bool

	claim, ok = obj.(*api.PersistentVolumeClaim)
	if !ok {
		if unknown, ok := obj.(cache.DeletedFinalStateUnknown); ok && unknown.Obj != nil {
			claim, ok = unknown.Obj.(*api.PersistentVolumeClaim)
			if !ok {
				glog.Errorf("Expected PersistentVolumeClaim but deleteClaim received %+v", unknown.Obj)
				return
			}
		} else {
			glog.Errorf("Expected PersistentVolumeClaim but deleteClaim received %+v", obj)
			return
		}
	}

	if !ok || claim == nil {
		return
	}

	if pvObj, exists, _ := ctrl.volumes.store.GetByKey(claim.Spec.VolumeName); exists {
		if volume, ok = pvObj.(*api.PersistentVolume); ok {
			// sync the volume when its claim is deleted.  Explicitly sync'ing the
			// volume here in response to claim deletion prevents the volume from
			// waiting until the next sync period for its Release.
			if volume != nil {
				err := ctrl.syncVolume(volume)
				if err != nil {
					glog.Errorf("PersistentVolumeController could not update volume %q from deleteClaim handler: %+v", volume.Name, err)
				}
			}
		} else {
			glog.Errorf("Cannot convert object from volume cache to volume %q!?: %+v", claim.Spec.VolumeName, pvObj)
		}
	}
}

// syncClaim is the main controller method to decide what to do with a claim.
// It's invoked by appropriate framework.Controller callbacks when a claim is
// created, updated or periodically synced. We do not differentiate between
// these events.
// For easier readability, it was split into syncUnboundClaim and syncBoundClaim
// methods.
func (ctrl *PersistentVolumeController) syncClaim(claim *api.PersistentVolumeClaim) error {
	glog.V(4).Infof("synchronizing PersistentVolumeClaim[%s]: %s", claimToClaimKey(claim), getClaimStatusForLogging(claim))

	if !hasAnnotation(claim.ObjectMeta, annBindCompleted) {
		return ctrl.syncUnboundClaim(claim)
	} else {
		return ctrl.syncBoundClaim(claim)
	}
}

// syncUnboundClaim is the main controller method to decide what to do with an
// unbound claim.
func (ctrl *PersistentVolumeController) syncUnboundClaim(claim *api.PersistentVolumeClaim) error {
	// This is a new PVC that has not completed binding
	// OBSERVATION: pvc is "Pending"
	if claim.Spec.VolumeName == "" {
		// User did not care which PV they get.
		// [Unit test set 1]
		volume, err := ctrl.volumes.findBestMatchForClaim(claim)
		if err != nil {
			glog.V(2).Infof("synchronizing unbound PersistentVolumeClaim[%s]: Error finding PV for claim: %v", claimToClaimKey(claim), err)
			return fmt.Errorf("Error finding PV for claim %q: %v", claimToClaimKey(claim), err)
		}
		if volume == nil {
			glog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: no volume found", claimToClaimKey(claim))
			// No PV could be found
			// OBSERVATION: pvc is "Pending", will retry
			if hasAnnotation(claim.ObjectMeta, annClass) {
				// TODO: provisioning
				//plugin := findProvisionerPluginForPV(pv) // Need to flesh this out
				//if plugin != nil {
				//FIXME: left off here
				// No match was found and provisioning was requested.
				//
				// maintain a map with the current provisioner goroutines that are running
				// if the key is already present in the map, return
				//
				// launch the goroutine that:
				// 1. calls plugin.Provision to make the storage asset
				// 2. gets back a PV object (partially filled)
				// 3. create the PV API object, with claimRef -> pvc
				// 4. deletes itself from the map when it's done
				// return
				//} else {
				// make an event calling out that no provisioner was configured
				// return, try later?
				//}
			}
			// Mark the claim as Pending and try to find a match in the next
			// periodic syncClaim
			if _, err = ctrl.updateClaimPhase(claim, api.ClaimPending); err != nil {
				return err
			}
			return nil
		} else /* pv != nil */ {
			// Found a PV for this claim
			// OBSERVATION: pvc is "Pending", pv is "Available"
			glog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume %q found: %s", claimToClaimKey(claim), volume.Name, getVolumeStatusForLogging(volume))
			if err = ctrl.bind(volume, claim); err != nil {
				// On any error saving the volume or the claim, subsequent
				// syncClaim will finish the binding.
				return err
			}
			// OBSERVATION: claim is "Bound", pv is "Bound"
			return nil
		}
	} else /* pvc.Spec.VolumeName != nil */ {
		// [Unit test set 2]
		// User asked for a specific PV.
		glog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume %q requested", claimToClaimKey(claim), claim.Spec.VolumeName)
		obj, found, err := ctrl.volumes.store.GetByKey(claim.Spec.VolumeName)
		if err != nil {
			return err
		}
		if !found {
			// User asked for a PV that does not exist.
			// OBSERVATION: pvc is "Pending"
			// Retry later.
			glog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume %q requested and not found, will try again next time", claimToClaimKey(claim), claim.Spec.VolumeName)
			if _, err = ctrl.updateClaimPhase(claim, api.ClaimPending); err != nil {
				return err
			}
			return nil
		} else {
			volume, ok := obj.(*api.PersistentVolume)
			if !ok {
				return fmt.Errorf("Cannot convert object from volume cache to volume %q!?: %+v", claim.Spec.VolumeName, obj)
			}
			glog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume %q requested and found: %s", claimToClaimKey(claim), claim.Spec.VolumeName, getVolumeStatusForLogging(volume))
			if volume.Spec.ClaimRef == nil {
				// User asked for a PV that is not claimed
				// OBSERVATION: pvc is "Pending", pv is "Available"
				glog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume is unbound, binding", claimToClaimKey(claim))
				if err = ctrl.bind(volume, claim); err != nil {
					// On any error saving the volume or the claim, subsequent
					// syncClaim will finish the binding.
					return err
				}
				// OBSERVATION: pvc is "Bound", pv is "Bound"
				return nil
			} else if isVolumeBoundToClaim(volume, claim) {
				// User asked for a PV that is claimed by this PVC
				// OBSERVATION: pvc is "Pending", pv is "Bound"
				glog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume already bound, finishing the binding", claimToClaimKey(claim))

				// Finish the volume binding by adding claim UID.
				if err = ctrl.bind(volume, claim); err != nil {
					return err
				}
				// OBSERVATION: pvc is "Bound", pv is "Bound"
				return nil
			} else {
				// User asked for a PV that is claimed by someone else
				// OBSERVATION: pvc is "Pending", pv is "Bound"
				if !hasAnnotation(claim.ObjectMeta, annBoundByController) {
					glog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume already bound to different claim by user, will retry later", claimToClaimKey(claim))
					// User asked for a specific PV, retry later
					if _, err = ctrl.updateClaimPhase(claim, api.ClaimPending); err != nil {
						return err
					}
					return nil
				} else {
					// This should never happen because someone had to remove
					// annBindCompleted annotation on the claim.
					otherClaimName := fmt.Sprintf("%s/%s", volume.Spec.ClaimRef.Namespace, volume.Spec.ClaimRef.Name)
					glog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume already bound to different claim %q by controller, THIS SHOULD NEVER HAPPEN", claimToClaimKey(claim), otherClaimName)
					return fmt.Errorf("Invalid binding of claim %q to volume %q: volume already claimed by %q", claimToClaimKey(claim), claim.Spec.VolumeName, otherClaimName)
				}
			}
		}
	}
}

// syncBoundClaim is the main controller method to decide what to do with a
// bound claim.
func (ctrl *PersistentVolumeController) syncBoundClaim(claim *api.PersistentVolumeClaim) error {
	return nil
}

// syncVolume is the main controller method to decide what to do with a volume.
// It's invoked by appropriate framework.Controller callbacks when a volume is
// created, updated or periodically synced. We do not differentiate between
// these events.
func (ctrl *PersistentVolumeController) syncVolume(volume *api.PersistentVolume) error {
	glog.V(4).Infof("synchronizing PersistentVolume[%s]: %s", volume.Name, getVolumeStatusForLogging(volume))

	return nil
}

// Run starts all of this controller's control loops
func (ctrl *PersistentVolumeController) Run() {
	glog.V(4).Infof("starting PersistentVolumeController")

	if ctrl.volumeControllerStopCh == nil {
		ctrl.volumeControllerStopCh = make(chan struct{})
		go ctrl.volumeController.Run(ctrl.volumeControllerStopCh)
	}

	if ctrl.claimControllerStopCh == nil {
		ctrl.claimControllerStopCh = make(chan struct{})
		go ctrl.claimController.Run(ctrl.claimControllerStopCh)
	}
}

// Stop gracefully shuts down this controller
func (ctrl *PersistentVolumeController) Stop() {
	glog.V(4).Infof("stopping PersistentVolumeController")
	close(ctrl.volumeControllerStopCh)
	close(ctrl.claimControllerStopCh)
}

// isFullySynced returns true, if both volume and claim caches are fully loaded
// after startup.
// We do not want to process events with not fully loaded caches - e.g. we might
// recycle/delete PVs that don't have corresponding claim in the cache yet.
func (ctrl *PersistentVolumeController) isFullySynced() bool {
	return ctrl.volumeController.HasSynced() && ctrl.claimController.HasSynced()
}

// updateClaimPhase saves new claim phase to API server.
func (ctrl *PersistentVolumeController) updateClaimPhase(claim *api.PersistentVolumeClaim, phase api.PersistentVolumeClaimPhase) (*api.PersistentVolumeClaim, error) {
	glog.V(4).Infof("updating PersistentVolumeClaim[%s]: set phase %s", claimToClaimKey(claim), phase)
	if claim.Status.Phase == phase {
		// Nothing to do.
		glog.V(4).Infof("updating PersistentVolumeClaim[%s]: phase %s already set", claimToClaimKey(claim), phase)
		return claim, nil
	}

	clone, err := conversion.NewCloner().DeepCopy(claim)
	if err != nil {
		return nil, fmt.Errorf("Error cloning claim: %v", err)
	}
	claimClone, ok := clone.(*api.PersistentVolumeClaim)
	if !ok {
		return nil, fmt.Errorf("Unexpected claim cast error : %v", claimClone)
	}

	claimClone.Status.Phase = phase
	newClaim, err := ctrl.kubeClient.Core().PersistentVolumeClaims(claimClone.Namespace).UpdateStatus(claimClone)
	if err != nil {
		glog.V(4).Infof("updating PersistentVolumeClaim[%s]: set phase %s failed: %v", claimToClaimKey(claim), phase, err)
		return newClaim, err
	}
	glog.V(2).Infof("claim %q entered phase %q", claimToClaimKey(claim), phase)
	return newClaim, nil
}

// updateVolumePhase saves new volume phase to API server.
func (ctrl *PersistentVolumeController) updateVolumePhase(volume *api.PersistentVolume, phase api.PersistentVolumePhase) (*api.PersistentVolume, error) {
	glog.V(4).Infof("updating PersistentVolume[%s]: set phase %s", volume.Name, phase)
	if volume.Status.Phase == phase {
		// Nothing to do.
		glog.V(4).Infof("updating PersistentVolume[%s]: phase %s already set", volume.Name, phase)
		return volume, nil
	}

	clone, err := conversion.NewCloner().DeepCopy(volume)
	if err != nil {
		return nil, fmt.Errorf("Error cloning claim: %v", err)
	}
	volumeClone, ok := clone.(*api.PersistentVolume)
	if !ok {
		return nil, fmt.Errorf("Unexpected volume cast error : %v", volumeClone)
	}

	volumeClone.Status.Phase = phase
	newVol, err := ctrl.kubeClient.Core().PersistentVolumes().UpdateStatus(volumeClone)
	if err != nil {
		glog.V(4).Infof("updating PersistentVolume[%s]: set phase %s failed: %v", volume.Name, phase, err)
		return newVol, err
	}
	glog.V(2).Infof("volume %q entered phase %q", volume.Name, phase)
	return newVol, err
}

// bindVolumeToClaim modifes given volume to be bound to a claim and saves it to
// API server. The claim is not modified in this method!
func (ctrl *PersistentVolumeController) bindVolumeToClaim(volume *api.PersistentVolume, claim *api.PersistentVolumeClaim) (*api.PersistentVolume, error) {
	glog.V(4).Infof("updating PersistentVolume[%s]: binding to %q", volume.Name, claimToClaimKey(claim))

	dirty := false

	// Check if the volume was already bound (either by user or by controller)
	shouldSetBoundByController := false
	if !isVolumeBoundToClaim(volume, claim) {
		shouldSetBoundByController = true
	}

	// The volume from method args can be pointing to watcher cache. We must not
	// modify these, therefore create a copy.
	clone, err := conversion.NewCloner().DeepCopy(volume)
	if err != nil {
		return nil, fmt.Errorf("Error cloning pv: %v", err)
	}
	volumeClone, ok := clone.(*api.PersistentVolume)
	if !ok {
		return nil, fmt.Errorf("Unexpected volume cast error : %v", volumeClone)
	}

	// Bind the volume to the claim if it is not bound yet
	if volume.Spec.ClaimRef == nil ||
		volume.Spec.ClaimRef.Name != claim.Name ||
		volume.Spec.ClaimRef.Namespace != claim.Namespace ||
		volume.Spec.ClaimRef.UID != claim.UID {

		claimRef, err := api.GetReference(claim)
		if err != nil {
			return nil, fmt.Errorf("Unexpected error getting claim reference: %v", err)
		}
		volumeClone.Spec.ClaimRef = claimRef
		dirty = true
	}

	// Set annBoundByController if it is not set yet
	if shouldSetBoundByController && !hasAnnotation(volumeClone.ObjectMeta, annBoundByController) {
		setAnnotation(&volumeClone.ObjectMeta, annBoundByController, "yes")
		dirty = true
	}

	// Save the volume only if something was changed
	if dirty {
		glog.V(2).Infof("claim %q bound to volume %q", claimToClaimKey(claim), volume.Name)
		newVol, err := ctrl.kubeClient.Core().PersistentVolumes().Update(volumeClone)
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

// bindClaimToVolume modifes given claim to be bound to a volume and saves it to
// API server. The volume is not modified in this method!
func (ctrl *PersistentVolumeController) bindClaimToVolume(claim *api.PersistentVolumeClaim, volume *api.PersistentVolume) (*api.PersistentVolumeClaim, error) {
	glog.V(4).Infof("updating PersistentVolumeClaim[%s]: binding to %q", claimToClaimKey(claim), volume.Name)

	dirty := false

	// Check if the claim was already bound (either by controller or by user)
	shouldSetBoundByController := false
	if volume.Name != claim.Spec.VolumeName {
		shouldSetBoundByController = true
	}

	// The claim from method args can be pointing to watcher cache. We must not
	// modify these, therefore create a copy.
	clone, err := conversion.NewCloner().DeepCopy(claim)
	if err != nil {
		return nil, fmt.Errorf("Error cloning claim: %v", err)
	}
	claimClone, ok := clone.(*api.PersistentVolumeClaim)
	if !ok {
		return nil, fmt.Errorf("Unexpected claim cast error : %v", claimClone)
	}

	// Bind the claim to the volume if it is not bound yet
	if claimClone.Spec.VolumeName != volume.Name {
		claimClone.Spec.VolumeName = volume.Name
		dirty = true
	}

	// Set annBoundByController if it is not set yet
	if shouldSetBoundByController && !hasAnnotation(claimClone.ObjectMeta, annBoundByController) {
		setAnnotation(&claimClone.ObjectMeta, annBoundByController, "yes")
		dirty = true
	}

	// Set annBindCompleted if it is not set yet
	if !hasAnnotation(claimClone.ObjectMeta, annBindCompleted) {
		setAnnotation(&claimClone.ObjectMeta, annBindCompleted, "yes")
		dirty = true
	}

	if dirty {
		glog.V(2).Infof("volume %q bound to claim %q", volume.Name, claimToClaimKey(claim))
		newClaim, err := ctrl.kubeClient.Core().PersistentVolumeClaims(claim.Namespace).Update(claimClone)
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

// bind saves binding information both to the volume and the claim and marks
// both objects as Bound. Volume is saved first.
// It returns on first error, it's up to the caller to implement some retry
// mechanism.
func (ctrl *PersistentVolumeController) bind(volume *api.PersistentVolume, claim *api.PersistentVolumeClaim) error {
	var err error

	glog.V(4).Infof("binding volume %q to claim %q", volume.Name, claimToClaimKey(claim))

	if volume, err = ctrl.bindVolumeToClaim(volume, claim); err != nil {
		glog.V(3).Infof("error binding volume %q to claim %q: failed saving the volume: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}

	if volume, err = ctrl.updateVolumePhase(volume, api.VolumeBound); err != nil {
		glog.V(3).Infof("error binding volume %q to claim %q: failed saving the volume status: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}

	if claim, err = ctrl.bindClaimToVolume(claim, volume); err != nil {
		glog.V(3).Infof("error binding volume %q to claim %q: failed saving the claim: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}

	if _, err = ctrl.updateClaimPhase(claim, api.ClaimBound); err != nil {
		glog.V(3).Infof("error binding volume %q to claim %q: failed saving the claim status: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}

	glog.V(4).Infof("volume %q bound to claim %q", volume.Name, claimToClaimKey(claim))
	glog.V(4).Infof("volume %q status after binding: %s", volume.Name, getVolumeStatusForLogging(volume))
	glog.V(4).Infof("claim %q status after binding: %s", claimToClaimKey(claim), getClaimStatusForLogging(claim))
	return nil
}

func hasAnnotation(obj api.ObjectMeta, ann string) bool {
	_, found := obj.Annotations[ann]
	return found
}

func setAnnotation(obj *api.ObjectMeta, ann string, value string) {
	if obj.Annotations == nil {
		obj.Annotations = make(map[string]string)
	}
	obj.Annotations[ann] = value
}

func getClaimStatusForLogging(claim *api.PersistentVolumeClaim) string {
	everBound := hasAnnotation(claim.ObjectMeta, annBindCompleted)
	boundByController := hasAnnotation(claim.ObjectMeta, annBoundByController)

	return fmt.Sprintf("phase: %s, bound to: %q, wasEverBound: %v, boundByController: %v", claim.Status.Phase, claim.Spec.VolumeName, everBound, boundByController)
}

func getVolumeStatusForLogging(volume *api.PersistentVolume) string {
	boundByController := hasAnnotation(volume.ObjectMeta, annBoundByController)
	claimName := ""
	if volume.Spec.ClaimRef != nil {
		claimName = fmt.Sprintf("%s/%s (uid: %s)", volume.Spec.ClaimRef.Namespace, volume.Spec.ClaimRef.Name, volume.Spec.ClaimRef.UID)
	}
	return fmt.Sprintf("phase: %s, bound to: %q, boundByController: %v", volume.Status.Phase, claimName, boundByController)
}

// isVolumeBoundToClaim returns true, if given volume is pre-bound or bound
// to specific claim. Both claim.Name and claim.Namespace must be equal.
// If claim.UID is present in volume.Spec.ClaimRef, it must be equal too.
func isVolumeBoundToClaim(volume *api.PersistentVolume, claim *api.PersistentVolumeClaim) bool {
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
