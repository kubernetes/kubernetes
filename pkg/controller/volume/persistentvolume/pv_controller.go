/*
Copyright 2016 The Kubernetes Authors.

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
	"reflect"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	storage "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	storagelisters "k8s.io/kubernetes/pkg/client/listers/storage/v1beta1"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/util/goroutinemap"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
	vol "k8s.io/kubernetes/pkg/volume"

	"github.com/golang/glog"
)

// ==================================================================
// PLEASE DO NOT ATTEMPT TO SIMPLIFY THIS CODE.
// KEEP THE SPACE SHUTTLE FLYING.
// ==================================================================
//
// This controller is intentionally written in a very verbose style.  You will
// notice:
//
// 1.  Every 'if' statement has a matching 'else' (exception: simple error
//     checks for a client API call)
// 2.  Things that may seem obvious are commented explicitly
//
// We call this style 'space shuttle style'.  Space shuttle style is meant to
// ensure that every branch and condition is considered and accounted for -
// the same way code is written at NASA for applications like the space
// shuttle.
//
// Originally, the work of this controller was split amongst three
// controllers.  This controller is the result a large effort to simplify the
// PV subsystem.  During that effort, it became clear that we needed to ensure
// that every single condition was handled and accounted for in the code, even
// if it resulted in no-op code branches.
//
// As a result, the controller code may seem overly verbose, commented, and
// 'branchy'.  However, a large amount of business knowledge and context is
// recorded here in order to ensure that future maintainers can correctly
// reason through the complexities of the binding behavior.  For that reason,
// changes to this file should preserve and add to the space shuttle style.
//
// ==================================================================
// PLEASE DO NOT ATTEMPT TO SIMPLIFY THIS CODE.
// KEEP THE SPACE SHUTTLE FLYING.
// ==================================================================

// Design:
//
// The fundamental key to this design is the bi-directional "pointer" between
// PersistentVolumes (PVs) and PersistentVolumeClaims (PVCs), which is
// represented here as pvc.Spec.VolumeName and pv.Spec.ClaimRef. The bi-
// directionality is complicated to manage in a transactionless system, but
// without it we can't ensure sane behavior in the face of different forms of
// trouble.  For example, a rogue HA controller instance could end up racing
// and making multiple bindings that are indistinguishable, resulting in
// potential data loss.
//
// This controller is designed to work in active-passive high availability
// mode. It *could* work also in active-active HA mode, all the object
// transitions are designed to cope with this, however performance could be
// lower as these two active controllers will step on each other toes
// frequently.
//
// This controller supports pre-bound (by the creator) objects in both
// directions: a PVC that wants a specific PV or a PV that is reserved for a
// specific PVC.
//
// The binding is two-step process. PV.Spec.ClaimRef is modified first and
// PVC.Spec.VolumeName second. At any point of this transaction, the PV or PVC
// can be modified by user or other controller or completelly deleted. Also,
// two (or more) controllers may try to bind different volumes to different
// claims at the same time. The controller must recover from any conflicts
// that may arise from these conditions.

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

// This annotation is added to a PV that has been dynamically provisioned by
// Kubernetes. Its value is name of volume plugin that created the volume.
// It serves both user (to show where a PV comes from) and Kubernetes (to
// recognize dynamically provisioned PVs in its decisions).
const annDynamicallyProvisioned = "pv.kubernetes.io/provisioned-by"

// This annotation is added to a PVC that is supposed to be dynamically
// provisioned. Its value is name of volume plugin that is supposed to provision
// a volume for this PVC.
const annStorageProvisioner = "volume.beta.kubernetes.io/storage-provisioner"

// Name of a tag attached to a real volume in cloud (e.g. AWS EBS or GCE PD)
// with namespace of a persistent volume claim used to create this volume.
const cloudVolumeCreatedForClaimNamespaceTag = "kubernetes.io/created-for/pvc/namespace"

// Name of a tag attached to a real volume in cloud (e.g. AWS EBS or GCE PD)
// with name of a persistent volume claim used to create this volume.
const cloudVolumeCreatedForClaimNameTag = "kubernetes.io/created-for/pvc/name"

// Name of a tag attached to a real volume in cloud (e.g. AWS EBS or GCE PD)
// with name of appropriate Kubernetes persistent volume .
const cloudVolumeCreatedForVolumeNameTag = "kubernetes.io/created-for/pv/name"

// Number of retries when we create a PV object for a provisioned volume.
const createProvisionedPVRetryCount = 5

// Interval between retries when we create a PV object for a provisioned volume.
const createProvisionedPVInterval = 10 * time.Second

// PersistentVolumeController is a controller that synchronizes
// PersistentVolumeClaims and PersistentVolumes. It starts two
// cache.Controllers that watch PersistentVolume and PersistentVolumeClaim
// changes.
type PersistentVolumeController struct {
	volumeLister       corelisters.PersistentVolumeLister
	volumeListerSynced cache.InformerSynced
	claimLister        corelisters.PersistentVolumeClaimLister
	claimListerSynced  cache.InformerSynced
	classLister        storagelisters.StorageClassLister
	classListerSynced  cache.InformerSynced

	kubeClient                clientset.Interface
	eventRecorder             record.EventRecorder
	cloud                     cloudprovider.Interface
	volumePluginMgr           vol.VolumePluginMgr
	enableDynamicProvisioning bool
	clusterName               string

	// Cache of the last known version of volumes and claims. This cache is
	// thread safe as long as the volumes/claims there are not modified, they
	// must be cloned before any modification. These caches get updated both by
	// "xxx added/updated/deleted" events from etcd and by the controller when
	// it saves newer version to etcd.
	// Why local cache: binding a volume to a claim generates 4 events, roughly
	// in this order (depends on goroutine ordering):
	// - volume.Spec update
	// - volume.Status update
	// - claim.Spec update
	// - claim.Status update
	// With these caches, the controller can check that it has already saved
	// volume.Status and claim.Spec+Status and does not need to do anything
	// when e.g. volume.Spec update event arrives before all the other events.
	// Without this cache, it would see the old version of volume.Status and
	// claim in the informers (it has not been updated from API server events
	// yet) and it would try to fix these objects to be bound together.
	// Any write to API server would fail with version conflict - these objects
	// have been already written.
	volumes persistentVolumeOrderedIndex
	claims  cache.Store

	// Work queues of claims and volumes to process. Every queue should have
	// exactly one worker thread, especially syncClaim() is not reentrant.
	// Two syncClaims could bind two different claims to the same volume or one
	// claim to two volumes. The controller would recover from this (due to
	// version errors in API server and other checks in this controller),
	// however overall speed of multi-worker controller would be lower than if
	// it runs single thread only.
	claimQueue  *workqueue.Type
	volumeQueue *workqueue.Type

	// Map of scheduled/running operations.
	runningOperations goroutinemap.GoRoutineMap

	// For testing only: hook to call before an asynchronous operation starts.
	// Not used when set to nil.
	preOperationHook func(operationName string)

	createProvisionedPVRetryCount int
	createProvisionedPVInterval   time.Duration

	// Provisioner for annAlphaClass.
	// TODO: remove in 1.5
	alphaProvisioner vol.ProvisionableVolumePlugin
}

// syncClaim is the main controller method to decide what to do with a claim.
// It's invoked by appropriate cache.Controller callbacks when a claim is
// created, updated or periodically synced. We do not differentiate between
// these events.
// For easier readability, it was split into syncUnboundClaim and syncBoundClaim
// methods.
func (ctrl *PersistentVolumeController) syncClaim(claim *v1.PersistentVolumeClaim) error {
	glog.V(4).Infof("synchronizing PersistentVolumeClaim[%s]: %s", claimToClaimKey(claim), getClaimStatusForLogging(claim))

	if !metav1.HasAnnotation(claim.ObjectMeta, annBindCompleted) {
		return ctrl.syncUnboundClaim(claim)
	} else {
		return ctrl.syncBoundClaim(claim)
	}
}

// syncUnboundClaim is the main controller method to decide what to do with an
// unbound claim.
func (ctrl *PersistentVolumeController) syncUnboundClaim(claim *v1.PersistentVolumeClaim) error {
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
			if v1.GetPersistentVolumeClaimClass(claim) != "" || metav1.HasAnnotation(claim.ObjectMeta, v1.AlphaStorageClassAnnotation) {
				if err = ctrl.provisionClaim(claim); err != nil {
					return err
				}
				return nil
			}
			// Mark the claim as Pending and try to find a match in the next
			// periodic syncClaim
			ctrl.eventRecorder.Event(claim, v1.EventTypeNormal, "FailedBinding", "no persistent volumes available for this claim and no storage class is set")
			if _, err = ctrl.updateClaimStatus(claim, v1.ClaimPending, nil); err != nil {
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
			if _, err = ctrl.updateClaimStatus(claim, v1.ClaimPending, nil); err != nil {
				return err
			}
			return nil
		} else {
			volume, ok := obj.(*v1.PersistentVolume)
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
				if !metav1.HasAnnotation(claim.ObjectMeta, annBoundByController) {
					glog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume already bound to different claim by user, will retry later", claimToClaimKey(claim))
					// User asked for a specific PV, retry later
					if _, err = ctrl.updateClaimStatus(claim, v1.ClaimPending, nil); err != nil {
						return err
					}
					return nil
				} else {
					// This should never happen because someone had to remove
					// annBindCompleted annotation on the claim.
					glog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume already bound to different claim %q by controller, THIS SHOULD NEVER HAPPEN", claimToClaimKey(claim), claimrefToClaimKey(volume.Spec.ClaimRef))
					return fmt.Errorf("Invalid binding of claim %q to volume %q: volume already claimed by %q", claimToClaimKey(claim), claim.Spec.VolumeName, claimrefToClaimKey(volume.Spec.ClaimRef))
				}
			}
		}
	}
}

// syncBoundClaim is the main controller method to decide what to do with a
// bound claim.
func (ctrl *PersistentVolumeController) syncBoundClaim(claim *v1.PersistentVolumeClaim) error {
	// HasAnnotation(pvc, annBindCompleted)
	// This PVC has previously been bound
	// OBSERVATION: pvc is not "Pending"
	// [Unit test set 3]
	if claim.Spec.VolumeName == "" {
		// Claim was bound before but not any more.
		if _, err := ctrl.updateClaimStatusWithEvent(claim, v1.ClaimLost, nil, v1.EventTypeWarning, "ClaimLost", "Bound claim has lost reference to PersistentVolume. Data on the volume is lost!"); err != nil {
			return err
		}
		return nil
	}
	obj, found, err := ctrl.volumes.store.GetByKey(claim.Spec.VolumeName)
	if err != nil {
		return err
	}
	if !found {
		// Claim is bound to a non-existing volume.
		if _, err = ctrl.updateClaimStatusWithEvent(claim, v1.ClaimLost, nil, v1.EventTypeWarning, "ClaimLost", "Bound claim has lost its PersistentVolume. Data on the volume is lost!"); err != nil {
			return err
		}
		return nil
	} else {
		volume, ok := obj.(*v1.PersistentVolume)
		if !ok {
			return fmt.Errorf("Cannot convert object from volume cache to volume %q!?: %#v", claim.Spec.VolumeName, obj)
		}

		glog.V(4).Infof("synchronizing bound PersistentVolumeClaim[%s]: volume %q found: %s", claimToClaimKey(claim), claim.Spec.VolumeName, getVolumeStatusForLogging(volume))
		if volume.Spec.ClaimRef == nil {
			// Claim is bound but volume has come unbound.
			// Or, a claim was bound and the controller has not received updated
			// volume yet. We can't distinguish these cases.
			// Bind the volume again and set all states to Bound.
			glog.V(4).Infof("synchronizing bound PersistentVolumeClaim[%s]: volume is unbound, fixing", claimToClaimKey(claim))
			if err = ctrl.bind(volume, claim); err != nil {
				// Objects not saved, next syncPV or syncClaim will try again
				return err
			}
			return nil
		} else if volume.Spec.ClaimRef.UID == claim.UID {
			// All is well
			// NOTE: syncPV can handle this so it can be left out.
			// NOTE: bind() call here will do nothing in most cases as
			// everything should be already set.
			glog.V(4).Infof("synchronizing bound PersistentVolumeClaim[%s]: claim is already correctly bound", claimToClaimKey(claim))
			if err = ctrl.bind(volume, claim); err != nil {
				// Objects not saved, next syncPV or syncClaim will try again
				return err
			}
			return nil
		} else {
			// Claim is bound but volume has a different claimant.
			// Set the claim phase to 'Lost', which is a terminal
			// phase.
			if _, err = ctrl.updateClaimStatusWithEvent(claim, v1.ClaimLost, nil, v1.EventTypeWarning, "ClaimMisbound", "Two claims are bound to the same volume, this one is bound incorrectly"); err != nil {
				return err
			}
			return nil
		}
	}
}

// syncVolume is the main controller method to decide what to do with a volume.
// It's invoked by appropriate cache.Controller callbacks when a volume is
// created, updated or periodically synced. We do not differentiate between
// these events.
func (ctrl *PersistentVolumeController) syncVolume(volume *v1.PersistentVolume) error {
	glog.V(4).Infof("synchronizing PersistentVolume[%s]: %s", volume.Name, getVolumeStatusForLogging(volume))

	// [Unit test set 4]
	if volume.Spec.ClaimRef == nil {
		// Volume is unused
		glog.V(4).Infof("synchronizing PersistentVolume[%s]: volume is unused", volume.Name)
		if _, err := ctrl.updateVolumePhase(volume, v1.VolumeAvailable, ""); err != nil {
			// Nothing was saved; we will fall back into the same
			// condition in the next call to this method
			return err
		}
		return nil
	} else /* pv.Spec.ClaimRef != nil */ {
		// Volume is bound to a claim.
		if volume.Spec.ClaimRef.UID == "" {
			// The PV is reserved for a PVC; that PVC has not yet been
			// bound to this PV; the PVC sync will handle it.
			glog.V(4).Infof("synchronizing PersistentVolume[%s]: volume is pre-bound to claim %s", volume.Name, claimrefToClaimKey(volume.Spec.ClaimRef))
			if _, err := ctrl.updateVolumePhase(volume, v1.VolumeAvailable, ""); err != nil {
				// Nothing was saved; we will fall back into the same
				// condition in the next call to this method
				return err
			}
			return nil
		}
		glog.V(4).Infof("synchronizing PersistentVolume[%s]: volume is bound to claim %s", volume.Name, claimrefToClaimKey(volume.Spec.ClaimRef))
		// Get the PVC by _name_
		var claim *v1.PersistentVolumeClaim
		claimName := claimrefToClaimKey(volume.Spec.ClaimRef)
		obj, found, err := ctrl.claims.GetByKey(claimName)
		if err != nil {
			return err
		}
		if !found {
			glog.V(4).Infof("synchronizing PersistentVolume[%s]: claim %s not found", volume.Name, claimrefToClaimKey(volume.Spec.ClaimRef))
			// Fall through with claim = nil
		} else {
			var ok bool
			claim, ok = obj.(*v1.PersistentVolumeClaim)
			if !ok {
				return fmt.Errorf("Cannot convert object from volume cache to volume %q!?: %#v", claim.Spec.VolumeName, obj)
			}
			glog.V(4).Infof("synchronizing PersistentVolume[%s]: claim %s found: %s", volume.Name, claimrefToClaimKey(volume.Spec.ClaimRef), getClaimStatusForLogging(claim))
		}
		if claim != nil && claim.UID != volume.Spec.ClaimRef.UID {
			// The claim that the PV was pointing to was deleted, and another
			// with the same name created.
			glog.V(4).Infof("synchronizing PersistentVolume[%s]: claim %s has different UID, the old one must have been deleted", volume.Name, claimrefToClaimKey(volume.Spec.ClaimRef))
			// Treat the volume as bound to a missing claim.
			claim = nil
		}

		if claim == nil {
			// If we get into this block, the claim must have been deleted;
			// NOTE: reclaimVolume may either release the PV back into the pool or
			// recycle it or do nothing (retain)

			// Do not overwrite previous Failed state - let the user see that
			// something went wrong, while we still re-try to reclaim the
			// volume.
			if volume.Status.Phase != v1.VolumeReleased && volume.Status.Phase != v1.VolumeFailed {
				// Also, log this only once:
				glog.V(2).Infof("volume %q is released and reclaim policy %q will be executed", volume.Name, volume.Spec.PersistentVolumeReclaimPolicy)
				if volume, err = ctrl.updateVolumePhase(volume, v1.VolumeReleased, ""); err != nil {
					// Nothing was saved; we will fall back into the same condition
					// in the next call to this method
					return err
				}
			}

			if err = ctrl.reclaimVolume(volume); err != nil {
				// Release failed, we will fall back into the same condition
				// in the next call to this method
				return err
			}
			return nil
		} else if claim.Spec.VolumeName == "" {
			if metav1.HasAnnotation(volume.ObjectMeta, annBoundByController) {
				// The binding is not completed; let PVC sync handle it
				glog.V(4).Infof("synchronizing PersistentVolume[%s]: volume not bound yet, waiting for syncClaim to fix it", volume.Name)
			} else {
				// Dangling PV; try to re-establish the link in the PVC sync
				glog.V(4).Infof("synchronizing PersistentVolume[%s]: volume was bound and got unbound (by user?), waiting for syncClaim to fix it", volume.Name)
			}
			// In both cases, the volume is Bound and the claim is Pending.
			// Next syncClaim will fix it. To speed it up, we enqueue the claim
			// into the controller, which results in syncClaim to be called
			// shortly (and in the right worker goroutine).
			// This speeds up binding of provisioned volumes - provisioner saves
			// only the new PV and it expects that next syncClaim will bind the
			// claim to it.
			ctrl.claimQueue.Add(claimToClaimKey(claim))
			return nil
		} else if claim.Spec.VolumeName == volume.Name {
			// Volume is bound to a claim properly, update status if necessary
			glog.V(4).Infof("synchronizing PersistentVolume[%s]: all is bound", volume.Name)
			if _, err = ctrl.updateVolumePhase(volume, v1.VolumeBound, ""); err != nil {
				// Nothing was saved; we will fall back into the same
				// condition in the next call to this method
				return err
			}
			return nil
		} else {
			// Volume is bound to a claim, but the claim is bound elsewhere
			if metav1.HasAnnotation(volume.ObjectMeta, annDynamicallyProvisioned) && volume.Spec.PersistentVolumeReclaimPolicy == v1.PersistentVolumeReclaimDelete {
				// This volume was dynamically provisioned for this claim. The
				// claim got bound elsewhere, and thus this volume is not
				// needed. Delete it.
				// Mark the volume as Released for external deleters and to let
				// the user know. Don't overwrite existing Failed status!
				if volume.Status.Phase != v1.VolumeReleased && volume.Status.Phase != v1.VolumeFailed {
					// Also, log this only once:
					glog.V(2).Infof("dynamically volume %q is released and it will be deleted", volume.Name)
					if volume, err = ctrl.updateVolumePhase(volume, v1.VolumeReleased, ""); err != nil {
						// Nothing was saved; we will fall back into the same condition
						// in the next call to this method
						return err
					}
				}
				if err = ctrl.reclaimVolume(volume); err != nil {
					// Deletion failed, we will fall back into the same condition
					// in the next call to this method
					return err
				}
				return nil
			} else {
				// Volume is bound to a claim, but the claim is bound elsewhere
				// and it's not dynamically provisioned.
				if metav1.HasAnnotation(volume.ObjectMeta, annBoundByController) {
					// This is part of the normal operation of the controller; the
					// controller tried to use this volume for a claim but the claim
					// was fulfilled by another volume. We did this; fix it.
					glog.V(4).Infof("synchronizing PersistentVolume[%s]: volume is bound by controller to a claim that is bound to another volume, unbinding", volume.Name)
					if err = ctrl.unbindVolume(volume); err != nil {
						return err
					}
					return nil
				} else {
					// The PV must have been created with this ptr; leave it alone.
					glog.V(4).Infof("synchronizing PersistentVolume[%s]: volume is bound by user to a claim that is bound to another volume, waiting for the claim to get unbound", volume.Name)
					// This just updates the volume phase and clears
					// volume.Spec.ClaimRef.UID. It leaves the volume pre-bound
					// to the claim.
					if err = ctrl.unbindVolume(volume); err != nil {
						return err
					}
					return nil
				}
			}
		}
	}
}

// updateClaimStatus saves new claim.Status to API server.
// Parameters:
//  claim - claim to update
//  phasephase - phase to set
//  volume - volume which Capacity is set into claim.Status.Capacity
func (ctrl *PersistentVolumeController) updateClaimStatus(claim *v1.PersistentVolumeClaim, phase v1.PersistentVolumeClaimPhase, volume *v1.PersistentVolume) (*v1.PersistentVolumeClaim, error) {
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

	newClaim, err := ctrl.kubeClient.Core().PersistentVolumeClaims(claimClone.Namespace).UpdateStatus(claimClone)
	if err != nil {
		glog.V(4).Infof("updating PersistentVolumeClaim[%s] status: set phase %s failed: %v", claimToClaimKey(claim), phase, err)
		return newClaim, err
	}
	_, err = ctrl.storeClaimUpdate(newClaim)
	if err != nil {
		glog.V(4).Infof("updating PersistentVolumeClaim[%s] status: cannot update internal cache: %v", claimToClaimKey(claim), err)
		return newClaim, err
	}
	glog.V(2).Infof("claim %q entered phase %q", claimToClaimKey(claim), phase)
	return newClaim, nil
}

// updateClaimStatusWithEvent saves new claim.Status to API server and emits
// given event on the claim. It saves the status and emits the event only when
// the status has actually changed from the version saved in API server.
// Parameters:
//   claim - claim to update
//   phasephase - phase to set
//   volume - volume which Capacity is set into claim.Status.Capacity
//   eventtype, reason, message - event to send, see EventRecorder.Event()
func (ctrl *PersistentVolumeController) updateClaimStatusWithEvent(claim *v1.PersistentVolumeClaim, phase v1.PersistentVolumeClaimPhase, volume *v1.PersistentVolume, eventtype, reason, message string) (*v1.PersistentVolumeClaim, error) {
	glog.V(4).Infof("updating updateClaimStatusWithEvent[%s]: set phase %s", claimToClaimKey(claim), phase)
	if claim.Status.Phase == phase {
		// Nothing to do.
		glog.V(4).Infof("updating updateClaimStatusWithEvent[%s]: phase %s already set", claimToClaimKey(claim), phase)
		return claim, nil
	}

	newClaim, err := ctrl.updateClaimStatus(claim, phase, volume)
	if err != nil {
		return nil, err
	}

	// Emit the event only when the status change happens, not every time
	// syncClaim is called.
	glog.V(3).Infof("claim %q changed status to %q: %s", claimToClaimKey(claim), phase, message)
	ctrl.eventRecorder.Event(newClaim, eventtype, reason, message)

	return newClaim, nil
}

// updateVolumePhase saves new volume phase to API server.
func (ctrl *PersistentVolumeController) updateVolumePhase(volume *v1.PersistentVolume, phase v1.PersistentVolumePhase, message string) (*v1.PersistentVolume, error) {
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

	newVol, err := ctrl.kubeClient.Core().PersistentVolumes().UpdateStatus(volumeClone)
	if err != nil {
		glog.V(4).Infof("updating PersistentVolume[%s]: set phase %s failed: %v", volume.Name, phase, err)
		return newVol, err
	}
	_, err = ctrl.storeVolumeUpdate(newVol)
	if err != nil {
		glog.V(4).Infof("updating PersistentVolume[%s]: cannot update internal cache: %v", volume.Name, err)
		return newVol, err
	}
	glog.V(2).Infof("volume %q entered phase %q", volume.Name, phase)
	return newVol, err
}

// updateVolumePhaseWithEvent saves new volume phase to API server and emits
// given event on the volume. It saves the phase and emits the event only when
// the phase has actually changed from the version saved in API server.
func (ctrl *PersistentVolumeController) updateVolumePhaseWithEvent(volume *v1.PersistentVolume, phase v1.PersistentVolumePhase, eventtype, reason, message string) (*v1.PersistentVolume, error) {
	glog.V(4).Infof("updating updateVolumePhaseWithEvent[%s]: set phase %s", volume.Name, phase)
	if volume.Status.Phase == phase {
		// Nothing to do.
		glog.V(4).Infof("updating updateVolumePhaseWithEvent[%s]: phase %s already set", volume.Name, phase)
		return volume, nil
	}

	newVol, err := ctrl.updateVolumePhase(volume, phase, message)
	if err != nil {
		return nil, err
	}

	// Emit the event only when the status change happens, not every time
	// syncClaim is called.
	glog.V(3).Infof("volume %q changed status to %q: %s", volume.Name, phase, message)
	ctrl.eventRecorder.Event(newVol, eventtype, reason, message)

	return newVol, nil
}

// bindVolumeToClaim modifes given volume to be bound to a claim and saves it to
// API server. The claim is not modified in this method!
func (ctrl *PersistentVolumeController) bindVolumeToClaim(volume *v1.PersistentVolume, claim *v1.PersistentVolumeClaim) (*v1.PersistentVolume, error) {
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
		newVol, err := ctrl.kubeClient.Core().PersistentVolumes().Update(volumeClone)
		if err != nil {
			glog.V(4).Infof("updating PersistentVolume[%s]: binding to %q failed: %v", volume.Name, claimToClaimKey(claim), err)
			return newVol, err
		}
		_, err = ctrl.storeVolumeUpdate(newVol)
		if err != nil {
			glog.V(4).Infof("updating PersistentVolume[%s]: cannot update internal cache: %v", volume.Name, err)
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
func (ctrl *PersistentVolumeController) bindClaimToVolume(claim *v1.PersistentVolumeClaim, volume *v1.PersistentVolume) (*v1.PersistentVolumeClaim, error) {
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
		newClaim, err := ctrl.kubeClient.Core().PersistentVolumeClaims(claim.Namespace).Update(claimClone)
		if err != nil {
			glog.V(4).Infof("updating PersistentVolumeClaim[%s]: binding to %q failed: %v", claimToClaimKey(claim), volume.Name, err)
			return newClaim, err
		}
		_, err = ctrl.storeClaimUpdate(newClaim)
		if err != nil {
			glog.V(4).Infof("updating PersistentVolumeClaim[%s]: cannot update internal cache: %v", claimToClaimKey(claim), err)
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
func (ctrl *PersistentVolumeController) bind(volume *v1.PersistentVolume, claim *v1.PersistentVolumeClaim) error {
	var err error
	// use updateClaim/updatedVolume to keep the original claim/volume for
	// logging in error cases.
	var updatedClaim *v1.PersistentVolumeClaim
	var updatedVolume *v1.PersistentVolume

	glog.V(4).Infof("binding volume %q to claim %q", volume.Name, claimToClaimKey(claim))

	if updatedVolume, err = ctrl.bindVolumeToClaim(volume, claim); err != nil {
		glog.V(3).Infof("error binding volume %q to claim %q: failed saving the volume: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}
	volume = updatedVolume

	if updatedVolume, err = ctrl.updateVolumePhase(volume, v1.VolumeBound, ""); err != nil {
		glog.V(3).Infof("error binding volume %q to claim %q: failed saving the volume status: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}
	volume = updatedVolume

	if updatedClaim, err = ctrl.bindClaimToVolume(claim, volume); err != nil {
		glog.V(3).Infof("error binding volume %q to claim %q: failed saving the claim: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}
	claim = updatedClaim

	if updatedClaim, err = ctrl.updateClaimStatus(claim, v1.ClaimBound, volume); err != nil {
		glog.V(3).Infof("error binding volume %q to claim %q: failed saving the claim status: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}
	claim = updatedClaim

	glog.V(4).Infof("volume %q bound to claim %q", volume.Name, claimToClaimKey(claim))
	glog.V(4).Infof("volume %q status after binding: %s", volume.Name, getVolumeStatusForLogging(volume))
	glog.V(4).Infof("claim %q status after binding: %s", claimToClaimKey(claim), getClaimStatusForLogging(claim))
	return nil
}

// unbindVolume rolls back previous binding of the volume. This may be necessary
// when two controllers bound two volumes to single claim - when we detect this,
// only one binding succeeds and the second one must be rolled back.
// This method updates both Spec and Status.
// It returns on first error, it's up to the caller to implement some retry
// mechanism.
func (ctrl *PersistentVolumeController) unbindVolume(volume *v1.PersistentVolume) error {
	glog.V(4).Infof("updating PersistentVolume[%s]: rolling back binding from %q", volume.Name, claimrefToClaimKey(volume.Spec.ClaimRef))

	// Save the PV only when any modification is neccessary.
	clone, err := api.Scheme.DeepCopy(volume)
	if err != nil {
		return fmt.Errorf("Error cloning pv: %v", err)
	}
	volumeClone, ok := clone.(*v1.PersistentVolume)
	if !ok {
		return fmt.Errorf("Unexpected volume cast error : %v", volumeClone)
	}

	if metav1.HasAnnotation(volume.ObjectMeta, annBoundByController) {
		// The volume was bound by the controller.
		volumeClone.Spec.ClaimRef = nil
		delete(volumeClone.Annotations, annBoundByController)
		if len(volumeClone.Annotations) == 0 {
			// No annotations look better than empty annotation map (and it's easier
			// to test).
			volumeClone.Annotations = nil
		}
	} else {
		// The volume was pre-bound by user. Clear only the binging UID.
		volumeClone.Spec.ClaimRef.UID = ""
	}

	newVol, err := ctrl.kubeClient.Core().PersistentVolumes().Update(volumeClone)
	if err != nil {
		glog.V(4).Infof("updating PersistentVolume[%s]: rollback failed: %v", volume.Name, err)
		return err
	}
	_, err = ctrl.storeVolumeUpdate(newVol)
	if err != nil {
		glog.V(4).Infof("updating PersistentVolume[%s]: cannot update internal cache: %v", volume.Name, err)
		return err
	}
	glog.V(4).Infof("updating PersistentVolume[%s]: rolled back", newVol.Name)

	// Update the status
	_, err = ctrl.updateVolumePhase(newVol, v1.VolumeAvailable, "")
	return err
}

// reclaimVolume implements volume.Spec.PersistentVolumeReclaimPolicy and
// starts appropriate reclaim action.
func (ctrl *PersistentVolumeController) reclaimVolume(volume *v1.PersistentVolume) error {
	switch volume.Spec.PersistentVolumeReclaimPolicy {
	case v1.PersistentVolumeReclaimRetain:
		glog.V(4).Infof("reclaimVolume[%s]: policy is Retain, nothing to do", volume.Name)

	case v1.PersistentVolumeReclaimRecycle:
		glog.V(4).Infof("reclaimVolume[%s]: policy is Recycle", volume.Name)
		opName := fmt.Sprintf("recycle-%s[%s]", volume.Name, string(volume.UID))
		ctrl.scheduleOperation(opName, func() error {
			ctrl.recycleVolumeOperation(volume)
			return nil
		})

	case v1.PersistentVolumeReclaimDelete:
		glog.V(4).Infof("reclaimVolume[%s]: policy is Delete", volume.Name)
		opName := fmt.Sprintf("delete-%s[%s]", volume.Name, string(volume.UID))
		ctrl.scheduleOperation(opName, func() error {
			return ctrl.deleteVolumeOperation(volume)
		})

	default:
		// Unknown PersistentVolumeReclaimPolicy
		if _, err := ctrl.updateVolumePhaseWithEvent(volume, v1.VolumeFailed, v1.EventTypeWarning, "VolumeUnknownReclaimPolicy", "Volume has unrecognized PersistentVolumeReclaimPolicy"); err != nil {
			return err
		}
	}
	return nil
}

// doRerecycleVolumeOperationcycleVolume recycles a volume. This method is
// running in standalone goroutine and already has all necessary locks.
func (ctrl *PersistentVolumeController) recycleVolumeOperation(arg interface{}) {
	volume, ok := arg.(*v1.PersistentVolume)
	if !ok {
		glog.Errorf("Cannot convert recycleVolumeOperation argument to volume, got %#v", arg)
		return
	}
	glog.V(4).Infof("recycleVolumeOperation [%s] started", volume.Name)

	// This method may have been waiting for a volume lock for some time.
	// Previous recycleVolumeOperation might just have saved an updated version,
	// so read current volume state now.
	newVolume, err := ctrl.kubeClient.Core().PersistentVolumes().Get(volume.Name, metav1.GetOptions{})
	if err != nil {
		glog.V(3).Infof("error reading peristent volume %q: %v", volume.Name, err)
		return
	}
	needsReclaim, err := ctrl.isVolumeReleased(newVolume)
	if err != nil {
		glog.V(3).Infof("error reading claim for volume %q: %v", volume.Name, err)
		return
	}
	if !needsReclaim {
		glog.V(3).Infof("volume %q no longer needs recycling, skipping", volume.Name)
		return
	}

	// Use the newest volume copy, this will save us from version conflicts on
	// saving.
	volume = newVolume

	// Find a plugin.
	spec := vol.NewSpecFromPersistentVolume(volume, false)
	plugin, err := ctrl.volumePluginMgr.FindRecyclablePluginBySpec(spec)
	if err != nil {
		// No recycler found. Emit an event and mark the volume Failed.
		if _, err = ctrl.updateVolumePhaseWithEvent(volume, v1.VolumeFailed, v1.EventTypeWarning, "VolumeFailedRecycle", "No recycler plugin found for the volume!"); err != nil {
			glog.V(4).Infof("recycleVolumeOperation [%s]: failed to mark volume as failed: %v", volume.Name, err)
			// Save failed, retry on the next deletion attempt
			return
		}
		// Despite the volume being Failed, the controller will retry recycling
		// the volume in every syncVolume() call.
		return
	}

	// Plugin found
	recorder := ctrl.newRecyclerEventRecorder(volume)

	if err = plugin.Recycle(volume.Name, spec, recorder); err != nil {
		// Recycler failed
		strerr := fmt.Sprintf("Recycle failed: %s", err)
		if _, err = ctrl.updateVolumePhaseWithEvent(volume, v1.VolumeFailed, v1.EventTypeWarning, "VolumeFailedRecycle", strerr); err != nil {
			glog.V(4).Infof("recycleVolumeOperation [%s]: failed to mark volume as failed: %v", volume.Name, err)
			// Save failed, retry on the next deletion attempt
			return
		}
		// Despite the volume being Failed, the controller will retry recycling
		// the volume in every syncVolume() call.
		return
	}

	glog.V(2).Infof("volume %q recycled", volume.Name)
	// Send an event
	ctrl.eventRecorder.Event(volume, v1.EventTypeNormal, "VolumeRecycled", "Volume recycled")
	// Make the volume available again
	if err = ctrl.unbindVolume(volume); err != nil {
		// Oops, could not save the volume and therefore the controller will
		// recycle the volume again on next update. We _could_ maintain a cache
		// of "recently recycled volumes" and avoid unnecessary recycling, this
		// is left out as future optimization.
		glog.V(3).Infof("recycleVolumeOperation [%s]: failed to make recycled volume 'Available' (%v), we will recycle the volume again", volume.Name, err)
		return
	}
	return
}

// deleteVolumeOperation deletes a volume. This method is running in standalone
// goroutine and already has all necessary locks.
func (ctrl *PersistentVolumeController) deleteVolumeOperation(arg interface{}) error {
	volume, ok := arg.(*v1.PersistentVolume)
	if !ok {
		glog.Errorf("Cannot convert deleteVolumeOperation argument to volume, got %#v", arg)
		return nil
	}

	glog.V(4).Infof("deleteVolumeOperation [%s] started", volume.Name)

	// This method may have been waiting for a volume lock for some time.
	// Previous deleteVolumeOperation might just have saved an updated version, so
	// read current volume state now.
	newVolume, err := ctrl.kubeClient.Core().PersistentVolumes().Get(volume.Name, metav1.GetOptions{})
	if err != nil {
		glog.V(3).Infof("error reading peristent volume %q: %v", volume.Name, err)
		return nil
	}
	needsReclaim, err := ctrl.isVolumeReleased(newVolume)
	if err != nil {
		glog.V(3).Infof("error reading claim for volume %q: %v", volume.Name, err)
		return nil
	}
	if !needsReclaim {
		glog.V(3).Infof("volume %q no longer needs deletion, skipping", volume.Name)
		return nil
	}

	deleted, err := ctrl.doDeleteVolume(volume)
	if err != nil {
		// Delete failed, update the volume and emit an event.
		glog.V(3).Infof("deletion of volume %q failed: %v", volume.Name, err)
		if vol.IsDeletedVolumeInUse(err) {
			// The plugin needs more time, don't mark the volume as Failed
			// and send Normal event only
			ctrl.eventRecorder.Event(volume, v1.EventTypeNormal, "VolumeDelete", err.Error())
		} else {
			// The plugin failed, mark the volume as Failed and send Warning
			// event
			if _, err := ctrl.updateVolumePhaseWithEvent(volume, v1.VolumeFailed, v1.EventTypeWarning, "VolumeFailedDelete", err.Error()); err != nil {
				glog.V(4).Infof("deleteVolumeOperation [%s]: failed to mark volume as failed: %v", volume.Name, err)
				// Save failed, retry on the next deletion attempt
				return err
			}
		}

		// Despite the volume being Failed, the controller will retry deleting
		// the volume in every syncVolume() call.
		return err
	}
	if !deleted {
		// The volume waits for deletion by an external plugin. Do nothing.
		return nil
	}

	glog.V(4).Infof("deleteVolumeOperation [%s]: success", volume.Name)
	// Delete the volume
	if err = ctrl.kubeClient.Core().PersistentVolumes().Delete(volume.Name, nil); err != nil {
		// Oops, could not delete the volume and therefore the controller will
		// try to delete the volume again on next update. We _could_ maintain a
		// cache of "recently deleted volumes" and avoid unnecessary deletion,
		// this is left out as future optimization.
		glog.V(3).Infof("failed to delete volume %q from database: %v", volume.Name, err)
		return nil
	}
	return nil
}

// isVolumeReleased returns true if given volume is released and can be recycled
// or deleted, based on its retain policy. I.e. the volume is bound to a claim
// and the claim does not exist or exists and is bound to different volume.
func (ctrl *PersistentVolumeController) isVolumeReleased(volume *v1.PersistentVolume) (bool, error) {
	// A volume needs reclaim if it has ClaimRef and appropriate claim does not
	// exist.
	if volume.Spec.ClaimRef == nil {
		glog.V(4).Infof("isVolumeReleased[%s]: ClaimRef is nil", volume.Name)
		return false, nil
	}
	if volume.Spec.ClaimRef.UID == "" {
		// This is a volume bound by user and the controller has not finished
		// binding to the real claim yet.
		glog.V(4).Infof("isVolumeReleased[%s]: ClaimRef is not bound", volume.Name)
		return false, nil
	}

	var claim *v1.PersistentVolumeClaim
	claimName := claimrefToClaimKey(volume.Spec.ClaimRef)
	obj, found, err := ctrl.claims.GetByKey(claimName)
	if err != nil {
		return false, err
	}
	if !found {
		// Fall through with claim = nil
	} else {
		var ok bool
		claim, ok = obj.(*v1.PersistentVolumeClaim)
		if !ok {
			return false, fmt.Errorf("Cannot convert object from claim cache to claim!?: %#v", obj)
		}
	}
	if claim != nil && claim.UID == volume.Spec.ClaimRef.UID {
		// the claim still exists and has the right UID

		if len(claim.Spec.VolumeName) > 0 && claim.Spec.VolumeName != volume.Name {
			// the claim is bound to another PV, this PV *is* released
			return true, nil
		}

		glog.V(4).Infof("isVolumeReleased[%s]: ClaimRef is still valid, volume is not released", volume.Name)
		return false, nil
	}

	glog.V(2).Infof("isVolumeReleased[%s]: volume is released", volume.Name)
	return true, nil
}

// doDeleteVolume finds appropriate delete plugin and deletes given volume. It
// returns 'true', when the volume was deleted and 'false' when the volume
// cannot be deleted because of the deleter is external. No error should be
// reported in this case.
func (ctrl *PersistentVolumeController) doDeleteVolume(volume *v1.PersistentVolume) (bool, error) {
	glog.V(4).Infof("doDeleteVolume [%s]", volume.Name)
	var err error

	plugin, err := ctrl.findDeletablePlugin(volume)
	if err != nil {
		return false, err
	}
	if plugin == nil {
		// External deleter is requested, do nothing
		glog.V(3).Infof("external deleter for volume %q requested, ignoring", volume.Name)
		return false, nil
	}

	// Plugin found
	glog.V(5).Infof("found a deleter plugin %q for volume %q", plugin.GetPluginName(), volume.Name)
	spec := vol.NewSpecFromPersistentVolume(volume, false)
	deleter, err := plugin.NewDeleter(spec)
	if err != nil {
		// Cannot create deleter
		return false, fmt.Errorf("Failed to create deleter for volume %q: %v", volume.Name, err)
	}

	if err = deleter.Delete(); err != nil {
		// Deleter failed
		return false, err
	}

	glog.V(2).Infof("volume %q deleted", volume.Name)
	return true, nil
}

// provisionClaim starts new asynchronous operation to provision a claim if
// provisioning is enabled.
func (ctrl *PersistentVolumeController) provisionClaim(claim *v1.PersistentVolumeClaim) error {
	if !ctrl.enableDynamicProvisioning {
		return nil
	}
	glog.V(4).Infof("provisionClaim[%s]: started", claimToClaimKey(claim))
	opName := fmt.Sprintf("provision-%s[%s]", claimToClaimKey(claim), string(claim.UID))
	ctrl.scheduleOperation(opName, func() error {
		ctrl.provisionClaimOperation(claim)
		return nil
	})
	return nil
}

// provisionClaimOperation provisions a volume. This method is running in
// standalone goroutine and already has all necessary locks.
func (ctrl *PersistentVolumeController) provisionClaimOperation(claimObj interface{}) {
	claim, ok := claimObj.(*v1.PersistentVolumeClaim)
	if !ok {
		glog.Errorf("Cannot convert provisionClaimOperation argument to claim, got %#v", claimObj)
		return
	}

	claimClass := v1.GetPersistentVolumeClaimClass(claim)
	glog.V(4).Infof("provisionClaimOperation [%s] started, class: %q", claimToClaimKey(claim), claimClass)

	plugin, storageClass, err := ctrl.findProvisionablePlugin(claim)
	if err != nil {
		ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, "ProvisioningFailed", err.Error())
		glog.V(2).Infof("error finding provisioning plugin for claim %s: %v", claimToClaimKey(claim), err)
		// The controller will retry provisioning the volume in every
		// syncVolume() call.
		return
	}

	if storageClass != nil {
		// Add provisioner annotation so external provisioners know when to start
		newClaim, err := ctrl.setClaimProvisioner(claim, storageClass)
		if err != nil {
			// Save failed, the controller will retry in the next sync
			glog.V(2).Infof("error saving claim %s: %v", claimToClaimKey(claim), err)
			return
		}
		claim = newClaim
	}

	if plugin == nil {
		// findProvisionablePlugin returned no error nor plugin.
		// This means that an unknown provisioner is requested. Report an event
		// and wait for the external provisioner
		if storageClass != nil {
			msg := fmt.Sprintf("cannot find provisioner %q, expecting that a volume for the claim is provisioned either manually or via external software", storageClass.Provisioner)
			ctrl.eventRecorder.Event(claim, v1.EventTypeNormal, "ExternalProvisioning", msg)
			glog.V(3).Infof("provisioning claim %q: %s", claimToClaimKey(claim), msg)
		} else {
			glog.V(3).Infof("cannot find storage class for claim %q", claimToClaimKey(claim))
		}
		return
	}

	// internal provisioning

	//  A previous doProvisionClaim may just have finished while we were waiting for
	//  the locks. Check that PV (with deterministic name) hasn't been provisioned
	//  yet.

	pvName := ctrl.getProvisionedVolumeNameForClaim(claim)
	volume, err := ctrl.kubeClient.Core().PersistentVolumes().Get(pvName, metav1.GetOptions{})
	if err == nil && volume != nil {
		// Volume has been already provisioned, nothing to do.
		glog.V(4).Infof("provisionClaimOperation [%s]: volume already exists, skipping", claimToClaimKey(claim))
		return
	}

	// Prepare a claimRef to the claim early (to fail before a volume is
	// provisioned)
	claimRef, err := v1.GetReference(api.Scheme, claim)
	if err != nil {
		glog.V(3).Infof("unexpected error getting claim reference: %v", err)
		return
	}

	// Gather provisioning options
	tags := make(map[string]string)
	tags[cloudVolumeCreatedForClaimNamespaceTag] = claim.Namespace
	tags[cloudVolumeCreatedForClaimNameTag] = claim.Name
	tags[cloudVolumeCreatedForVolumeNameTag] = pvName

	options := vol.VolumeOptions{
		PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimDelete,
		CloudTags:                     &tags,
		ClusterName:                   ctrl.clusterName,
		PVName:                        pvName,
		PVC:                           claim,
		Parameters:                    storageClass.Parameters,
	}

	// Provision the volume
	provisioner, err := plugin.NewProvisioner(options)
	if err != nil {
		strerr := fmt.Sprintf("Failed to create provisioner: %v", err)
		glog.V(2).Infof("failed to create provisioner for claim %q with StorageClass %q: %v", claimToClaimKey(claim), storageClass.Name, err)
		ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, "ProvisioningFailed", strerr)
		return
	}

	volume, err = provisioner.Provision()
	if err != nil {
		strerr := fmt.Sprintf("Failed to provision volume with StorageClass %q: %v", storageClass.Name, err)
		glog.V(2).Infof("failed to provision volume for claim %q with StorageClass %q: %v", claimToClaimKey(claim), storageClass.Name, err)
		ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, "ProvisioningFailed", strerr)
		return
	}

	glog.V(3).Infof("volume %q for claim %q created", volume.Name, claimToClaimKey(claim))

	// Create Kubernetes PV object for the volume.
	if volume.Name == "" {
		volume.Name = pvName
	}
	// Bind it to the claim
	volume.Spec.ClaimRef = claimRef
	volume.Status.Phase = v1.VolumeBound

	// Add annBoundByController (used in deleting the volume)
	metav1.SetMetaDataAnnotation(&volume.ObjectMeta, annBoundByController, "yes")
	metav1.SetMetaDataAnnotation(&volume.ObjectMeta, annDynamicallyProvisioned, plugin.GetPluginName())
	// For Alpha provisioning behavior, do not add storage.BetaStorageClassAnnotations for volumes created
	// by storage.AlphaStorageClassAnnotation
	// TODO: remove this check in 1.5, storage.StorageClassAnnotation will be always non-empty there.
	if claimClass != "" {
		volume.Spec.StorageClassName = claimClass
	}

	// Try to create the PV object several times
	for i := 0; i < ctrl.createProvisionedPVRetryCount; i++ {
		glog.V(4).Infof("provisionClaimOperation [%s]: trying to save volume %s", claimToClaimKey(claim), volume.Name)
		var newVol *v1.PersistentVolume
		if newVol, err = ctrl.kubeClient.Core().PersistentVolumes().Create(volume); err == nil {
			// Save succeeded.
			glog.V(3).Infof("volume %q for claim %q saved", volume.Name, claimToClaimKey(claim))

			_, err = ctrl.storeVolumeUpdate(newVol)
			if err != nil {
				// We will get an "volume added" event soon, this is not a big error
				glog.V(4).Infof("provisionClaimOperation [%s]: cannot update internal cache: %v", volume.Name, err)
			}
			break
		}
		// Save failed, try again after a while.
		glog.V(3).Infof("failed to save volume %q for claim %q: %v", volume.Name, claimToClaimKey(claim), err)
		time.Sleep(ctrl.createProvisionedPVInterval)
	}

	if err != nil {
		// Save failed. Now we have a storage asset outside of Kubernetes,
		// but we don't have appropriate PV object for it.
		// Emit some event here and try to delete the storage asset several
		// times.
		strerr := fmt.Sprintf("Error creating provisioned PV object for claim %s: %v. Deleting the volume.", claimToClaimKey(claim), err)
		glog.V(3).Info(strerr)
		ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, "ProvisioningFailed", strerr)

		for i := 0; i < ctrl.createProvisionedPVRetryCount; i++ {
			deleted, err := ctrl.doDeleteVolume(volume)
			if err == nil && deleted {
				// Delete succeeded
				glog.V(4).Infof("provisionClaimOperation [%s]: cleaning volume %s succeeded", claimToClaimKey(claim), volume.Name)
				break
			}
			if !deleted {
				// This is unreachable code, the volume was provisioned by an
				// internal plugin and therefore there MUST be an internal
				// plugin that deletes it.
				glog.Errorf("Error finding internal deleter for volume plugin %q", plugin.GetPluginName())
				break
			}
			// Delete failed, try again after a while.
			glog.V(3).Infof("failed to delete volume %q: %v", volume.Name, err)
			time.Sleep(ctrl.createProvisionedPVInterval)
		}

		if err != nil {
			// Delete failed several times. There is an orphaned volume and there
			// is nothing we can do about it.
			strerr := fmt.Sprintf("Error cleaning provisioned volume for claim %s: %v. Please delete manually.", claimToClaimKey(claim), err)
			glog.V(2).Info(strerr)
			ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, "ProvisioningCleanupFailed", strerr)
		}
	} else {
		glog.V(2).Infof("volume %q provisioned for claim %q", volume.Name, claimToClaimKey(claim))
		msg := fmt.Sprintf("Successfully provisioned volume %s using %s", volume.Name, plugin.GetPluginName())
		ctrl.eventRecorder.Event(claim, v1.EventTypeNormal, "ProvisioningSucceeded", msg)
	}
}

// getProvisionedVolumeNameForClaim returns PV.Name for the provisioned volume.
// The name must be unique.
func (ctrl *PersistentVolumeController) getProvisionedVolumeNameForClaim(claim *v1.PersistentVolumeClaim) string {
	return "pvc-" + string(claim.UID)
}

// scheduleOperation starts given asynchronous operation on given volume. It
// makes sure the operation is already not running.
func (ctrl *PersistentVolumeController) scheduleOperation(operationName string, operation func() error) {
	glog.V(4).Infof("scheduleOperation[%s]", operationName)

	// Poke test code that an operation is just about to get started.
	if ctrl.preOperationHook != nil {
		ctrl.preOperationHook(operationName)
	}

	err := ctrl.runningOperations.Run(operationName, operation)
	if err != nil {
		switch {
		case goroutinemap.IsAlreadyExists(err):
			glog.V(4).Infof("operation %q is already running, skipping", operationName)
		case exponentialbackoff.IsExponentialBackoff(err):
			glog.V(4).Infof("operation %q postponed due to exponential backoff", operationName)
		default:
			glog.Errorf("error scheduling operation %q: %v", operationName, err)
		}
	}
}

// newRecyclerEventRecorder returns a RecycleEventRecorder that sends all events
// to given volume.
func (ctrl *PersistentVolumeController) newRecyclerEventRecorder(volume *v1.PersistentVolume) vol.RecycleEventRecorder {
	return func(eventtype, message string) {
		ctrl.eventRecorder.Eventf(volume, eventtype, "RecyclerPod", "Recycler pod: %s", message)
	}
}

// findProvisionablePlugin finds a provisioner plugin for a given claim.
// It returns either the provisioning plugin or nil when an external
// provisioner is requested.
func (ctrl *PersistentVolumeController) findProvisionablePlugin(claim *v1.PersistentVolumeClaim) (vol.ProvisionableVolumePlugin, *storage.StorageClass, error) {
	// TODO: remove this alpha behavior in 1.5
	alpha := metav1.HasAnnotation(claim.ObjectMeta, v1.AlphaStorageClassAnnotation)
	if alpha && v1.PersistentVolumeClaimHasClass(claim) {
		// Both Alpha annotation and storage class name is set. Use the storage
		// class name.
		alpha = false
		msg := fmt.Sprintf("both %q annotation and storageClassName are present, using storageClassName", v1.AlphaStorageClassAnnotation)
		ctrl.eventRecorder.Event(claim, v1.EventTypeNormal, "ProvisioningIgnoreAlpha", msg)
	}
	if alpha {
		// Fall back to fixed list of provisioner plugins
		return ctrl.findAlphaProvisionablePlugin()
	}

	// provisionClaim() which leads here is never called with claimClass=="", we
	// can save some checks.
	claimClass := v1.GetPersistentVolumeClaimClass(claim)
	class, err := ctrl.classLister.Get(claimClass)
	if err != nil {
		return nil, nil, err
	}

	// Find a plugin for the class
	plugin, err := ctrl.volumePluginMgr.FindProvisionablePluginByName(class.Provisioner)
	if err != nil {
		if !strings.HasPrefix(class.Provisioner, "kubernetes.io/") {
			// External provisioner is requested, do not report error
			return nil, class, nil
		}
		return nil, class, err
	}
	return plugin, class, nil
}

// findAlphaProvisionablePlugin returns a volume plugin compatible with
// Kubernetes 1.3.
// TODO: remove in Kubernetes 1.5
func (ctrl *PersistentVolumeController) findAlphaProvisionablePlugin() (vol.ProvisionableVolumePlugin, *storage.StorageClass, error) {
	if ctrl.alphaProvisioner == nil {
		return nil, nil, fmt.Errorf("cannot find volume plugin for alpha provisioning")
	}

	// Return a dummy StorageClass instance with no parameters
	storageClass := &storage.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "",
		},
		Provisioner: ctrl.alphaProvisioner.GetPluginName(),
	}
	glog.V(4).Infof("using alpha provisioner %s", ctrl.alphaProvisioner.GetPluginName())
	return ctrl.alphaProvisioner, storageClass, nil
}

// findDeletablePlugin finds a deleter plugin for a given volume. It returns
// either the deleter plugin or nil when an external deleter is requested.
func (ctrl *PersistentVolumeController) findDeletablePlugin(volume *v1.PersistentVolume) (vol.DeletableVolumePlugin, error) {
	// Find a plugin. Try to find the same plugin that provisioned the volume
	var plugin vol.DeletableVolumePlugin
	if metav1.HasAnnotation(volume.ObjectMeta, annDynamicallyProvisioned) {
		provisionPluginName := volume.Annotations[annDynamicallyProvisioned]
		if provisionPluginName != "" {
			plugin, err := ctrl.volumePluginMgr.FindDeletablePluginByName(provisionPluginName)
			if err != nil {
				if !strings.HasPrefix(provisionPluginName, "kubernetes.io/") {
					// External provisioner is requested, do not report error
					return nil, nil
				}
				return nil, err
			}
			return plugin, nil
		}
	}

	// The plugin that provisioned the volume was not found or the volume
	// was not dynamically provisioned. Try to find a plugin by spec.
	spec := vol.NewSpecFromPersistentVolume(volume, false)
	plugin, err := ctrl.volumePluginMgr.FindDeletablePluginBySpec(spec)
	if err != nil {
		// No deleter found. Emit an event and mark the volume Failed.
		return nil, fmt.Errorf("Error getting deleter volume plugin for volume %q: %v", volume.Name, err)
	}
	return plugin, nil
}
