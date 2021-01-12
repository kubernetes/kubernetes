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
	"context"
	"fmt"
	"reflect"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	corelisters "k8s.io/client-go/listers/core/v1"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	ref "k8s.io/client-go/tools/reference"
	"k8s.io/client-go/util/workqueue"
	cloudprovider "k8s.io/cloud-provider"
	volerr "k8s.io/cloud-provider/volume/errors"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/controller/volume/common"
	"k8s.io/kubernetes/pkg/controller/volume/events"
	"k8s.io/kubernetes/pkg/controller/volume/persistentvolume/metrics"
	pvutil "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/util"
	"k8s.io/kubernetes/pkg/features"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	"k8s.io/kubernetes/pkg/util/goroutinemap"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
	vol "k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/recyclerclient"

	"k8s.io/klog/v2"
)

// ==================================================================
// PLEASE DO NOT ATTEMPT TO SIMPLIFY THIS CODE.
// KEEP THE SPACE SHUTTLE FLYING.
// ==================================================================
//
// This controller is intentionally written in a very verbose style. You will
// notice:
//
// 1. Every 'if' statement has a matching 'else' (exception: simple error
//    checks for a client API call)
// 2. Things that may seem obvious are commented explicitly
//
// We call this style 'space shuttle style'. Space shuttle style is meant to
// ensure that every branch and condition is considered and accounted for -
// the same way code is written at NASA for applications like the space
// shuttle.
//
// Originally, the work of this controller was split amongst three
// controllers. This controller is the result a large effort to simplify the
// PV subsystem. During that effort, it became clear that we needed to ensure
// that every single condition was handled and accounted for in the code, even
// if it resulted in no-op code branches.
//
// As a result, the controller code may seem overly verbose, commented, and
// 'branchy'. However, a large amount of business knowledge and context is
// recorded here in order to ensure that future maintainers can correctly
// reason through the complexities of the binding behavior. For that reason,
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
// trouble. For example, a rogue HA controller instance could end up racing
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
// can be modified by user or other controller or completely deleted. Also,
// two (or more) controllers may try to bind different volumes to different
// claims at the same time. The controller must recover from any conflicts
// that may arise from these conditions.

// CloudVolumeCreatedForClaimNamespaceTag is a name of a tag attached to a real volume in cloud (e.g. AWS EBS or GCE PD)
// with namespace of a persistent volume claim used to create this volume.
const CloudVolumeCreatedForClaimNamespaceTag = "kubernetes.io/created-for/pvc/namespace"

// CloudVolumeCreatedForClaimNameTag is a name of a tag attached to a real volume in cloud (e.g. AWS EBS or GCE PD)
// with name of a persistent volume claim used to create this volume.
const CloudVolumeCreatedForClaimNameTag = "kubernetes.io/created-for/pvc/name"

// CloudVolumeCreatedForVolumeNameTag is a name of a tag attached to a real volume in cloud (e.g. AWS EBS or GCE PD)
// with name of appropriate Kubernetes persistent volume .
const CloudVolumeCreatedForVolumeNameTag = "kubernetes.io/created-for/pv/name"

// Number of retries when we create a PV object for a provisioned volume.
const createProvisionedPVRetryCount = 5

// Interval between retries when we create a PV object for a provisioned volume.
const createProvisionedPVInterval = 10 * time.Second

// CSINameTranslator can get the CSI Driver name based on the in-tree plugin name
type CSINameTranslator interface {
	GetCSINameFromInTreeName(pluginName string) (string, error)
}

// CSIMigratedPluginManager keeps track of CSI migration status of a plugin
type CSIMigratedPluginManager interface {
	IsMigrationEnabledForPlugin(pluginName string) bool
}

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
	podLister          corelisters.PodLister
	podListerSynced    cache.InformerSynced
	podIndexer         cache.Indexer
	NodeLister         corelisters.NodeLister
	NodeListerSynced   cache.InformerSynced

	kubeClient                clientset.Interface
	eventRecorder             record.EventRecorder
	cloud                     cloudprovider.Interface
	volumePluginMgr           vol.VolumePluginMgr
	enableDynamicProvisioning bool
	clusterName               string
	resyncPeriod              time.Duration

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

	// operationTimestamps caches start timestamp of operations
	// (currently provision + binding/deletion) for metric recording.
	// Detailed lifecycle/key for each operation
	// 1. provision + binding
	//     key:        claimKey
	//     start time: user has NOT provide any volume ref in the claim AND
	//                 there is no existing volume found for the claim,
	//                 "provisionClaim" is called with a valid plugin/external provisioner
	//                 to provision a volume
	//     end time:   after a volume has been provisioned and bound to the claim successfully
	//                 the corresponding timestamp entry will be deleted from cache
	//     abort:      claim has not been bound to a volume yet but a claim deleted event
	//                 has been received from API server
	// 2. deletion
	//     key:        volumeName
	//     start time: when "reclaimVolume" process a volume with reclaim policy
	//                 set to be "PersistentVolumeReclaimDelete"
	//     end time:   after a volume deleted event has been received from API server
	//                 the corresponding timestamp entry will be deleted from cache
	//     abort:      N.A.
	operationTimestamps metrics.OperationStartTimeCache

	translator               CSINameTranslator
	csiMigratedPluginManager CSIMigratedPluginManager

	// filteredDialOptions configures any dialing done by the controller.
	filteredDialOptions *proxyutil.FilteredDialOptions
}

// syncClaim is the main controller method to decide what to do with a claim.
// It's invoked by appropriate cache.Controller callbacks when a claim is
// created, updated or periodically synced. We do not differentiate between
// these events.
// For easier readability, it was split into syncUnboundClaim and syncBoundClaim
// methods.
func (ctrl *PersistentVolumeController) syncClaim(claim *v1.PersistentVolumeClaim) error {
	klog.V(4).Infof("synchronizing PersistentVolumeClaim[%s]: %s", claimToClaimKey(claim), getClaimStatusForLogging(claim))

	// Set correct "migrated-to" annotations on PVC and update in API server if
	// necessary
	newClaim, err := ctrl.updateClaimMigrationAnnotations(claim)
	if err != nil {
		// Nothing was saved; we will fall back into the same
		// condition in the next call to this method
		return err
	}
	claim = newClaim

	if !metav1.HasAnnotation(claim.ObjectMeta, pvutil.AnnBindCompleted) {
		return ctrl.syncUnboundClaim(claim)
	} else {
		return ctrl.syncBoundClaim(claim)
	}
}

// checkVolumeSatisfyClaim checks if the volume requested by the claim satisfies the requirements of the claim
func checkVolumeSatisfyClaim(volume *v1.PersistentVolume, claim *v1.PersistentVolumeClaim) error {
	requestedQty := claim.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	requestedSize := requestedQty.Value()

	// check if PV's DeletionTimeStamp is set, if so, return error.
	if utilfeature.DefaultFeatureGate.Enabled(features.StorageObjectInUseProtection) {
		if volume.ObjectMeta.DeletionTimestamp != nil {
			return fmt.Errorf("the volume is marked for deletion %q", volume.Name)
		}
	}

	volumeQty := volume.Spec.Capacity[v1.ResourceStorage]
	volumeSize := volumeQty.Value()
	if volumeSize < requestedSize {
		return fmt.Errorf("requested PV is too small")
	}

	requestedClass := v1helper.GetPersistentVolumeClaimClass(claim)
	if v1helper.GetPersistentVolumeClass(volume) != requestedClass {
		return fmt.Errorf("storageClassName does not match")
	}

	if pvutil.CheckVolumeModeMismatches(&claim.Spec, &volume.Spec) {
		return fmt.Errorf("incompatible volumeMode")
	}

	if !pvutil.CheckAccessModes(claim, volume) {
		return fmt.Errorf("incompatible accessMode")
	}

	return nil
}

// emitEventForUnboundDelayBindingClaim generates informative event for claim
// if it's in delay binding mode and not bound yet.
func (ctrl *PersistentVolumeController) emitEventForUnboundDelayBindingClaim(claim *v1.PersistentVolumeClaim) error {
	reason := events.WaitForFirstConsumer
	message := "waiting for first consumer to be created before binding"
	podNames, err := ctrl.findNonScheduledPodsByPVC(claim)
	if err != nil {
		return err
	}
	if len(podNames) > 0 {
		reason = events.WaitForPodScheduled
		if len(podNames) > 1 {
			// Although only one pod is taken into account in
			// volume scheduling, more than one pods can reference
			// the PVC at the same time. We can't know which pod is
			// used in scheduling, all pods are included.
			message = fmt.Sprintf("waiting for pods %s to be scheduled", strings.Join(podNames, ","))
		} else {
			message = fmt.Sprintf("waiting for pod %s to be scheduled", podNames[0])
		}
	}
	ctrl.eventRecorder.Event(claim, v1.EventTypeNormal, reason, message)
	return nil
}

// syncUnboundClaim is the main controller method to decide what to do with an
// unbound claim.
func (ctrl *PersistentVolumeController) syncUnboundClaim(claim *v1.PersistentVolumeClaim) error {
	// This is a new PVC that has not completed binding
	// OBSERVATION: pvc is "Pending"
	if claim.Spec.VolumeName == "" {
		// User did not care which PV they get.
		delayBinding, err := pvutil.IsDelayBindingMode(claim, ctrl.classLister)
		if err != nil {
			return err
		}

		// [Unit test set 1]
		volume, err := ctrl.volumes.findBestMatchForClaim(claim, delayBinding)
		if err != nil {
			klog.V(2).Infof("synchronizing unbound PersistentVolumeClaim[%s]: Error finding PV for claim: %v", claimToClaimKey(claim), err)
			return fmt.Errorf("Error finding PV for claim %q: %v", claimToClaimKey(claim), err)
		}
		if volume == nil {
			klog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: no volume found", claimToClaimKey(claim))
			// No PV could be found
			// OBSERVATION: pvc is "Pending", will retry
			switch {
			case delayBinding && !pvutil.IsDelayBindingProvisioning(claim):
				if err = ctrl.emitEventForUnboundDelayBindingClaim(claim); err != nil {
					return err
				}
			case v1helper.GetPersistentVolumeClaimClass(claim) != "":
				if err = ctrl.provisionClaim(claim); err != nil {
					return err
				}
				return nil
			default:
				ctrl.eventRecorder.Event(claim, v1.EventTypeNormal, events.FailedBinding, "no persistent volumes available for this claim and no storage class is set")
			}

			// Mark the claim as Pending and try to find a match in the next
			// periodic syncClaim
			if _, err = ctrl.updateClaimStatus(claim, v1.ClaimPending, nil); err != nil {
				return err
			}
			return nil
		} else /* pv != nil */ {
			// Found a PV for this claim
			// OBSERVATION: pvc is "Pending", pv is "Available"
			claimKey := claimToClaimKey(claim)
			klog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume %q found: %s", claimKey, volume.Name, getVolumeStatusForLogging(volume))
			if err = ctrl.bind(volume, claim); err != nil {
				// On any error saving the volume or the claim, subsequent
				// syncClaim will finish the binding.
				// record count error for provision if exists
				// timestamp entry will remain in cache until a success binding has happened
				metrics.RecordMetric(claimKey, &ctrl.operationTimestamps, err)
				return err
			}
			// OBSERVATION: claim is "Bound", pv is "Bound"
			// if exists a timestamp entry in cache, record end to end provision latency and clean up cache
			// End of the provision + binding operation lifecycle, cache will be cleaned by "RecordMetric"
			// [Unit test 12-1, 12-2, 12-4]
			metrics.RecordMetric(claimKey, &ctrl.operationTimestamps, nil)
			return nil
		}
	} else /* pvc.Spec.VolumeName != nil */ {
		// [Unit test set 2]
		// User asked for a specific PV.
		klog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume %q requested", claimToClaimKey(claim), claim.Spec.VolumeName)
		obj, found, err := ctrl.volumes.store.GetByKey(claim.Spec.VolumeName)
		if err != nil {
			return err
		}
		if !found {
			// User asked for a PV that does not exist.
			// OBSERVATION: pvc is "Pending"
			// Retry later.
			klog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume %q requested and not found, will try again next time", claimToClaimKey(claim), claim.Spec.VolumeName)
			if _, err = ctrl.updateClaimStatus(claim, v1.ClaimPending, nil); err != nil {
				return err
			}
			return nil
		} else {
			volume, ok := obj.(*v1.PersistentVolume)
			if !ok {
				return fmt.Errorf("Cannot convert object from volume cache to volume %q!?: %+v", claim.Spec.VolumeName, obj)
			}
			klog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume %q requested and found: %s", claimToClaimKey(claim), claim.Spec.VolumeName, getVolumeStatusForLogging(volume))
			if volume.Spec.ClaimRef == nil {
				// User asked for a PV that is not claimed
				// OBSERVATION: pvc is "Pending", pv is "Available"
				klog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume is unbound, binding", claimToClaimKey(claim))
				if err = checkVolumeSatisfyClaim(volume, claim); err != nil {
					klog.V(4).Infof("Can't bind the claim to volume %q: %v", volume.Name, err)
					// send an event
					msg := fmt.Sprintf("Cannot bind to requested volume %q: %s", volume.Name, err)
					ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, events.VolumeMismatch, msg)
					// volume does not satisfy the requirements of the claim
					if _, err = ctrl.updateClaimStatus(claim, v1.ClaimPending, nil); err != nil {
						return err
					}
				} else if err = ctrl.bind(volume, claim); err != nil {
					// On any error saving the volume or the claim, subsequent
					// syncClaim will finish the binding.
					return err
				}
				// OBSERVATION: pvc is "Bound", pv is "Bound"
				return nil
			} else if pvutil.IsVolumeBoundToClaim(volume, claim) {
				// User asked for a PV that is claimed by this PVC
				// OBSERVATION: pvc is "Pending", pv is "Bound"
				klog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume already bound, finishing the binding", claimToClaimKey(claim))

				// Finish the volume binding by adding claim UID.
				if err = ctrl.bind(volume, claim); err != nil {
					return err
				}
				// OBSERVATION: pvc is "Bound", pv is "Bound"
				return nil
			} else {
				// User asked for a PV that is claimed by someone else
				// OBSERVATION: pvc is "Pending", pv is "Bound"
				if !metav1.HasAnnotation(claim.ObjectMeta, pvutil.AnnBoundByController) {
					klog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume already bound to different claim by user, will retry later", claimToClaimKey(claim))
					claimMsg := fmt.Sprintf("volume %q already bound to a different claim.", volume.Name)
					ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, events.FailedBinding, claimMsg)
					// User asked for a specific PV, retry later
					if _, err = ctrl.updateClaimStatus(claim, v1.ClaimPending, nil); err != nil {
						return err
					}
					return nil
				} else {
					// This should never happen because someone had to remove
					// AnnBindCompleted annotation on the claim.
					klog.V(4).Infof("synchronizing unbound PersistentVolumeClaim[%s]: volume already bound to different claim %q by controller, THIS SHOULD NEVER HAPPEN", claimToClaimKey(claim), claimrefToClaimKey(volume.Spec.ClaimRef))
					claimMsg := fmt.Sprintf("volume %q already bound to a different claim.", volume.Name)
					ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, events.FailedBinding, claimMsg)

					return fmt.Errorf("Invalid binding of claim %q to volume %q: volume already claimed by %q", claimToClaimKey(claim), claim.Spec.VolumeName, claimrefToClaimKey(volume.Spec.ClaimRef))
				}
			}
		}
	}
}

// syncBoundClaim is the main controller method to decide what to do with a
// bound claim.
func (ctrl *PersistentVolumeController) syncBoundClaim(claim *v1.PersistentVolumeClaim) error {
	// HasAnnotation(pvc, pvutil.AnnBindCompleted)
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

		klog.V(4).Infof("synchronizing bound PersistentVolumeClaim[%s]: volume %q found: %s", claimToClaimKey(claim), claim.Spec.VolumeName, getVolumeStatusForLogging(volume))
		if volume.Spec.ClaimRef == nil {
			// Claim is bound but volume has come unbound.
			// Or, a claim was bound and the controller has not received updated
			// volume yet. We can't distinguish these cases.
			// Bind the volume again and set all states to Bound.
			klog.V(4).Infof("synchronizing bound PersistentVolumeClaim[%s]: volume is unbound, fixing", claimToClaimKey(claim))
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
			klog.V(4).Infof("synchronizing bound PersistentVolumeClaim[%s]: claim is already correctly bound", claimToClaimKey(claim))
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
	klog.V(4).Infof("synchronizing PersistentVolume[%s]: %s", volume.Name, getVolumeStatusForLogging(volume))

	// Set correct "migrated-to" annotations on PV and update in API server if
	// necessary
	newVolume, err := ctrl.updateVolumeMigrationAnnotations(volume)
	if err != nil {
		// Nothing was saved; we will fall back into the same
		// condition in the next call to this method
		return err
	}
	volume = newVolume

	// [Unit test set 4]
	if volume.Spec.ClaimRef == nil {
		// Volume is unused
		klog.V(4).Infof("synchronizing PersistentVolume[%s]: volume is unused", volume.Name)
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
			klog.V(4).Infof("synchronizing PersistentVolume[%s]: volume is pre-bound to claim %s", volume.Name, claimrefToClaimKey(volume.Spec.ClaimRef))
			if _, err := ctrl.updateVolumePhase(volume, v1.VolumeAvailable, ""); err != nil {
				// Nothing was saved; we will fall back into the same
				// condition in the next call to this method
				return err
			}
			return nil
		}
		klog.V(4).Infof("synchronizing PersistentVolume[%s]: volume is bound to claim %s", volume.Name, claimrefToClaimKey(volume.Spec.ClaimRef))
		// Get the PVC by _name_
		var claim *v1.PersistentVolumeClaim
		claimName := claimrefToClaimKey(volume.Spec.ClaimRef)
		obj, found, err := ctrl.claims.GetByKey(claimName)
		if err != nil {
			return err
		}
		if !found {
			// If the PV was created by an external PV provisioner or
			// bound by external PV binder (e.g. kube-scheduler), it's
			// possible under heavy load that the corresponding PVC is not synced to
			// controller local cache yet. So we need to double-check PVC in
			//   1) informer cache
			//   2) apiserver if not found in informer cache
			// to make sure we will not reclaim a PV wrongly.
			// Note that only non-released and non-failed volumes will be
			// updated to Released state when PVC does not exist.
			if volume.Status.Phase != v1.VolumeReleased && volume.Status.Phase != v1.VolumeFailed {
				obj, err = ctrl.claimLister.PersistentVolumeClaims(volume.Spec.ClaimRef.Namespace).Get(volume.Spec.ClaimRef.Name)
				if err != nil && !apierrors.IsNotFound(err) {
					return err
				}
				found = !apierrors.IsNotFound(err)
				if !found {
					obj, err = ctrl.kubeClient.CoreV1().PersistentVolumeClaims(volume.Spec.ClaimRef.Namespace).Get(context.TODO(), volume.Spec.ClaimRef.Name, metav1.GetOptions{})
					if err != nil && !apierrors.IsNotFound(err) {
						return err
					}
					found = !apierrors.IsNotFound(err)
				}
			}
		}
		if !found {
			klog.V(4).Infof("synchronizing PersistentVolume[%s]: claim %s not found", volume.Name, claimrefToClaimKey(volume.Spec.ClaimRef))
			// Fall through with claim = nil
		} else {
			var ok bool
			claim, ok = obj.(*v1.PersistentVolumeClaim)
			if !ok {
				return fmt.Errorf("Cannot convert object from volume cache to volume %q!?: %#v", claim.Spec.VolumeName, obj)
			}
			klog.V(4).Infof("synchronizing PersistentVolume[%s]: claim %s found: %s", volume.Name, claimrefToClaimKey(volume.Spec.ClaimRef), getClaimStatusForLogging(claim))
		}
		if claim != nil && claim.UID != volume.Spec.ClaimRef.UID {
			// The claim that the PV was pointing to was deleted, and another
			// with the same name created.
			klog.V(4).Infof("synchronizing PersistentVolume[%s]: claim %s has different UID, the old one must have been deleted", volume.Name, claimrefToClaimKey(volume.Spec.ClaimRef))
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
				klog.V(2).Infof("volume %q is released and reclaim policy %q will be executed", volume.Name, volume.Spec.PersistentVolumeReclaimPolicy)
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
			if volume.Spec.PersistentVolumeReclaimPolicy == v1.PersistentVolumeReclaimRetain {
				// volume is being retained, it references a claim that does not exist now.
				klog.V(4).Infof("PersistentVolume[%s] references a claim %q (%s) that is not found", volume.Name, claimrefToClaimKey(volume.Spec.ClaimRef), volume.Spec.ClaimRef.UID)
			}
			return nil
		} else if claim.Spec.VolumeName == "" {
			if pvutil.CheckVolumeModeMismatches(&claim.Spec, &volume.Spec) {
				// Binding for the volume won't be called in syncUnboundClaim,
				// because findBestMatchForClaim won't return the volume due to volumeMode mismatch.
				volumeMsg := fmt.Sprintf("Cannot bind PersistentVolume to requested PersistentVolumeClaim %q due to incompatible volumeMode.", claim.Name)
				ctrl.eventRecorder.Event(volume, v1.EventTypeWarning, events.VolumeMismatch, volumeMsg)
				claimMsg := fmt.Sprintf("Cannot bind PersistentVolume %q to requested PersistentVolumeClaim due to incompatible volumeMode.", volume.Name)
				ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, events.VolumeMismatch, claimMsg)
				// Skipping syncClaim
				return nil
			}

			if metav1.HasAnnotation(volume.ObjectMeta, pvutil.AnnBoundByController) {
				// The binding is not completed; let PVC sync handle it
				klog.V(4).Infof("synchronizing PersistentVolume[%s]: volume not bound yet, waiting for syncClaim to fix it", volume.Name)
			} else {
				// Dangling PV; try to re-establish the link in the PVC sync
				klog.V(4).Infof("synchronizing PersistentVolume[%s]: volume was bound and got unbound (by user?), waiting for syncClaim to fix it", volume.Name)
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
			klog.V(4).Infof("synchronizing PersistentVolume[%s]: all is bound", volume.Name)
			if _, err = ctrl.updateVolumePhase(volume, v1.VolumeBound, ""); err != nil {
				// Nothing was saved; we will fall back into the same
				// condition in the next call to this method
				return err
			}
			return nil
		} else {
			// Volume is bound to a claim, but the claim is bound elsewhere
			if metav1.HasAnnotation(volume.ObjectMeta, pvutil.AnnDynamicallyProvisioned) && volume.Spec.PersistentVolumeReclaimPolicy == v1.PersistentVolumeReclaimDelete {
				// This volume was dynamically provisioned for this claim. The
				// claim got bound elsewhere, and thus this volume is not
				// needed. Delete it.
				// Mark the volume as Released for external deleters and to let
				// the user know. Don't overwrite existing Failed status!
				if volume.Status.Phase != v1.VolumeReleased && volume.Status.Phase != v1.VolumeFailed {
					// Also, log this only once:
					klog.V(2).Infof("dynamically volume %q is released and it will be deleted", volume.Name)
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
				if metav1.HasAnnotation(volume.ObjectMeta, pvutil.AnnBoundByController) {
					// This is part of the normal operation of the controller; the
					// controller tried to use this volume for a claim but the claim
					// was fulfilled by another volume. We did this; fix it.
					klog.V(4).Infof("synchronizing PersistentVolume[%s]: volume is bound by controller to a claim that is bound to another volume, unbinding", volume.Name)
					if err = ctrl.unbindVolume(volume); err != nil {
						return err
					}
					return nil
				} else {
					// The PV must have been created with this ptr; leave it alone.
					klog.V(4).Infof("synchronizing PersistentVolume[%s]: volume is bound by user to a claim that is bound to another volume, waiting for the claim to get unbound", volume.Name)
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
//  phase - phase to set
//  volume - volume which Capacity is set into claim.Status.Capacity
func (ctrl *PersistentVolumeController) updateClaimStatus(claim *v1.PersistentVolumeClaim, phase v1.PersistentVolumeClaimPhase, volume *v1.PersistentVolume) (*v1.PersistentVolumeClaim, error) {
	klog.V(4).Infof("updating PersistentVolumeClaim[%s] status: set phase %s", claimToClaimKey(claim), phase)

	dirty := false

	claimClone := claim.DeepCopy()
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

		// Update Capacity if the claim is becoming Bound, not if it was already.
		// A discrepancy can be intentional to mean that the PVC filesystem size
		// doesn't match the PV block device size, so don't clobber it
		if claim.Status.Phase != phase {
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
	}

	if !dirty {
		// Nothing to do.
		klog.V(4).Infof("updating PersistentVolumeClaim[%s] status: phase %s already set", claimToClaimKey(claim), phase)
		return claim, nil
	}

	newClaim, err := ctrl.kubeClient.CoreV1().PersistentVolumeClaims(claimClone.Namespace).UpdateStatus(context.TODO(), claimClone, metav1.UpdateOptions{})
	if err != nil {
		klog.V(4).Infof("updating PersistentVolumeClaim[%s] status: set phase %s failed: %v", claimToClaimKey(claim), phase, err)
		return newClaim, err
	}
	_, err = ctrl.storeClaimUpdate(newClaim)
	if err != nil {
		klog.V(4).Infof("updating PersistentVolumeClaim[%s] status: cannot update internal cache: %v", claimToClaimKey(claim), err)
		return newClaim, err
	}
	klog.V(2).Infof("claim %q entered phase %q", claimToClaimKey(claim), phase)
	return newClaim, nil
}

// updateClaimStatusWithEvent saves new claim.Status to API server and emits
// given event on the claim. It saves the status and emits the event only when
// the status has actually changed from the version saved in API server.
// Parameters:
//   claim - claim to update
//   phase - phase to set
//   volume - volume which Capacity is set into claim.Status.Capacity
//   eventtype, reason, message - event to send, see EventRecorder.Event()
func (ctrl *PersistentVolumeController) updateClaimStatusWithEvent(claim *v1.PersistentVolumeClaim, phase v1.PersistentVolumeClaimPhase, volume *v1.PersistentVolume, eventtype, reason, message string) (*v1.PersistentVolumeClaim, error) {
	klog.V(4).Infof("updating updateClaimStatusWithEvent[%s]: set phase %s", claimToClaimKey(claim), phase)
	if claim.Status.Phase == phase {
		// Nothing to do.
		klog.V(4).Infof("updating updateClaimStatusWithEvent[%s]: phase %s already set", claimToClaimKey(claim), phase)
		return claim, nil
	}

	newClaim, err := ctrl.updateClaimStatus(claim, phase, volume)
	if err != nil {
		return nil, err
	}

	// Emit the event only when the status change happens, not every time
	// syncClaim is called.
	klog.V(3).Infof("claim %q changed status to %q: %s", claimToClaimKey(claim), phase, message)
	ctrl.eventRecorder.Event(newClaim, eventtype, reason, message)

	return newClaim, nil
}

// updateVolumePhase saves new volume phase to API server.
func (ctrl *PersistentVolumeController) updateVolumePhase(volume *v1.PersistentVolume, phase v1.PersistentVolumePhase, message string) (*v1.PersistentVolume, error) {
	klog.V(4).Infof("updating PersistentVolume[%s]: set phase %s", volume.Name, phase)
	if volume.Status.Phase == phase {
		// Nothing to do.
		klog.V(4).Infof("updating PersistentVolume[%s]: phase %s already set", volume.Name, phase)
		return volume, nil
	}

	volumeClone := volume.DeepCopy()
	volumeClone.Status.Phase = phase
	volumeClone.Status.Message = message

	newVol, err := ctrl.kubeClient.CoreV1().PersistentVolumes().UpdateStatus(context.TODO(), volumeClone, metav1.UpdateOptions{})
	if err != nil {
		klog.V(4).Infof("updating PersistentVolume[%s]: set phase %s failed: %v", volume.Name, phase, err)
		return newVol, err
	}
	_, err = ctrl.storeVolumeUpdate(newVol)
	if err != nil {
		klog.V(4).Infof("updating PersistentVolume[%s]: cannot update internal cache: %v", volume.Name, err)
		return newVol, err
	}
	klog.V(2).Infof("volume %q entered phase %q", volume.Name, phase)
	return newVol, err
}

// updateVolumePhaseWithEvent saves new volume phase to API server and emits
// given event on the volume. It saves the phase and emits the event only when
// the phase has actually changed from the version saved in API server.
func (ctrl *PersistentVolumeController) updateVolumePhaseWithEvent(volume *v1.PersistentVolume, phase v1.PersistentVolumePhase, eventtype, reason, message string) (*v1.PersistentVolume, error) {
	klog.V(4).Infof("updating updateVolumePhaseWithEvent[%s]: set phase %s", volume.Name, phase)
	if volume.Status.Phase == phase {
		// Nothing to do.
		klog.V(4).Infof("updating updateVolumePhaseWithEvent[%s]: phase %s already set", volume.Name, phase)
		return volume, nil
	}

	newVol, err := ctrl.updateVolumePhase(volume, phase, message)
	if err != nil {
		return nil, err
	}

	// Emit the event only when the status change happens, not every time
	// syncClaim is called.
	klog.V(3).Infof("volume %q changed status to %q: %s", volume.Name, phase, message)
	ctrl.eventRecorder.Event(newVol, eventtype, reason, message)

	return newVol, nil
}

// bindVolumeToClaim modifies given volume to be bound to a claim and saves it to
// API server. The claim is not modified in this method!
func (ctrl *PersistentVolumeController) bindVolumeToClaim(volume *v1.PersistentVolume, claim *v1.PersistentVolumeClaim) (*v1.PersistentVolume, error) {
	klog.V(4).Infof("updating PersistentVolume[%s]: binding to %q", volume.Name, claimToClaimKey(claim))

	volumeClone, dirty, err := pvutil.GetBindVolumeToClaim(volume, claim)
	if err != nil {
		return nil, err
	}

	// Save the volume only if something was changed
	if dirty {
		return ctrl.updateBindVolumeToClaim(volumeClone, true)
	}

	klog.V(4).Infof("updating PersistentVolume[%s]: already bound to %q", volume.Name, claimToClaimKey(claim))
	return volume, nil
}

// updateBindVolumeToClaim modifies given volume to be bound to a claim and saves it to
// API server. The claim is not modified in this method!
func (ctrl *PersistentVolumeController) updateBindVolumeToClaim(volumeClone *v1.PersistentVolume, updateCache bool) (*v1.PersistentVolume, error) {
	claimKey := claimrefToClaimKey(volumeClone.Spec.ClaimRef)
	klog.V(2).Infof("claim %q bound to volume %q", claimKey, volumeClone.Name)
	newVol, err := ctrl.kubeClient.CoreV1().PersistentVolumes().Update(context.TODO(), volumeClone, metav1.UpdateOptions{})
	if err != nil {
		klog.V(4).Infof("updating PersistentVolume[%s]: binding to %q failed: %v", volumeClone.Name, claimKey, err)
		return newVol, err
	}
	if updateCache {
		_, err = ctrl.storeVolumeUpdate(newVol)
		if err != nil {
			klog.V(4).Infof("updating PersistentVolume[%s]: cannot update internal cache: %v", volumeClone.Name, err)
			return newVol, err
		}
	}
	klog.V(4).Infof("updating PersistentVolume[%s]: bound to %q", newVol.Name, claimKey)
	return newVol, nil
}

// bindClaimToVolume modifies the given claim to be bound to a volume and
// saves it to API server. The volume is not modified in this method!
func (ctrl *PersistentVolumeController) bindClaimToVolume(claim *v1.PersistentVolumeClaim, volume *v1.PersistentVolume) (*v1.PersistentVolumeClaim, error) {
	klog.V(4).Infof("updating PersistentVolumeClaim[%s]: binding to %q", claimToClaimKey(claim), volume.Name)

	dirty := false

	// Check if the claim was already bound (either by controller or by user)
	shouldBind := false
	if volume.Name != claim.Spec.VolumeName {
		shouldBind = true
	}

	// The claim from method args can be pointing to watcher cache. We must not
	// modify these, therefore create a copy.
	claimClone := claim.DeepCopy()

	if shouldBind {
		dirty = true
		// Bind the claim to the volume
		claimClone.Spec.VolumeName = volume.Name

		// Set AnnBoundByController if it is not set yet
		if !metav1.HasAnnotation(claimClone.ObjectMeta, pvutil.AnnBoundByController) {
			metav1.SetMetaDataAnnotation(&claimClone.ObjectMeta, pvutil.AnnBoundByController, "yes")
		}
	}

	// Set AnnBindCompleted if it is not set yet
	if !metav1.HasAnnotation(claimClone.ObjectMeta, pvutil.AnnBindCompleted) {
		metav1.SetMetaDataAnnotation(&claimClone.ObjectMeta, pvutil.AnnBindCompleted, "yes")
		dirty = true
	}

	if dirty {
		klog.V(2).Infof("volume %q bound to claim %q", volume.Name, claimToClaimKey(claim))
		newClaim, err := ctrl.kubeClient.CoreV1().PersistentVolumeClaims(claim.Namespace).Update(context.TODO(), claimClone, metav1.UpdateOptions{})
		if err != nil {
			klog.V(4).Infof("updating PersistentVolumeClaim[%s]: binding to %q failed: %v", claimToClaimKey(claim), volume.Name, err)
			return newClaim, err
		}
		_, err = ctrl.storeClaimUpdate(newClaim)
		if err != nil {
			klog.V(4).Infof("updating PersistentVolumeClaim[%s]: cannot update internal cache: %v", claimToClaimKey(claim), err)
			return newClaim, err
		}
		klog.V(4).Infof("updating PersistentVolumeClaim[%s]: bound to %q", claimToClaimKey(claim), volume.Name)
		return newClaim, nil
	}

	klog.V(4).Infof("updating PersistentVolumeClaim[%s]: already bound to %q", claimToClaimKey(claim), volume.Name)
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

	klog.V(4).Infof("binding volume %q to claim %q", volume.Name, claimToClaimKey(claim))

	if updatedVolume, err = ctrl.bindVolumeToClaim(volume, claim); err != nil {
		klog.V(3).Infof("error binding volume %q to claim %q: failed saving the volume: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}
	volume = updatedVolume

	if updatedVolume, err = ctrl.updateVolumePhase(volume, v1.VolumeBound, ""); err != nil {
		klog.V(3).Infof("error binding volume %q to claim %q: failed saving the volume status: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}
	volume = updatedVolume

	if updatedClaim, err = ctrl.bindClaimToVolume(claim, volume); err != nil {
		klog.V(3).Infof("error binding volume %q to claim %q: failed saving the claim: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}
	claim = updatedClaim

	if updatedClaim, err = ctrl.updateClaimStatus(claim, v1.ClaimBound, volume); err != nil {
		klog.V(3).Infof("error binding volume %q to claim %q: failed saving the claim status: %v", volume.Name, claimToClaimKey(claim), err)
		return err
	}
	claim = updatedClaim

	klog.V(4).Infof("volume %q bound to claim %q", volume.Name, claimToClaimKey(claim))
	klog.V(4).Infof("volume %q status after binding: %s", volume.Name, getVolumeStatusForLogging(volume))
	klog.V(4).Infof("claim %q status after binding: %s", claimToClaimKey(claim), getClaimStatusForLogging(claim))
	return nil
}

// unbindVolume rolls back previous binding of the volume. This may be necessary
// when two controllers bound two volumes to single claim - when we detect this,
// only one binding succeeds and the second one must be rolled back.
// This method updates both Spec and Status.
// It returns on first error, it's up to the caller to implement some retry
// mechanism.
func (ctrl *PersistentVolumeController) unbindVolume(volume *v1.PersistentVolume) error {
	klog.V(4).Infof("updating PersistentVolume[%s]: rolling back binding from %q", volume.Name, claimrefToClaimKey(volume.Spec.ClaimRef))

	// Save the PV only when any modification is necessary.
	volumeClone := volume.DeepCopy()

	if metav1.HasAnnotation(volume.ObjectMeta, pvutil.AnnBoundByController) {
		// The volume was bound by the controller.
		volumeClone.Spec.ClaimRef = nil
		delete(volumeClone.Annotations, pvutil.AnnBoundByController)
		if len(volumeClone.Annotations) == 0 {
			// No annotations look better than empty annotation map (and it's easier
			// to test).
			volumeClone.Annotations = nil
		}
	} else {
		// The volume was pre-bound by user. Clear only the binding UID.
		volumeClone.Spec.ClaimRef.UID = ""
	}

	newVol, err := ctrl.kubeClient.CoreV1().PersistentVolumes().Update(context.TODO(), volumeClone, metav1.UpdateOptions{})
	if err != nil {
		klog.V(4).Infof("updating PersistentVolume[%s]: rollback failed: %v", volume.Name, err)
		return err
	}
	_, err = ctrl.storeVolumeUpdate(newVol)
	if err != nil {
		klog.V(4).Infof("updating PersistentVolume[%s]: cannot update internal cache: %v", volume.Name, err)
		return err
	}
	klog.V(4).Infof("updating PersistentVolume[%s]: rolled back", newVol.Name)

	// Update the status
	_, err = ctrl.updateVolumePhase(newVol, v1.VolumeAvailable, "")
	return err
}

// reclaimVolume implements volume.Spec.PersistentVolumeReclaimPolicy and
// starts appropriate reclaim action.
func (ctrl *PersistentVolumeController) reclaimVolume(volume *v1.PersistentVolume) error {
	if migrated := volume.Annotations[pvutil.AnnMigratedTo]; len(migrated) > 0 {
		// PV is Migrated. The PV controller should stand down and the external
		// provisioner will handle this PV
		return nil
	}
	switch volume.Spec.PersistentVolumeReclaimPolicy {
	case v1.PersistentVolumeReclaimRetain:
		klog.V(4).Infof("reclaimVolume[%s]: policy is Retain, nothing to do", volume.Name)

	case v1.PersistentVolumeReclaimRecycle:
		klog.V(4).Infof("reclaimVolume[%s]: policy is Recycle", volume.Name)
		opName := fmt.Sprintf("recycle-%s[%s]", volume.Name, string(volume.UID))
		ctrl.scheduleOperation(opName, func() error {
			ctrl.recycleVolumeOperation(volume)
			return nil
		})

	case v1.PersistentVolumeReclaimDelete:
		klog.V(4).Infof("reclaimVolume[%s]: policy is Delete", volume.Name)
		opName := fmt.Sprintf("delete-%s[%s]", volume.Name, string(volume.UID))
		// create a start timestamp entry in cache for deletion operation if no one exists with
		// key = volume.Name, pluginName = provisionerName, operation = "delete"
		ctrl.operationTimestamps.AddIfNotExist(volume.Name, ctrl.getProvisionerNameFromVolume(volume), "delete")
		ctrl.scheduleOperation(opName, func() error {
			_, err := ctrl.deleteVolumeOperation(volume)
			if err != nil {
				// only report error count to "volume_operation_total_errors"
				// latency reporting will happen when the volume get finally
				// deleted and a volume deleted event is captured
				metrics.RecordMetric(volume.Name, &ctrl.operationTimestamps, err)
			}
			return err
		})

	default:
		// Unknown PersistentVolumeReclaimPolicy
		if _, err := ctrl.updateVolumePhaseWithEvent(volume, v1.VolumeFailed, v1.EventTypeWarning, "VolumeUnknownReclaimPolicy", "Volume has unrecognized PersistentVolumeReclaimPolicy"); err != nil {
			return err
		}
	}
	return nil
}

// recycleVolumeOperation recycles a volume. This method is running in
// standalone goroutine and already has all necessary locks.
func (ctrl *PersistentVolumeController) recycleVolumeOperation(volume *v1.PersistentVolume) {
	klog.V(4).Infof("recycleVolumeOperation [%s] started", volume.Name)

	// This method may have been waiting for a volume lock for some time.
	// Previous recycleVolumeOperation might just have saved an updated version,
	// so read current volume state now.
	newVolume, err := ctrl.kubeClient.CoreV1().PersistentVolumes().Get(context.TODO(), volume.Name, metav1.GetOptions{})
	if err != nil {
		klog.V(3).Infof("error reading persistent volume %q: %v", volume.Name, err)
		return
	}
	needsReclaim, err := ctrl.isVolumeReleased(newVolume)
	if err != nil {
		klog.V(3).Infof("error reading claim for volume %q: %v", volume.Name, err)
		return
	}
	if !needsReclaim {
		klog.V(3).Infof("volume %q no longer needs recycling, skipping", volume.Name)
		return
	}
	pods, used, err := ctrl.isVolumeUsed(newVolume)
	if err != nil {
		klog.V(3).Infof("can't recycle volume %q: %v", volume.Name, err)
		return
	}

	// Verify the claim is in cache: if so, then it is a different PVC with the same name
	// since the volume is known to be released at this moment. The new (cached) PVC must use
	// a different PV -- we checked that the PV is unused in isVolumeReleased.
	// So the old PV is safe to be recycled.
	claimName := claimrefToClaimKey(volume.Spec.ClaimRef)
	_, claimCached, err := ctrl.claims.GetByKey(claimName)
	if err != nil {
		klog.V(3).Infof("error getting the claim %s from cache", claimName)
		return
	}

	if used && !claimCached {
		msg := fmt.Sprintf("Volume is used by pods: %s", strings.Join(pods, ","))
		klog.V(3).Infof("can't recycle volume %q: %s", volume.Name, msg)
		ctrl.eventRecorder.Event(volume, v1.EventTypeNormal, events.VolumeFailedRecycle, msg)
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
		if _, err = ctrl.updateVolumePhaseWithEvent(volume, v1.VolumeFailed, v1.EventTypeWarning, events.VolumeFailedRecycle, "No recycler plugin found for the volume!"); err != nil {
			klog.V(4).Infof("recycleVolumeOperation [%s]: failed to mark volume as failed: %v", volume.Name, err)
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
		if _, err = ctrl.updateVolumePhaseWithEvent(volume, v1.VolumeFailed, v1.EventTypeWarning, events.VolumeFailedRecycle, strerr); err != nil {
			klog.V(4).Infof("recycleVolumeOperation [%s]: failed to mark volume as failed: %v", volume.Name, err)
			// Save failed, retry on the next deletion attempt
			return
		}
		// Despite the volume being Failed, the controller will retry recycling
		// the volume in every syncVolume() call.
		return
	}

	klog.V(2).Infof("volume %q recycled", volume.Name)
	// Send an event
	ctrl.eventRecorder.Event(volume, v1.EventTypeNormal, events.VolumeRecycled, "Volume recycled")
	// Make the volume available again
	if err = ctrl.unbindVolume(volume); err != nil {
		// Oops, could not save the volume and therefore the controller will
		// recycle the volume again on next update. We _could_ maintain a cache
		// of "recently recycled volumes" and avoid unnecessary recycling, this
		// is left out as future optimization.
		klog.V(3).Infof("recycleVolumeOperation [%s]: failed to make recycled volume 'Available' (%v), we will recycle the volume again", volume.Name, err)
		return
	}
	return
}

// deleteVolumeOperation deletes a volume. This method is running in standalone
// goroutine and already has all necessary locks.
func (ctrl *PersistentVolumeController) deleteVolumeOperation(volume *v1.PersistentVolume) (string, error) {
	klog.V(4).Infof("deleteVolumeOperation [%s] started", volume.Name)

	// This method may have been waiting for a volume lock for some time.
	// Previous deleteVolumeOperation might just have saved an updated version, so
	// read current volume state now.
	newVolume, err := ctrl.kubeClient.CoreV1().PersistentVolumes().Get(context.TODO(), volume.Name, metav1.GetOptions{})
	if err != nil {
		klog.V(3).Infof("error reading persistent volume %q: %v", volume.Name, err)
		return "", nil
	}

	if newVolume.GetDeletionTimestamp() != nil {
		klog.V(3).Infof("Volume %q is already being deleted", volume.Name)
		return "", nil
	}
	needsReclaim, err := ctrl.isVolumeReleased(newVolume)
	if err != nil {
		klog.V(3).Infof("error reading claim for volume %q: %v", volume.Name, err)
		return "", nil
	}
	if !needsReclaim {
		klog.V(3).Infof("volume %q no longer needs deletion, skipping", volume.Name)
		return "", nil
	}

	pluginName, deleted, err := ctrl.doDeleteVolume(volume)
	if err != nil {
		// Delete failed, update the volume and emit an event.
		klog.V(3).Infof("deletion of volume %q failed: %v", volume.Name, err)
		if volerr.IsDeletedVolumeInUse(err) {
			// The plugin needs more time, don't mark the volume as Failed
			// and send Normal event only
			ctrl.eventRecorder.Event(volume, v1.EventTypeNormal, events.VolumeDelete, err.Error())
		} else {
			// The plugin failed, mark the volume as Failed and send Warning
			// event
			if _, err := ctrl.updateVolumePhaseWithEvent(volume, v1.VolumeFailed, v1.EventTypeWarning, events.VolumeFailedDelete, err.Error()); err != nil {
				klog.V(4).Infof("deleteVolumeOperation [%s]: failed to mark volume as failed: %v", volume.Name, err)
				// Save failed, retry on the next deletion attempt
				return pluginName, err
			}
		}

		// Despite the volume being Failed, the controller will retry deleting
		// the volume in every syncVolume() call.
		return pluginName, err
	}
	if !deleted {
		// The volume waits for deletion by an external plugin. Do nothing.
		return pluginName, nil
	}

	klog.V(4).Infof("deleteVolumeOperation [%s]: success", volume.Name)
	// Delete the volume
	if err = ctrl.kubeClient.CoreV1().PersistentVolumes().Delete(context.TODO(), volume.Name, metav1.DeleteOptions{}); err != nil {
		// Oops, could not delete the volume and therefore the controller will
		// try to delete the volume again on next update. We _could_ maintain a
		// cache of "recently deleted volumes" and avoid unnecessary deletion,
		// this is left out as future optimization.
		klog.V(3).Infof("failed to delete volume %q from database: %v", volume.Name, err)
		return pluginName, nil
	}
	return pluginName, nil
}

// isVolumeReleased returns true if given volume is released and can be recycled
// or deleted, based on its retain policy. I.e. the volume is bound to a claim
// and the claim does not exist or exists and is bound to different volume.
func (ctrl *PersistentVolumeController) isVolumeReleased(volume *v1.PersistentVolume) (bool, error) {
	// A volume needs reclaim if it has ClaimRef and appropriate claim does not
	// exist.
	if volume.Spec.ClaimRef == nil {
		klog.V(4).Infof("isVolumeReleased[%s]: ClaimRef is nil", volume.Name)
		return false, nil
	}
	if volume.Spec.ClaimRef.UID == "" {
		// This is a volume bound by user and the controller has not finished
		// binding to the real claim yet.
		klog.V(4).Infof("isVolumeReleased[%s]: ClaimRef is not bound", volume.Name)
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

		klog.V(4).Infof("isVolumeReleased[%s]: ClaimRef is still valid, volume is not released", volume.Name)
		return false, nil
	}

	klog.V(2).Infof("isVolumeReleased[%s]: volume is released", volume.Name)
	return true, nil
}

func (ctrl *PersistentVolumeController) findPodsByPVCKey(key string) ([]*v1.Pod, error) {
	pods := []*v1.Pod{}
	objs, err := ctrl.podIndexer.ByIndex(common.PodPVCIndex, key)
	if err != nil {
		return pods, err
	}
	for _, obj := range objs {
		pod, ok := obj.(*v1.Pod)
		if !ok {
			continue
		}
		pods = append(pods, pod)
	}
	return pods, err
}

// isVolumeUsed returns list of active pods that use given PV.
func (ctrl *PersistentVolumeController) isVolumeUsed(pv *v1.PersistentVolume) ([]string, bool, error) {
	if pv.Spec.ClaimRef == nil {
		return nil, false, nil
	}
	podNames := sets.NewString()
	pvcKey := fmt.Sprintf("%s/%s", pv.Spec.ClaimRef.Namespace, pv.Spec.ClaimRef.Name)
	pods, err := ctrl.findPodsByPVCKey(pvcKey)
	if err != nil {
		return nil, false, fmt.Errorf("error finding pods by pvc %q: %s", pvcKey, err)
	}
	for _, pod := range pods {
		if util.IsPodTerminated(pod, pod.Status) {
			continue
		}
		podNames.Insert(pod.Namespace + "/" + pod.Name)
	}
	return podNames.List(), podNames.Len() != 0, nil
}

// findNonScheduledPodsByPVC returns list of non-scheduled active pods that reference given PVC.
func (ctrl *PersistentVolumeController) findNonScheduledPodsByPVC(pvc *v1.PersistentVolumeClaim) ([]string, error) {
	pvcKey := fmt.Sprintf("%s/%s", pvc.Namespace, pvc.Name)
	pods, err := ctrl.findPodsByPVCKey(pvcKey)
	if err != nil {
		return nil, err
	}
	podNames := []string{}
	for _, pod := range pods {
		if util.IsPodTerminated(pod, pod.Status) {
			continue
		}
		if len(pod.Spec.NodeName) == 0 {
			podNames = append(podNames, pod.Name)
		}
	}
	return podNames, nil
}

// doDeleteVolume finds appropriate delete plugin and deletes given volume, returning
// the volume plugin name. Also, it returns 'true', when the volume was deleted and
// 'false' when the volume cannot be deleted because the deleter is external. No
// error should be reported in this case.
func (ctrl *PersistentVolumeController) doDeleteVolume(volume *v1.PersistentVolume) (string, bool, error) {
	klog.V(4).Infof("doDeleteVolume [%s]", volume.Name)
	var err error

	plugin, err := ctrl.findDeletablePlugin(volume)
	if err != nil {
		return "", false, err
	}
	if plugin == nil {
		// External deleter is requested, do nothing
		klog.V(3).Infof("external deleter for volume %q requested, ignoring", volume.Name)
		return "", false, nil
	}

	// Plugin found
	pluginName := plugin.GetPluginName()
	klog.V(5).Infof("found a deleter plugin %q for volume %q", pluginName, volume.Name)
	spec := vol.NewSpecFromPersistentVolume(volume, false)
	deleter, err := plugin.NewDeleter(spec)
	if err != nil {
		// Cannot create deleter
		return pluginName, false, fmt.Errorf("Failed to create deleter for volume %q: %v", volume.Name, err)
	}

	opComplete := util.OperationCompleteHook(pluginName, "volume_delete")
	err = deleter.Delete()
	opComplete(&err)
	if err != nil {
		// Deleter failed
		return pluginName, false, err
	}

	klog.V(2).Infof("volume %q deleted", volume.Name)
	return pluginName, true, nil
}

// provisionClaim starts new asynchronous operation to provision a claim if
// provisioning is enabled.
func (ctrl *PersistentVolumeController) provisionClaim(claim *v1.PersistentVolumeClaim) error {
	if !ctrl.enableDynamicProvisioning {
		return nil
	}
	klog.V(4).Infof("provisionClaim[%s]: started", claimToClaimKey(claim))
	opName := fmt.Sprintf("provision-%s[%s]", claimToClaimKey(claim), string(claim.UID))
	plugin, storageClass, err := ctrl.findProvisionablePlugin(claim)
	// findProvisionablePlugin does not return err for external provisioners
	if err != nil {
		ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, events.ProvisioningFailed, err.Error())
		klog.Errorf("error finding provisioning plugin for claim %s: %v", claimToClaimKey(claim), err)
		// failed to find the requested provisioning plugin, directly return err for now.
		// controller will retry the provisioning in every syncUnboundClaim() call
		// retain the original behavior of returning nil from provisionClaim call
		return nil
	}
	ctrl.scheduleOperation(opName, func() error {
		// create a start timestamp entry in cache for provision operation if no one exists with
		// key = claimKey, pluginName = provisionerName, operation = "provision"
		claimKey := claimToClaimKey(claim)
		ctrl.operationTimestamps.AddIfNotExist(claimKey, ctrl.getProvisionerName(plugin, storageClass), "provision")
		var err error
		if plugin == nil {
			_, err = ctrl.provisionClaimOperationExternal(claim, storageClass)
		} else {
			_, err = ctrl.provisionClaimOperation(claim, plugin, storageClass)
		}
		// if error happened, record an error count metric
		// timestamp entry will remain in cache until a success binding has happened
		if err != nil {
			metrics.RecordMetric(claimKey, &ctrl.operationTimestamps, err)
		}
		return err
	})
	return nil
}

// provisionClaimOperation provisions a volume. This method is running in
// standalone goroutine and already has all necessary locks.
func (ctrl *PersistentVolumeController) provisionClaimOperation(
	claim *v1.PersistentVolumeClaim,
	plugin vol.ProvisionableVolumePlugin,
	storageClass *storage.StorageClass) (string, error) {
	claimClass := v1helper.GetPersistentVolumeClaimClass(claim)
	klog.V(4).Infof("provisionClaimOperation [%s] started, class: %q", claimToClaimKey(claim), claimClass)

	// called from provisionClaim(), in this case, plugin MUST NOT be nil
	// NOTE: checks on plugin/storageClass has been saved
	pluginName := plugin.GetPluginName()
	provisionerName := storageClass.Provisioner

	// Add provisioner annotation to be consistent with external provisioner workflow
	newClaim, err := ctrl.setClaimProvisioner(claim, provisionerName)
	if err != nil {
		// Save failed, the controller will retry in the next sync
		klog.V(2).Infof("error saving claim %s: %v", claimToClaimKey(claim), err)
		return pluginName, err
	}
	claim = newClaim

	// internal provisioning

	//  A previous provisionClaimOperation may just have finished while we were waiting for
	//  the locks. Check that PV (with deterministic name) hasn't been provisioned
	//  yet.

	pvName := ctrl.getProvisionedVolumeNameForClaim(claim)
	volume, err := ctrl.kubeClient.CoreV1().PersistentVolumes().Get(context.TODO(), pvName, metav1.GetOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		klog.V(3).Infof("error reading persistent volume %q: %v", pvName, err)
		return pluginName, err
	}
	if err == nil && volume != nil {
		// Volume has been already provisioned, nothing to do.
		klog.V(4).Infof("provisionClaimOperation [%s]: volume already exists, skipping", claimToClaimKey(claim))
		return pluginName, err
	}

	// Prepare a claimRef to the claim early (to fail before a volume is
	// provisioned)
	claimRef, err := ref.GetReference(scheme.Scheme, claim)
	if err != nil {
		klog.V(3).Infof("unexpected error getting claim reference: %v", err)
		return pluginName, err
	}

	// Gather provisioning options
	tags := make(map[string]string)
	tags[CloudVolumeCreatedForClaimNamespaceTag] = claim.Namespace
	tags[CloudVolumeCreatedForClaimNameTag] = claim.Name
	tags[CloudVolumeCreatedForVolumeNameTag] = pvName

	options := vol.VolumeOptions{
		PersistentVolumeReclaimPolicy: *storageClass.ReclaimPolicy,
		MountOptions:                  storageClass.MountOptions,
		CloudTags:                     &tags,
		ClusterName:                   ctrl.clusterName,
		PVName:                        pvName,
		PVC:                           claim,
		Parameters:                    storageClass.Parameters,
	}

	// Refuse to provision if the plugin doesn't support mount options, creation
	// of PV would be rejected by validation anyway
	if !plugin.SupportsMountOption() && len(options.MountOptions) > 0 {
		strerr := fmt.Sprintf("Mount options are not supported by the provisioner but StorageClass %q has mount options %v", storageClass.Name, options.MountOptions)
		klog.V(2).Infof("Mount options are not supported by the provisioner but claim %q's StorageClass %q has mount options %v", claimToClaimKey(claim), storageClass.Name, options.MountOptions)
		ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, events.ProvisioningFailed, strerr)
		return pluginName, fmt.Errorf("provisioner %q doesn't support mount options", plugin.GetPluginName())
	}

	// Provision the volume
	provisioner, err := plugin.NewProvisioner(options)
	if err != nil {
		strerr := fmt.Sprintf("Failed to create provisioner: %v", err)
		klog.V(2).Infof("failed to create provisioner for claim %q with StorageClass %q: %v", claimToClaimKey(claim), storageClass.Name, err)
		ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, events.ProvisioningFailed, strerr)
		return pluginName, err
	}

	var selectedNode *v1.Node = nil
	if nodeName, ok := claim.Annotations[pvutil.AnnSelectedNode]; ok {
		selectedNode, err = ctrl.NodeLister.Get(nodeName)
		if err != nil {
			strerr := fmt.Sprintf("Failed to get target node: %v", err)
			klog.V(3).Infof("unexpected error getting target node %q for claim %q: %v", nodeName, claimToClaimKey(claim), err)
			ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, events.ProvisioningFailed, strerr)
			return pluginName, err
		}
	}
	allowedTopologies := storageClass.AllowedTopologies

	opComplete := util.OperationCompleteHook(plugin.GetPluginName(), "volume_provision")
	volume, err = provisioner.Provision(selectedNode, allowedTopologies)
	opComplete(&err)
	if err != nil {
		// Other places of failure have nothing to do with VolumeScheduling,
		// so just let controller retry in the next sync. We'll only call func
		// rescheduleProvisioning here when the underlying provisioning actually failed.
		ctrl.rescheduleProvisioning(claim)

		strerr := fmt.Sprintf("Failed to provision volume with StorageClass %q: %v", storageClass.Name, err)
		klog.V(2).Infof("failed to provision volume for claim %q with StorageClass %q: %v", claimToClaimKey(claim), storageClass.Name, err)
		ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, events.ProvisioningFailed, strerr)
		return pluginName, err
	}

	klog.V(3).Infof("volume %q for claim %q created", volume.Name, claimToClaimKey(claim))

	// Create Kubernetes PV object for the volume.
	if volume.Name == "" {
		volume.Name = pvName
	}
	// Bind it to the claim
	volume.Spec.ClaimRef = claimRef
	volume.Status.Phase = v1.VolumeBound
	volume.Spec.StorageClassName = claimClass

	// Add AnnBoundByController (used in deleting the volume)
	metav1.SetMetaDataAnnotation(&volume.ObjectMeta, pvutil.AnnBoundByController, "yes")
	metav1.SetMetaDataAnnotation(&volume.ObjectMeta, pvutil.AnnDynamicallyProvisioned, plugin.GetPluginName())

	// Try to create the PV object several times
	for i := 0; i < ctrl.createProvisionedPVRetryCount; i++ {
		klog.V(4).Infof("provisionClaimOperation [%s]: trying to save volume %s", claimToClaimKey(claim), volume.Name)
		var newVol *v1.PersistentVolume
		if newVol, err = ctrl.kubeClient.CoreV1().PersistentVolumes().Create(context.TODO(), volume, metav1.CreateOptions{}); err == nil || apierrors.IsAlreadyExists(err) {
			// Save succeeded.
			if err != nil {
				klog.V(3).Infof("volume %q for claim %q already exists, reusing", volume.Name, claimToClaimKey(claim))
				err = nil
			} else {
				klog.V(3).Infof("volume %q for claim %q saved", volume.Name, claimToClaimKey(claim))

				_, updateErr := ctrl.storeVolumeUpdate(newVol)
				if updateErr != nil {
					// We will get an "volume added" event soon, this is not a big error
					klog.V(4).Infof("provisionClaimOperation [%s]: cannot update internal cache: %v", volume.Name, updateErr)
				}
			}
			break
		}
		// Save failed, try again after a while.
		klog.V(3).Infof("failed to save volume %q for claim %q: %v", volume.Name, claimToClaimKey(claim), err)
		time.Sleep(ctrl.createProvisionedPVInterval)
	}

	if err != nil {
		// Save failed. Now we have a storage asset outside of Kubernetes,
		// but we don't have appropriate PV object for it.
		// Emit some event here and try to delete the storage asset several
		// times.
		strerr := fmt.Sprintf("Error creating provisioned PV object for claim %s: %v. Deleting the volume.", claimToClaimKey(claim), err)
		klog.V(3).Info(strerr)
		ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, events.ProvisioningFailed, strerr)

		var deleteErr error
		var deleted bool
		for i := 0; i < ctrl.createProvisionedPVRetryCount; i++ {
			_, deleted, deleteErr = ctrl.doDeleteVolume(volume)
			if deleteErr == nil && deleted {
				// Delete succeeded
				klog.V(4).Infof("provisionClaimOperation [%s]: cleaning volume %s succeeded", claimToClaimKey(claim), volume.Name)
				break
			}
			if !deleted {
				// This is unreachable code, the volume was provisioned by an
				// internal plugin and therefore there MUST be an internal
				// plugin that deletes it.
				klog.Errorf("Error finding internal deleter for volume plugin %q", plugin.GetPluginName())
				break
			}
			// Delete failed, try again after a while.
			klog.V(3).Infof("failed to delete volume %q: %v", volume.Name, deleteErr)
			time.Sleep(ctrl.createProvisionedPVInterval)
		}

		if deleteErr != nil {
			// Delete failed several times. There is an orphaned volume and there
			// is nothing we can do about it.
			strerr := fmt.Sprintf("Error cleaning provisioned volume for claim %s: %v. Please delete manually.", claimToClaimKey(claim), deleteErr)
			klog.V(2).Info(strerr)
			ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, events.ProvisioningCleanupFailed, strerr)
		}
	} else {
		klog.V(2).Infof("volume %q provisioned for claim %q", volume.Name, claimToClaimKey(claim))
		msg := fmt.Sprintf("Successfully provisioned volume %s using %s", volume.Name, plugin.GetPluginName())
		ctrl.eventRecorder.Event(claim, v1.EventTypeNormal, events.ProvisioningSucceeded, msg)
	}
	return pluginName, nil
}

// provisionClaimOperationExternal provisions a volume using external provisioner async-ly
// This method will be running in a standalone go-routine scheduled in "provisionClaim"
func (ctrl *PersistentVolumeController) provisionClaimOperationExternal(
	claim *v1.PersistentVolumeClaim,
	storageClass *storage.StorageClass) (string, error) {
	claimClass := v1helper.GetPersistentVolumeClaimClass(claim)
	klog.V(4).Infof("provisionClaimOperationExternal [%s] started, class: %q", claimToClaimKey(claim), claimClass)
	// Set provisionerName to external provisioner name by setClaimProvisioner
	var err error
	provisionerName := storageClass.Provisioner
	if ctrl.csiMigratedPluginManager.IsMigrationEnabledForPlugin(storageClass.Provisioner) {
		// update the provisioner name to use the migrated CSI plugin name
		provisionerName, err = ctrl.translator.GetCSINameFromInTreeName(storageClass.Provisioner)
		if err != nil {
			strerr := fmt.Sprintf("error getting CSI name for In tree plugin %s: %v", storageClass.Provisioner, err)
			klog.V(2).Infof("%s", strerr)
			ctrl.eventRecorder.Event(claim, v1.EventTypeWarning, events.ProvisioningFailed, strerr)
			return provisionerName, err
		}
	}
	// Add provisioner annotation so external provisioners know when to start
	newClaim, err := ctrl.setClaimProvisioner(claim, provisionerName)
	if err != nil {
		// Save failed, the controller will retry in the next sync
		klog.V(2).Infof("error saving claim %s: %v", claimToClaimKey(claim), err)
		return provisionerName, err
	}
	claim = newClaim
	msg := fmt.Sprintf("waiting for a volume to be created, either by external provisioner %q or manually created by system administrator", provisionerName)
	// External provisioner has been requested for provisioning the volume
	// Report an event and wait for external provisioner to finish
	ctrl.eventRecorder.Event(claim, v1.EventTypeNormal, events.ExternalProvisioning, msg)
	klog.V(3).Infof("provisionClaimOperationExternal provisioning claim %q: %s", claimToClaimKey(claim), msg)
	// return provisioner name here for metric reporting
	return provisionerName, nil
}

// rescheduleProvisioning signal back to the scheduler to retry dynamic provisioning
// by removing the AnnSelectedNode annotation
func (ctrl *PersistentVolumeController) rescheduleProvisioning(claim *v1.PersistentVolumeClaim) {
	if _, ok := claim.Annotations[pvutil.AnnSelectedNode]; !ok {
		// Provisioning not triggered by the scheduler, skip
		return
	}

	// The claim from method args can be pointing to watcher cache. We must not
	// modify these, therefore create a copy.
	newClaim := claim.DeepCopy()
	delete(newClaim.Annotations, pvutil.AnnSelectedNode)
	// Try to update the PVC object
	if _, err := ctrl.kubeClient.CoreV1().PersistentVolumeClaims(newClaim.Namespace).Update(context.TODO(), newClaim, metav1.UpdateOptions{}); err != nil {
		klog.V(4).Infof("Failed to delete annotation 'pvutil.AnnSelectedNode' for PersistentVolumeClaim %q: %v", claimToClaimKey(newClaim), err)
		return
	}
	if _, err := ctrl.storeClaimUpdate(newClaim); err != nil {
		// We will get an "claim updated" event soon, this is not a big error
		klog.V(4).Infof("Updating PersistentVolumeClaim %q: cannot update internal cache: %v", claimToClaimKey(newClaim), err)
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
	klog.V(4).Infof("scheduleOperation[%s]", operationName)

	// Poke test code that an operation is just about to get started.
	if ctrl.preOperationHook != nil {
		ctrl.preOperationHook(operationName)
	}

	err := ctrl.runningOperations.Run(operationName, operation)
	if err != nil {
		switch {
		case goroutinemap.IsAlreadyExists(err):
			klog.V(4).Infof("operation %q is already running, skipping", operationName)
		case exponentialbackoff.IsExponentialBackoff(err):
			klog.V(4).Infof("operation %q postponed due to exponential backoff", operationName)
		default:
			klog.Errorf("error scheduling operation %q: %v", operationName, err)
		}
	}
}

// newRecyclerEventRecorder returns a RecycleEventRecorder that sends all events
// to given volume.
func (ctrl *PersistentVolumeController) newRecyclerEventRecorder(volume *v1.PersistentVolume) recyclerclient.RecycleEventRecorder {
	return func(eventtype, message string) {
		ctrl.eventRecorder.Eventf(volume, eventtype, events.RecyclerPod, "Recycler pod: %s", message)
	}
}

// findProvisionablePlugin finds a provisioner plugin for a given claim.
// It returns either the provisioning plugin or nil when an external
// provisioner is requested.
func (ctrl *PersistentVolumeController) findProvisionablePlugin(claim *v1.PersistentVolumeClaim) (vol.ProvisionableVolumePlugin, *storage.StorageClass, error) {
	// provisionClaim() which leads here is never called with claimClass=="", we
	// can save some checks.
	claimClass := v1helper.GetPersistentVolumeClaimClass(claim)
	class, err := ctrl.classLister.Get(claimClass)
	if err != nil {
		return nil, nil, err
	}

	// Find a plugin for the class
	if ctrl.csiMigratedPluginManager.IsMigrationEnabledForPlugin(class.Provisioner) {
		// CSI migration scenario - do not depend on in-tree plugin
		return nil, class, nil
	}
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

// findDeletablePlugin finds a deleter plugin for a given volume. It returns
// either the deleter plugin or nil when an external deleter is requested.
func (ctrl *PersistentVolumeController) findDeletablePlugin(volume *v1.PersistentVolume) (vol.DeletableVolumePlugin, error) {
	// Find a plugin. Try to find the same plugin that provisioned the volume
	var plugin vol.DeletableVolumePlugin
	if metav1.HasAnnotation(volume.ObjectMeta, pvutil.AnnDynamicallyProvisioned) {
		provisionPluginName := volume.Annotations[pvutil.AnnDynamicallyProvisioned]
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

// obtain provisioner/deleter name for a volume
func (ctrl *PersistentVolumeController) getProvisionerNameFromVolume(volume *v1.PersistentVolume) string {
	plugin, err := ctrl.findDeletablePlugin(volume)
	if err != nil {
		return "N/A"
	}
	if plugin != nil {
		return plugin.GetPluginName()
	}
	// If reached here, Either an external provisioner was used for provisioning
	// or a plugin has been migrated to CSI.
	// If an external provisioner was used, i.e., plugin == nil, instead of using
	// the AnnDynamicallyProvisioned annotation value, use the storageClass's Provisioner
	// field to avoid explosion of the metric in the cases like local storage provisioner
	// tagging a volume with arbitrary provisioner names
	storageClass := v1helper.GetPersistentVolumeClass(volume)
	class, err := ctrl.classLister.Get(storageClass)
	if err != nil {
		return "N/A"
	}
	if ctrl.csiMigratedPluginManager.IsMigrationEnabledForPlugin(class.Provisioner) {
		provisionerName, err := ctrl.translator.GetCSINameFromInTreeName(class.Provisioner)
		if err != nil {
			return "N/A"
		}
		return provisionerName
	}
	return class.Provisioner
}

// obtain plugin/external provisioner name from plugin and storage class for timestamp logging purposes
func (ctrl *PersistentVolumeController) getProvisionerName(plugin vol.ProvisionableVolumePlugin, storageClass *storage.StorageClass) string {
	// non CSI-migrated in-tree plugin, returns the plugin's name
	if plugin != nil {
		return plugin.GetPluginName()
	}
	if ctrl.csiMigratedPluginManager.IsMigrationEnabledForPlugin(storageClass.Provisioner) {
		// get the name of the CSI plugin that the in-tree storage class
		// provisioner has migrated to
		provisionerName, err := ctrl.translator.GetCSINameFromInTreeName(storageClass.Provisioner)
		if err != nil {
			return "N/A"
		}
		return provisionerName
	}
	return storageClass.Provisioner
}
