/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package volumeclaimbinder

import (
	"fmt"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/framework"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// PersistentVolumeClaimBinder is a controller that synchronizes PersistentVolumeClaims.
type PersistentVolumeClaimBinder struct {
	volumeIndex      *persistentVolumeOrderedIndex
	volumeController *framework.Controller
	claimController  *framework.Controller
	client           binderClient
	stopChannels     map[string]chan struct{}
	lock             sync.RWMutex
}

// NewPersistentVolumeClaimBinder creates a new PersistentVolumeClaimBinder
func NewPersistentVolumeClaimBinder(kubeClient client.Interface, syncPeriod time.Duration) *PersistentVolumeClaimBinder {
	volumeIndex := NewPersistentVolumeOrderedIndex()
	binderClient := NewBinderClient(kubeClient)
	binder := &PersistentVolumeClaimBinder{
		volumeIndex: volumeIndex,
		client:      binderClient,
	}

	_, volumeController := framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return kubeClient.PersistentVolumes().List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return kubeClient.PersistentVolumes().Watch(labels.Everything(), fields.Everything(), resourceVersion)
			},
		},
		&api.PersistentVolume{},
		syncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    binder.addVolume,
			UpdateFunc: binder.updateVolume,
			DeleteFunc: binder.deleteVolume,
		},
	)
	_, claimController := framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return kubeClient.PersistentVolumeClaims(api.NamespaceAll).List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return kubeClient.PersistentVolumeClaims(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), resourceVersion)
			},
		},
		&api.PersistentVolumeClaim{},
		syncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    binder.addClaim,
			UpdateFunc: binder.updateClaim,
			// no DeleteFunc needed.  a claim requires no clean-up.
			// syncVolume handles the missing claim
		},
	)

	binder.claimController = claimController
	binder.volumeController = volumeController

	return binder
}

func (binder *PersistentVolumeClaimBinder) addVolume(obj interface{}) {
	binder.lock.Lock()
	defer binder.lock.Unlock()
	volume := obj.(*api.PersistentVolume)
	err := syncVolume(binder.volumeIndex, binder.client, volume)
	if err != nil {
		glog.Errorf("PVClaimBinder could not add volume %s: %+v", volume.Name, err)
	}
}

func (binder *PersistentVolumeClaimBinder) updateVolume(oldObj, newObj interface{}) {
	binder.lock.Lock()
	defer binder.lock.Unlock()
	newVolume := newObj.(*api.PersistentVolume)
	binder.volumeIndex.Update(newVolume)
	err := syncVolume(binder.volumeIndex, binder.client, newVolume)
	if err != nil {
		glog.Errorf("PVClaimBinder could not update volume %s: %+v", newVolume.Name, err)
	}
}

func (binder *PersistentVolumeClaimBinder) deleteVolume(obj interface{}) {
	binder.lock.Lock()
	defer binder.lock.Unlock()
	volume := obj.(*api.PersistentVolume)
	binder.volumeIndex.Delete(volume)
}

func (binder *PersistentVolumeClaimBinder) addClaim(obj interface{}) {
	binder.lock.Lock()
	defer binder.lock.Unlock()
	claim := obj.(*api.PersistentVolumeClaim)
	err := syncClaim(binder.volumeIndex, binder.client, claim)
	if err != nil {
		glog.Errorf("PVClaimBinder could not add claim %s: %+v", claim.Name, err)
	}
}

func (binder *PersistentVolumeClaimBinder) updateClaim(oldObj, newObj interface{}) {
	binder.lock.Lock()
	defer binder.lock.Unlock()
	newClaim := newObj.(*api.PersistentVolumeClaim)
	err := syncClaim(binder.volumeIndex, binder.client, newClaim)
	if err != nil {
		glog.Errorf("PVClaimBinder could not update claim %s: %+v", newClaim.Name, err)
	}
}

func syncVolume(volumeIndex *persistentVolumeOrderedIndex, binderClient binderClient, volume *api.PersistentVolume) (err error) {
	glog.V(5).Infof("Synchronizing PersistentVolume[%s], current phase: %s\n", volume.Name, volume.Status.Phase)

	// volumes can be in one of the following states:
	//
	// VolumePending -- default value -- not bound to a claim and not yet processed through this controller.
	// VolumeAvailable -- not bound to a claim, but processed at least once and found in this controller's volumeIndex.
	// VolumeBound -- bound to a claim because volume.Spec.ClaimRef != nil.   Claim status may not be correct.
	// VolumeReleased -- volume.Spec.ClaimRef != nil but the claim has been deleted by the user.
	// VolumeFailed -- volume.Spec.ClaimRef != nil and the volume failed processing in the recycler
	currentPhase := volume.Status.Phase
	nextPhase := currentPhase

	switch currentPhase {
	// pending volumes are available only after indexing in order to be matched to claims.
	case api.VolumePending:
		if volume.Spec.ClaimRef != nil {
			// Pending volumes that have a ClaimRef were recently recycled.  The Recycler set the phase to VolumePending
			// to start the volume again at the beginning of this lifecycle.
			// ClaimRef is the last bind between persistent volume and claim.
			// The claim has already been deleted by the user at this point
			oldClaimRef := volume.Spec.ClaimRef
			volume.Spec.ClaimRef = nil
			_, err = binderClient.UpdatePersistentVolume(volume)
			if err != nil {
				// rollback on error, keep the ClaimRef until we can successfully update the volume
				volume.Spec.ClaimRef = oldClaimRef
				return fmt.Errorf("Unexpected error saving PersistentVolume: %+v", err)
			}
		}

		_, exists, err := volumeIndex.Get(volume)
		if err != nil {
			return err
		}
		if !exists {
			volumeIndex.Add(volume)
		}
		glog.V(5).Infof("PersistentVolume[%s] is now available\n", volume.Name)
		nextPhase = api.VolumeAvailable

	// available volumes await a claim
	case api.VolumeAvailable:
		if volume.Spec.ClaimRef != nil {
			_, err := binderClient.GetPersistentVolumeClaim(volume.Spec.ClaimRef.Namespace, volume.Spec.ClaimRef.Name)
			if err == nil {
				// change of phase will trigger an update event with the newly bound volume
				glog.V(5).Infof("PersistentVolume[%s] is now bound\n", volume.Name)
				nextPhase = api.VolumeBound
			} else {
				if errors.IsNotFound(err) {
					nextPhase = api.VolumeReleased
				}
			}
		}

	//bound volumes require verification of their bound claims
	case api.VolumeBound:
		if volume.Spec.ClaimRef == nil {
			return fmt.Errorf("PersistentVolume[%s] expected to be bound but found nil claimRef: %+v", volume.Name, volume)
		} else {
			_, err := binderClient.GetPersistentVolumeClaim(volume.Spec.ClaimRef.Namespace, volume.Spec.ClaimRef.Name)
			if err != nil {
				if errors.IsNotFound(err) {
					nextPhase = api.VolumeReleased
				} else {
					return err
				}
			}
		}

	// released volumes require recycling
	case api.VolumeReleased:
		if volume.Spec.ClaimRef == nil {
			return fmt.Errorf("PersistentVolume[%s] expected to be bound but found nil claimRef: %+v", volume.Name, volume)
		} else {
			// another process is watching for released volumes.
			// PersistentVolumeReclaimPolicy is set per PersistentVolume
		}

	// volumes are removed by processes external to this binder and must be removed from the cluster
	case api.VolumeFailed:
		if volume.Spec.ClaimRef == nil {
			return fmt.Errorf("PersistentVolume[%s] expected to be bound but found nil claimRef: %+v", volume)
		} else {
			glog.V(5).Infof("PersistentVolume[%s] previously failed recycling.  Skipping.\n", volume.Name)
		}
	}

	if currentPhase != nextPhase {
		volume.Status.Phase = nextPhase

		// a change in state will trigger another update through this controller.
		// each pass through this controller evaluates current phase and decides whether or not to change to the next phase
		glog.V(5).Infof("PersistentVolume[%s] changing phase from %s to %s\n", volume.Name, currentPhase, nextPhase)
		volume, err := binderClient.UpdatePersistentVolumeStatus(volume)
		if err != nil {
			// Rollback to previous phase
			volume.Status.Phase = currentPhase
		}
		volumeIndex.Update(volume)
	}

	return nil
}

func syncClaim(volumeIndex *persistentVolumeOrderedIndex, binderClient binderClient, claim *api.PersistentVolumeClaim) (err error) {
	glog.V(5).Infof("Synchronizing PersistentVolumeClaim[%s]\n", claim.Name)

	// claims can be in one of the following states:
	//
	// ClaimPending -- default value -- not bound to a claim.  A volume that matches the claim may not exist.
	// ClaimBound -- bound to a volume.  claim.Status.VolumeRef != nil
	currentPhase := claim.Status.Phase
	nextPhase := currentPhase

	switch currentPhase {
	// pending claims await a matching volume
	case api.ClaimPending:
		volume, err := volumeIndex.FindBestMatchForClaim(claim)
		if err != nil {
			return err
		}
		if volume == nil {
			return fmt.Errorf("A volume match does not exist for persistent claim: %s", claim.Name)
		}

		// make a binding reference to the claim.
		// triggers update of the claim in this controller, which builds claim status
		claim.Spec.VolumeName = volume.Name
		// TODO: make this similar to Pod's binding both with BindingREST subresource and GuaranteedUpdate helper in etcd.go
		claim, err = binderClient.UpdatePersistentVolumeClaim(claim)
		if err == nil {
			nextPhase = api.ClaimBound
			glog.V(5).Infof("PersistentVolumeClaim[%s] is bound\n", claim.Name)
		} else {
			// Rollback by unsetting the ClaimRef on the volume pointer.
			// the volume in the index will be unbound again and ready to be matched.
			claim.Spec.VolumeName = ""
			// Rollback by restoring original phase to claim pointer
			nextPhase = api.ClaimPending
			return fmt.Errorf("Error updating volume: %+v\n", err)
		}

	case api.ClaimBound:
		volume, err := binderClient.GetPersistentVolume(claim.Spec.VolumeName)
		if err != nil {
			return fmt.Errorf("Unexpected error getting persistent volume: %v\n", err)
		}

		if volume.Spec.ClaimRef == nil {
			glog.V(5).Infof("Rebuilding bind on pv.Spec.ClaimRef\n")
			claimRef, err := api.GetReference(claim)
			if err != nil {
				return fmt.Errorf("Unexpected error getting claim reference: %v\n", err)
			}
			volume.Spec.ClaimRef = claimRef
			_, err = binderClient.UpdatePersistentVolume(volume)
			if err != nil {
				return fmt.Errorf("Unexpected error saving PersistentVolume.Status: %+v", err)
			}
		}

		// all "actuals" are transferred from PV to PVC so the user knows what
		// type of volume they actually got for their claim.
		// Volumes cannot have zero AccessModes, so checking that a claim has access modes
		// is sufficient to tell us if these values have already been set.
		if len(claim.Status.AccessModes) == 0 {
			claim.Status.Phase = api.ClaimBound
			claim.Status.AccessModes = volume.Spec.AccessModes
			claim.Status.Capacity = volume.Spec.Capacity
			_, err := binderClient.UpdatePersistentVolumeClaimStatus(claim)
			if err != nil {
				return fmt.Errorf("Unexpected error saving claim status: %+v", err)
			}
		}
	}

	if currentPhase != nextPhase {
		claim.Status.Phase = nextPhase
		binderClient.UpdatePersistentVolumeClaimStatus(claim)
	}
	return nil
}

// Run starts all of this binder's control loops
func (controller *PersistentVolumeClaimBinder) Run() {
	glog.V(5).Infof("Starting PersistentVolumeClaimBinder\n")
	if controller.stopChannels == nil {
		controller.stopChannels = make(map[string]chan struct{})
	}

	if _, exists := controller.stopChannels["volumes"]; !exists {
		controller.stopChannels["volumes"] = make(chan struct{})
		go controller.volumeController.Run(controller.stopChannels["volumes"])
	}

	if _, exists := controller.stopChannels["claims"]; !exists {
		controller.stopChannels["claims"] = make(chan struct{})
		go controller.claimController.Run(controller.stopChannels["claims"])
	}
}

// Stop gracefully shuts down this binder
func (controller *PersistentVolumeClaimBinder) Stop() {
	glog.V(5).Infof("Stopping PersistentVolumeClaimBinder\n")
	for name, stopChan := range controller.stopChannels {
		close(stopChan)
		delete(controller.stopChannels, name)
	}
}

// binderClient abstracts access to PVs and PVCs
type binderClient interface {
	GetPersistentVolume(name string) (*api.PersistentVolume, error)
	UpdatePersistentVolume(volume *api.PersistentVolume) (*api.PersistentVolume, error)
	DeletePersistentVolume(volume *api.PersistentVolume) error
	UpdatePersistentVolumeStatus(volume *api.PersistentVolume) (*api.PersistentVolume, error)
	GetPersistentVolumeClaim(namespace, name string) (*api.PersistentVolumeClaim, error)
	UpdatePersistentVolumeClaim(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error)
	UpdatePersistentVolumeClaimStatus(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error)
}

func NewBinderClient(c client.Interface) binderClient {
	return &realBinderClient{c}
}

type realBinderClient struct {
	client client.Interface
}

func (c *realBinderClient) GetPersistentVolume(name string) (*api.PersistentVolume, error) {
	return c.client.PersistentVolumes().Get(name)
}

func (c *realBinderClient) UpdatePersistentVolume(volume *api.PersistentVolume) (*api.PersistentVolume, error) {
	return c.client.PersistentVolumes().Update(volume)
}

func (c *realBinderClient) DeletePersistentVolume(volume *api.PersistentVolume) error {
	return c.client.PersistentVolumes().Delete(volume.Name)
}

func (c *realBinderClient) UpdatePersistentVolumeStatus(volume *api.PersistentVolume) (*api.PersistentVolume, error) {
	return c.client.PersistentVolumes().UpdateStatus(volume)
}

func (c *realBinderClient) GetPersistentVolumeClaim(namespace, name string) (*api.PersistentVolumeClaim, error) {
	return c.client.PersistentVolumeClaims(namespace).Get(name)
}

func (c *realBinderClient) UpdatePersistentVolumeClaim(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	return c.client.PersistentVolumeClaims(claim.Namespace).Update(claim)
}

func (c *realBinderClient) UpdatePersistentVolumeClaimStatus(claim *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	return c.client.PersistentVolumeClaims(claim.Namespace).UpdateStatus(claim)
}
