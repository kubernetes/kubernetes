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
			// the missing claim itself is the release of the resource.
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
	syncVolume(binder.volumeIndex, binder.client, volume)
}

func (binder *PersistentVolumeClaimBinder) updateVolume(oldObj, newObj interface{}) {
	binder.lock.Lock()
	defer binder.lock.Unlock()
	newVolume := newObj.(*api.PersistentVolume)
	binder.volumeIndex.Update(newVolume)
	syncVolume(binder.volumeIndex, binder.client, newVolume)
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
	syncClaim(binder.volumeIndex, binder.client, claim)
}

func (binder *PersistentVolumeClaimBinder) updateClaim(oldObj, newObj interface{}) {
	binder.lock.Lock()
	defer binder.lock.Unlock()
	newClaim := newObj.(*api.PersistentVolumeClaim)
	syncClaim(binder.volumeIndex, binder.client, newClaim)
}

func syncVolume(volumeIndex *persistentVolumeOrderedIndex, binderClient binderClient, volume *api.PersistentVolume) (err error) {
	glog.V(5).Infof("Synchronizing PersistentVolume[%s]\n", volume.Name)

	// volumes can be in one of the following states:
	//
	// VolumePending -- default value -- not bound to a claim and not yet processed through this controller.
	// VolumeAvailable -- not bound to a claim, but processed at least once and found in this controller's volumeIndex.
	// VolumeBound -- bound to a claim because volume.Spec.ClaimRef != nil.   Claim status may not be correct.
	// VolumeReleased -- volume.Spec.ClaimRef != nil but the claim has been deleted by the user.
	currentPhase := volume.Status.Phase
	nextPhase := currentPhase

	switch currentPhase {
	// pending volumes are available only after indexing in order to be matched to claims.
	case api.VolumePending:
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
			return fmt.Errorf("PersistentVolume[%s] expected to be bound but found nil claimRef: %+v", volume)
		} else {
			claim, err := binderClient.GetPersistentVolumeClaim(volume.Spec.ClaimRef.Namespace, volume.Spec.ClaimRef.Name)
			if err == nil {
				// bound and active.  Build claim status as needed.
				if claim.Status.VolumeRef == nil {
					// syncClaimStatus sets VolumeRef, attempts to persist claim status,
					// and does a rollback as needed on claim.Status
					syncClaimStatus(binderClient, volume, claim)
				}
			} else {
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
			return fmt.Errorf("PersistentVolume[%s] expected to be bound but found nil claimRef: %+v", volume)
		} else {
			// TODO: implement Recycle method on plugins
		}
	}

	if currentPhase != nextPhase {
		volume.Status.Phase = nextPhase

		// a change in state will trigger another update through this controller.
		// each pass through this controller evaluates current phase and decides whether or not to change to the next phase
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

		claimRef, err := api.GetReference(claim)
		if err != nil {
			return fmt.Errorf("Unexpected error getting claim reference: %v\n", err)
		}

		// make a binding reference to the claim.
		// triggers update of the volume in this controller, which builds claim status
		volume.Spec.ClaimRef = claimRef
		volume, err = binderClient.UpdatePersistentVolume(volume)
		if err == nil {
			nextPhase = api.ClaimBound
		}
		if err != nil {
			// Rollback by unsetting the ClaimRef on the volume pointer.
			// the volume in the index will be unbound again and ready to be matched.
			volume.Spec.ClaimRef = nil
			// Rollback by restoring original phase to claim pointer
			nextPhase = api.ClaimPending
			return fmt.Errorf("Error updating volume: %+v\n", err)
		}

	// bound claims requires no maintenance.  Deletion by the user is the last lifecycle phase.
	case api.ClaimBound:
		// This is the end of a claim's lifecycle.
		// After claim deletion, a volume is recycled when it verifies its claim is unbound
		glog.V(5).Infof("PersistentVolumeClaime[%s] is bound\n", claim.Name)
	}

	if currentPhase != nextPhase {
		claim.Status.Phase = nextPhase
		binderClient.UpdatePersistentVolumeClaimStatus(claim)
	}
	return nil
}

func syncClaimStatus(binderClient binderClient, volume *api.PersistentVolume, claim *api.PersistentVolumeClaim) (err error) {
	volumeRef, err := api.GetReference(volume)
	if err != nil {
		return fmt.Errorf("Unexpected error getting volume reference: %v\n", err)
	}

	// all "actuals" are transferred from PV to PVC so the user knows what
	// type of volume they actually got for their claim
	claim.Status.Phase = api.ClaimBound
	claim.Status.VolumeRef = volumeRef
	claim.Status.AccessModes = volume.Spec.AccessModes
	claim.Status.Capacity = volume.Spec.Capacity

	_, err = binderClient.UpdatePersistentVolumeClaimStatus(claim)
	if err != nil {
		claim.Status.Phase = api.ClaimPending
		claim.Status.VolumeRef = nil
		claim.Status.AccessModes = nil
		claim.Status.Capacity = nil
	}
	return err
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
