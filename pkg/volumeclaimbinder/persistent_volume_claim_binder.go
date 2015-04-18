/*
Copyright 2014 Google Inc. All rights reserved.

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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/framework"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/golang/glog"
	"reflect"
)

// PersistentVolumeClaimBinder is a controller that synchronizes PersistentVolumeClaims.
type PersistentVolumeClaimBinder struct {
	volumeIndex      *persistentVolumeOrderedIndex
	volumeController *framework.Controller
	claimController  *framework.Controller
	client           binderClient
}

// NewPersistentVolumeClaimBinder creates a new PersistentVolumeClaimBinder
func NewPersistentVolumeClaimBinder(kubeClient client.Interface, syncPeriod time.Duration) *PersistentVolumeClaimBinder {
	volumeIndex := NewPersistentVolumeOrderedIndex()
	binderClient := NewBinderClient(kubeClient)

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
			AddFunc: func(obj interface{}) {
				volume := obj.(*api.PersistentVolume)
				volumeIndex.Indexer.Add(volume)
				syncVolume(binderClient, volume)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				oldVolume := oldObj.(*api.PersistentVolume)
				newVolume := newObj.(*api.PersistentVolume)
				volumeIndex.Indexer.Update(newVolume)
				if updateRequired(oldVolume, newVolume) {
					syncVolume(binderClient, newVolume)
				}
			},
			DeleteFunc: func(obj interface{}) {
				volume := obj.(*api.PersistentVolume)
				volumeIndex.Indexer.Delete(volume)
			},
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
			AddFunc: func(obj interface{}) {
				claim := obj.(*api.PersistentVolumeClaim)
				syncClaim(volumeIndex, binderClient, claim)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				//				oldClaim := newObj.(*api.PersistentVolumeClaim)
				newClaim := newObj.(*api.PersistentVolumeClaim)
				if newClaim.Status.VolumeRef == nil {
					syncClaim(volumeIndex, binderClient, newClaim)
				}
			},
		},
	)

	binder := &PersistentVolumeClaimBinder{
		volumeController: volumeController,
		claimController:  claimController,
		volumeIndex:      volumeIndex,
		client:           binderClient,
	}

	return binder
}

func updateRequired(oldVolume, newVolume *api.PersistentVolume) bool {
	// Spec changes affect indexing and sorting volumes
	if !reflect.DeepEqual(oldVolume.Spec, newVolume.Spec) {
		return true
	}
	if !reflect.DeepEqual(oldVolume.Status, newVolume.Status) {
		return true
	}
	return false
}

func syncVolume(binderClient binderClient, volume *api.PersistentVolume) (err error) {
	glog.V(5).Infof("Synchronizing PersistentVolume[%s]\n", volume.Name)

	if volume.Spec.ClaimRef != nil {
		if volume.Status.Phase == api.VolumeAvailable {
			volume.Status.Phase = api.VolumeBound
			_, err := binderClient.UpdatePersistentVolumeStatus(volume)
			if err != nil {
				return fmt.Errorf("Error updating pv.status: %v\n", err)
			}
		}

		// verify the volume is still claimed by a user
		if claim, err := binderClient.GetPersistentVolumeClaim(volume.Spec.ClaimRef.Namespace, volume.Spec.ClaimRef.Name); err == nil {
			glog.V(5).Infof("PersistentVolume[%s] is bound to PersistentVolumeClaim[%s]\n", volume.Name, volume.Spec.ClaimRef.Name)
			// rebuild the Claim's Status as needed
			if claim.Status.VolumeRef == nil {
				syncClaimStatus(binderClient, volume, claim)
			}
		} else {
			//claim was deleted by user.
			glog.V(3).Infof("PersistentVolumeClaim[%s] unbound from PersistentVolume[%s]\n", volume.Spec.ClaimRef.Name, volume.Name)
			// volume.Spec.ClaimRef is deliberately left non-nil so that another process can recycle the newly released volume
			volume.Status.Phase = api.VolumeReleased
			volume, err = binderClient.UpdatePersistentVolumeStatus(volume)
			if err != nil {
				return fmt.Errorf("Error updating pv: %+v\n", err)
			}
		}
	} else {
		volume.Status.Phase = api.VolumeAvailable
		_, err := binderClient.UpdatePersistentVolumeStatus(volume)
		if err != nil {
			return fmt.Errorf("Error updating pv.status: %v\n", err)
		}
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

func syncClaim(volumeIndex *persistentVolumeOrderedIndex, binderClient binderClient, claim *api.PersistentVolumeClaim) (err error) {
	glog.V(5).Infof("Synchronizing PersistentVolumeClaim[%s]\n", claim.Name)

	if claim.Status.VolumeRef != nil {
		glog.V(5).Infof("PersistentVolumeClaim[%s] is bound to PersistentVolume[%s]\n", claim.Name, claim.Status.VolumeRef.Name)
		return nil
	}

	volume, err := volumeIndex.FindBestMatchForClaim(claim)
	if err != nil {
		return err
	}

	if volume != nil {
		claimRef, err := api.GetReference(claim)
		if err != nil {
			return fmt.Errorf("Unexpected error getting claim reference: %v\n", err)
		}

		// make a binding reference to the claim
		volume.Spec.ClaimRef = claimRef
		volume, err = binderClient.UpdatePersistentVolume(volume)

		if err != nil {
			// volume no longer bound
			volume.Spec.ClaimRef = nil
			return fmt.Errorf("Error updating volume: %+v\n", err)
		} else {
			err = syncClaimStatus(binderClient, volume, claim)
			if err != nil {
				return fmt.Errorf("Error update claim.status: %+v\n", err)
			}
		}
	} else {
		glog.V(5).Infof("No volume match found for PersistentVolumeClaim[%s]\n", claim.UID)
		if claim.Status.Phase != api.ClaimPending {
			claim.Status.Phase = api.ClaimPending
			_, err := binderClient.UpdatePersistentVolumeClaimStatus(claim)
			if err != nil {
				return fmt.Errorf("Error updating pvclaim.status: %v\n", err)
			}
		}
	}
	return nil
}

func (controller *PersistentVolumeClaimBinder) Run() {
	glog.V(5).Infof("Starting PersistentVolumeClaimBinder\n")
	go controller.claimController.Run(make(chan struct{}))
	go controller.volumeController.Run(make(chan struct{}))
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
