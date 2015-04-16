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
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

// PersistentVolumeClaimBinder is a controller that synchronizes PersistentVolumeClaims.
type PersistentVolumeClaimBinder struct {
	volumeStore *persistentVolumeOrderedIndex
	claimStore  cache.Store
	client      client.Interface
	// protects access to binding
	lock sync.RWMutex
}

// NewPersistentVolumeClaimBinder creates a new PersistentVolumeClaimBinder
func NewPersistentVolumeClaimBinder(kubeClient client.Interface) *PersistentVolumeClaimBinder {
	volumeStore := NewPersistentVolumeOrderedIndex()
	volumeReflector := cache.NewReflector(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return kubeClient.PersistentVolumes().List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return kubeClient.PersistentVolumes().Watch(labels.Everything(), fields.Everything(), resourceVersion)
			},
		},
		&api.PersistentVolume{},
		volumeStore,
		0,
	)
	volumeReflector.Run()

	claimStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	claimReflector := cache.NewReflector(
		&cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return kubeClient.PersistentVolumeClaims(api.NamespaceAll).List(labels.Everything(), fields.Everything())
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return kubeClient.PersistentVolumeClaims(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), resourceVersion)
			},
		},
		&api.PersistentVolumeClaim{},
		claimStore,
		0,
	)
	claimReflector.Run()

	binder := &PersistentVolumeClaimBinder{
		volumeStore: volumeStore,
		claimStore:  claimStore,
		client:      kubeClient,
	}

	return binder
}

// syncPersistentVolume inspects all bound PVs to determine if their bound PersistentVolumeClaim still exists.
func (controller *PersistentVolumeClaimBinder) syncPersistentVolume(obj interface{}) error {
	volume := obj.(*api.PersistentVolume)
	glog.V(5).Infof("Synchronizing PersistentVolume[%s]%s\n", volume.Name)

	if volume.Spec.ClaimRef != nil {
		if volume.Status.Phase == api.VolumeAvailable {
			volume.Status.Phase = api.VolumeBound
			_, err := controller.client.PersistentVolumes().Update(volume)
			if err != nil {
				return fmt.Errorf("Error updating pv.status: %v\n", err)
			}
		}

		// verify the volume is still claimed by a user
		if claim, err := controller.client.PersistentVolumeClaims(volume.Spec.ClaimRef.Namespace).Get(volume.Spec.ClaimRef.Name); err == nil {
			glog.V(5).Infof("PersistentVolume[%s] is bound to PersistentVolumeClaim[%s]\n", volume.Name, volume.Spec.ClaimRef.Name)
			controller.syncPersistentVolumeClaimStatus(volume, claim)
		} else {
			//claim was deleted by user.
			glog.V(3).Infof("PersistentVolumeClaim[%s] unbound from PersistentVolume[%s]\n", volume.Spec.ClaimRef.Name, volume.Name)
			// volume.Spec.ClaimRef is deliberately left non-nil so that another process can recycle the newly release volume
			volume.Status.Phase = api.VolumeReleased
			volume, err = controller.client.PersistentVolumes().UpdateStatus(volume)
			if err != nil {
				return fmt.Errorf("Error updating pv: %+v\n", err)
			}
		}
	}
	return nil
}

func (controller *PersistentVolumeClaimBinder) syncPersistentVolumeClaim(obj interface{}) error {
	controller.lock.Lock()
	defer controller.lock.Unlock()

	claim := obj.(*api.PersistentVolumeClaim)
	glog.V(5).Infof("Synchronizing PersistentVolumeClaim[%s]\n", claim.Name)

	if claim.Status.VolumeRef != nil {
		glog.V(5).Infof("PersistentVolumeClaim[%s] is bound to PersistentVolume[%s]\n", claim.Name, claim.Status.VolumeRef.Name)
		return nil
	}

	pv, err := controller.volumeStore.FindBestMatchForClaim(claim)
	if err != nil {
		return err
	}

	if pv != nil {
		claimRef, err := api.GetReference(claim)
		if err != nil {
			return fmt.Errorf("Unexpected error getting claim reference: %v\n", err)
		}

		// make a binding reference to the claim
		pv.Spec.ClaimRef = claimRef
		pv, err = controller.client.PersistentVolumes().Update(pv)

		if err != nil {
			// volume no longer bound
			pv.Spec.ClaimRef = nil
			return fmt.Errorf("Error updating volume: %+v\n", err)
		} else {
			glog.V(3).Infof("PersistentVolumeClaim[%s] bound to PersistentVolume[%s]\n", claim.Name, pv.Name)
			pv.Status.Phase = api.VolumeBound
			err := controller.syncPersistentVolumeClaimStatus(pv, claim)
			if err != nil {
				return fmt.Errorf("Error updating pvclaim.status: %v\n", err)
			}
		}
	} else {
		glog.V(5).Infof("No volume match found for PersistentVolumeClaim[%s]\n", claim.UID)
	}
	return nil
}

// syncPersistentVolumeClaimStatus builds and persistens a PVClaim's Status, rolling back to empty values if the update fails
func (controller *PersistentVolumeClaimBinder) syncPersistentVolumeClaimStatus(volume *api.PersistentVolume, claim *api.PersistentVolumeClaim) error {
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

	_, err = controller.client.PersistentVolumeClaims(claim.Namespace).UpdateStatus(claim)

	if err != nil {
		claim.Status.Phase = api.ClaimPending
		claim.Status.VolumeRef = nil
		claim.Status.AccessModes = nil
		claim.Status.Capacity = nil
	}

	return err
}

func (controller *PersistentVolumeClaimBinder) Run(period time.Duration) {
	glog.V(5).Infof("Starting PersistentVolumeClaimBinder\n")
	go util.Forever(func() { controller.synchronize() }, period)
}

// Synchronizer is a generic List/ProcessFunc used by the Reconcile function & reconciliation loop,
// because we're reconciling two Kinds in this component and I didn't want to dupe the loop
type Synchronizer struct {
	ListFunc      func() []interface{}
	ReconcileFunc func(interface{}) error
}

func (controller *PersistentVolumeClaimBinder) synchronize() {
	volumeSynchronizer := Synchronizer{
		ListFunc:      controller.volumeStore.List,
		ReconcileFunc: controller.syncPersistentVolume,
	}

	claimsSynchronizer := Synchronizer{
		ListFunc:      controller.claimStore.List,
		ReconcileFunc: controller.syncPersistentVolumeClaim,
	}

	controller.reconcile(volumeSynchronizer, claimsSynchronizer)
}

func (controller *PersistentVolumeClaimBinder) reconcile(synchronizers ...Synchronizer) {
	for _, synchronizer := range synchronizers {
		items := synchronizer.ListFunc()
		if len(items) == 0 {
			continue
		}
		wg := sync.WaitGroup{}
		wg.Add(len(items))
		for ix := range items {
			func(ix int) {
				defer wg.Done()
				obj := items[ix]
				glog.V(5).Infof("Reconciliation of %v", obj)
				err := synchronizer.ReconcileFunc(obj)
				if err != nil {
					glog.Errorf("Error reconciling: %v", err)
				}
			}(ix)
		}
		wg.Wait()
	}
}
