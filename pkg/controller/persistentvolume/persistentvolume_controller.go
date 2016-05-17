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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	vol "k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

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

func (ctrl *PersistentVolumeController) syncVolume(volume *api.PersistentVolume) error {
	glog.V(4).Infof("synchronizing PersistentVolume[%s], current phase: %s", volume.Name, volume.Status.Phase)

	return nil
}

func (ctrl *PersistentVolumeController) syncClaim(claim *api.PersistentVolumeClaim) error {
	glog.V(4).Infof("synchronizing PersistentVolumeClaim[%s], current phase: %s", claim.Name, claim.Status.Phase)

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
