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
	"strconv"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	unversioned_core "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/goroutinemap"
	vol "k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// This file contains the controller base functionality, i.e. framework to
// process PV/PVC added/updated/deleted events. The real binding, provisioning,
// recycling and deleting is done in controller.go

// NewPersistentVolumeController creates a new PersistentVolumeController
func NewPersistentVolumeController(
	kubeClient clientset.Interface,
	syncPeriod time.Duration,
	provisioner vol.ProvisionableVolumePlugin,
	recyclers []vol.VolumePlugin,
	cloud cloudprovider.Interface,
	clusterName string,
	volumeSource, claimSource cache.ListerWatcher,
	eventRecorder record.EventRecorder,
	enableDynamicProvisioning bool,
) *PersistentVolumeController {

	if eventRecorder == nil {
		broadcaster := record.NewBroadcaster()
		broadcaster.StartRecordingToSink(&unversioned_core.EventSinkImpl{Interface: kubeClient.Core().Events("")})
		eventRecorder = broadcaster.NewRecorder(api.EventSource{Component: "persistentvolume-controller"})
	}

	controller := &PersistentVolumeController{
		volumes:                       newPersistentVolumeOrderedIndex(),
		claims:                        cache.NewStore(framework.DeletionHandlingMetaNamespaceKeyFunc),
		kubeClient:                    kubeClient,
		eventRecorder:                 eventRecorder,
		runningOperations:             goroutinemap.NewGoRoutineMap(false /* exponentialBackOffOnError */),
		cloud:                         cloud,
		provisioner:                   provisioner,
		enableDynamicProvisioning:     enableDynamicProvisioning,
		clusterName:                   clusterName,
		createProvisionedPVRetryCount: createProvisionedPVRetryCount,
		createProvisionedPVInterval:   createProvisionedPVInterval,
	}

	controller.recyclePluginMgr.InitPlugins(recyclers, controller)
	if controller.provisioner != nil {
		if err := controller.provisioner.Init(controller); err != nil {
			glog.Errorf("PersistentVolumeController: error initializing provisioner plugin: %v", err)
		}
	}

	if volumeSource == nil {
		volumeSource = &cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return kubeClient.Core().PersistentVolumes().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return kubeClient.Core().PersistentVolumes().Watch(options)
			},
		}
	}
	controller.volumeSource = volumeSource

	if claimSource == nil {
		claimSource = &cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return kubeClient.Core().PersistentVolumeClaims(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return kubeClient.Core().PersistentVolumeClaims(api.NamespaceAll).Watch(options)
			},
		}
	}
	controller.claimSource = claimSource

	_, controller.volumeController = framework.NewIndexerInformer(
		volumeSource,
		&api.PersistentVolume{},
		syncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    controller.addVolume,
			UpdateFunc: controller.updateVolume,
			DeleteFunc: controller.deleteVolume,
		},
		cache.Indexers{"accessmodes": accessModesIndexFunc},
	)
	_, controller.claimController = framework.NewInformer(
		claimSource,
		&api.PersistentVolumeClaim{},
		syncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    controller.addClaim,
			UpdateFunc: controller.updateClaim,
			DeleteFunc: controller.deleteClaim,
		},
	)
	return controller
}

// initializeCaches fills all controller caches with initial data from etcd in
// order to have the caches already filled when first addClaim/addVolume to
// perform initial synchronization of the controller.
func (ctrl *PersistentVolumeController) initializeCaches(volumeSource, claimSource cache.ListerWatcher) {
	volumeListObj, err := volumeSource.List(api.ListOptions{})
	if err != nil {
		glog.Errorf("PersistentVolumeController can't initialize caches: %v", err)
		return
	}
	volumeList, ok := volumeListObj.(*api.PersistentVolumeList)
	if !ok {
		glog.Errorf("PersistentVolumeController can't initialize caches, expected list of volumes, got: %#v", volumeListObj)
		return
	}
	for _, volume := range volumeList.Items {
		// Ignore template volumes from kubernetes 1.2
		deleted := ctrl.upgradeVolumeFrom1_2(&volume)
		if !deleted {
			clone, err := conversion.NewCloner().DeepCopy(&volume)
			if err != nil {
				glog.Errorf("error cloning volume %q: %v", volume.Name, err)
				continue
			}
			volumeClone := clone.(*api.PersistentVolume)
			ctrl.storeVolumeUpdate(volumeClone)
		}
	}

	claimListObj, err := claimSource.List(api.ListOptions{})
	if err != nil {
		glog.Errorf("PersistentVolumeController can't initialize caches: %v", err)
		return
	}
	claimList, ok := claimListObj.(*api.PersistentVolumeClaimList)
	if !ok {
		glog.Errorf("PersistentVolumeController can't initialize caches, expected list of claims, got: %#v", claimListObj)
		return
	}
	for _, claim := range claimList.Items {
		clone, err := conversion.NewCloner().DeepCopy(&claim)
		if err != nil {
			glog.Errorf("error cloning claim %q: %v", claimToClaimKey(&claim), err)
			continue
		}
		claimClone := clone.(*api.PersistentVolumeClaim)
		ctrl.storeClaimUpdate(claimClone)
	}
	glog.V(4).Infof("controller initialized")
}

func (ctrl *PersistentVolumeController) storeVolumeUpdate(volume *api.PersistentVolume) (bool, error) {
	return storeObjectUpdate(ctrl.volumes.store, volume, "volume")
}

func (ctrl *PersistentVolumeController) storeClaimUpdate(claim *api.PersistentVolumeClaim) (bool, error) {
	return storeObjectUpdate(ctrl.claims, claim, "claim")
}

// addVolume is callback from framework.Controller watching PersistentVolume
// events.
func (ctrl *PersistentVolumeController) addVolume(obj interface{}) {
	pv, ok := obj.(*api.PersistentVolume)
	if !ok {
		glog.Errorf("expected PersistentVolume but handler received %#v", obj)
		return
	}

	if ctrl.upgradeVolumeFrom1_2(pv) {
		// volume deleted
		return
	}

	// Store the new volume version in the cache and do not process it if this
	// is an old version.
	new, err := ctrl.storeVolumeUpdate(pv)
	if err != nil {
		glog.Errorf("%v", err)
	}
	if !new {
		return
	}

	if err := ctrl.syncVolume(pv); err != nil {
		if errors.IsConflict(err) {
			// Version conflict error happens quite often and the controller
			// recovers from it easily.
			glog.V(3).Infof("PersistentVolumeController could not add volume %q: %+v", pv.Name, err)
		} else {
			glog.Errorf("PersistentVolumeController could not add volume %q: %+v", pv.Name, err)
		}
	}
}

// updateVolume is callback from framework.Controller watching PersistentVolume
// events.
func (ctrl *PersistentVolumeController) updateVolume(oldObj, newObj interface{}) {
	newVolume, ok := newObj.(*api.PersistentVolume)
	if !ok {
		glog.Errorf("Expected PersistentVolume but handler received %#v", newObj)
		return
	}

	if ctrl.upgradeVolumeFrom1_2(newVolume) {
		// volume deleted
		return
	}

	// Store the new volume version in the cache and do not process it if this
	// is an old version.
	new, err := ctrl.storeVolumeUpdate(newVolume)
	if err != nil {
		glog.Errorf("%v", err)
	}
	if !new {
		return
	}

	if err := ctrl.syncVolume(newVolume); err != nil {
		if errors.IsConflict(err) {
			// Version conflict error happens quite often and the controller
			// recovers from it easily.
			glog.V(3).Infof("PersistentVolumeController could not update volume %q: %+v", newVolume.Name, err)
		} else {
			glog.Errorf("PersistentVolumeController could not update volume %q: %+v", newVolume.Name, err)
		}
	}
}

// deleteVolume is callback from framework.Controller watching PersistentVolume
// events.
func (ctrl *PersistentVolumeController) deleteVolume(obj interface{}) {
	_ = ctrl.volumes.store.Delete(obj)

	var volume *api.PersistentVolume
	var ok bool
	volume, ok = obj.(*api.PersistentVolume)
	if !ok {
		if unknown, ok := obj.(cache.DeletedFinalStateUnknown); ok && unknown.Obj != nil {
			volume, ok = unknown.Obj.(*api.PersistentVolume)
			if !ok {
				glog.Errorf("Expected PersistentVolume but deleteVolume received %#v", unknown.Obj)
				return
			}
		} else {
			glog.Errorf("Expected PersistentVolume but deleteVolume received %+v", obj)
			return
		}
	}

	if !ok || volume == nil || volume.Spec.ClaimRef == nil {
		return
	}

	glog.V(4).Infof("volume %q deleted", volume.Name)

	if claimObj, exists, _ := ctrl.claims.GetByKey(claimrefToClaimKey(volume.Spec.ClaimRef)); exists {
		if claim, ok := claimObj.(*api.PersistentVolumeClaim); ok && claim != nil {
			// sync the claim when its volume is deleted. Explicitly syncing the
			// claim here in response to volume deletion prevents the claim from
			// waiting until the next sync period for its Lost status.
			err := ctrl.syncClaim(claim)
			if err != nil {
				if errors.IsConflict(err) {
					// Version conflict error happens quite often and the
					// controller recovers from it easily.
					glog.V(3).Infof("PersistentVolumeController could not update volume %q from deleteVolume handler: %+v", claimToClaimKey(claim), err)
				} else {
					glog.Errorf("PersistentVolumeController could not update volume %q from deleteVolume handler: %+v", claimToClaimKey(claim), err)
				}
			}
		} else {
			glog.Errorf("Cannot convert object from claim cache to claim %q!?: %#v", claimrefToClaimKey(volume.Spec.ClaimRef), claimObj)
		}
	}
}

// addClaim is callback from framework.Controller watching PersistentVolumeClaim
// events.
func (ctrl *PersistentVolumeController) addClaim(obj interface{}) {
	// Store the new claim version in the cache and do not process it if this is
	// an old version.
	claim, ok := obj.(*api.PersistentVolumeClaim)
	if !ok {
		glog.Errorf("Expected PersistentVolumeClaim but addClaim received %+v", obj)
		return
	}

	new, err := ctrl.storeClaimUpdate(claim)
	if err != nil {
		glog.Errorf("%v", err)
	}
	if !new {
		return
	}

	if err := ctrl.syncClaim(claim); err != nil {
		if errors.IsConflict(err) {
			// Version conflict error happens quite often and the controller
			// recovers from it easily.
			glog.V(3).Infof("PersistentVolumeController could not add claim %q: %+v", claimToClaimKey(claim), err)
		} else {
			glog.Errorf("PersistentVolumeController could not add claim %q: %+v", claimToClaimKey(claim), err)
		}
	}
}

// updateClaim is callback from framework.Controller watching PersistentVolumeClaim
// events.
func (ctrl *PersistentVolumeController) updateClaim(oldObj, newObj interface{}) {
	// Store the new claim version in the cache and do not process it if this is
	// an old version.
	newClaim, ok := newObj.(*api.PersistentVolumeClaim)
	if !ok {
		glog.Errorf("Expected PersistentVolumeClaim but updateClaim received %+v", newObj)
		return
	}

	new, err := ctrl.storeClaimUpdate(newClaim)
	if err != nil {
		glog.Errorf("%v", err)
	}
	if !new {
		return
	}

	if err := ctrl.syncClaim(newClaim); err != nil {
		if errors.IsConflict(err) {
			// Version conflict error happens quite often and the controller
			// recovers from it easily.
			glog.V(3).Infof("PersistentVolumeController could not update claim %q: %+v", claimToClaimKey(newClaim), err)
		} else {
			glog.Errorf("PersistentVolumeController could not update claim %q: %+v", claimToClaimKey(newClaim), err)
		}
	}
}

// deleteClaim is callback from framework.Controller watching PersistentVolumeClaim
// events.
func (ctrl *PersistentVolumeController) deleteClaim(obj interface{}) {
	_ = ctrl.claims.Delete(obj)

	var volume *api.PersistentVolume
	var claim *api.PersistentVolumeClaim
	var ok bool

	claim, ok = obj.(*api.PersistentVolumeClaim)
	if !ok {
		if unknown, ok := obj.(cache.DeletedFinalStateUnknown); ok && unknown.Obj != nil {
			claim, ok = unknown.Obj.(*api.PersistentVolumeClaim)
			if !ok {
				glog.Errorf("Expected PersistentVolumeClaim but deleteClaim received %#v", unknown.Obj)
				return
			}
		} else {
			glog.Errorf("Expected PersistentVolumeClaim but deleteClaim received %#v", obj)
			return
		}
	}

	if !ok || claim == nil {
		return
	}
	glog.V(4).Infof("claim %q deleted", claimToClaimKey(claim))

	if pvObj, exists, _ := ctrl.volumes.store.GetByKey(claim.Spec.VolumeName); exists {
		if volume, ok = pvObj.(*api.PersistentVolume); ok {
			// sync the volume when its claim is deleted.  Explicitly sync'ing the
			// volume here in response to claim deletion prevents the volume from
			// waiting until the next sync period for its Release.
			if volume != nil {
				err := ctrl.syncVolume(volume)
				if err != nil {
					if errors.IsConflict(err) {
						// Version conflict error happens quite often and the
						// controller recovers from it easily.
						glog.V(3).Infof("PersistentVolumeController could not update volume %q from deleteClaim handler: %+v", volume.Name, err)
					} else {
						glog.Errorf("PersistentVolumeController could not update volume %q from deleteClaim handler: %+v", volume.Name, err)
					}
				}
			}
		} else {
			glog.Errorf("Cannot convert object from volume cache to volume %q!?: %#v", claim.Spec.VolumeName, pvObj)
		}
	}
}

// Run starts all of this controller's control loops
func (ctrl *PersistentVolumeController) Run() {
	glog.V(4).Infof("starting PersistentVolumeController")

	ctrl.initializeCaches(ctrl.volumeSource, ctrl.claimSource)

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

const (
	// these pair of constants are used by the provisioner in Kubernetes 1.2.
	pvProvisioningRequiredAnnotationKey    = "volume.experimental.kubernetes.io/provisioning-required"
	pvProvisioningCompletedAnnotationValue = "volume.experimental.kubernetes.io/provisioning-completed"
)

// upgradeVolumeFrom1_2 updates PV from Kubernetes 1.2 to 1.3 and newer. In 1.2,
// we used template PersistentVolume instances for dynamic provisioning. In 1.3
// and later, these template (and not provisioned) instances must be removed to
// make the controller to provision a new PV.
// It returns true if the volume was deleted.
// TODO: remove this function when upgrade from 1.2 becomes unsupported.
func (ctrl *PersistentVolumeController) upgradeVolumeFrom1_2(volume *api.PersistentVolume) bool {
	annValue, found := volume.Annotations[pvProvisioningRequiredAnnotationKey]
	if !found {
		// The volume is not template
		return false
	}
	if annValue == pvProvisioningCompletedAnnotationValue {
		// The volume is already fully provisioned. The new controller will
		// ignore this annotation and it will obey its ReclaimPolicy, which is
		// likely to delete the volume when appropriate claim is deleted.
		return false
	}
	glog.V(2).Infof("deleting unprovisioned template volume %q from Kubernetes 1.2.", volume.Name)
	err := ctrl.kubeClient.Core().PersistentVolumes().Delete(volume.Name, nil)
	if err != nil {
		glog.Errorf("cannot delete unprovisioned template volume %q: %v", volume.Name, err)
	}
	// Remove from local cache
	err = ctrl.volumes.store.Delete(volume)
	if err != nil {
		glog.Errorf("cannot remove volume %q from local cache: %v", volume.Name, err)
	}

	return true
}

// Stateless functions

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
	bound := hasAnnotation(claim.ObjectMeta, annBindCompleted)
	boundByController := hasAnnotation(claim.ObjectMeta, annBoundByController)

	return fmt.Sprintf("phase: %s, bound to: %q, bindCompleted: %v, boundByController: %v", claim.Status.Phase, claim.Spec.VolumeName, bound, boundByController)
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

// storeObjectUpdate updates given cache with a new object version from Informer
// callback (i.e. with events from etcd) or with an object modified by the
// controller itself. Returns "true", if the cache was updated, false if the
// object is an old version and should be ignored.
func storeObjectUpdate(store cache.Store, obj interface{}, className string) (bool, error) {
	objAccessor, err := meta.Accessor(obj)
	if err != nil {
		return false, fmt.Errorf("Error reading cache of %s: %v", className, err)
	}
	objName := objAccessor.GetNamespace() + "/" + objAccessor.GetName()

	oldObj, found, err := store.Get(obj)
	if err != nil {
		return false, fmt.Errorf("Error finding %s %q in controller cache: %v", className, objName, err)
	}

	if !found {
		// This is a new object
		glog.V(4).Infof("storeObjectUpdate: adding %s %q, version %s", className, objName, objAccessor.GetResourceVersion())
		if err = store.Add(obj); err != nil {
			return false, fmt.Errorf("Error adding %s %q to controller cache: %v", className, objName, err)
		}
		return true, nil
	}

	oldObjAccessor, err := meta.Accessor(oldObj)
	if err != nil {
		return false, err
	}

	objResourceVersion, err := strconv.ParseInt(objAccessor.GetResourceVersion(), 10, 64)
	if err != nil {
		return false, fmt.Errorf("Error parsing ResourceVersion %q of %s %q: %s", objAccessor.GetResourceVersion(), className, objName, err)
	}
	oldObjResourceVersion, err := strconv.ParseInt(oldObjAccessor.GetResourceVersion(), 10, 64)
	if err != nil {
		return false, fmt.Errorf("Error parsing old ResourceVersion %q of %s %q: %s", oldObjAccessor.GetResourceVersion(), className, objName, err)
	}

	// Throw away only older version, let the same version pass - we do want to
	// get periodic sync events.
	if oldObjResourceVersion > objResourceVersion {
		glog.V(4).Infof("storeObjectUpdate: ignoring %s %q version %s", className, objName, objAccessor.GetResourceVersion())
		return false, nil
	}

	glog.V(4).Infof("storeObjectUpdate updating %s %q with version %s", className, objName, objAccessor.GetResourceVersion())
	if err = store.Update(obj); err != nil {
		return false, fmt.Errorf("Error updating %s %q in controller cache: %v", className, objName, err)
	}
	return true, nil
}
