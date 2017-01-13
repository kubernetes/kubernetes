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

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	storage "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/goroutinemap"
	"k8s.io/kubernetes/pkg/util/workqueue"
	vol "k8s.io/kubernetes/pkg/volume"

	"github.com/golang/glog"
)

// This file contains the controller base functionality, i.e. framework to
// process PV/PVC added/updated/deleted events. The real binding, provisioning,
// recycling and deleting is done in pv_controller.go

// ControllerParameters contains arguments for creation of a new
// PersistentVolume controller.
type ControllerParameters struct {
	KubeClient                             clientset.Interface
	SyncPeriod                             time.Duration
	AlphaProvisioner                       vol.ProvisionableVolumePlugin
	VolumePlugins                          []vol.VolumePlugin
	Cloud                                  cloudprovider.Interface
	ClusterName                            string
	VolumeSource, ClaimSource, ClassSource cache.ListerWatcher
	EventRecorder                          record.EventRecorder
	EnableDynamicProvisioning              bool
}

// NewController creates a new PersistentVolume controller
func NewController(p ControllerParameters) *PersistentVolumeController {
	eventRecorder := p.EventRecorder
	if eventRecorder == nil {
		broadcaster := record.NewBroadcaster()
		broadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{Interface: p.KubeClient.Core().Events("")})
		eventRecorder = broadcaster.NewRecorder(v1.EventSource{Component: "persistentvolume-controller"})
	}

	controller := &PersistentVolumeController{
		volumes:           newPersistentVolumeOrderedIndex(),
		claims:            cache.NewStore(cache.DeletionHandlingMetaNamespaceKeyFunc),
		kubeClient:        p.KubeClient,
		eventRecorder:     eventRecorder,
		runningOperations: goroutinemap.NewGoRoutineMap(true /* exponentialBackOffOnError */),
		cloud:             p.Cloud,
		enableDynamicProvisioning:     p.EnableDynamicProvisioning,
		clusterName:                   p.ClusterName,
		createProvisionedPVRetryCount: createProvisionedPVRetryCount,
		createProvisionedPVInterval:   createProvisionedPVInterval,
		alphaProvisioner:              p.AlphaProvisioner,
		claimQueue:                    workqueue.NewNamed("claims"),
		volumeQueue:                   workqueue.NewNamed("volumes"),
	}

	controller.volumePluginMgr.InitPlugins(p.VolumePlugins, controller)
	if controller.alphaProvisioner != nil {
		if err := controller.alphaProvisioner.Init(controller); err != nil {
			glog.Errorf("PersistentVolumeController: error initializing alpha provisioner plugin: %v", err)
		}
	}

	volumeSource := p.VolumeSource
	if volumeSource == nil {
		volumeSource = &cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				return p.KubeClient.Core().PersistentVolumes().List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return p.KubeClient.Core().PersistentVolumes().Watch(options)
			},
		}
	}
	controller.volumeSource = volumeSource

	claimSource := p.ClaimSource
	if claimSource == nil {
		claimSource = &cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				return p.KubeClient.Core().PersistentVolumeClaims(v1.NamespaceAll).List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return p.KubeClient.Core().PersistentVolumeClaims(v1.NamespaceAll).Watch(options)
			},
		}
	}
	controller.claimSource = claimSource

	classSource := p.ClassSource
	if classSource == nil {
		classSource = &cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				return p.KubeClient.Storage().StorageClasses().List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return p.KubeClient.Storage().StorageClasses().Watch(options)
			},
		}
	}
	controller.classSource = classSource

	controller.volumeInformer, controller.volumeController = cache.NewIndexerInformer(
		volumeSource,
		&v1.PersistentVolume{},
		p.SyncPeriod,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    func(obj interface{}) { controller.enqueueWork(controller.volumeQueue, obj) },
			UpdateFunc: func(oldObj, newObj interface{}) { controller.enqueueWork(controller.volumeQueue, newObj) },
			DeleteFunc: func(obj interface{}) { controller.enqueueWork(controller.volumeQueue, obj) },
		},
		cache.Indexers{"accessmodes": accessModesIndexFunc},
	)
	controller.claimInformer, controller.claimController = cache.NewInformer(
		claimSource,
		&v1.PersistentVolumeClaim{},
		p.SyncPeriod,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    func(obj interface{}) { controller.enqueueWork(controller.claimQueue, obj) },
			UpdateFunc: func(oldObj, newObj interface{}) { controller.enqueueWork(controller.claimQueue, newObj) },
			DeleteFunc: func(obj interface{}) { controller.enqueueWork(controller.claimQueue, obj) },
		},
	)

	// This is just a cache of StorageClass instances, no special actions are
	// needed when a class is created/deleted/updated.
	controller.classes = cache.NewStore(cache.DeletionHandlingMetaNamespaceKeyFunc)
	controller.classReflector = cache.NewReflector(
		classSource,
		&storage.StorageClass{},
		controller.classes,
		p.SyncPeriod,
	)
	return controller
}

// initializeCaches fills all controller caches with initial data from etcd in
// order to have the caches already filled when first addClaim/addVolume to
// perform initial synchronization of the controller.
func (ctrl *PersistentVolumeController) initializeCaches(volumeSource, claimSource cache.ListerWatcher) {
	volumeListObj, err := volumeSource.List(v1.ListOptions{})
	if err != nil {
		glog.Errorf("PersistentVolumeController can't initialize caches: %v", err)
		return
	}
	volumeList, ok := volumeListObj.(*v1.PersistentVolumeList)
	if !ok {
		glog.Errorf("PersistentVolumeController can't initialize caches, expected list of volumes, got: %#v", volumeListObj)
		return
	}
	for _, volume := range volumeList.Items {
		// Ignore template volumes from kubernetes 1.2
		deleted := ctrl.upgradeVolumeFrom1_2(&volume)
		if !deleted {
			clone, err := api.Scheme.DeepCopy(&volume)
			if err != nil {
				glog.Errorf("error cloning volume %q: %v", volume.Name, err)
				continue
			}
			volumeClone := clone.(*v1.PersistentVolume)
			ctrl.storeVolumeUpdate(volumeClone)
		}
	}

	claimListObj, err := claimSource.List(v1.ListOptions{})
	if err != nil {
		glog.Errorf("PersistentVolumeController can't initialize caches: %v", err)
		return
	}
	claimList, ok := claimListObj.(*v1.PersistentVolumeClaimList)
	if !ok {
		glog.Errorf("PersistentVolumeController can't initialize caches, expected list of claims, got: %#v", claimListObj)
		return
	}
	for _, claim := range claimList.Items {
		clone, err := api.Scheme.DeepCopy(&claim)
		if err != nil {
			glog.Errorf("error cloning claim %q: %v", claimToClaimKey(&claim), err)
			continue
		}
		claimClone := clone.(*v1.PersistentVolumeClaim)
		ctrl.storeClaimUpdate(claimClone)
	}
	glog.V(4).Infof("controller initialized")
}

// enqueueWork adds volume or claim to given work queue.
func (ctrl *PersistentVolumeController) enqueueWork(queue workqueue.Interface, obj interface{}) {
	// Beware of "xxx deleted" events
	if unknown, ok := obj.(cache.DeletedFinalStateUnknown); ok && unknown.Obj != nil {
		obj = unknown.Obj
	}
	objName, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("failed to get key from object: %v", err)
		return
	}
	glog.V(5).Infof("enqueued %q for sync", objName)
	queue.Add(objName)
}

func (ctrl *PersistentVolumeController) storeVolumeUpdate(volume interface{}) (bool, error) {
	return storeObjectUpdate(ctrl.volumes.store, volume, "volume")
}

func (ctrl *PersistentVolumeController) storeClaimUpdate(claim interface{}) (bool, error) {
	return storeObjectUpdate(ctrl.claims, claim, "claim")
}

// updateVolume runs in worker thread and handles "volume added",
// "volume updated" and "periodic sync" events.
func (ctrl *PersistentVolumeController) updateVolume(volume *v1.PersistentVolume) {
	if deleted := ctrl.upgradeVolumeFrom1_2(volume); deleted {
		// volume deleted
		return
	}

	// Store the new volume version in the cache and do not process it if this
	// is an old version.
	new, err := ctrl.storeVolumeUpdate(volume)
	if err != nil {
		glog.Errorf("%v", err)
	}
	if !new {
		return
	}

	err = ctrl.syncVolume(volume)
	if err != nil {
		if errors.IsConflict(err) {
			// Version conflict error happens quite often and the controller
			// recovers from it easily.
			glog.V(3).Infof("could not sync volume %q: %+v", volume.Name, err)
		} else {
			glog.Errorf("could not sync volume %q: %+v", volume.Name, err)
		}
	}
}

// deleteVolume runs in worker thread and handles "volume deleted" event.
func (ctrl *PersistentVolumeController) deleteVolume(volume *v1.PersistentVolume) {
	_ = ctrl.volumes.store.Delete(volume)
	glog.V(4).Infof("volume %q deleted", volume.Name)

	if volume.Spec.ClaimRef == nil {
		return
	}
	// sync the claim when its volume is deleted. Explicitly syncing the
	// claim here in response to volume deletion prevents the claim from
	// waiting until the next sync period for its Lost status.
	claimKey := claimrefToClaimKey(volume.Spec.ClaimRef)
	glog.V(5).Infof("deleteVolume[%s]: scheduling sync of claim %q", volume.Name, claimKey)
	ctrl.claimQueue.Add(claimKey)
}

// updateClaim runs in worker thread and handles "claim added",
// "claim updated" and "periodic sync" events.
func (ctrl *PersistentVolumeController) updateClaim(claim *v1.PersistentVolumeClaim) {
	// Store the new claim version in the cache and do not process it if this is
	// an old version.
	new, err := ctrl.storeClaimUpdate(claim)
	if err != nil {
		glog.Errorf("%v", err)
	}
	if !new {
		return
	}
	err = ctrl.syncClaim(claim)
	if err != nil {
		if errors.IsConflict(err) {
			// Version conflict error happens quite often and the controller
			// recovers from it easily.
			glog.V(3).Infof("could not sync claim %q: %+v", claimToClaimKey(claim), err)
		} else {
			glog.Errorf("could not sync volume %q: %+v", claimToClaimKey(claim), err)
		}
	}
}

// deleteClaim runs in worker thread and handles "claim deleted" event.
func (ctrl *PersistentVolumeController) deleteClaim(claim *v1.PersistentVolumeClaim) {
	_ = ctrl.claims.Delete(claim)
	glog.V(4).Infof("claim %q deleted", claimToClaimKey(claim))

	// sync the volume when its claim is deleted.  Explicitly sync'ing the
	// volume here in response to claim deletion prevents the volume from
	// waiting until the next sync period for its Release.
	volumeName := claim.Spec.VolumeName
	glog.V(5).Infof("deleteClaim[%s]: scheduling sync of volume %q", claimToClaimKey(claim), volumeName)
	ctrl.volumeQueue.Add(volumeName)
}

// Run starts all of this controller's control loops
func (ctrl *PersistentVolumeController) Run(stopCh <-chan struct{}) {
	glog.V(1).Infof("starting PersistentVolumeController")
	ctrl.initializeCaches(ctrl.volumeSource, ctrl.claimSource)
	go ctrl.volumeController.Run(stopCh)
	go ctrl.claimController.Run(stopCh)
	go ctrl.classReflector.RunUntil(stopCh)
	go wait.Until(ctrl.volumeWorker, time.Second, stopCh)
	go wait.Until(ctrl.claimWorker, time.Second, stopCh)

	<-stopCh

	ctrl.claimQueue.ShutDown()
	ctrl.volumeQueue.ShutDown()
}

// volumeWorker processes items from volumeQueue. It must run only once,
// syncVolume is not assured to be reentrant.
func (ctrl *PersistentVolumeController) volumeWorker() {
	workFunc := func() bool {
		keyObj, quit := ctrl.volumeQueue.Get()
		if quit {
			return true
		}
		defer ctrl.volumeQueue.Done(keyObj)
		key := keyObj.(string)
		glog.V(5).Infof("volumeWorker[%s]", key)

		volumeObj, found, err := ctrl.volumeInformer.GetByKey(key)
		if err != nil {
			glog.V(2).Infof("error getting volume %q from informer: %v", key, err)
			return false
		}

		if found {
			// The volume still exists in informer cache, the event must have
			// been add/update/sync
			volume, ok := volumeObj.(*v1.PersistentVolume)
			if !ok {
				glog.Errorf("expected volume, got %+v", volumeObj)
				return false
			}
			ctrl.updateVolume(volume)
			return false
		}

		// The volume is not in informer cache, the event must have been
		// "delete"
		volumeObj, found, err = ctrl.volumes.store.GetByKey(key)
		if err != nil {
			glog.V(2).Infof("error getting volume %q from cache: %v", key, err)
			return false
		}
		if !found {
			// The controller has already processed the delete event and
			// deleted the volume from its cache
			glog.V(2).Infof("deletion of volume %q was already processed", key)
			return false
		}
		volume, ok := volumeObj.(*v1.PersistentVolume)
		if !ok {
			glog.Errorf("expected volume, got %+v", volumeObj)
			return false
		}
		ctrl.deleteVolume(volume)
		return false
	}
	for {
		if quit := workFunc(); quit {
			glog.Infof("volume worker queue shutting down")
			return
		}
	}
}

// claimWorker processes items from claimQueue. It must run only once,
// syncClaim is not reentrant.
func (ctrl *PersistentVolumeController) claimWorker() {
	workFunc := func() bool {
		keyObj, quit := ctrl.claimQueue.Get()
		if quit {
			return true
		}
		defer ctrl.claimQueue.Done(keyObj)
		key := keyObj.(string)
		glog.V(5).Infof("claimWorker[%s]", key)

		claimObj, found, err := ctrl.claimInformer.GetByKey(key)
		if err != nil {
			glog.V(2).Infof("error getting claim %q from informer: %v", key, err)
			return false
		}

		if found {
			// The claim still exists in informer cache, the event must have
			// been add/update/sync
			claim, ok := claimObj.(*v1.PersistentVolumeClaim)
			if !ok {
				glog.Errorf("expected claim, got %+v", claimObj)
				return false
			}
			ctrl.updateClaim(claim)
			return false
		}

		// The claim is not in informer cache, the event must have been "delete"
		claimObj, found, err = ctrl.claims.GetByKey(key)
		if err != nil {
			glog.V(2).Infof("error getting claim %q from cache: %v", key, err)
			return false
		}
		if !found {
			// The controller has already processed the delete event and
			// deleted the claim from its cache
			glog.V(2).Infof("deletion of claim %q was already processed", key)
			return false
		}
		claim, ok := claimObj.(*v1.PersistentVolumeClaim)
		if !ok {
			glog.Errorf("expected claim, got %+v", claimObj)
			return false
		}
		ctrl.deleteClaim(claim)
		return false
	}
	for {
		if quit := workFunc(); quit {
			glog.Infof("claim worker queue shutting down")
			return
		}
	}
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
func (ctrl *PersistentVolumeController) upgradeVolumeFrom1_2(volume *v1.PersistentVolume) bool {
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

// setClaimProvisioner saves
// claim.Annotations[annStorageProvisioner] = class.Provisioner
func (ctrl *PersistentVolumeController) setClaimProvisioner(claim *v1.PersistentVolumeClaim, class *storage.StorageClass) (*v1.PersistentVolumeClaim, error) {
	if val, ok := claim.Annotations[annDynamicallyProvisioned]; ok && val == class.Provisioner {
		// annotation is already set, nothing to do
		return claim, nil
	}

	// The volume from method args can be pointing to watcher cache. We must not
	// modify these, therefore create a copy.
	clone, err := api.Scheme.DeepCopy(claim)
	if err != nil {
		return nil, fmt.Errorf("Error cloning pv: %v", err)
	}
	claimClone, ok := clone.(*v1.PersistentVolumeClaim)
	if !ok {
		return nil, fmt.Errorf("Unexpected claim cast error : %v", claimClone)
	}
	v1.SetMetaDataAnnotation(&claimClone.ObjectMeta, annStorageProvisioner, class.Provisioner)
	newClaim, err := ctrl.kubeClient.Core().PersistentVolumeClaims(claim.Namespace).Update(claimClone)
	if err != nil {
		return newClaim, err
	}
	_, err = ctrl.storeClaimUpdate(newClaim)
	if err != nil {
		return newClaim, err
	}
	return newClaim, nil
}

// Stateless functions

func getClaimStatusForLogging(claim *v1.PersistentVolumeClaim) string {
	bound := v1.HasAnnotation(claim.ObjectMeta, annBindCompleted)
	boundByController := v1.HasAnnotation(claim.ObjectMeta, annBoundByController)

	return fmt.Sprintf("phase: %s, bound to: %q, bindCompleted: %v, boundByController: %v", claim.Status.Phase, claim.Spec.VolumeName, bound, boundByController)
}

func getVolumeStatusForLogging(volume *v1.PersistentVolume) string {
	boundByController := v1.HasAnnotation(volume.ObjectMeta, annBoundByController)
	claimName := ""
	if volume.Spec.ClaimRef != nil {
		claimName = fmt.Sprintf("%s/%s (uid: %s)", volume.Spec.ClaimRef.Namespace, volume.Spec.ClaimRef.Name, volume.Spec.ClaimRef.UID)
	}
	return fmt.Sprintf("phase: %s, bound to: %q, boundByController: %v", volume.Status.Phase, claimName, boundByController)
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

// storeObjectUpdate updates given cache with a new object version from Informer
// callback (i.e. with events from etcd) or with an object modified by the
// controller itself. Returns "true", if the cache was updated, false if the
// object is an old version and should be ignored.
func storeObjectUpdate(store cache.Store, obj interface{}, className string) (bool, error) {
	objName, err := controller.KeyFunc(obj)
	if err != nil {
		return false, fmt.Errorf("Couldn't get key for object %+v: %v", obj, err)
	}
	oldObj, found, err := store.Get(obj)
	if err != nil {
		return false, fmt.Errorf("Error finding %s %q in controller cache: %v", className, objName, err)
	}

	objAccessor, err := meta.Accessor(obj)
	if err != nil {
		return false, err
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
