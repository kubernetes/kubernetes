/*
Copyright 2018 The Kubernetes Authors.

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

package snapshot

import (
	"fmt"
	"strings"
	"time"

	"github.com/golang/glog"
	authenticationv1 "k8s.io/api/authentication/v1"
	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	storage "k8s.io/api/storage/v1alpha1"
	"k8s.io/apimachinery/pkg/api/errors"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	storageinformers "k8s.io/client-go/informers/storage/v1alpha1"
	"k8s.io/client-go/kubernetes"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	storagelisters "k8s.io/client-go/listers/storage/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	ref "k8s.io/client-go/tools/reference"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/volume/events"
	"k8s.io/kubernetes/pkg/util/goroutinemap"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
	vol "k8s.io/kubernetes/pkg/volume"
)

// Number of retries when we create a VSD object for a created snapshot.
const createSnapshotDataRetryCount = 5

// Interval between retries when we create a VSD object for a provisioned snapshot.
const createSnapshotDataInterval = 10 * time.Second

// defaultSnapshotSyncPeriod is the default period for syncing volume snapshots
// and volume snapshot datas.
const defaultSnapshotSyncPeriod = 20 * time.Second

// annBindCompleted annotation applies to snapshots. It indicates that the lifecycle
// of the snapshot has passed through the initial setup. This information changes how
// we interpret some observations of the state of the objects. Value of this
// annotation does not matter.
const annBindCompleted = "vsd.kubernetes.io/bind-completed"

// This annotation is added to a VSD that has been dynamically created by
// Kubernetes. Its value is name of volume plugin that created the snapshot data.
// It serves both user (to show where a VSD comes from) and Kubernetes (to
// recognize dynamically created VSDs in its decisions).
const annDynamicallyCreatedBy = "vsd.kubernetes.io/created-by"

// annBoundByController annotation applies to VSDs and Vss.  It indicates that
// the binding (VSD->VS or VS->VSD) was installed by the controller.  The
// absence of this annotation means the binding was done by the user (i.e.
// pre-bound). Value of this annotation does not matter.
const annBoundByController = "vsd.kubernetes.io/bound-by-controller"

// This annotation is added to a volumeSnapshot that is supposed to create snapshot
// Its value is name of volume plugin that is supposed to create
// a snapshot data for this snapshot.
const annStorageSnapshotter = "snapshot.beta.kubernetes.io/storage-snapshotter"

// annSnapshotDataShouldDelete is added to a VSD that will be deleted. It serves external
// snapshotter to delete the backend snapshot data. Value of this annotation does not matter.
const annSnapshotDataShouldDelete = "vsd.kubernetes.io/should-delete"

// ControllerParameters contains arguments for creation of a new
// SnapshotController controller.
type ControllerParameters struct {
	KubeClient                 clientset.Interface
	SyncPeriod                 time.Duration
	VolumePlugins              []vol.VolumePlugin
	Cloud                      cloudprovider.Interface
	ClusterName                string
	VolumeInformer             coreinformers.PersistentVolumeInformer
	ClaimInformer              coreinformers.PersistentVolumeClaimInformer
	VolumeSnapshotInformer     storageinformers.VolumeSnapshotInformer
	VolumeSnapshotDataInformer storageinformers.VolumeSnapshotDataInformer
	EventRecorder              record.EventRecorder
}

type SnapshotController struct {
	vsLister           storagelisters.VolumeSnapshotLister
	vsListerSynced     cache.InformerSynced
	vsdLister          storagelisters.VolumeSnapshotDataLister
	vsdListerSynced    cache.InformerSynced
	volumeLister       corelisters.PersistentVolumeLister
	volumeListerSynced cache.InformerSynced
	claimLister        corelisters.PersistentVolumeClaimLister
	claimListerSynced  cache.InformerSynced

	client          kubernetes.Interface
	eventRecorder   record.EventRecorder
	resyncPeriod    time.Duration
	volumePluginMgr vol.VolumePluginMgr
	// cloud provider used by volume host
	cloud cloudprovider.Interface

	// Map of scheduled/running operations.
	runningOperations            goroutinemap.GoRoutineMap
	createSnapshotDataRetryCount int
	createSnapshotDataInterval   time.Duration

	snapshotQueue     workqueue.RateLimitingInterface
	snapshotDataQueue workqueue.RateLimitingInterface

	snapshotStore     cache.Store
	snapshotDataStore cache.Store
}

// NewController creates a new VolumeSnapshot controller
func NewSnapshotController(p ControllerParameters) (*SnapshotController, error) {
	eventRecorder := p.EventRecorder
	if eventRecorder == nil {
		broadcaster := record.NewBroadcaster()
		broadcaster.StartLogging(glog.Infof)
		broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: p.KubeClient.CoreV1().Events("")})
		eventRecorder = broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "volumesnapshot-controller"})
	}

	ctrl := &SnapshotController{
		client:                       p.KubeClient,
		cloud:                        p.Cloud,
		eventRecorder:                eventRecorder,
		resyncPeriod:                 p.SyncPeriod,
		createSnapshotDataRetryCount: createSnapshotDataRetryCount,
		createSnapshotDataInterval:   createSnapshotDataInterval,
		runningOperations:            goroutinemap.NewGoRoutineMap(true),
		snapshotStore:                cache.NewStore(cache.DeletionHandlingMetaNamespaceKeyFunc),
		snapshotDataStore:            cache.NewStore(cache.DeletionHandlingMetaNamespaceKeyFunc),
		snapshotQueue:                workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "csi-snapshotter-vs"),
		snapshotDataQueue:            workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "csi-snapshotter-vsd"),
	}

	// Prober is nil because vsd is not aware of Flexvolume.
	if err := ctrl.volumePluginMgr.InitPlugins(p.VolumePlugins, nil /* prober */, ctrl); err != nil {
		return nil, fmt.Errorf("could not initialize volume plugins for Snapshot Controller: %v", err)
	}

	p.VolumeSnapshotInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    func(obj interface{}) { ctrl.enqueueWork(ctrl.snapshotQueue, obj) },
			UpdateFunc: func(oldObj, newObj interface{}) { ctrl.enqueueWork(ctrl.snapshotQueue, newObj) },
			DeleteFunc: func(obj interface{}) { ctrl.enqueueWork(ctrl.snapshotQueue, obj) },
		},
		defaultSnapshotSyncPeriod,
	)
	ctrl.vsLister = p.VolumeSnapshotInformer.Lister()
	ctrl.vsListerSynced = p.VolumeSnapshotInformer.Informer().HasSynced

	p.VolumeSnapshotDataInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    func(obj interface{}) { ctrl.enqueueWork(ctrl.snapshotDataQueue, obj) },
			UpdateFunc: func(oldObj, newObj interface{}) { ctrl.enqueueWork(ctrl.snapshotDataQueue, newObj) },
			DeleteFunc: func(obj interface{}) { ctrl.enqueueWork(ctrl.snapshotDataQueue, obj) },
		},
		defaultSnapshotSyncPeriod,
	)
	ctrl.vsdLister = p.VolumeSnapshotDataInformer.Lister()
	ctrl.vsdListerSynced = p.VolumeSnapshotDataInformer.Informer().HasSynced

	ctrl.volumeLister = p.VolumeInformer.Lister()
	ctrl.volumeListerSynced = p.VolumeInformer.Informer().HasSynced

	ctrl.claimLister = p.ClaimInformer.Lister()
	ctrl.claimListerSynced = p.ClaimInformer.Informer().HasSynced

	return ctrl, nil
}

// enqueueWork adds vs or vsd to given work queue.
func (ctrl *SnapshotController) enqueueWork(queue workqueue.Interface, obj interface{}) {
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

func (ctrl *SnapshotController) Run(workers int, stopCh <-chan struct{}) {
	defer ctrl.snapshotQueue.ShutDown()
	defer ctrl.snapshotDataQueue.ShutDown()

	glog.Infof("Starting snapshot controller")
	defer glog.Infof("Shutting snapshot controller")

	if !cache.WaitForCacheSync(stopCh, ctrl.vsListerSynced, ctrl.vsdListerSynced, ctrl.volumeListerSynced, ctrl.claimListerSynced) {
		glog.Errorf("Cannot sync caches")
		return
	}

	ctrl.initializeCaches(ctrl.vsLister, ctrl.vsdLister)

	for i := 0; i < workers; i++ {
		go wait.Until(ctrl.snapshotWorker, 0, stopCh)
		go wait.Until(ctrl.snapshotDataWorker, 0, stopCh)
	}

	<-stopCh
}

// vsWorker processes items from snapshotQueue. It must run only once,
// syncVolume is not assured to be reentrant.
func (ctrl *SnapshotController) snapshotWorker() {
	workFunc := func() bool {
		keyObj, quit := ctrl.snapshotQueue.Get()
		if quit {
			return true
		}
		defer ctrl.snapshotQueue.Done(keyObj)
		key := keyObj.(string)
		glog.V(5).Infof("vsWorker[%s]", key)

		namespace, name, err := cache.SplitMetaNamespaceKey(key)
		if err != nil {
			glog.V(4).Infof("error getting namespace & name of snapshot %q to get snapshot from informer: %v", key, err)
			return false
		}
		snapshot, err := ctrl.vsLister.VolumeSnapshots(namespace).Get(name)
		if err == nil {
			// The volume snapshot still exists in informer cache, the event must have
			// been add/update/sync
			ctrl.updateSnapshot(snapshot)
			return false
		}
		if !errors.IsNotFound(err) {
			glog.V(2).Infof("error getting snapshot %q from informer: %v", key, err)
			return false
		}
		// The snapshot is not in informer cache, the event must have been "delete"
		vsObj, found, err := ctrl.snapshotStore.GetByKey(key)
		if err != nil {
			glog.V(2).Infof("error getting snapshot %q from cache: %v", key, err)
			return false
		}
		if !found {
			// The controller has already processed the delete event and
			// deleted the snapshot from its cache
			glog.V(2).Infof("deletion of vs %q was already processed", key)
			return false
		}
		snapshot, ok := vsObj.(*storage.VolumeSnapshot)
		if !ok {
			glog.Errorf("expected vs, got %+v", vsObj)
			return false
		}

		ctrl.deleteSnapshot(snapshot)

		return false
	}

	for {
		if quit := workFunc(); quit {
			glog.Infof("snapshot worker queue shutting down")
			return
		}
	}
}

// snapshotDataWorker processes items from snapshotDataQueue. It must run only once,
// syncVsd is not assured to be reentrant.
func (ctrl *SnapshotController) snapshotDataWorker() {
	workFunc := func() bool {
		keyObj, quit := ctrl.snapshotDataQueue.Get()
		if quit {
			return true
		}
		defer ctrl.snapshotDataQueue.Done(keyObj)
		key := keyObj.(string)
		glog.V(5).Infof("snapshotDataWorker[%s]", key)

		_, name, err := cache.SplitMetaNamespaceKey(key)
		if err != nil {
			glog.V(4).Infof("error getting name of snapshotData %q to get snapshotData from informer: %v", key, err)
			return false
		}
		vsd, err := ctrl.vsdLister.Get(name)
		if err == nil {
			// The snapshotData still exists in informer cache, the event must have
			// been add/update/sync
			ctrl.updateSnapshotData(vsd)
			return false
		}
		if !errors.IsNotFound(err) {
			glog.V(2).Infof("error getting vsd %q from informer: %v", key, err)
			return false
		}

		// The snapshotData is not in informer cache, the event must have been
		// "delete"
		vsdObj, found, err := ctrl.snapshotDataStore.GetByKey(key)
		if err != nil {
			glog.V(2).Infof("error getting vsd %q from cache: %v", key, err)
			return false
		}
		if !found {
			// The controller has already processed the delete event and
			// deleted the snapshotData from its cache
			glog.V(2).Infof("deletion of snapshotData %q was already processed", key)
			return false
		}
		vsd, ok := vsdObj.(*storage.VolumeSnapshotData)
		if !ok {
			glog.Errorf("expected snapshotData, got %+v", vsd)
			return false
		}
		ctrl.deleteSnapshotData(vsd)
		return false
	}

	for {
		if quit := workFunc(); quit {
			glog.Infof("vsd worker queue shutting down")
			return
		}
	}
}

// initializeCaches fills all controller caches with initial data from etcd in
// order to have the caches already filled when first addVS/addVSD to
// perform initial synchronization of the controller.
func (ctrl *SnapshotController) initializeCaches(vsLister storagelisters.VolumeSnapshotLister, vsdLister storagelisters.VolumeSnapshotDataLister) {
	vsList, err := vsLister.List(labels.Everything())
	if err != nil {
		glog.Errorf("SnapshotController can't initialize caches: %v", err)
		return
	}
	for _, vs := range vsList {
		vsClone := vs.DeepCopy()
		if _, err = ctrl.storeSnapshotUpdate(vsClone); err != nil {
			glog.Errorf("error updating volume snapshot cache: %v", err)
		}
	}

	vsdList, err := vsdLister.List(labels.Everything())
	if err != nil {
		glog.Errorf("SnapshotController can't initialize caches: %v", err)
		return
	}
	for _, vsd := range vsdList {
		vsdClone := vsd.DeepCopy()
		if _, err = ctrl.storeSnapshotDataUpdate(vsdClone); err != nil {
			glog.Errorf("error updating volume snapshot cache: %v", err)
		}
	}

	glog.V(4).Infof("controller initialized")
}

func (ctrl *SnapshotController) storeSnapshotUpdate(vs interface{}) (bool, error) {
	return storeObjectUpdate(ctrl.snapshotStore, vs, "vs")
}

func (ctrl *SnapshotController) storeSnapshotDataUpdate(vsd interface{}) (bool, error) {
	return storeObjectUpdate(ctrl.snapshotDataStore, vsd, "vsd")
}

// deleteVSD runs in worker thread and handles "snapshot deleted" event.
func (ctrl *SnapshotController) deleteSnapshotData(vsd *storage.VolumeSnapshotData) {
	_ = ctrl.snapshotDataStore.Delete(vsd)
	glog.V(4).Infof("vsd %q deleted", vsd.Name)

	snapshotName := VsRefToVsKey(vsd.Spec.VolumeSnapshotRef)
	if snapshotName == "" {
		glog.V(5).Infof("deleteVSD[%q]: vsd not bound", vsd.Name)
		return
	}
	// sync the vs when its vs is deleted.  Explicitly sync'ing the
	// vs here in response to vsd deletion prevents the vs from
	// waiting until the next sync period for its Release.
	glog.V(5).Infof("deleteVSD[%q]: scheduling sync of vs %s", vsd.Name, snapshotName)
	ctrl.snapshotDataQueue.Add(snapshotName)
}

// deleteVS runs in worker thread and handles "snapshot deleted" event.
func (ctrl *SnapshotController) deleteSnapshot(vs *storage.VolumeSnapshot) {
	_ = ctrl.snapshotStore.Delete(vs)
	glog.V(4).Infof("vs %q deleted", VsToVsKey(vs))

	snapshotDataName := vs.Spec.SnapshotDataName
	if snapshotDataName == "" {
		glog.V(5).Infof("deleteVS[%q]: vsd not bound", VsToVsKey(vs))
		return
	}
	// sync the vsd when its vs is deleted.  Explicitly sync'ing the
	// vsd here in response to vs deletion prevents the vsd from
	// waiting until the next sync period for its Release.
	glog.V(5).Infof("deleteSnapshot[%q]: scheduling sync of vsd %s", VsToVsKey(vs), snapshotDataName)
	ctrl.snapshotDataQueue.Add(snapshotDataName)
}

// updateSnapshot runs in worker thread and handles "snapshot added",
// "snapshot updated" and "periodic sync" events.
func (ctrl *SnapshotController) updateSnapshot(vs *storage.VolumeSnapshot) {
	// Store the new vs version in the cache and do not process it if this is
	// an old version.
	new, err := ctrl.storeSnapshotUpdate(vs)
	if err != nil {
		glog.Errorf("%v", err)
	}
	if !new {
		return
	}
	err = ctrl.syncSnapshot(vs)
	if err != nil {
		if errors.IsConflict(err) {
			// Version conflict error happens quite often and the controller
			// recovers from it easily.
			glog.V(3).Infof("could not sync snapshot %q: %+v", VsToVsKey(vs), err)
		} else {
			glog.Errorf("could not sync snapshot %q: %+v", VsToVsKey(vs), err)
		}
	}
}

// updateVsd runs in worker thread and handles "vsd added",
// "vsd updated" and "periodic sync" events.
func (ctrl *SnapshotController) updateSnapshotData(vsd *storage.VolumeSnapshotData) {
	// Store the new vs version in the cache and do not process it if this is
	// an old version.
	new, err := ctrl.storeSnapshotDataUpdate(vsd)
	if err != nil {
		glog.Errorf("%v", err)
	}
	if !new {
		return
	}
	err = ctrl.syncSnapshotData(vsd)
	if err != nil {
		if errors.IsConflict(err) {
			// Version conflict error happens quite often and the controller
			// recovers from it easily.
			glog.V(3).Infof("could not sync vsd %q: %+v", vsd.Name, err)
		} else {
			glog.Errorf("could not sync vsd %q: %+v", vsd.Name, err)
		}
	}
}

// syncVSD deals with one key off the queue.  It returns false when it's time to quit.
func (ctrl *SnapshotController) syncSnapshotData(vsd *storage.VolumeSnapshotData) error {
	glog.V(4).Infof("synchronizing VolumeSnapshotData[%s]", vsd.Name)

	// VolumeSnapshotData is not bind to any VolumeSnapshot, this case rare and we just return err
	if vsd.Spec.VolumeSnapshotRef == nil {
		// Vsd is not bind
		glog.V(4).Infof("synchronizing VolumeSnapshotData[%s]: vsd is not bind", vsd.Name)
		return fmt.Errorf("volumeSnapshotData %s is not bind to any VolumeSnapshot", vsd.Name)
	} else {
		glog.V(4).Infof("synchronizing VolumeSnapshotData[%s]: vsd is bound to vs %s", vsd.Name, VsRefToVsKey(vsd.Spec.VolumeSnapshotRef))
		// Get the VS by _name_
		var vs *storage.VolumeSnapshot
		vsName := VsRefToVsKey(vsd.Spec.VolumeSnapshotRef)
		obj, found, err := ctrl.snapshotStore.GetByKey(vsName)
		if err != nil {
			return err
		}
		if !found {
			glog.V(4).Infof("synchronizing VolumeSnapshotData[%s]: vs %s not found", vsd.Name, VsRefToVsKey(vsd.Spec.VolumeSnapshotRef))
			// Fall through with vs = nil
		} else {
			var ok bool
			vs, ok = obj.(*storage.VolumeSnapshot)
			if !ok {
				return fmt.Errorf("cannot convert object from vs cache to vs %q!?: %#v", vsd.Name, obj)
			}
			glog.V(4).Infof("synchronizing VolumeSnapshotData[%s]: vs %s found", vsd.Name, VsRefToVsKey(vsd.Spec.VolumeSnapshotRef))
		}
		if vs != nil && vs.UID != vsd.Spec.VolumeSnapshotRef.UID {
			// The vs that the vsd was pointing to was deleted, and another
			// with the same name created.
			glog.V(4).Infof("synchronizing VolumeSnapshotData[%s]: vsd %s has different UID, the old one must have been deleted", vsd.Name, VsRefToVsKey(vsd.Spec.VolumeSnapshotRef))
			// Treat the snapshot data as bound to a missing snapshot.
			vs = nil
		}
		if vs == nil {
			// first we mark the snapshotData is deletable, so that external snapshotter
			// can start delete the backend snapshotData.
			if _, err := ctrl.markSnapshotDataDelete(vsd); err != nil {
				return err
			}
			// If we get into this block, the snapshot must have been deleted, so we just delete the snapshot data.
			glog.V(4).Infof("delete backend storage snapshot data [%s]", vsd.Name)
			opName := fmt.Sprintf("delete-%s[%s]", vsd.Name, string(vsd.UID))
			ctrl.scheduleOperation(opName, func() error {
				return ctrl.deleteSnapshotDataOperation(vsd)
			})
			return nil
		} else if vs.Spec.SnapshotDataName == "" {
			if metav1.HasAnnotation(vsd.ObjectMeta, annBoundByController) {
				// The binding is not completed; let VS sync handle it
				glog.V(4).Infof("synchronizing VolumeSnapshotData[%s]: snapshot data not bound yet, waiting for syncSnapshot to fix it", vsd.Name)
			} else {
				// Dangling VSD; try to re-establish the link in the VS sync
				glog.V(4).Infof("synchronizing VolumeSnapshotData[%s]: snapshot data was bound and got unbound (by user?), waiting for syncSnapshot to fix it", vsd.Name)
			}
			// In both cases, the snapshotData is Bound and the snapshot is not bound to it.
			// Next syncSnapshot will fix it. To speed it up, we enqueue the snapshot
			// into the controller, which results in syncSnapshot to be called
			// shortly (and in the right worker goroutine).
			// This speeds up binding of created snapshotDatas - snapshotter saves
			// only the new VSD and it expects that next syncSnapshot will bind the
			// snapshot to it.
			ctrl.snapshotQueue.Add(VsToVsKey(vs))
			return nil
		} else if vs.Spec.SnapshotDataName == vsd.Name {
			// SnapshotData is bound to a snapshot properly, update status if necessary
			glog.V(4).Infof("synchronizing VolumeSnapshotData[%s]: all is bound", vsd.Name)

			// SnapshotData is already ready, everything is fine
			if IsSnapshotDataReady(vsd) {
				return nil
			}

			snapshotDataCond, err := ctrl.pollSnapshotDataStatus(vsd)
			if err != nil {
				// Nothing was saved; we will fall back into the same
				// condition in the next call to this method
				return err
			}
			if _, err := ctrl.snapshotDataConditionUpdate(vsd, snapshotDataCond); err != nil {
				return err
			}
			return nil
		} else {
			// SnapshotData is bound to a snapshot, but the snapshot is bound elsewhere
			if metav1.HasAnnotation(vsd.ObjectMeta, annDynamicallyCreatedBy) {
				// first we mark the snapshotData is deletable, so that external snapshotter
				// can start delete the backend snapshotData.
				if _, err := ctrl.markSnapshotDataDelete(vsd); err != nil {
					return err
				}
				// This snapshotData was dynamically created for this snapshot. The
				// snapshot got bound elsewhere, and thus this snapshotData is not
				// needed. Delete it.
				glog.V(4).Infof("delete backend storage snapshot data [%s]", vsd.Name)
				opName := fmt.Sprintf("delete-%s[%s]", vsd.Name, string(vsd.UID))
				ctrl.scheduleOperation(opName, func() error {
					return ctrl.deleteSnapshotDataOperation(vsd)
				})
				return nil
			} else {
				// snapshotData is bound to a snapshot, but the snapshot is bound elsewhere
				// and it's not dynamically created.
				if metav1.HasAnnotation(vsd.ObjectMeta, annBoundByController) {
					// This is part of the normal operation of the controller; the
					// controller tried to use this vsd for a snapshot but the snapshot
					// was fulfilled by another vsd. We did this; fix it.
					glog.V(4).Infof("synchronizing VolumeSnapshotData[%s]: snapshotData is bound by controller to a snapshot that is bound to another snapshotData, unbinding", vsd.Name)
					if err = ctrl.unbindSnapshotData(vsd); err != nil {
						return err
					}
					return nil
				} else {
					// The VSD must have been created with this ptr; leave it alone.
					glog.V(4).Infof("synchronizing PersistentVolume[%s]: snapshotData is bound by user to a snapshot that is bound to another snapshotData, waiting for the snapshot to get unbound", vsd.Name)
					// This just updates clears snapshotData.Spec.VolumeSnapshotRef.UID. It leaves the
					// snapshotData pre-bound to the snapshot.
					if err = ctrl.unbindSnapshotData(vsd); err != nil {
						return err
					}
					return nil
				}
			}
		}
	}
}

// syncSnapshot is the main controller method to decide what to do with a snapshot.
// It's invoked by appropriate cache.Controller callbacks when a snapshot is
// created, updated or periodically synced. We do not differentiate between
// these events.
// For easier readability, it was split into syncUnboundSnapshot and syncBoundSnapshot
// methods.
func (ctrl *SnapshotController) syncSnapshot(snapshot *storage.VolumeSnapshot) error {
	glog.V(4).Infof("synchronizing VolumeSnapshot[%s]: %s", VsToVsKey(snapshot))

	if !metav1.HasAnnotation(snapshot.ObjectMeta, annBindCompleted) {
		return ctrl.syncUnboundSnapshot(snapshot)
	} else {
		return ctrl.syncBoundSnapshot(snapshot)
	}
}

// syncUnboundSnapshot is the main controller method to decide what to do with an
// unbound snapshot.
func (ctrl *SnapshotController) syncUnboundSnapshot(snapshot *storage.VolumeSnapshot) error {
	// This is a new Snapshot that has not completed binding
	if snapshot.Spec.SnapshotDataName == "" {
		snapshotData := ctrl.findMatchSnapshotData(snapshot)
		if snapshotData == nil {
			glog.V(4).Infof("synchronizing unbound VolumeSnapshot[%s]: no SnapshotData found", VsToVsKey(snapshot))
			// No VSD could be found
			if err := ctrl.createSnapshot(snapshot); err != nil {
				return err
			}
			return nil
		} else /* vsd != nil */ {
			// Found a VSD for this snapshot
			glog.V(4).Infof("synchronizing unbound VolumeSnapshot[%s]: snapshot data %q found: %s", VsToVsKey(snapshot), snapshotData.Name)
			if err := ctrl.bind(snapshot, snapshotData); err != nil {
				// On any error saving the snapshotData or the snapshot, subsequent
				// syncSnapshot will finish the binding.
				return err
			}
			// OBSERVATION: snapshot is "Bound", snapshotData is "Bound"
			return nil
		}
	} else /* vs.Spec.SnapshotDataName != nil */ {
		// [Unit test set 2]
		// User asked for a specific VSD.
		glog.V(4).Infof("synchronizing unbound VolumeSnapshot[%s]: snapshotData %q requested", VsToVsKey(snapshot), snapshot.Spec.SnapshotDataName)
		obj, found, err := ctrl.snapshotDataStore.GetByKey(snapshot.Spec.SnapshotDataName)
		if err != nil {
			return err
		}
		if !found {
			// User asked for a VSD that does not exist.
			// OBSERVATION: VS is "Pending"
			// Retry later.
			glog.V(4).Infof("synchronizing unbound VolumeSnapshot[%s]: SnapshotData %q requested and not found, will try again next time", VsToVsKey(snapshot), snapshot.Spec.SnapshotDataName)
			condition := storage.VolumeSnapshotCondition{
				Type:    storage.VolumeSnapshotConditionUploading,
				Status:  v1.ConditionTrue,
				Message: "Requested SnapshotData not found",
			}

			if _, err := ctrl.snapshotConditionUpdate(snapshot, &condition); err != nil {
				return err
			}
			return nil
		} else {
			vsd, ok := obj.(*storage.VolumeSnapshotData)
			if !ok {
				return fmt.Errorf("cannot convert object from VolumeSnapshotData cache to VolumeSnapshotData %q!?: %+v", snapshot.Spec.SnapshotDataName, obj)
			}
			glog.V(4).Infof("synchronizing unbound VolumeSnapshot[%s]: SnapshotData %q requested and found", VsToVsKey(snapshot), snapshot.Spec.SnapshotDataName)
			if vsd.Spec.VolumeSnapshotRef == nil {
				// User asked for a VSD that is not bound
				if err = ctrl.bind(snapshot, vsd); err != nil {
					// On any error saving the snapshotData or the snapshot, subsequent
					// syncSnapshot will finish the binding.
					return err
				}
				// OBSERVATION: vs is "Bound", vsd is "Bound"
				return nil
			} else if isSnapshotDataBoundToSnapshot(vsd, snapshot) {
				// User asked for a vsd that is bound by this vs
				glog.V(4).Infof("synchronizing unbound VolumeSnapshot[%s]: SnapshotData already bound, finishing the binding", VsToVsKey(snapshot))

				// Finish the snapshotData binding by adding snapshot UID.
				if err = ctrl.bind(snapshot, vsd); err != nil {
					return err
				}
				// OBSERVATION: vs is "Bound", vsd is "Bound"
				return nil
			} else {
				// User asked for a VSD that is bound by someone else
				// OBSERVATION: vs is "Pending", vsd is "Bound"
				if !metav1.HasAnnotation(snapshot.ObjectMeta, annBoundByController) {
					glog.V(4).Infof("synchronizing unbound VolumeSnapshot[%s]: SnapshotData already bound to different snapshot by user, will retry later", VsToVsKey(snapshot))
					// User asked for a specific VSD, retry later
					condition := storage.VolumeSnapshotCondition{
						Type:    storage.VolumeSnapshotConditionUploading,
						Status:  v1.ConditionTrue,
						Message: "Requested SnapshotData is bound to other snapshot",
					}

					if _, err := ctrl.snapshotConditionUpdate(snapshot, &condition); err != nil {
						return err
					}
					return nil
				} else {
					// This should never happen because someone had to remove
					// annBindCompleted annotation on the snapshot.
					glog.V(4).Infof("synchronizing unbound VolumeSnapshot[%s]: SnapshotData already bound to different snapshot %q by controller, THIS SHOULD NEVER HAPPEN", VsToVsKey(snapshot), VsRefToVsKey(vsd.Spec.VolumeSnapshotRef))
					return fmt.Errorf("invalid binding of snapshot %q to SnapshotData %q: SnapshotData already bound by %q", VsToVsKey(snapshot), snapshot.Spec.SnapshotDataName, VsRefToVsKey(vsd.Spec.VolumeSnapshotRef))
				}
			}
		}
	}
}

// syncBoundSnapshot is the main controller method to decide what to do with a
// bound snapshot.
func (ctrl *SnapshotController) syncBoundSnapshot(snapshot *storage.VolumeSnapshot) error {
	// HasAnnotation(snapshot, annBindCompleted)
	// This snapshot has previously been bound
	if snapshot.Spec.SnapshotDataName == "" {
		// Snapshot was bound before but not any more.
		condition := &storage.VolumeSnapshotCondition{
			Type:    storage.VolumeSnapshotConditionError,
			Status:  v1.ConditionTrue,
			Reason:  "SnapshotLost",
			Message: "Bound snapshot has lost reference to VolumeSnapshotData. Data on the snapshot is lost!",
		}
		if _, err := ctrl.snapshotConditionUpdate(snapshot, condition); err != nil {
			return err
		}
		return nil
	}
	obj, found, err := ctrl.snapshotDataStore.GetByKey(snapshot.Spec.SnapshotDataName)
	if err != nil {
		return err
	}
	if !found {
		// Snapshot is bound to a non-existing snapshot data.
		condition := &storage.VolumeSnapshotCondition{
			Type:    storage.VolumeSnapshotConditionError,
			Status:  v1.ConditionTrue,
			Reason:  "SnapshotLost",
			Message: "Bound snapshot has lost its VolumeSnapshotData. Data on the snapshot is lost!",
		}
		if _, err := ctrl.snapshotConditionUpdate(snapshot, condition); err != nil {
			return err
		}
		return nil
	} else {
		snapshotData, ok := obj.(*storage.VolumeSnapshotData)
		if !ok {
			return fmt.Errorf("cannot convert object from snapshotData cache to snapshotData %q!?: %#v", snapshot.Spec.SnapshotDataName, obj)
		}

		glog.V(4).Infof("synchronizing bound VolumeSnapshot[%s]: snapshotData %q", VsToVsKey(snapshot), snapshot.Spec.SnapshotDataName)
		if snapshotData.Spec.VolumeSnapshotRef == nil {
			// VolumeSnapshot is bound but snapshotData has come unbound.
			// Or, a snapshot was bound and the controller has not received updated
			// snapshotData yet. We can't distinguish these cases.
			// Bind the snapshotData again.
			glog.V(4).Infof("synchronizing bound VolumeSnapshot[%s]: snapshotData is unbound, fixing", VsToVsKey(snapshot))
			if err = ctrl.bind(snapshot, snapshotData); err != nil {
				// Objects not saved, next syncSnapshotData or syncSnapshot will try again
				return err
			}
			return nil
		} else if snapshotData.Spec.VolumeSnapshotRef.UID == snapshot.UID {
			// All is well
			// everything should be already set.
			glog.V(4).Infof("synchronizing bound VolumeSnapshot[%s]: snapshot is already correctly bound", VsToVsKey(snapshot))

			if _, err := ctrl.syncCondition(snapshot, snapshotData); err != nil {
				// Objects not saved, next syncSnapshotData or syncSnapshot will try again
				return err
			}

			if err = ctrl.bind(snapshot, snapshotData); err != nil {
				// Objects not saved, next syncSnapshotData or syncSnapshot will try again
				return err
			}
			return nil
		} else {
			// Snapshot is bound but snapshotData has a different snapshot.
			condition := &storage.VolumeSnapshotCondition{
				Type:    storage.VolumeSnapshotConditionError,
				Status:  v1.ConditionTrue,
				Reason:  "SnapshotMisbound",
				Message: "Two snapshots are bound to the same snapshotData, this one is bound incorrectly!",
			}
			if _, err := ctrl.snapshotConditionUpdate(snapshot, condition); err != nil {
				return err
			}
			return nil
		}
	}
}

// findMatchSnapshotData looks up VolumeSnapshotData for a VolumeSnapshot named snapshotName
func (ctrl *SnapshotController) findMatchSnapshotData(vs *storage.VolumeSnapshot) *storage.VolumeSnapshotData {
	objs := ctrl.snapshotDataStore.List()
	for _, obj := range objs {
		vsd := obj.(*storage.VolumeSnapshotData)
		if isSnapshotDataBoundToSnapshot(vsd, vs) {
			glog.V(4).Infof("Error: no VolumeSnapshotData for VolumeSnapshot %s found", VsToVsKey(vs))
			return vsd
		}
	}
	return nil
}

// unbindSnapshotData rolls back previous binding of the snapshotData. This may be necessary
// when two controllers bound two snapshotDatas to single snapshot - when we detect this,
// only one binding succeeds and the second one must be rolled back.
// This method updates both Spec and Status.
// It returns on first error, it's up to the caller to implement some retry
// mechanism.
func (ctrl *SnapshotController) unbindSnapshotData(snapshotData *storage.VolumeSnapshotData) error {
	glog.V(4).Infof("updating VolumeSnapshotData[%s]: rolling back binding from %q", snapshotData.Name, VsRefToVsKey(snapshotData.Spec.VolumeSnapshotRef))

	// Save the VSD only when any modification is necessary.
	snapshotDataClone := snapshotData.DeepCopy()

	if metav1.HasAnnotation(snapshotData.ObjectMeta, annBoundByController) {
		// The snapshotData was bound by the controller.
		snapshotDataClone.Spec.VolumeSnapshotRef = nil
		delete(snapshotDataClone.Annotations, annBoundByController)
		if len(snapshotDataClone.Annotations) == 0 {
			// No annotations look better than empty annotation map (and it's easier
			// to test).
			snapshotDataClone.Annotations = nil
		}
	} else {
		// The snapshotData was pre-bound by user. Clear only the binging UID.
		snapshotDataClone.Spec.VolumeSnapshotRef.UID = ""
	}

	newVsd, err := ctrl.client.StorageV1alpha1().VolumeSnapshotDatas().Update(snapshotDataClone)
	if err != nil {
		glog.V(4).Infof("updating VolumeSnapshotData[%s]: rollback failed: %v", snapshotData.Name, err)
		return err
	}
	_, err = ctrl.storeSnapshotDataUpdate(newVsd)
	if err != nil {
		glog.V(4).Infof("updating VolumeSnapshotData[%s]: cannot update internal cache: %v", snapshotData.Name, err)
		return err
	}
	glog.V(4).Infof("updating VolumeSnapshotData[%s]: rolled back", newVsd.Name)

	return nil
}

// bind saves binding information both to the snapshotData and the snapshot and marks
// both objects as Bound. snapshotData is saved first.
// It returns on first error, it's up to the caller to implement some retry
// mechanism.
func (ctrl *SnapshotController) bind(snapshot *storage.VolumeSnapshot, snapshotData *storage.VolumeSnapshotData) error {
	var err error
	// use updatedSnapshot/updatedSnapshotData to keep the original snapshot/snapshotData for
	// logging in error cases.
	var updatedSnapshot *storage.VolumeSnapshot
	var updatedSnapshotData *storage.VolumeSnapshotData

	glog.V(4).Infof("binding snapshotData %q to snapshot %q", snapshotData.Name, VsToVsKey(snapshot))

	if updatedSnapshotData, err = ctrl.bindSnapshotDataToSnapshot(snapshotData, snapshot); err != nil {
		glog.V(3).Infof("error binding snapshotData %q to snapshot %q: failed saving the snapshotData: %v", snapshotData.Name, VsToVsKey(snapshot), err)
		return err
	}
	snapshotData = updatedSnapshotData

	if updatedSnapshot, err = ctrl.bindSnapshotToSnapshotData(snapshot, snapshotData); err != nil {
		glog.V(3).Infof("error binding snapshotData %q to snapshot %q: failed saving the snapshot: %v", snapshotData.Name, VsToVsKey(snapshot), err)
		return err
	}
	snapshot = updatedSnapshot

	glog.V(4).Infof("snapshotData %q bound to snapshot %q", snapshotData.Name, VsToVsKey(snapshot))
	return nil
}

// bindSnapshotToSnapshotData modifies the given snapshot to be bound to a SnapshotData and
// saves it to API server. The SnapshotData is not modified in this method!
func (ctrl *SnapshotController) bindSnapshotToSnapshotData(snapshot *storage.VolumeSnapshot, snapshotData *storage.VolumeSnapshotData) (*storage.VolumeSnapshot, error) {
	glog.V(4).Infof("updating VolumeSnapshot[%s]: binding to %q", VsToVsKey(snapshot), snapshotData.Name)

	dirty := false

	// Check if the snapshot was already bound (either by controller or by user)
	shouldBind := false
	if snapshotData.Name != snapshot.Spec.SnapshotDataName {
		shouldBind = true
	}

	// The snapshot from method args can be pointing to watcher cache. We must not
	// modify these, therefore create a copy.
	snapshotClone := snapshot.DeepCopy()

	if shouldBind {
		dirty = true
		// Bind the snapshot to the snapshotData
		snapshotClone.Spec.SnapshotDataName = snapshotData.Name

		// Set annBoundByController if it is not set yet
		if !metav1.HasAnnotation(snapshotClone.ObjectMeta, annBoundByController) {
			metav1.SetMetaDataAnnotation(&snapshotClone.ObjectMeta, annBoundByController, "yes")
		}
	}

	// Set annBindCompleted if it is not set yet
	if !metav1.HasAnnotation(snapshotClone.ObjectMeta, annBindCompleted) {
		metav1.SetMetaDataAnnotation(&snapshotClone.ObjectMeta, annBindCompleted, "yes")
		dirty = true
	}

	if dirty {
		glog.V(2).Infof("snapshotData %q bound to snapshot %q", snapshotData.Name, VsToVsKey(snapshot))
		newSnapshot, err := ctrl.client.StorageV1alpha1().VolumeSnapshots(snapshot.Namespace).Update(snapshotClone)
		if err != nil {
			glog.V(4).Infof("updating VolumeSnapshot[%s]: binding to %q failed: %v", VsToVsKey(snapshot), snapshotData.Name, err)
			return newSnapshot, err
		}
		_, err = ctrl.storeSnapshotUpdate(snapshotClone)
		if err != nil {
			glog.V(4).Infof("updating VolumeSnapshot[%s]: cannot update internal cache: %v", VsToVsKey(snapshot), err)
			return newSnapshot, err
		}
		glog.V(4).Infof("updating VolumeSnapshot[%s]: bound to %q", VsToVsKey(snapshot), snapshotData.Name)
		return newSnapshot, nil
	}

	glog.V(4).Infof("updating VolumeSnapshot[%s]: already bound to %q", VsToVsKey(snapshot), snapshotData.Name)
	return snapshot, nil
}

// bindSnapshotDataToSnapshot modifies given SnapshotData to be bound to a Snapshot and saves it to
// API server. The Snapshot is not modified in this method!
func (ctrl *SnapshotController) bindSnapshotDataToSnapshot(snapshotData *storage.VolumeSnapshotData, snapshot *storage.VolumeSnapshot) (*storage.VolumeSnapshotData, error) {
	glog.V(4).Infof("updating PersistentVolume[%s]: binding to %q", snapshotData.Name, VsToVsKey(snapshot))

	snapshotDataClone, dirty, err := ctrl.getBindSnapshotDataToSnapshot(snapshotData, snapshot)
	if err != nil {
		return nil, err
	}

	// Save the snapshotData only if something was changed
	if dirty {
		return ctrl.updateBindSnapshotDataToSnapshot(snapshotDataClone, snapshot, true)
	}

	glog.V(4).Infof("updating PersistentVolume[%s]: already bound to %q", snapshotData.Name, VsToVsKey(snapshot))
	return snapshotData, nil
}

// updateBindSnapshotDataToSnapshot modifies given snapshotData to be bound to a snapshot and saves it to
// API server. The snapshot is not modified in this method!
func (ctrl *SnapshotController) updateBindSnapshotDataToSnapshot(snapshotDataClone *storage.VolumeSnapshotData, snapshot *storage.VolumeSnapshot, updateCache bool) (*storage.VolumeSnapshotData, error) {
	glog.V(2).Infof("snapshot %q bound to snapshotData %q", VsToVsKey(snapshot), snapshotDataClone.Name)
	newVsd, err := ctrl.client.StorageV1alpha1().VolumeSnapshotDatas().Update(snapshotDataClone)
	if err != nil {
		glog.V(4).Infof("updating VolumeSnapshotData[%s]: binding to %q failed: %v", snapshotDataClone.Name, VsToVsKey(snapshot), err)
		return newVsd, err
	}
	if updateCache {
		_, err = ctrl.storeSnapshotDataUpdate(newVsd)
		if err != nil {
			glog.V(4).Infof("updating VolumeSnapshotData[%s]: cannot update internal cache: %v", snapshotDataClone.Name, err)
			return newVsd, err
		}
	}
	glog.V(4).Infof("updating VolumeSnapshotData[%s]: bound to %q", newVsd.Name, VsToVsKey(snapshot))
	return newVsd, nil
}

// Get new VSD object only, no API or cache update
func (ctrl *SnapshotController) getBindSnapshotDataToSnapshot(snapshotData *storage.VolumeSnapshotData, snapshot *storage.VolumeSnapshot) (*storage.VolumeSnapshotData, bool, error) {
	dirty := false

	// Check if the snapshotData was already bound (either by user or by controller)
	shouldSetBoundByController := false
	if !isSnapshotDataBoundToSnapshot(snapshotData, snapshot) {
		shouldSetBoundByController = true
	}

	// The snapshotData from method args can be pointing to watcher cache. We must not
	// modify these, therefore create a copy.
	snapshotDataClone := snapshotData.DeepCopy()

	// Bind the snapshotData to the snapshot if it is not bound yet
	if snapshotData.Spec.VolumeSnapshotRef == nil ||
		snapshotData.Spec.VolumeSnapshotRef.Name != snapshot.Name ||
		snapshotData.Spec.VolumeSnapshotRef.Namespace != snapshot.Namespace ||
		snapshotData.Spec.VolumeSnapshotRef.UID != snapshot.UID {

		snapshotRef, err := ref.GetReference(scheme.Scheme, snapshot)
		if err != nil {
			return nil, false, fmt.Errorf("unexpected error getting snapshot reference: %v", err)
		}
		snapshotDataClone.Spec.VolumeSnapshotRef = snapshotRef
		dirty = true
	}

	// Set annBoundByController if it is not set yet
	if shouldSetBoundByController && !metav1.HasAnnotation(snapshotDataClone.ObjectMeta, annBoundByController) {
		metav1.SetMetaDataAnnotation(&snapshotDataClone.ObjectMeta, annBoundByController, "yes")
		dirty = true
	}

	return snapshotDataClone, dirty, nil
}

// pollSnapshotDataStatus looks up snapshot data status.
func (ctrl *SnapshotController) pollSnapshotDataStatus(vsd *storage.VolumeSnapshotData) (*storage.VolumeSnapshotDataCondition, error) {
	plugin, err := ctrl.findSnapshotablePluginByVsd(vsd)
	if err != nil {
		return nil, err
	}
	if plugin == nil {
		// External snapshotter is requested, do nothing
		glog.V(3).Infof("external snapshotter for VolumeSnapshotData %q requested, ignoring", vsd.Name)
		return nil, nil
	}

	snapshotter, err := plugin.NewSnapshotter()
	if err != nil {
		return nil, err
	}

	return snapshotter.GetSnapshot(vsd)
}

func (ctrl *SnapshotController) deleteSnapshotDataOperation(vsd *storage.VolumeSnapshotData) error {
	glog.V(4).Infof("deleteSnapshotOperation [%s] started", vsd.Name)
	var err error

	plugin, err := ctrl.findSnapshotablePluginByVsd(vsd)
	if err != nil {
		return err
	}
	if plugin == nil {
		// External snapshotter is requested, do nothing
		glog.V(3).Infof("external snapshotter for VolumeSnapshotData %q requested, ignoring", vsd.Name)
		return nil
	}

	snapshotter, err := plugin.NewSnapshotter()
	if err != nil {
		return err
	}

	err = snapshotter.DeleteSnapshot(vsd)
	if err != nil {
		return fmt.Errorf("failed to delete snapshot data %#v, err: %v", vsd, err)
	}
	glog.Infof("snapshot data %#v deleted", vsd)

	err = ctrl.client.StorageV1alpha1().VolumeSnapshotDatas().Delete(vsd.Name, &metav1.DeleteOptions{})
	if err != nil {
		// Oops, could not delete the snapshot data and therefore the controller will
		// try to delete the snapshot data again on next update. We _could_ maintain a
		// cache of "recently deleted snapshot data" and avoid unnecessary deletion,
		// this is left out as future optimization.
		glog.V(3).Infof("failed to delete snapshot data %q from database: %v", vsd.Name, err)
		return nil
	}

	return nil
}

// scheduleOperation starts given asynchronous operation on given snapshot. It
// makes sure the operation is already not running.
func (ctrl *SnapshotController) scheduleOperation(operationName string, operation func() error) {
	glog.V(4).Infof("scheduleOperation[%s]", operationName)

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

// createSnapshot starts new asynchronous operation to create snapshot data for snapshot
func (ctrl *SnapshotController) createSnapshot(vs *storage.VolumeSnapshot) error {
	glog.V(4).Infof("createSnapshot[%s]: started", VsToVsKey(vs))
	opName := fmt.Sprintf("create-%s[%s]", VsToVsKey(vs), string(vs.UID))
	ctrl.scheduleOperation(opName, func() error {
		ctrl.createSnapshotOperation(vs)
		return nil
	})
	return nil
}

// createSnapshotOperation create a VolumeSnapshotData. This method is running in
// standalone goroutine and already has all necessary locks.
func (ctrl *SnapshotController) createSnapshotOperation(snapshot *storage.VolumeSnapshot) {
	glog.V(4).Infof("createSnapshotOperation [%s] started", VsToVsKey(snapshot))

	plugin, storageClass, err := ctrl.findSnapshotablePluginByVs(snapshot)
	if err != nil {
		ctrl.eventRecorder.Event(snapshot, v1.EventTypeWarning, events.CreateSnapshotFailed, err.Error())
		glog.V(2).Infof("error finding snapshotter plugin for snapshot %s: %v", VsToVsKey(snapshot), err)
		// The controller will retry create snapshot in every
		// syncSnapshot() call.
		return
	}
	// Add snapshotter annotation so external snapshotter know when to start
	newVs, err := ctrl.setSnapshotSnapshotter(snapshot, storageClass)
	if err != nil {
		// Save failed, the controller will retry in the next sync
		glog.V(2).Infof("error saving snapshot %s: %v", VsToVsKey(newVs), err)
		return
	}
	snapshot = newVs

	if plugin == nil {
		// findSnapshotablePluginByVs returned no error nor plugin.
		// This means that an unknown snapshotter is requested. Report an event
		// and wait for the external snapshotter
		msg := fmt.Sprintf("waiting for a snapshot data to be created, either by external snapshotter %q or manually created by system administrator", storageClass.Snapshotter)
		ctrl.eventRecorder.Event(snapshot, v1.EventTypeNormal, events.ExternalSnapshotter, msg)
		glog.V(3).Infof("create snapshot data for snapshot %q: %s", VsToVsKey(snapshot), msg)
		return
	}

	// internal snapshotter

	//  A previous createSnapshot may just have finished while we were waiting for
	//  the locks. Check that snapshot data (with deterministic name) hasn't been created
	//  yet.
	snapDataName := GetSnapshotDataNameForSnapshot(snapshot)
	snapshotData, err := ctrl.client.StorageV1alpha1().VolumeSnapshotDatas().Get(snapDataName, metav1.GetOptions{})
	if err == nil && snapshotData != nil {
		// Volume snapshot data has been already created, nothing to do.
		glog.V(4).Infof("createSnapshot [%s]: volume snapshot data already exists, skipping", VsToVsKey(snapshot))
		return
	}

	volume, err := ctrl.getVolumeFromVolumeSnapshot(snapshot)
	if err != nil {
		glog.V(3).Infof(err.Error())
		return
	}

	// Prepare a volumeSnapshotRef and persistentVolumeRef early (to fail before a snapshotData is
	// created)
	volumeSnapshotRef, err := ref.GetReference(scheme.Scheme, snapshot)
	if err != nil {
		glog.V(3).Infof("unexpected error getting snapshot reference: %v", err)
		return
	}
	persistentVolumeRef, err := ref.GetReference(scheme.Scheme, volume)
	if err != nil {
		glog.V(3).Infof("unexpected error getting volume reference: %v", err)
		return
	}

	snapshotter, err := plugin.NewSnapshotter()
	if err != nil {
		strerr := fmt.Sprintf("Failed to create snapshotter: %v", err)
		glog.V(2).Infof("failed to create snapshotter for snapshot %q with StorageClass %q: %v", VsToVsKey(snapshot), storageClass.Name, err)
		ctrl.eventRecorder.Event(snapshot, v1.EventTypeWarning, events.CreateSnapshotFailed, strerr)
		return
	}

	snapshotData, err = snapshotter.CreateSnapshot(snapDataName, volume, storageClass.Parameters)
	if err != nil {
		strerr := fmt.Sprintf("Failed to create snapshot data with StorageClass %q: %v", storageClass.Name, err)
		glog.V(2).Infof("failed to create snapshot data for snapshot %q with StorageClass %q: %v", VsToVsKey(snapshot), storageClass.Name, err)
		ctrl.eventRecorder.Event(snapshot, v1.EventTypeWarning, events.CreateSnapshotFailed, strerr)
		return
	}

	glog.V(3).Infof("VolumeSnapshotData %q for VolumeSnapshot %q created", snapshotData.Name, VsToVsKey(snapshot))

	// Bind it to the VolumeSnapshot
	snapshotData.Spec.VolumeSnapshotRef = volumeSnapshotRef
	snapshotData.Spec.PersistentVolumeRef = persistentVolumeRef

	// Add annBoundByController (used in deleting the VolumeSnapshotData)
	metav1.SetMetaDataAnnotation(&snapshotData.ObjectMeta, annBoundByController, "yes")
	metav1.SetMetaDataAnnotation(&snapshotData.ObjectMeta, annDynamicallyCreatedBy, storageClass.Snapshotter)

	// Try to create the VSD object several times
	for i := 0; i < ctrl.createSnapshotDataRetryCount; i++ {
		glog.V(4).Infof("createSnapshot [%s]: trying to save volume snapshot data %s", VsToVsKey(snapshot), snapshotData.Name)
		if _, err = ctrl.client.StorageV1alpha1().VolumeSnapshotDatas().Create(snapshotData); err == nil || apierrs.IsAlreadyExists(err) {
			// Save succeeded.
			if err != nil {
				glog.V(3).Infof("volume snapshot data %q for snapshot %q already exists, reusing", snapshotData.Name, VsToVsKey(snapshot))
				err = nil
			} else {
				glog.V(3).Infof("volume snapshot data %q for snapshot %q saved", snapshotData.Name, VsToVsKey(snapshot))
			}
			break
		}
		// Save failed, try again after a while.
		glog.V(3).Infof("failed to save volume snapshot data %q for snapshot %q: %v", snapshotData.Name, VsToVsKey(snapshot), err)
		time.Sleep(ctrl.createSnapshotDataInterval)
	}

	if err != nil {
		// Save failed. Now we have a storage asset outside of Kubernetes,
		// but we don't have appropriate volumesnapshotdata object for it.
		// Emit some event here and try to delete the storage asset several
		// times.
		strerr := fmt.Sprintf("Error creating volume snapshot data object for snapshot %s: %v. Deleting the snapshot data.", VsToVsKey(snapshot), err)
		glog.V(3).Info(strerr)
		ctrl.eventRecorder.Event(snapshot, v1.EventTypeWarning, "CreateSnapshotDataFailed", strerr)

		for i := 0; i < ctrl.createSnapshotDataRetryCount; i++ {
			if err = ctrl.deleteSnapshotDataOperation(snapshotData); err == nil {
				// Delete succeeded
				glog.V(4).Infof("createSnapshot [%s]: cleaning snapshot data %s succeeded", VsToVsKey(snapshot), snapshotData.Name)
				break
			}
			// Delete failed, try again after a while.
			glog.Infof("failed to delete snapshot data %q: %v", snapshotData.Name, err)
			time.Sleep(ctrl.createSnapshotDataInterval)
		}

		if err != nil {
			// Delete failed several times. There is an orphaned volume snapshot data and there
			// is nothing we can do about it.
			strerr := fmt.Sprintf("Error cleaning volume snapshot data for snapshot %s: %v. Please delete manually.", VsToVsKey(snapshot), err)
			glog.Error(strerr)
			ctrl.eventRecorder.Event(snapshot, v1.EventTypeWarning, events.SnapshotDataCleanupFailed, strerr)
		}
	} else {
		glog.V(2).Infof("VolumeSnapshotData %q created for VolumeSnapshot %q", snapshotData.Name, VsToVsKey(snapshot))
		msg := fmt.Sprintf("Successfully create snapshot data  %s using %s", snapshotData.Name, plugin.GetPluginName())
		ctrl.eventRecorder.Event(snapshot, v1.EventTypeNormal, events.CreateSnapshotSucceeded, msg)
	}
}

// markSnapshotDataDelete saves
// snapshotData.Annotations[annSnapshotDataShouldDelete] = "yes"
func (ctrl *SnapshotController) markSnapshotDataDelete(snapshotData *storage.VolumeSnapshotData) (*storage.VolumeSnapshotData, error) {
	if _, ok := snapshotData.Annotations[annSnapshotDataShouldDelete]; ok {
		// annotation is already set, nothing to do
		return snapshotData, nil
	}

	// The snapshotData from method args can be pointing to watcher cache. We must not
	// modify these, therefore create a copy.
	vsdClone := snapshotData.DeepCopy()
	metav1.SetMetaDataAnnotation(&vsdClone.ObjectMeta, annSnapshotDataShouldDelete, "yes")
	newVsd, err := ctrl.client.StorageV1alpha1().VolumeSnapshotDatas().Update(vsdClone)
	if err != nil {
		return newVsd, err
	}
	_, err = ctrl.storeSnapshotDataUpdate(newVsd)
	if err != nil {
		return newVsd, err
	}
	return newVsd, nil
}

// setSnapshotSnapshotter saves
// snapshot.Annotations[annStorageSnapshotter] = class.Snapshotter
func (ctrl *SnapshotController) setSnapshotSnapshotter(snapshot *storage.VolumeSnapshot, class *storagev1.StorageClass) (*storage.VolumeSnapshot, error) {
	if val, ok := snapshot.Annotations[annStorageSnapshotter]; ok && val == class.Snapshotter {
		// annotation is already set, nothing to do
		return snapshot, nil
	}

	// The snapshot from method args can be pointing to watcher cache. We must not
	// modify these, therefore create a copy.
	vsClone := snapshot.DeepCopy()
	metav1.SetMetaDataAnnotation(&vsClone.ObjectMeta, annStorageSnapshotter, class.Snapshotter)
	newVs, err := ctrl.client.StorageV1alpha1().VolumeSnapshots(snapshot.Namespace).Update(vsClone)
	if err != nil {
		return newVs, err
	}
	_, err = ctrl.storeSnapshotUpdate(newVs)
	if err != nil {
		return newVs, err
	}
	return newVs, nil
}

// findSnapshottablePlugin finds a snapshottable plugin for a given VolumeSnapshotData. It returns
// either the snapshottable plugin or nil when an external snapshotter is requested.
func (ctrl *SnapshotController) findSnapshotablePluginByVsd(vsd *storage.VolumeSnapshotData) (vol.SnapshotableVolumePlugin, error) {
	// Find a plugin. Try to find the same plugin that create the snapshotData
	var plugin vol.SnapshotableVolumePlugin
	if metav1.HasAnnotation(vsd.ObjectMeta, annDynamicallyCreatedBy) {
		snapshotPluginName := vsd.Annotations[annDynamicallyCreatedBy]
		if snapshotPluginName != "" {
			plugin, err := ctrl.volumePluginMgr.FindSnapshottablePluginByName(snapshotPluginName)
			if err != nil {
				if !strings.HasPrefix(snapshotPluginName, "kubernetes.io/") {
					// External snapshotter is requested, do not report error.
					return nil, nil
				}
				return nil, err
			}
			return plugin, nil
		}
	}

	volume, err := GetPvFromSnapshotData(vsd, ctrl.client)
	if err != nil {
		return nil, err
	}
	// The plugin that create the snapshot data was not found or the snapshot data
	// was not dynamically created. Try to find a plugin by spec.
	spec := vol.NewSpecFromPersistentVolume(volume, false)
	plugin, err = ctrl.volumePluginMgr.FindSnapshottablePluginBySpec(spec)
	if err != nil {
		// No deleter found. Emit an event and mark the snapshot Failed.
		return nil, fmt.Errorf("error getting Snapshottable volume plugin for volume %q: %v", volume.Name, err)
	}
	return plugin, nil
}

// findSnapshotablePluginByVs finds a snapshotter plugin for a given snapshot.
// It returns either the snapshotter plugin or nil when an external
// snapshotter is requested.
func (ctrl *SnapshotController) findSnapshotablePluginByVs(snapshot *storage.VolumeSnapshot) (vol.SnapshotableVolumePlugin, *storagev1.StorageClass, error) {
	class, err := ctrl.getClassFromVolumeSnapshot(snapshot)
	if err != nil {
		return nil, nil, err
	}

	// Find a plugin for the class
	plugin, err := ctrl.volumePluginMgr.FindSnapshottablePluginByName(class.Provisioner)
	if err != nil {
		if !strings.HasPrefix(class.Provisioner, "kubernetes.io/") {
			// External provisioner is requested, do not report error
			return nil, class, nil
		}
		return nil, class, err
	}
	return plugin, class, nil
}

// getClassFromVolumeSnapshot is a helper function to get storage class from VolumeSnapshot.
func (ctrl *SnapshotController) getClassFromVolumeSnapshot(snapshot *storage.VolumeSnapshot) (*storagev1.StorageClass, error) {
	pvc, err := ctrl.getClaimFromVolumeSnapshot(snapshot)
	if err != nil {
		return nil, err
	}

	className := pvc.Spec.StorageClassName
	class, err := ctrl.client.StorageV1().StorageClasses().Get(*className, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve storage class %s from the API server: %q", className, err)
	}
	return class, nil
}

// getClaimFromVolumeSnapshot is a helper function to get PV from VolumeSnapshot.
func (ctrl *SnapshotController) getClaimFromVolumeSnapshot(snapshot *storage.VolumeSnapshot) (*v1.PersistentVolumeClaim, error) {
	pvcName := snapshot.Spec.PersistentVolumeClaimName
	if pvcName == "" {
		return nil, fmt.Errorf("the PVC name is not specified in snapshot %s", VsToVsKey(snapshot))
	}

	pvc, err := ctrl.claimLister.PersistentVolumeClaims(snapshot.Namespace).Get(pvcName)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve PVC %s from the API server: %q", pvcName, err)
	}
	if pvc.Status.Phase != v1.ClaimBound {
		return nil, fmt.Errorf("the PVC %s not yet bound to a PV, will not attempt to take a snapshot yet", pvcName)
	}

	return pvc, nil
}

// getVolumeFromVolumeSnapshot is a helper function to get PV from VolumeSnapshot.
func (ctrl *SnapshotController) getVolumeFromVolumeSnapshot(snapshot *storage.VolumeSnapshot) (*v1.PersistentVolume, error) {
	pvc, err := ctrl.getClaimFromVolumeSnapshot(snapshot)
	if err != nil {
		return nil, err
	}

	pvName := pvc.Spec.VolumeName
	pv, err := ctrl.volumeLister.Get(pvName)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve PV %s from the API server: %q", pvName, err)
	}
	return pv, nil
}

// syncCondition syncs condition between snapshot and snapshotData.
func (ctrl *SnapshotController) syncCondition(snapshot *storage.VolumeSnapshot, snapshotData *storage.VolumeSnapshotData) (*storage.VolumeSnapshot, error) {
	conditions := snapshotData.Status.Conditions
	if conditions == nil || len(conditions) == 0 {
		glog.V(4).Infof("syncCondition: snapshotData %v condition is empty, no need to sync snapshot %v", snapshotData.Name, VsToVsKey(snapshot))
		return snapshot, nil
	}

	lastCondition := conditions[len(conditions)-1]
	snapshotCond := TransformDataCondition2SnapshotCondition(&lastCondition)

	return ctrl.snapshotConditionUpdate(snapshot, snapshotCond)
}

func (ctrl *SnapshotController) snapshotConditionUpdate(snapshot *storage.VolumeSnapshot, condition *storage.VolumeSnapshotCondition) (*storage.VolumeSnapshot, error) {
	if condition == nil {
		return snapshot, nil
	}

	snapshotCopy := snapshot.DeepCopy()

	// no condition changes.
	if !UpdateSnapshotCondition(&snapshotCopy.Status, condition) {
		return snapshotCopy, nil
	}

	glog.V(2).Infof("Updating snapshot condition for %s/%s to (%s==%s)", snapshot.Namespace, snapshot.Name, condition.Type, condition.Status)

	newVs, err := ctrl.client.StorageV1alpha1().VolumeSnapshots(snapshot.Namespace).UpdateStatus(snapshotCopy)
	if err != nil {
		glog.V(4).Infof("updating VolumeSnapshot[%s] status failed: %v", VsToVsKey(snapshot), err)
		return newVs, err
	}
	_, err = ctrl.storeSnapshotUpdate(newVs)
	if err != nil {
		glog.V(4).Infof("updating VolumeSnapshot[%s] status: cannot update internal cache: %v", VsToVsKey(snapshot), err)
		return newVs, err
	}

	glog.V(2).Infof("VolumeSnapshot %q status update success", VsToVsKey(snapshot))
	return newVs, nil
}

func (ctrl *SnapshotController) snapshotDataConditionUpdate(snapshotData *storage.VolumeSnapshotData, condition *storage.VolumeSnapshotDataCondition) (*storage.VolumeSnapshotData, error) {
	if condition == nil {
		return snapshotData, nil
	}

	snapshotDataCopy := snapshotData.DeepCopy()

	// no condition changes.
	if !UpdateSnapshotDataCondition(&snapshotDataCopy.Status, condition) {
		return snapshotDataCopy, nil
	}

	glog.V(2).Infof("Updating snapshot condition for %s to (%s==%s)", snapshotData.Name, condition.Type, condition.Status)

	newVsd, err := ctrl.client.StorageV1alpha1().VolumeSnapshotDatas().UpdateStatus(snapshotDataCopy)
	if err != nil {
		glog.V(4).Infof("updating VolumeSnapshotData[%s] status failed: %v", newVsd.Name, err)
		return newVsd, err
	}
	_, err = ctrl.storeSnapshotDataUpdate(newVsd)
	if err != nil {
		glog.V(4).Infof("updating VolumeSnapshotData[%s] status: cannot update internal cache: %v", snapshotData.Name, err)
		return newVsd, err
	}

	glog.V(2).Infof("VolumeSnapshotData %q status update success", snapshotData.Name)
	return newVsd, nil
}

func (ctrl *SnapshotController) GetPodsDir() string {
	return ""
}

func (ctrl *SnapshotController) GetServiceAccountTokenFunc() func(_, _ string, _ *authenticationv1.TokenRequest) (*authenticationv1.TokenRequest, error) {
	return func(_, _ string, _ *authenticationv1.TokenRequest) (*authenticationv1.TokenRequest, error) {
		return nil, fmt.Errorf("GetServiceAccountToken unsupported in SnapshotController")
	}
}
