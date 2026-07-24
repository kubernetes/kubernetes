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
	"slices"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	storageinformers "k8s.io/client-go/informers/storage/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	storagehelpers "k8s.io/component-helpers/storage/volume"
	"k8s.io/component-helpers/storage/volume/assumecache"
	csitrans "k8s.io/csi-translation-lib"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/volume/common"
	"k8s.io/kubernetes/pkg/controller/volume/persistentvolume/metrics"
	"k8s.io/kubernetes/pkg/util/goroutinemap"
	"k8s.io/kubernetes/pkg/util/slice"
	vol "k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"

	"k8s.io/klog/v2"
)

// This file contains the controller base functionality, i.e. framework to
// process PV/PVC added/updated/deleted events. The real binding, provisioning,
// recycling and deleting is done in pv_controller.go

// ControllerParameters contains arguments for creation of a new
// PersistentVolume controller.
type ControllerParameters struct {
	KubeClient                clientset.Interface
	SyncPeriod                time.Duration
	VolumePlugins             []vol.VolumePlugin
	VolumeInformer            coreinformers.PersistentVolumeInformer
	ClaimInformer             coreinformers.PersistentVolumeClaimInformer
	ClassInformer             storageinformers.StorageClassInformer
	PodInformer               coreinformers.PodInformer
	NodeInformer              coreinformers.NodeInformer
	EnableDynamicProvisioning bool
}

// NewController creates a new PersistentVolume controller
func NewController(ctx context.Context, p ControllerParameters) (*PersistentVolumeController, error) {
	eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
	eventRecorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "persistentvolume-controller"})

	controller := &PersistentVolumeController{
		kubeClient:                    p.KubeClient,
		eventBroadcaster:              eventBroadcaster,
		eventRecorder:                 eventRecorder,
		runningOperations:             goroutinemap.NewGoRoutineMap(true /* exponentialBackOffOnError */),
		enableDynamicProvisioning:     p.EnableDynamicProvisioning,
		createProvisionedPVRetryCount: createProvisionedPVRetryCount,
		createProvisionedPVInterval:   createProvisionedPVInterval,
		claimQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "claims"},
		),
		volumeQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "volumes"},
		),
		resyncPeriod:        p.SyncPeriod,
		operationTimestamps: metrics.NewOperationStartTimeCache(),
	}

	// Prober is nil because PV is not aware of Flexvolume.
	if err := controller.volumePluginMgr.InitPlugins(p.VolumePlugins, nil /* prober */, controller); err != nil {
		return nil, fmt.Errorf("could not initialize volume plugins for PersistentVolume Controller: %w", err)
	}

	logger := klog.FromContext(ctx)

	if err := p.VolumeInformer.Informer().AddIndexers(cache.Indexers{accessModesIndex: accessModesIndexFunc}); err != nil {
		return nil, fmt.Errorf("could not add accessmodes indexer to the PV informer: %w", err)
	}
	pvCache, err := assumecache.NewAssumeCache[*v1.PersistentVolume](logger, p.VolumeInformer.Informer(), schema.GroupResource{Resource: "persistentvolumes"})
	if err != nil {
		return nil, fmt.Errorf("could not create the PV assume cache: %w", err)
	}
	controller.volumes = persistentVolumeOrderedIndex{pvCache}
	pvcCache, err := assumecache.NewAssumeCache[*v1.PersistentVolumeClaim](logger, p.ClaimInformer.Informer(), schema.GroupResource{Resource: "persistentvolumeclaims"})
	if err != nil {
		return nil, fmt.Errorf("could not create the PVC assume cache: %w", err)
	}
	controller.claims = pvcCache

	_, err = p.VolumeInformer.Informer().AddEventHandlerWithOptions(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    func(obj interface{}) { controller.enqueueWork(ctx, controller.volumeQueue, obj) },
			UpdateFunc: func(oldObj, newObj interface{}) { controller.enqueueWork(ctx, controller.volumeQueue, newObj) },
			DeleteFunc: func(obj interface{}) { controller.volumeDeleted(ctx, obj) },
		},
		cache.HandlerOptions{Logger: &logger},
	)
	if err != nil {
		return nil, fmt.Errorf("could not add volume event handler: %w", err)
	}
	controller.volumeLister = p.VolumeInformer.Lister()
	controller.volumeListerSynced = p.VolumeInformer.Informer().HasSynced

	_, err = p.ClaimInformer.Informer().AddEventHandlerWithOptions(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    func(obj interface{}) { controller.enqueueWork(ctx, controller.claimQueue, obj) },
			UpdateFunc: func(oldObj, newObj interface{}) { controller.enqueueWork(ctx, controller.claimQueue, newObj) },
			DeleteFunc: func(obj interface{}) { controller.claimDeleted(ctx, obj) },
		},
		cache.HandlerOptions{Logger: &logger},
	)
	if err != nil {
		return nil, fmt.Errorf("could not add claim event handler: %w", err)
	}
	controller.claimLister = p.ClaimInformer.Lister()
	controller.claimListerSynced = p.ClaimInformer.Informer().HasSynced

	controller.classLister = p.ClassInformer.Lister()
	controller.classListerSynced = p.ClassInformer.Informer().HasSynced
	controller.podLister = p.PodInformer.Lister()
	controller.podIndexer = p.PodInformer.Informer().GetIndexer()
	controller.podListerSynced = p.PodInformer.Informer().HasSynced
	controller.NodeLister = p.NodeInformer.Lister()
	controller.NodeListerSynced = p.NodeInformer.Informer().HasSynced

	// This custom indexer will index pods by its PVC keys. Then we don't need
	// to iterate all pods every time to find pods which reference given PVC.
	if err := common.AddPodPVCIndexerIfNotPresent(controller.podIndexer); err != nil {
		return nil, fmt.Errorf("could not initialize attach detach controller: %w", err)
	}

	csiTranslator := csitrans.New()
	controller.translator = csiTranslator
	controller.csiMigratedPluginManager = csimigration.NewPluginManager(csiTranslator)

	return controller, nil
}

// enqueueWork adds volume or claim to given work queue.
func (ctrl *PersistentVolumeController) enqueueWork(ctx context.Context, queue workqueue.TypedInterface[string], obj interface{}) {
	// Beware of "xxx deleted" events
	logger := klog.FromContext(ctx)
	if unknown, ok := obj.(cache.DeletedFinalStateUnknown); ok && unknown.Obj != nil {
		obj = unknown.Obj
	}
	objName, err := controller.KeyFunc(obj)
	if err != nil {
		logger.Error(err, "Failed to get key from object")
		return
	}
	logger.V(5).Info("Enqueued for sync", "objName", objName)
	queue.Add(objName)
}

// updateVolume runs in worker thread and handles "volume added",
// "volume updated" and "periodic sync" events.
func (ctrl *PersistentVolumeController) updateVolume(ctx context.Context, volume *v1.PersistentVolume) error {
	logger := klog.FromContext(ctx)
	err := ctrl.syncVolume(ctx, volume)
	if err != nil {
		if errors.IsConflict(err) {
			// Version conflict error happens quite often and the controller
			// recovers from it easily.
			logger.V(3).Info("Could not sync volume", "volumeName", volume.Name, "err", err)
		} else {
			logger.Error(err, "Could not sync volume", "volumeName", volume.Name)
		}
		return err
	}
	return nil
}

// volumeDeleted handles the informer "volume deleted" event.
func (ctrl *PersistentVolumeController) volumeDeleted(ctx context.Context, obj interface{}) {
	logger := klog.FromContext(ctx)
	volume, ok := obj.(*v1.PersistentVolume)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleErrorWithLogger(logger, nil, "Couldn't get object from tombstone", "obj", obj)
			return
		}
		volume, ok = tombstone.Obj.(*v1.PersistentVolume)
		if !ok {
			utilruntime.HandleErrorWithLogger(logger, nil, "Tombstone contained object that is not a volume", "obj", tombstone.Obj)
			return
		}
	}
	logger.V(4).Info("volume deleted", "volumeName", volume.Name)
	// record deletion metric if a deletion start timestamp is in the cache
	// the following calls will be a no-op if there is nothing for this volume in the cache
	// end of timestamp cache entry lifecycle, "RecordMetric" will do the clean
	metrics.RecordMetric(volume.Name, &ctrl.operationTimestamps, nil)

	if volume.Spec.ClaimRef == nil {
		return
	}
	// sync the claim when its volume is deleted. Explicitly syncing the
	// claim here in response to volume deletion prevents the claim from
	// waiting until the next sync period for its Lost status.
	claimKey := claimrefToClaimKey(volume.Spec.ClaimRef)
	logger.V(5).Info("volumeDeleted: scheduling sync of claim", "PVC", klog.KRef(volume.Spec.ClaimRef.Namespace, volume.Spec.ClaimRef.Name), "volumeName", volume.Name)
	ctrl.claimQueue.Add(claimKey)
}

// updateClaim runs in worker thread and handles "claim added",
// "claim updated" and "periodic sync" events.
func (ctrl *PersistentVolumeController) updateClaim(ctx context.Context, claim *v1.PersistentVolumeClaim) error {
	logger := klog.FromContext(ctx)
	err := ctrl.syncClaim(ctx, claim)
	if err != nil {
		if errors.IsConflict(err) {
			// Version conflict error happens quite often and the controller
			// recovers from it easily.
			logger.V(3).Info("Could not sync claim", "PVC", klog.KObj(claim), "err", err)
		} else {
			logger.Error(err, "Could not sync claim", "PVC", klog.KObj(claim))
		}
		return err
	}
	return nil
}

// Unit test [5-5] [5-6] [5-7]
// claimDeleted handles the informer "claim deleted" event.
func (ctrl *PersistentVolumeController) claimDeleted(ctx context.Context, obj interface{}) {
	logger := klog.FromContext(ctx)
	claim, ok := obj.(*v1.PersistentVolumeClaim)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleErrorWithLogger(logger, nil, "Couldn't get object from tombstone", "obj", obj)
			return
		}
		claim, ok = tombstone.Obj.(*v1.PersistentVolumeClaim)
		if !ok {
			utilruntime.HandleErrorWithLogger(logger, nil, "Tombstone contained object that is not a claim", "obj", tombstone.Obj)
			return
		}
	}
	claimKey := claimToClaimKey(claim)
	logger.V(4).Info("Claim deleted", "PVC", klog.KObj(claim))
	// clean any possible unfinished provision start timestamp from cache
	// Unit test [5-8] [5-9]
	ctrl.operationTimestamps.Delete(claimKey)

	volumeName := claim.Spec.VolumeName
	if volumeName == "" {
		logger.V(5).Info("claimDeleted: volume not bound", "PVC", klog.KObj(claim))
		return
	}

	// sync the volume when its claim is deleted.  Explicitly sync'ing the
	// volume here in response to claim deletion prevents the volume from
	// waiting until the next sync period for its Release.
	logger.V(5).Info("claimDeleted: scheduling sync of volume", "PVC", klog.KObj(claim), "volumeName", volumeName)
	ctrl.volumeQueue.Add(volumeName)
}

// Run starts all of this controller's control loops
func (ctrl *PersistentVolumeController) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()

	// Start events processing pipeline.
	ctrl.eventBroadcaster.StartStructuredLogging(3)
	ctrl.eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: ctrl.kubeClient.CoreV1().Events("")})
	defer ctrl.eventBroadcaster.Shutdown()

	logger := klog.FromContext(ctx)
	logger.Info("Starting persistent volume controller")

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down persistent volume controller")
		ctrl.claimQueue.ShutDown()
		ctrl.volumeQueue.ShutDown()
		wg.Wait()
	}()

	if !cache.WaitForNamedCacheSyncWithContext(ctx, ctrl.volumeListerSynced, ctrl.claimListerSynced, ctrl.classListerSynced, ctrl.podListerSynced, ctrl.NodeListerSynced) {
		return
	}

	metrics.Register(ctrl.volumes, ctrl.claims, &ctrl.volumePluginMgr)
	volumeutil.RegisterMetrics()

	wg.Go(func() {
		wait.Until(func() { ctrl.resync(ctx) }, ctrl.resyncPeriod, ctx.Done())
	})
	// Only the volume worker fans out; the claim worker stays single (serial
	// match-and-reserve).
	wg.Go(func() {
		wait.UntilWithContext(ctx, ctrl.volumeWorker, time.Second)
	})
	wg.Go(func() {
		wait.UntilWithContext(ctx, ctrl.claimWorker, time.Second)
	})
	<-ctx.Done()
}

func (ctrl *PersistentVolumeController) updateClaimMigrationAnnotations(ctx context.Context,
	claim *v1.PersistentVolumeClaim) (*v1.PersistentVolumeClaim, error) {
	// TODO: update[Claim|Volume]MigrationAnnotations can be optimized to not
	// copy the claim/volume if no modifications are required. Though this
	// requires some refactoring as well as an interesting change in the
	// semantics of the function which may be undesirable. If no copy is made
	// when no modifications are required this function could sometimes return a
	// copy of the volume and sometimes return a ref to the original
	claimClone := claim.DeepCopy()
	logger := klog.FromContext(ctx)
	modified := updateMigrationAnnotations(logger, ctrl.csiMigratedPluginManager, ctrl.translator, claimClone.Annotations, true)
	if !modified {
		return claimClone, nil
	}
	newClaim, err := ctrl.kubeClient.CoreV1().PersistentVolumeClaims(claimClone.Namespace).Update(ctx, claimClone, metav1.UpdateOptions{})
	if err != nil {
		return nil, fmt.Errorf("persistent Volume Controller can't anneal migration annotations: %v", err)
	}
	if err = ctrl.claims.AssumeWritten(newClaim); err != nil {
		return nil, fmt.Errorf("persistent Volume Controller can't anneal migration annotations: %v", err)
	}
	return newClaim, nil
}

func (ctrl *PersistentVolumeController) updateVolumeMigrationAnnotationsAndFinalizers(ctx context.Context,
	volume *v1.PersistentVolume) (*v1.PersistentVolume, error) {
	volumeClone := volume.DeepCopy()
	logger := klog.FromContext(ctx)
	annModified := updateMigrationAnnotations(logger, ctrl.csiMigratedPluginManager, ctrl.translator, volumeClone.Annotations, false)
	modifiedFinalizers, finalizersModified := modifyDeletionFinalizers(logger, ctrl.csiMigratedPluginManager, volumeClone)
	if !annModified && !finalizersModified {
		return volumeClone, nil
	}
	if finalizersModified {
		volumeClone.ObjectMeta.SetFinalizers(modifiedFinalizers)
	}
	newVol, err := ctrl.kubeClient.CoreV1().PersistentVolumes().Update(ctx, volumeClone, metav1.UpdateOptions{})
	if err != nil {
		return nil, fmt.Errorf("persistent Volume Controller can't anneal migration annotations or finalizer: %v", err)
	}
	if err = ctrl.volumes.AssumeWritten(newVol); err != nil {
		return nil, fmt.Errorf("persistent Volume Controller can't anneal migration annotations or finalizer: %v", err)
	}
	return newVol, nil
}

// modifyDeletionFinalizers updates the finalizers based on the reclaim policy and if it is a in-tree volume or not.
// The in-tree PV deletion protection finalizer is only added if the reclaimPolicy associated with the PV is `Delete`.
// The in-tree PV deletion protection finalizer is removed if the reclaimPolicy associated with the PV is `Retain` or
// `Recycle`, removing the finalizer is necessary to reflect the recalimPolicy updates on the PV.
// The method also removes any external PV Deletion Protection finalizers added on the PV, this represents CSI migration
// rollback/disable scenarios.
func modifyDeletionFinalizers(logger klog.Logger, cmpm CSIMigratedPluginManager, volume *v1.PersistentVolume) ([]string, bool) {
	modified := false
	var outFinalizers []string
	if !metav1.HasAnnotation(volume.ObjectMeta, storagehelpers.AnnDynamicallyProvisioned) {
		// PV deletion protection finalizer is currently supported only for dynamically
		// provisioned volumes.
		return volume.Finalizers, false
	}
	if volume.Finalizers != nil {
		outFinalizers = append(outFinalizers, volume.Finalizers...)
	}
	provisioner := volume.Annotations[storagehelpers.AnnDynamicallyProvisioned]
	if cmpm.IsMigrationEnabledForPlugin(provisioner) {
		// Remove in-tree delete finalizer on the PV as migration is enabled.
		if slices.Contains(outFinalizers, storagehelpers.PVDeletionInTreeProtectionFinalizer) {
			outFinalizers = slice.RemoveString(outFinalizers, storagehelpers.PVDeletionInTreeProtectionFinalizer, nil)
			modified = true
		}
		return outFinalizers, modified
	}
	// Check if it is a in-tree volume.
	if !strings.HasPrefix(provisioner, "kubernetes.io/") {
		// The provision plugin does not begin with known in-tree plugin volume prefix annotation.
		return volume.Finalizers, false
	}
	reclaimPolicy := volume.Spec.PersistentVolumeReclaimPolicy
	// Add back the in-tree PV deletion protection finalizer if does not already exists
	if reclaimPolicy == v1.PersistentVolumeReclaimDelete && !slices.Contains(outFinalizers, storagehelpers.PVDeletionInTreeProtectionFinalizer) {
		logger.V(4).Info("Adding in-tree pv deletion protection finalizer on volume", "volumeName", volume.Name)
		outFinalizers = append(outFinalizers, storagehelpers.PVDeletionInTreeProtectionFinalizer)
		modified = true
	} else if (reclaimPolicy == v1.PersistentVolumeReclaimRetain || reclaimPolicy == v1.PersistentVolumeReclaimRecycle) && slices.Contains(outFinalizers, storagehelpers.PVDeletionInTreeProtectionFinalizer) {
		// Remove the in-tree PV deletion protection finalizer if the reclaim policy is 'Retain' or 'Recycle'
		logger.V(4).Info("Removing in-tree pv deletion protection finalizer on volume", "volumeName", volume.Name)
		outFinalizers = slice.RemoveString(outFinalizers, storagehelpers.PVDeletionInTreeProtectionFinalizer, nil)
		modified = true
	}
	// Remove the external PV deletion protection finalizer
	if slices.Contains(outFinalizers, storagehelpers.PVDeletionProtectionFinalizer) {
		logger.V(4).Info("Removing external pv deletion protection finalizer on volume", "volumeName", volume.Name)
		outFinalizers = slice.RemoveString(outFinalizers, storagehelpers.PVDeletionProtectionFinalizer, nil)
		modified = true
	}
	return outFinalizers, modified
}

// updateMigrationAnnotations takes an Annotations map and checks for a
// provisioner name using the provisionerKey. It will then add a
// "pv.kubernetes.io/migrated-to" annotation if migration with the CSI
// driver name for that provisioner is "on" based on feature flags, it will also
// remove the annotation is migration is "off" for that provisioner in rollback
// scenarios. Returns true if the annotations map was modified and false otherwise.
func updateMigrationAnnotations(logger klog.Logger, cmpm CSIMigratedPluginManager, translator CSINameTranslator, ann map[string]string, claim bool) bool {
	var csiDriverName string
	var err error

	if ann == nil {
		// No annotations so we can't get the provisioner and don't know whether
		// this is migrated - no change
		return false
	}
	var provisionerKey string
	if claim {
		provisionerKey = storagehelpers.AnnStorageProvisioner
	} else {
		provisionerKey = storagehelpers.AnnDynamicallyProvisioned
	}
	provisioner, ok := ann[provisionerKey]
	if !ok {
		if claim {
			// Also check beta AnnStorageProvisioner annontation to make sure
			provisioner, ok = ann[storagehelpers.AnnBetaStorageProvisioner]
			if !ok {
				return false
			}
		} else {
			// Volume Statically provisioned.
			return false
		}
	}

	migratedToDriver := ann[storagehelpers.AnnMigratedTo]
	if cmpm.IsMigrationEnabledForPlugin(provisioner) {
		csiDriverName, err = translator.GetCSINameFromInTreeName(provisioner)
		if err != nil {
			logger.Error(err, "Could not update volume migration annotations. Migration enabled for plugin but could not find corresponding driver name", "plugin", provisioner)
			return false
		}
		if migratedToDriver != csiDriverName {
			ann[storagehelpers.AnnMigratedTo] = csiDriverName
			return true
		}
	} else {
		if migratedToDriver != "" {
			// Migration annotation exists but the driver isn't migrated currently
			delete(ann, storagehelpers.AnnMigratedTo)
			return true
		}
	}
	return false
}

// volumeWorker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (ctrl *PersistentVolumeController) volumeWorker(ctx context.Context) {
	for ctrl.processNextVolumeWorkItem(ctx) {
	}
}

// processNextVolumeWorkItem deals with one key off the volumeQueue. It returns false when it's time to quit.
func (ctrl *PersistentVolumeController) processNextVolumeWorkItem(ctx context.Context) bool {
	key, quit := ctrl.volumeQueue.Get()
	if quit {
		return false
	}
	defer ctrl.volumeQueue.Done(key)

	err := ctrl.syncVolumeByKey(ctx, key)
	if err == nil {
		ctrl.volumeQueue.Forget(key)
		return true
	}

	ctrl.volumeQueue.AddRateLimited(key)

	return true
}

// syncVolumeByKey processes a single volume identified by key from the queue.
func (ctrl *PersistentVolumeController) syncVolumeByKey(ctx context.Context, key string) error {
	logger := klog.FromContext(ctx)
	logger.V(5).Info("syncVolumeByKey", "volumeKey", key)

	volume, assumed, err := ctrl.volumes.GetAssumed(key)
	if err != nil {
		if errors.IsNotFound(err) {
			// The volume no longer exists; its deletion (and any follow-up
			// work) was already handled by the informer delete handler.
			logger.V(4).Info("Volume deleted", "volumeKey", key)
			return nil
		}
		logger.V(2).Info("Error getting volume from cache", "volumeKey", key, "err", err)
		return err
	}
	if assumed {
		// Serving our own write not yet observed by the informer.
		// The informer catching up will re-enqueue.
		return nil
	}
	return ctrl.updateVolume(ctx, volume)
}

// claimWorker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (ctrl *PersistentVolumeController) claimWorker(ctx context.Context) {
	for ctrl.processNextClaimWorkItem(ctx) {
	}
}

// processNextClaimWorkItem deals with one key off the claimQueue. It returns false when it's time to quit.
func (ctrl *PersistentVolumeController) processNextClaimWorkItem(ctx context.Context) bool {
	key, quit := ctrl.claimQueue.Get()
	if quit {
		return false
	}
	defer ctrl.claimQueue.Done(key)

	err := ctrl.syncClaimByKey(ctx, key)
	if err == nil {
		ctrl.claimQueue.Forget(key)
		return true
	}

	ctrl.claimQueue.AddRateLimited(key)

	return true
}

// syncClaimByKey processes a single claim identified by key from the queue.
func (ctrl *PersistentVolumeController) syncClaimByKey(ctx context.Context, key string) error {
	logger := klog.FromContext(ctx)
	logger.V(5).Info("syncClaimByKey", "claimKey", key)

	claim, assumed, err := ctrl.claims.GetAssumed(key)
	if err != nil {
		if errors.IsNotFound(err) {
			// The claim no longer exists; its deletion (and any follow-up
			// work) was already handled by the informer delete handler.
			logger.V(4).Info("Claim deleted", "claimKey", key)
			return nil
		}
		logger.V(2).Info("Error getting claim from cache", "claimKey", key, "err", err)
		return err
	}
	if assumed {
		// Serving our own write not yet observed by the informer.
		// The informer catching up will re-enqueue.
		return nil
	}
	return ctrl.updateClaim(ctx, claim)
}

// resync supplements short resync period of shared informers - we don't want
// all consumers of PV/PVC shared informer to have a short resync period,
// therefore we do our own.
func (ctrl *PersistentVolumeController) resync(ctx context.Context) {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Resyncing PV controller")

	pvcs, err := ctrl.claimLister.List(labels.NewSelector())
	if err != nil {
		logger.Info("Cannot list claims", "err", err)
		return
	}
	for _, pvc := range pvcs {
		ctrl.enqueueWork(ctx, ctrl.claimQueue, pvc)
	}

	pvs, err := ctrl.volumeLister.List(labels.NewSelector())
	if err != nil {
		logger.Info("Cannot list persistent volumes", "err", err)
		return
	}
	for _, pv := range pvs {
		ctrl.enqueueWork(ctx, ctrl.volumeQueue, pv)
	}
}

// setClaimProvisioner saves
// claim.Annotations["volume.kubernetes.io/storage-provisioner"] = class.Provisioner
func (ctrl *PersistentVolumeController) setClaimProvisioner(ctx context.Context, claim *v1.PersistentVolumeClaim, provisionerName string) (*v1.PersistentVolumeClaim, error) {
	if val, ok := claim.Annotations[storagehelpers.AnnStorageProvisioner]; ok && val == provisionerName {
		// annotation is already set, nothing to do
		return claim, nil
	}

	// The volume from method args can be pointing to watcher cache. We must not
	// modify these, therefore create a copy.
	claimClone := claim.DeepCopy()
	// TODO: remove the beta storage provisioner anno after the deprecation period
	logger := klog.FromContext(ctx)
	metav1.SetMetaDataAnnotation(&claimClone.ObjectMeta, storagehelpers.AnnBetaStorageProvisioner, provisionerName)
	metav1.SetMetaDataAnnotation(&claimClone.ObjectMeta, storagehelpers.AnnStorageProvisioner, provisionerName)
	updateMigrationAnnotations(logger, ctrl.csiMigratedPluginManager, ctrl.translator, claimClone.Annotations, true)
	newClaim, err := ctrl.kubeClient.CoreV1().PersistentVolumeClaims(claim.Namespace).Update(ctx, claimClone, metav1.UpdateOptions{})
	if err != nil {
		return newClaim, err
	}
	if err = ctrl.claims.AssumeWritten(newClaim); err != nil {
		return newClaim, err
	}
	return newClaim, nil
}

// Stateless functions

func getClaimStatusForLogging(claim *v1.PersistentVolumeClaim) string {
	bound := metav1.HasAnnotation(claim.ObjectMeta, storagehelpers.AnnBindCompleted)
	boundByController := metav1.HasAnnotation(claim.ObjectMeta, storagehelpers.AnnBoundByController)

	return fmt.Sprintf("phase: %s, bound to: %q, bindCompleted: %v, boundByController: %v", claim.Status.Phase, claim.Spec.VolumeName, bound, boundByController)
}

func getVolumeStatusForLogging(volume *v1.PersistentVolume) string {
	boundByController := metav1.HasAnnotation(volume.ObjectMeta, storagehelpers.AnnBoundByController)
	claimName := ""
	if volume.Spec.ClaimRef != nil {
		claimName = fmt.Sprintf("%s/%s (uid: %s)", volume.Spec.ClaimRef.Namespace, volume.Spec.ClaimRef.Name, volume.Spec.ClaimRef.UID)
	}
	return fmt.Sprintf("phase: %s, bound to: %q, boundByController: %v", volume.Status.Phase, claimName, boundByController)
}
