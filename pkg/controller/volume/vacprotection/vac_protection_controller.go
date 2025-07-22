/*
Copyright 2024 The Kubernetes Authors.

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

package vacprotection

import (
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	storageinformers "k8s.io/client-go/informers/storage/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	storagelisters "k8s.io/client-go/listers/storage/v1beta1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/volume/protectionutil"
	"k8s.io/kubernetes/pkg/util/slice"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/utils/ptr"
)

const (
	vacNameKeyIndex = "volumeAttributesClassName"
)

// Controller is controller that adds and removes VACProtectionFinalizer
// from VACs that are not used by any PV or PVC.
//
// This controller only use informers, so it may remove the finalizer too early.
//
// One scenario is:
//  1. There is a VolumeAttributesClass that is not used by any PVC. This
//     VolumeAttributesClass is synced to all informers (external-provisioner,
//     external-resizer, KCM)
//
//  2. At the same time:
//
//       * User creates a PVC that uses this VolumeAttributesClass.
//
//       * Another user deletes the VolumeAttributesClass.
//
//  3. VolumeAttributesClass deletion event with DeletionTimestamp reaches
//     this controller. Because the PVC creation event has not yet
//     reached KCM informers, the controller lets the VolumeAttributesClass
//     to be deleted by removing the finalizer. PVC creation event reaches
//     the external-provisioner, before VolumeAttributesClass update. The
//     external-provisioner will try to provision a new volume using the
//     VolumeAttributesClass that will get deleted soon.
//
//       * If the external-provisioner gets the VolumeAttributesClass before
//     deletion in the informer, the provisioning will succeed.
//
//       * Otherwise the external-prosivioner will fail the provisioning.
//
// Solving this scenario properly requires to Get/List requests to the API server,
// which will cause performance issue in larger cluster similar to the existing
// PVC protection controller - related issue https://github.com/kubernetes/kubernetes/issues/109282

type Controller struct {
	client clientset.Interface

	pvcSynced cache.InformerSynced
	pvSynced  cache.InformerSynced
	vacLister storagelisters.VolumeAttributesClassLister
	vacSynced cache.InformerSynced

	getPVsAssignedToVAC  func(vacName string) ([]*v1.PersistentVolume, error)
	getPVCsAssignedToVAC func(vacName string) ([]*v1.PersistentVolumeClaim, error)

	queue workqueue.TypedRateLimitingInterface[string]
}

// NewVACProtectionController returns a new *Controller.
func NewVACProtectionController(logger klog.Logger,
	client clientset.Interface,
	pvcInformer coreinformers.PersistentVolumeClaimInformer,
	pvInformer coreinformers.PersistentVolumeInformer,
	vacInformer storageinformers.VolumeAttributesClassInformer) (*Controller, error) {
	c := &Controller{
		client:    client,
		pvcSynced: pvcInformer.Informer().HasSynced,
		pvSynced:  pvInformer.Informer().HasSynced,
		vacLister: vacInformer.Lister(),
		vacSynced: vacInformer.Informer().HasSynced,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: "vacprotection",
			},
		),
	}

	_, _ = vacInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.vacAddedUpdated(logger, obj)
		},
		UpdateFunc: func(old, new interface{}) {
			c.vacAddedUpdated(logger, new)
		},
	})

	err := pvInformer.Informer().AddIndexers(cache.Indexers{vacNameKeyIndex: func(obj interface{}) ([]string, error) {
		pv, ok := obj.(*v1.PersistentVolume)
		if !ok {
			return []string{}, nil
		}
		return getPVReferencedVACNames(pv), nil
	}})
	if err != nil {
		return nil, fmt.Errorf("failed to add index to PV informer: %w", err)
	}

	pvIndexer := pvInformer.Informer().GetIndexer()
	c.getPVsAssignedToVAC = func(vacName string) ([]*v1.PersistentVolume, error) {
		objs, err := pvIndexer.ByIndex(vacNameKeyIndex, vacName)
		if err != nil {
			return nil, err
		}
		pvcs := make([]*v1.PersistentVolume, 0, len(objs))
		for _, obj := range objs {
			pvc, ok := obj.(*v1.PersistentVolume)
			if !ok {
				continue
			}
			pvcs = append(pvcs, pvc)
		}
		return pvcs, nil
	}

	err = pvcInformer.Informer().AddIndexers(cache.Indexers{vacNameKeyIndex: func(obj interface{}) ([]string, error) {
		pvc, ok := obj.(*v1.PersistentVolumeClaim)
		if !ok {
			return []string{}, nil
		}
		return getPVCReferencedVACNames(pvc), nil
	}})
	if err != nil {
		return nil, fmt.Errorf("failed to add index to PVC informer: %w", err)
	}

	pvcIndexer := pvcInformer.Informer().GetIndexer()
	c.getPVCsAssignedToVAC = func(vacName string) ([]*v1.PersistentVolumeClaim, error) {
		objs, err := pvcIndexer.ByIndex(vacNameKeyIndex, vacName)
		if err != nil {
			return nil, err
		}
		pvcs := make([]*v1.PersistentVolumeClaim, 0, len(objs))
		for _, obj := range objs {
			pvc, ok := obj.(*v1.PersistentVolumeClaim)
			if !ok {
				continue
			}
			pvcs = append(pvcs, pvc)
		}
		return pvcs, nil
	}

	_, _ = pvcInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(old, new interface{}) {
			c.pvcUpdated(logger, old, new)
		},
		DeleteFunc: func(obj interface{}) {
			c.pvcDeleted(logger, obj)
		},
	})

	_, _ = pvInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(old, new interface{}) {
			c.pvUpdated(logger, old, new)
		},
		DeleteFunc: func(obj interface{}) {
			c.pvDeleted(logger, obj)
		},
	})
	return c, nil
}

// Run runs the controller goroutines.
func (c *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	logger := klog.FromContext(ctx)
	logger.Info("Starting VAC protection controller")
	defer logger.Info("Shutting down VAC protection controller")

	if !cache.WaitForNamedCacheSync("VAC protection", ctx.Done(), c.pvSynced, c.pvcSynced, c.vacSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	}

	<-ctx.Done()
}

func (c *Controller) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

// processNextWorkItem deals with one pvcKey off the queue.  It returns false when it's time to quit.
func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	vacKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(vacKey)

	err := c.processVAC(ctx, vacKey)
	if err == nil {
		c.queue.Forget(vacKey)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("VAC %v failed with : %w", vacKey, err))
	c.queue.AddRateLimited(vacKey)

	return true
}

func (c *Controller) processVAC(ctx context.Context, vacName string) error {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Processing VAC", "VAC", klog.KRef("", vacName))
	startTime := time.Now()
	defer func() {
		logger.V(4).Info("Finished processing VAC", "VAC", klog.KRef("", vacName), "cost", time.Since(startTime))
	}()

	vac, err := c.vacLister.Get(vacName)
	if err != nil {
		if apierrors.IsNotFound(err) {
			logger.V(4).Info("VAC not found, ignoring", "VAC", klog.KRef("", vacName))
			return nil
		}
		return err
	}

	if protectionutil.IsDeletionCandidate(vac, volumeutil.VACProtectionFinalizer) {
		// VAC should be deleted. Check if it's used and remove finalizer if
		// it's not.
		isUsed := c.isBeingUsed(ctx, vac)
		if !isUsed {
			return c.removeFinalizer(ctx, vac)
		}
		logger.V(4).Info("Keeping VAC because it is being used", "PVC", klog.KRef("", vacName))
	}

	if protectionutil.NeedToAddFinalizer(vac, volumeutil.VACProtectionFinalizer) {
		return c.addFinalizer(ctx, vac)
	}

	return nil
}

func (c *Controller) addFinalizer(ctx context.Context, vac *storagev1beta1.VolumeAttributesClass) error {
	vacClone := vac.DeepCopy()
	vacClone.ObjectMeta.Finalizers = append(vacClone.ObjectMeta.Finalizers, volumeutil.VACProtectionFinalizer)
	_, err := c.client.StorageV1beta1().VolumeAttributesClasses().Update(ctx, vacClone, metav1.UpdateOptions{})
	logger := klog.FromContext(ctx)
	if err != nil {
		logger.V(3).Info("Error adding protection finalizer to VAC", "VAC", klog.KObj(vac), "err", err)
		return err
	}
	logger.V(3).Info("Added protection finalizer to VAC", "VAC", klog.KObj(vac))
	return nil
}

func (c *Controller) removeFinalizer(ctx context.Context, vac *storagev1beta1.VolumeAttributesClass) error {
	vacClone := vac.DeepCopy()
	vacClone.ObjectMeta.Finalizers = slice.RemoveString(vacClone.ObjectMeta.Finalizers, volumeutil.VACProtectionFinalizer, nil)
	_, err := c.client.StorageV1beta1().VolumeAttributesClasses().Update(ctx, vacClone, metav1.UpdateOptions{})
	logger := klog.FromContext(ctx)
	if err != nil {
		logger.V(3).Info("Error removing protection finalizer from VAC", "VAC", klog.KObj(vac), "err", err)
		return err
	}
	logger.V(3).Info("Removed protection finalizer from VAC", "VAC", klog.KObj(vac))
	return nil
}

func (c *Controller) isBeingUsed(ctx context.Context, vac *storagev1beta1.VolumeAttributesClass) bool {
	logger := klog.FromContext(ctx)

	pvs, err := c.getPVsAssignedToVAC(vac.Name)
	if err != nil {
		logger.Error(err, "Error getting PVs assigned to VAC", "VAC", klog.KObj(vac))
		return true
	}
	if len(pvs) > 0 {
		return true
	}

	pvcs, err := c.getPVCsAssignedToVAC(vac.Name)
	if err != nil {
		logger.Error(err, "Error getting PVCs assigned to VAC", "VAC", klog.KObj(vac))
		return true
	}
	if len(pvcs) > 0 {
		return true
	}
	return false
}

// pvAddedUpdated reacts to vac added/updated events
func (c *Controller) vacAddedUpdated(logger klog.Logger, obj interface{}) {
	vac, ok := obj.(*storagev1beta1.VolumeAttributesClass)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("VAC informer returned non-VAC object: %#v", obj))
		return
	}
	logger.V(4).Info("Got event on VAC", "VAC", klog.KObj(vac))

	if protectionutil.NeedToAddFinalizer(vac, volumeutil.VACProtectionFinalizer) || protectionutil.IsDeletionCandidate(vac, volumeutil.VACProtectionFinalizer) {
		c.queue.Add(vac.Name)
	}
}

// pvcDeleted reacts to pvc deleted events
func (c *Controller) pvcDeleted(logger klog.Logger, obj interface{}) {
	pvc, ok := obj.(*v1.PersistentVolumeClaim)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("PVC informer returned non-PVC object: %#v", obj))
		return
	}
	logger.V(4).Info("Got event on PVC", "PVC", klog.KObj(pvc))
	vacNames := getPVCReferencedVACNames(pvc)
	for _, vacName := range vacNames {
		c.queue.Add(vacName)
	}
}

// pvcUpdated reacts to pvc updated events
func (c *Controller) pvcUpdated(logger klog.Logger, old, new interface{}) {
	oldPVC, ok := old.(*v1.PersistentVolumeClaim)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("PVC informer returned non-PVC object: %#v", old))
		return
	}
	newPVC, ok := new.(*v1.PersistentVolumeClaim)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("PVC informer returned non-PVC object: %#v", new))
		return
	}

	logger.V(4).Info("Got event on PVC", "PVC", klog.KObj(newPVC))

	vavNames := sets.New(getPVCReferencedVACNames(oldPVC)...).Delete(getPVCReferencedVACNames(newPVC)...).UnsortedList()
	for _, vacName := range vavNames {
		c.queue.Add(vacName)
	}
}

// pvUpdated reacts to pv updated events
func (c *Controller) pvUpdated(logger klog.Logger, old, new interface{}) {
	oldPV, ok := old.(*v1.PersistentVolume)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("PV informer returned non-PV object: %#v", old))
		return
	}
	newPV, ok := new.(*v1.PersistentVolume)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("PV informer returned non-PV object: %#v", new))
		return
	}

	logger.V(4).Info("Got event on PV", "PV", klog.KObj(newPV))
	vavNames := sets.New(getPVReferencedVACNames(oldPV)...).Delete(getPVReferencedVACNames(newPV)...).UnsortedList()
	for _, vacName := range vavNames {
		c.queue.Add(vacName)
	}
}

// pvDeleted reacts to pv deleted events
func (c *Controller) pvDeleted(logger klog.Logger, obj interface{}) {
	pv, ok := obj.(*v1.PersistentVolume)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("PV informer returned non-PV object: %#v", obj))
		return
	}
	logger.V(4).Info("Got event on PV", "PV", klog.KObj(pv))
	vacNames := getPVReferencedVACNames(pv)
	for _, vacName := range vacNames {
		c.queue.Add(vacName)
	}
}

// getPVCReferencedVACNames returns a list of VAC names that are referenced by the PVC.
func getPVCReferencedVACNames(pvc *v1.PersistentVolumeClaim) []string {
	keys := sets.New[string]()
	vacName := ptr.Deref(pvc.Spec.VolumeAttributesClassName, "")
	if vacName != "" {
		keys.Insert(vacName)
	}
	vacName = ptr.Deref(pvc.Status.CurrentVolumeAttributesClassName, "")
	if vacName != "" {
		keys.Insert(vacName)
	}
	status := pvc.Status.ModifyVolumeStatus
	if status != nil && status.TargetVolumeAttributesClassName != "" {
		keys.Insert(status.TargetVolumeAttributesClassName)
	}
	return keys.UnsortedList()
}

// getPVReferencedVACNames returns a list of VAC names that are referenced by the PV.
func getPVReferencedVACNames(pv *v1.PersistentVolume) []string {
	result := []string{}
	vacName := ptr.Deref(pv.Spec.VolumeAttributesClassName, "")
	if vacName != "" {
		result = append(result, vacName)
	}
	return result
}
