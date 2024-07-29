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

package pvprotection

import (
	"context"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/volume/protectionutil"
	"k8s.io/kubernetes/pkg/util/slice"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

// Controller is controller that removes PVProtectionFinalizer
// from PVs that are not bound to PVCs.
type Controller struct {
	client clientset.Interface

	pvLister       corelisters.PersistentVolumeLister
	pvListerSynced cache.InformerSynced

	queue workqueue.TypedRateLimitingInterface[string]
}

// NewPVProtectionController returns a new *Controller.
func NewPVProtectionController(logger klog.Logger, pvInformer coreinformers.PersistentVolumeInformer, cl clientset.Interface) *Controller {
	e := &Controller{
		client: cl,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "pvprotection"},
		),
	}

	e.pvLister = pvInformer.Lister()
	e.pvListerSynced = pvInformer.Informer().HasSynced
	pvInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			e.pvAddedUpdated(logger, obj)
		},
		UpdateFunc: func(old, new interface{}) {
			e.pvAddedUpdated(logger, new)
		},
	})

	return e
}

// Run runs the controller goroutines.
func (c *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrashWithContext(ctx)
	defer c.queue.ShutDown()

	logger := klog.FromContext(ctx)
	logger.Info("Starting PV protection controller")
	defer logger.Info("Shutting down PV protection controller")

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.pvListerSynced) {
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
	pvKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(pvKey)

	pvName := pvKey

	err := c.processPV(ctx, pvName)
	if err == nil {
		c.queue.Forget(pvKey)
		return true
	}

	utilruntime.HandleErrorWithContext(ctx, err, "Syncing PV", "key", pvKey)
	c.queue.AddRateLimited(pvKey)

	return true
}

func (c *Controller) processPV(ctx context.Context, pvName string) error {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Processing PV", "PV", klog.KRef("", pvName))
	startTime := time.Now()
	defer func() {
		logger.V(4).Info("Finished processing PV", "PV", klog.KRef("", pvName), "cost", time.Since(startTime))
	}()

	pv, err := c.pvLister.Get(pvName)
	if apierrors.IsNotFound(err) {
		logger.V(4).Info("PV not found, ignoring", "PV", klog.KRef("", pvName))
		return nil
	}
	if err != nil {
		return err
	}

	if protectionutil.IsDeletionCandidate(pv, volumeutil.PVProtectionFinalizer) {
		// PV should be deleted. Check if it's used and remove finalizer if
		// it's not.
		isUsed := c.isBeingUsed(pv)
		if !isUsed {
			return c.removeFinalizer(ctx, pv)
		}
		logger.V(4).Info("Keeping PV because it is being used", "PV", klog.KRef("", pvName))
	}

	if protectionutil.NeedToAddFinalizer(pv, volumeutil.PVProtectionFinalizer) {
		// PV is not being deleted -> it should have the finalizer. The
		// finalizer should be added by admission plugin, this is just to add
		// the finalizer to old PVs that were created before the admission
		// plugin was enabled.
		return c.addFinalizer(ctx, pv)
	}
	return nil
}

func (c *Controller) addFinalizer(ctx context.Context, pv *v1.PersistentVolume) error {
	pvClone := pv.DeepCopy()
	pvClone.ObjectMeta.Finalizers = append(pvClone.ObjectMeta.Finalizers, volumeutil.PVProtectionFinalizer)
	_, err := c.client.CoreV1().PersistentVolumes().Update(ctx, pvClone, metav1.UpdateOptions{})
	logger := klog.FromContext(ctx)
	if err != nil {
		logger.V(3).Info("Error adding protection finalizer to PV", "PV", klog.KObj(pv), "err", err)
		return err
	}
	logger.V(3).Info("Added protection finalizer to PV", "PV", klog.KObj(pv))
	return nil
}

func (c *Controller) removeFinalizer(ctx context.Context, pv *v1.PersistentVolume) error {
	pvClone := pv.DeepCopy()
	pvClone.ObjectMeta.Finalizers = slice.RemoveString(pvClone.ObjectMeta.Finalizers, volumeutil.PVProtectionFinalizer, nil)
	_, err := c.client.CoreV1().PersistentVolumes().Update(ctx, pvClone, metav1.UpdateOptions{})
	logger := klog.FromContext(ctx)
	if err != nil {
		logger.V(3).Info("Error removing protection finalizer from PV", "PV", klog.KObj(pv), "err", err)
		return err
	}
	logger.V(3).Info("Removed protection finalizer from PV", "PV", klog.KObj(pv))
	return nil
}

func (c *Controller) isBeingUsed(pv *v1.PersistentVolume) bool {
	// check if PV is being bound to a PVC by its status
	// the status will be updated by PV controller
	if pv.Status.Phase == v1.VolumeBound {
		// the PV is being used now
		return true
	}

	return false
}

// pvAddedUpdated reacts to pv added/updated events
func (c *Controller) pvAddedUpdated(logger klog.Logger, obj interface{}) {
	pv, ok := obj.(*v1.PersistentVolume)
	if !ok {
		utilruntime.HandleErrorWithContext(klog.NewContext(context.Background(), logger), nil, "PV informer returned non-PV object", "obj", klog.Format(obj))
		return
	}
	logger.V(4).Info("Got event on PV", "PV", klog.KObj(pv))

	if protectionutil.NeedToAddFinalizer(pv, volumeutil.PVProtectionFinalizer) || protectionutil.IsDeletionCandidate(pv, volumeutil.PVProtectionFinalizer) {
		c.queue.Add(pv.Name)
	}
}
