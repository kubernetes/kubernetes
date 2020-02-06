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
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-base/metrics/prometheus/ratelimiter"
	"k8s.io/klog"
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

	queue workqueue.RateLimitingInterface

	// allows overriding of StorageObjectInUseProtection feature Enabled/Disabled for testing
	storageObjectInUseProtectionEnabled bool
}

// NewPVProtectionController returns a new *Controller.
func NewPVProtectionController(pvInformer coreinformers.PersistentVolumeInformer, cl clientset.Interface, storageObjectInUseProtectionFeatureEnabled bool) *Controller {
	e := &Controller{
		client:                              cl,
		queue:                               workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "pvprotection"),
		storageObjectInUseProtectionEnabled: storageObjectInUseProtectionFeatureEnabled,
	}
	if cl != nil && cl.CoreV1().RESTClient().GetRateLimiter() != nil {
		ratelimiter.RegisterMetricAndTrackRateLimiterUsage("persistentvolume_protection_controller", cl.CoreV1().RESTClient().GetRateLimiter())
	}

	e.pvLister = pvInformer.Lister()
	e.pvListerSynced = pvInformer.Informer().HasSynced
	pvInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: e.pvAddedUpdated,
		UpdateFunc: func(old, new interface{}) {
			e.pvAddedUpdated(new)
		},
	})

	return e
}

// Run runs the controller goroutines.
func (c *Controller) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting PV protection controller")
	defer klog.Infof("Shutting down PV protection controller")

	if !cache.WaitForNamedCacheSync("PV protection", stopCh, c.pvListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(c.runWorker, time.Second, stopCh)
	}

	<-stopCh
}

func (c *Controller) runWorker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one pvcKey off the queue.  It returns false when it's time to quit.
func (c *Controller) processNextWorkItem() bool {
	pvKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(pvKey)

	pvName := pvKey.(string)

	err := c.processPV(pvName)
	if err == nil {
		c.queue.Forget(pvKey)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("PV %v failed with : %v", pvKey, err))
	c.queue.AddRateLimited(pvKey)

	return true
}

func (c *Controller) processPV(pvName string) error {
	klog.V(4).Infof("Processing PV %s", pvName)
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished processing PV %s (%v)", pvName, time.Since(startTime))
	}()

	pv, err := c.pvLister.Get(pvName)
	if apierrors.IsNotFound(err) {
		klog.V(4).Infof("PV %s not found, ignoring", pvName)
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
			return c.removeFinalizer(pv)
		}
	}

	if protectionutil.NeedToAddFinalizer(pv, volumeutil.PVProtectionFinalizer) {
		// PV is not being deleted -> it should have the finalizer. The
		// finalizer should be added by admission plugin, this is just to add
		// the finalizer to old PVs that were created before the admission
		// plugin was enabled.
		return c.addFinalizer(pv)
	}
	return nil
}

func (c *Controller) addFinalizer(pv *v1.PersistentVolume) error {
	// Skip adding Finalizer in case the StorageObjectInUseProtection feature is not enabled
	if !c.storageObjectInUseProtectionEnabled {
		return nil
	}
	pvClone := pv.DeepCopy()
	pvClone.ObjectMeta.Finalizers = append(pvClone.ObjectMeta.Finalizers, volumeutil.PVProtectionFinalizer)
	_, err := c.client.CoreV1().PersistentVolumes().Update(pvClone)
	if err != nil {
		klog.V(3).Infof("Error adding protection finalizer to PV %s: %v", pv.Name, err)
		return err
	}
	klog.V(3).Infof("Added protection finalizer to PV %s", pv.Name)
	return nil
}

func (c *Controller) removeFinalizer(pv *v1.PersistentVolume) error {
	pvClone := pv.DeepCopy()
	pvClone.ObjectMeta.Finalizers = slice.RemoveString(pvClone.ObjectMeta.Finalizers, volumeutil.PVProtectionFinalizer, nil)
	_, err := c.client.CoreV1().PersistentVolumes().Update(pvClone)
	if err != nil {
		klog.V(3).Infof("Error removing protection finalizer from PV %s: %v", pv.Name, err)
		return err
	}
	klog.V(3).Infof("Removed protection finalizer from PV %s", pv.Name)
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
func (c *Controller) pvAddedUpdated(obj interface{}) {
	pv, ok := obj.(*v1.PersistentVolume)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("PV informer returned non-PV object: %#v", obj))
		return
	}
	klog.V(4).Infof("Got event on PV %s", pv.Name)

	if protectionutil.NeedToAddFinalizer(pv, volumeutil.PVProtectionFinalizer) || protectionutil.IsDeletionCandidate(pv, volumeutil.PVProtectionFinalizer) {
		c.queue.Add(pv.Name)
	}
}
