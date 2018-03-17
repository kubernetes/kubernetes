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

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/util/metrics"
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
	// allows overriding for testing
	features utilfeature.FeatureGate
}

// NewPVProtectionController returns a new *Controller.
func NewPVProtectionController(pvInformer coreinformers.PersistentVolumeInformer, cl clientset.Interface) *Controller {
	e := &Controller{
		client:   cl,
		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "pvprotection"),
		features: utilfeature.DefaultFeatureGate,
	}

	if cl != nil && cl.CoreV1().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("persistentvolume_protection_controller", cl.CoreV1().RESTClient().GetRateLimiter())
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

	glog.Infof("Starting PV protection controller")
	defer glog.Infof("Shutting down PV protection controller")

	if !controller.WaitForCacheSync("PV protection", stopCh, c.pvListerSynced) {
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
	glog.V(4).Infof("Processing PV %s", pvName)
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished processing PV %s (%v)", pvName, time.Now().Sub(startTime))
	}()

	pv, err := c.pvLister.Get(pvName)
	if apierrs.IsNotFound(err) {
		glog.V(4).Infof("PV %s not found, ignoring", pvName)
		return nil
	}
	if err != nil {
		return err
	}

	if isDeletionCandidate(pv) {
		// PV should be deleted. Check if it's used and remove finalizer if
		// it's not.
		isUsed := c.isBeingUsed(pv)
		if !isUsed {
			return c.removeFinalizer(pv)
		}
	}

	return nil
}

func (c *Controller) removeFinalizer(pv *v1.PersistentVolume) error {
	pvClone := pv.DeepCopy()
	pvClone.ObjectMeta.Finalizers = slice.RemoveString(pvClone.ObjectMeta.Finalizers, volumeutil.PVProtectionFinalizer, nil)
	_, err := c.client.CoreV1().PersistentVolumes().Update(pvClone)
	if err != nil {
		glog.V(3).Infof("Error removing protection finalizer from PV %s: %v", pv.Name, err)
		return err
	}
	glog.V(3).Infof("Removed protection finalizer from PV %s", pv.Name)
	return nil
}

func (c *Controller) isBeingUsed(pv *v1.PersistentVolume) bool {
	// if PVC protection is disabled - we do not care about PV and PVC binding
	if !c.features.Enabled(features.PVCProtection) {
		return false
	}

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
	glog.V(4).Infof("Got event on PV %s", pv.Name)

	if isDeletionCandidate(pv) {
		c.queue.Add(pv.Name)
	}
}

func isDeletionCandidate(pv *v1.PersistentVolume) bool {
	return pv.ObjectMeta.DeletionTimestamp != nil && slice.ContainsString(pv.ObjectMeta.Finalizers, volumeutil.PVProtectionFinalizer, nil)
}
