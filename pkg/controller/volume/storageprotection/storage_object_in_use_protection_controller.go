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

package storageprotection

import (
	"fmt"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/metrics"
	"k8s.io/kubernetes/pkg/util/slice"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

// Controller is controller that removes PVCProtectionFinalizer/PVProtectionFinalizer
// from PVCs/PVs that are used by no pods.
type Controller struct {
	client clientset.Interface

	pvLister       corelisters.PersistentVolumeLister
	pvListerSynced cache.InformerSynced

	pvcLister       corelisters.PersistentVolumeClaimLister
	pvcListerSynced cache.InformerSynced

	podLister       corelisters.PodLister
	podListerSynced cache.InformerSynced

	pvQueue  workqueue.RateLimitingInterface
	pvcQueue workqueue.RateLimitingInterface
}

// NewStorageObjectInUseProtectionController returns a new *Controller.
func NewStorageObjectInUseProtectionController(pvInformer coreinformers.PersistentVolumeInformer, pvcInformer coreinformers.PersistentVolumeClaimInformer, podInformer coreinformers.PodInformer, cl clientset.Interface) *Controller {
	e := &Controller{
		client:   cl,
		pvcQueue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "pvcqueue"),
		pvQueue:  workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "pvqueue"),
	}
	if cl != nil && cl.CoreV1().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("storage_object_in_use_protection_controller", cl.CoreV1().RESTClient().GetRateLimiter())
	}

	e.pvLister = pvInformer.Lister()
	e.pvListerSynced = pvInformer.Informer().HasSynced
	pvInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: e.pvAddedUpdated,
		UpdateFunc: func(old, new interface{}) {
			e.pvAddedUpdated(new)
		},
	})

	e.pvcLister = pvcInformer.Lister()
	e.pvcListerSynced = pvcInformer.Informer().HasSynced
	pvcInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: e.pvcAddedUpdated,
		UpdateFunc: func(old, new interface{}) {
			e.pvcAddedUpdated(new)
		},
	})

	e.podLister = podInformer.Lister()
	e.podListerSynced = podInformer.Informer().HasSynced
	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			e.podAddedDeletedUpdated(obj, false)
		},
		DeleteFunc: func(obj interface{}) {
			e.podAddedDeletedUpdated(obj, true)
		},
		UpdateFunc: func(old, new interface{}) {
			e.podAddedDeletedUpdated(new, false)
		},
	})

	return e
}

// Run runs the controller goroutines.
func (c *Controller) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.pvQueue.ShutDown()
	defer c.pvcQueue.ShutDown()

	glog.Infof("Starting storage object in use protection controller")
	defer glog.Infof("Shutting down storage object in use protection controller")

	if !controller.WaitForCacheSync("storage object in use protection", stopCh, c.pvListerSynced, c.pvcListerSynced, c.podListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(c.runVolumeWorker, time.Second, stopCh)
		go wait.Until(c.runClaimWorker, time.Second, stopCh)
	}

	<-stopCh
}

func (c *Controller) runVolumeWorker() {
	workFunc := func() bool {
		pvKey, quit := c.pvQueue.Get()
		if quit {
			return false
		}
		defer c.pvQueue.Done(pvKey)

		pvName := pvKey.(string)

		err := c.processPV(pvName)
		if err == nil {
			c.pvQueue.Forget(pvKey)
			return true
		}

		utilruntime.HandleError(fmt.Errorf("PV %v failed with : %v", pvKey, err))
		c.pvQueue.AddRateLimited(pvKey)

		return true
	}
	for {
		if ok := workFunc(); !ok {
			glog.Infof("volume worker queue shutting down")
			return
		}
	}
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
		isUsed := c.isPVBeingUsed(pv)
		if !isUsed {
			return c.removeFinalizer(pv)
		}
	}

	if needToAddFinalizer(pv) {
		// PV is not being deleted -> it should have the finalizer. The
		// finalizer should be added by admission plugin, this is just to add
		// the finalizer to old PVs that were created before the admission
		// plugin was enabled.
		return c.addFinalizer(pv)
	}
	return nil
}

func (c *Controller) runClaimWorker() {
	workFunc := func() bool {
		pvcKey, quit := c.pvcQueue.Get()
		if quit {
			return false
		}
		defer c.pvcQueue.Done(pvcKey)

		pvcNamespace, pvcName, err := cache.SplitMetaNamespaceKey(pvcKey.(string))
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("error parsing PVC key %q: %v", pvcKey, err))
			return true
		}

		err = c.processPVC(pvcNamespace, pvcName)
		if err == nil {
			c.pvcQueue.Forget(pvcKey)
			return true
		}

		utilruntime.HandleError(fmt.Errorf("PVC %v failed with : %v", pvcKey, err))
		c.pvcQueue.AddRateLimited(pvcKey)

		return true
	}
	for {
		if ok := workFunc(); !ok {
			glog.Infof("claim worker queue shutting down")
			return
		}
	}
}

func (c *Controller) processPVC(pvcNamespace, pvcName string) error {
	glog.V(4).Infof("Processing PVC %s/%s", pvcNamespace, pvcName)
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished processing PVC %s/%s (%v)", pvcNamespace, pvcName, time.Now().Sub(startTime))
	}()

	pvc, err := c.pvcLister.PersistentVolumeClaims(pvcNamespace).Get(pvcName)
	if apierrs.IsNotFound(err) {
		glog.V(4).Infof("PVC %s/%s not found, ignoring", pvcNamespace, pvcName)
		return nil
	}
	if err != nil {
		return err
	}

	if isDeletionCandidate(pvc) {
		// PVC should be deleted. Check if it's used and remove finalizer if
		// it's not.
		isUsed, err := c.isPVCBeingUsed(pvc)
		if err != nil {
			return err
		}
		if !isUsed {
			return c.removeFinalizer(pvc)
		}
	}

	if needToAddFinalizer(pvc) {
		// PVC is not being deleted -> it should have the finalizer. The
		// finalizer should be added by admission plugin, this is just to add
		// the finalizer to old PVCs that were created before the admission
		// plugin was enabled.
		return c.addFinalizer(pvc)
	}
	return nil
}

func (c *Controller) addFinalizer(obj interface{}) error {
	switch obj.(type) {
	case *v1.PersistentVolume:
		pv, _ := obj.(*v1.PersistentVolume)
		pvClone := pv.DeepCopy()
		pvClone.ObjectMeta.Finalizers = append(pvClone.ObjectMeta.Finalizers, volumeutil.PVProtectionFinalizer)
		_, err := c.client.CoreV1().PersistentVolumes().Update(pvClone)
		if err != nil {
			glog.V(3).Infof("Error adding protection finalizer to PV %s: %v", pv.Name)
			return err
		}
		glog.V(3).Infof("Added protection finalizer to PV %s", pv.Name)
	case *v1.PersistentVolumeClaim:
		pvc, _ := obj.(*v1.PersistentVolumeClaim)
		claimClone := pvc.DeepCopy()
		claimClone.ObjectMeta.Finalizers = append(claimClone.ObjectMeta.Finalizers, volumeutil.PVCProtectionFinalizer)
		_, err := c.client.CoreV1().PersistentVolumeClaims(claimClone.Namespace).Update(claimClone)
		if err != nil {
			glog.V(3).Infof("Error adding protection finalizer to PVC %s/%s: %v", pvc.Namespace, pvc.Name, err)
			return err
		}
		glog.V(3).Infof("Added protection finalizer to PVC %s/%s", pvc.Namespace, pvc.Name)
	default:
		return fmt.Errorf("obj type error, expected: *v1.PersistentVolume or *v1.PersistentVolumeClaim, got: %v", obj)
	}
	return nil
}

func (c *Controller) removeFinalizer(obj interface{}) error {
	switch obj.(type) {
	case *v1.PersistentVolume:
		pv, _ := obj.(*v1.PersistentVolume)
		pvClone := pv.DeepCopy()
		pvClone.ObjectMeta.Finalizers = slice.RemoveString(pvClone.ObjectMeta.Finalizers, volumeutil.PVProtectionFinalizer, nil)
		_, err := c.client.CoreV1().PersistentVolumes().Update(pvClone)
		if err != nil {
			glog.V(3).Infof("Error removing protection finalizer from PV %s: %v", pv.Name, err)
			return err
		}
		glog.V(3).Infof("Removed protection finalizer from PV %s", pv.Name)
	case *v1.PersistentVolumeClaim:
		pvc, _ := obj.(*v1.PersistentVolumeClaim)
		claimClone := pvc.DeepCopy()
		claimClone.ObjectMeta.Finalizers = slice.RemoveString(claimClone.ObjectMeta.Finalizers, volumeutil.PVCProtectionFinalizer, nil)
		_, err := c.client.CoreV1().PersistentVolumeClaims(claimClone.Namespace).Update(claimClone)
		if err != nil {
			glog.V(3).Infof("Error removing protection finalizer from PVC %s/%s: %v", pvc.Namespace, pvc.Name, err)
			return err
		}
		glog.V(3).Infof("Removed protection finalizer from PVC %s/%s", pvc.Namespace, pvc.Name)
	default:
		return fmt.Errorf("obj type error, expected: *v1.PersistentVolume or *v1.PersistentVolumeClaim, got: %v", obj)
	}
	return nil
}

func (c *Controller) isPVBeingUsed(pv *v1.PersistentVolume) bool {
	// check if PV is being bound to a PVC by its status
	// the status will be updated by PV controller
	if pv.Status.Phase == v1.VolumeBound {
		// the PV is being used now
		return true
	}

	return false
}

func (c *Controller) isPVCBeingUsed(pvc *v1.PersistentVolumeClaim) (bool, error) {
	pods, err := c.podLister.Pods(pvc.Namespace).List(labels.Everything())
	if err != nil {
		return false, err
	}
	for _, pod := range pods {
		if pod.Spec.NodeName == "" {
			// This pod is not scheduled. We have a predicated in scheduler that
			// prevents scheduling pods with deletion timestamp, so we can be
			// pretty sure it won't be scheduled in parallel to this check.
			// Therefore this pod does not block the PVC from deletion.
			glog.V(4).Infof("Skipping unscheduled pod %s when checking PVC %s/%s", pod.Name, pvc.Namespace, pvc.Name)
			continue
		}
		if volumeutil.IsPodTerminated(pod, pod.Status) {
			// This pod is being unmounted/detached or is already
			// unmounted/detached. It does not block the PVC from deletion.
			continue
		}
		for _, volume := range pod.Spec.Volumes {
			if volume.PersistentVolumeClaim == nil {
				continue
			}
			if volume.PersistentVolumeClaim.ClaimName == pvc.Name {
				glog.V(2).Infof("Keeping PVC %s/%s, it is used by pod %s/%s", pvc.Namespace, pvc.Name, pod.Namespace, pod.Name)
				return true, nil
			}
		}
	}

	glog.V(3).Infof("PVC %s/%s is unused", pvc.Namespace, pvc.Name)
	return false, nil
}

// pvAddedUpdated reacts to pv added/updated events
func (c *Controller) pvAddedUpdated(obj interface{}) {
	pv, ok := obj.(*v1.PersistentVolume)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("PV informer returned non-PV object: %#v", obj))
		return
	}
	glog.V(4).Infof("Got event on PV %s", pv.Name)

	if needToAddFinalizer(pv) || isDeletionCandidate(pv) {
		c.pvQueue.Add(pv.Name)
	}
}

// pvcAddedUpdated reacts to pvc added/updated/deleted events
func (c *Controller) pvcAddedUpdated(obj interface{}) {
	pvc, ok := obj.(*v1.PersistentVolumeClaim)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("PVC informer returned non-PVC object: %#v", obj))
		return
	}
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(pvc)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for Persistent Volume Claim %#v: %v", pvc, err))
		return
	}
	glog.V(4).Infof("Got event on PVC %s", key)

	if needToAddFinalizer(pvc) || isDeletionCandidate(pvc) {
		c.pvcQueue.Add(key)
	}
}

// podAddedDeletedUpdated reacts to Pod events
func (c *Controller) podAddedDeletedUpdated(obj interface{}, deleted bool) {
	pod, ok := obj.(*v1.Pod)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Couldn't get object from tombstone %#v", obj))
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("Tombstone contained object that is not a Pod %#v", obj))
			return
		}
	}

	// Filter out pods that can't help us to remove a finalizer on PVC
	if !deleted && !volumeutil.IsPodTerminated(pod, pod.Status) && pod.Spec.NodeName != "" {
		return
	}

	glog.V(4).Infof("Got event on pod %s/%s", pod.Namespace, pod.Name)

	// Enqueue all PVCs that the pod uses
	for _, volume := range pod.Spec.Volumes {
		if volume.PersistentVolumeClaim != nil {
			c.pvcQueue.Add(pod.Namespace + "/" + volume.PersistentVolumeClaim.ClaimName)
		}
	}
}

func isDeletionCandidate(obj interface{}) bool {
	switch obj.(type) {
	case *v1.PersistentVolume:
		pv, _ := obj.(*v1.PersistentVolume)
		return pv.ObjectMeta.DeletionTimestamp != nil && slice.ContainsString(pv.ObjectMeta.Finalizers, volumeutil.PVProtectionFinalizer, nil)
	case *v1.PersistentVolumeClaim:
		pvc, _ := obj.(*v1.PersistentVolumeClaim)
		return pvc.ObjectMeta.DeletionTimestamp != nil && slice.ContainsString(pvc.ObjectMeta.Finalizers, volumeutil.PVCProtectionFinalizer, nil)
	}
	return false
}

func needToAddFinalizer(obj interface{}) bool {
	switch obj.(type) {
	case *v1.PersistentVolume:
		pv, _ := obj.(*v1.PersistentVolume)
		return pv.ObjectMeta.DeletionTimestamp == nil && !slice.ContainsString(pv.ObjectMeta.Finalizers, volumeutil.PVProtectionFinalizer, nil)
	case *v1.PersistentVolumeClaim:
		pvc, _ := obj.(*v1.PersistentVolumeClaim)
		return pvc.ObjectMeta.DeletionTimestamp == nil && !slice.ContainsString(pvc.ObjectMeta.Finalizers, volumeutil.PVCProtectionFinalizer, nil)
	}
	return false
}
