/*
Copyright 2017 The Kubernetes Authors.

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

package pvcprotection

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

// Controller is controller that removes PVCProtectionFinalizer
// from PVCs that are used by no pods.
type Controller struct {
	client clientset.Interface

	pvcLister       corelisters.PersistentVolumeClaimLister
	pvcListerSynced cache.InformerSynced

	podLister       corelisters.PodLister
	podListerSynced cache.InformerSynced

	queue workqueue.RateLimitingInterface

	// allows overriding of StorageObjectInUseProtection feature Enabled/Disabled for testing
	storageObjectInUseProtectionEnabled bool
}

// NewPVCProtectionController returns a new instance of PVCProtectionController.
func NewPVCProtectionController(pvcInformer coreinformers.PersistentVolumeClaimInformer, podInformer coreinformers.PodInformer, cl clientset.Interface, storageObjectInUseProtectionFeatureEnabled bool) *Controller {
	e := &Controller{
		client: cl,
		queue:  workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "pvcprotection"),
		storageObjectInUseProtectionEnabled: storageObjectInUseProtectionFeatureEnabled,
	}
	if cl != nil && cl.CoreV1().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("persistentvolumeclaim_protection_controller", cl.CoreV1().RESTClient().GetRateLimiter())
	}

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
	defer c.queue.ShutDown()

	glog.Infof("Starting PVC protection controller")
	defer glog.Infof("Shutting down PVC protection controller")

	if !controller.WaitForCacheSync("PVC protection", stopCh, c.pvcListerSynced, c.podListerSynced) {
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
	pvcKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(pvcKey)

	pvcNamespace, pvcName, err := cache.SplitMetaNamespaceKey(pvcKey.(string))
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Error parsing PVC key %q: %v", pvcKey, err))
		return true
	}

	err = c.processPVC(pvcNamespace, pvcName)
	if err == nil {
		c.queue.Forget(pvcKey)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("PVC %v failed with : %v", pvcKey, err))
	c.queue.AddRateLimited(pvcKey)

	return true
}

func (c *Controller) processPVC(pvcNamespace, pvcName string) error {
	glog.V(4).Infof("Processing PVC %s/%s", pvcNamespace, pvcName)
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished processing PVC %s/%s (%v)", pvcNamespace, pvcName, time.Since(startTime))
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
		isUsed, err := c.isBeingUsed(pvc)
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

func (c *Controller) addFinalizer(pvc *v1.PersistentVolumeClaim) error {
	// Skip adding Finalizer in case the StorageObjectInUseProtection feature is not enabled
	if !c.storageObjectInUseProtectionEnabled {
		return nil
	}
	claimClone := pvc.DeepCopy()
	claimClone.ObjectMeta.Finalizers = append(claimClone.ObjectMeta.Finalizers, volumeutil.PVCProtectionFinalizer)
	_, err := c.client.CoreV1().PersistentVolumeClaims(claimClone.Namespace).Update(claimClone)
	if err != nil {
		glog.V(3).Infof("Error adding protection finalizer to PVC %s/%s: %v", pvc.Namespace, pvc.Name, err)
		return err
	}
	glog.V(3).Infof("Added protection finalizer to PVC %s/%s", pvc.Namespace, pvc.Name)
	return nil
}

func (c *Controller) removeFinalizer(pvc *v1.PersistentVolumeClaim) error {
	claimClone := pvc.DeepCopy()
	claimClone.ObjectMeta.Finalizers = slice.RemoveString(claimClone.ObjectMeta.Finalizers, volumeutil.PVCProtectionFinalizer, nil)
	_, err := c.client.CoreV1().PersistentVolumeClaims(claimClone.Namespace).Update(claimClone)
	if err != nil {
		glog.V(3).Infof("Error removing protection finalizer from PVC %s/%s: %v", pvc.Namespace, pvc.Name, err)
		return err
	}
	glog.V(3).Infof("Removed protection finalizer from PVC %s/%s", pvc.Namespace, pvc.Name)
	return nil
}

func (c *Controller) isBeingUsed(pvc *v1.PersistentVolumeClaim) (bool, error) {
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
		c.queue.Add(key)
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
			c.queue.Add(pod.Namespace + "/" + volume.PersistentVolumeClaim.ClaimName)
		}
	}
}

func isDeletionCandidate(pvc *v1.PersistentVolumeClaim) bool {
	return pvc.ObjectMeta.DeletionTimestamp != nil && slice.ContainsString(pvc.ObjectMeta.Finalizers, volumeutil.PVCProtectionFinalizer, nil)
}

func needToAddFinalizer(pvc *v1.PersistentVolumeClaim) bool {
	return pvc.ObjectMeta.DeletionTimestamp == nil && !slice.ContainsString(pvc.ObjectMeta.Finalizers, volumeutil.PVCProtectionFinalizer, nil)
}
