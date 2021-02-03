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
	"context"
	"fmt"
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
	"k8s.io/component-base/metrics/prometheus/ratelimiter"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/volume/common"
	"k8s.io/kubernetes/pkg/controller/volume/protectionutil"
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
	podIndexer      cache.Indexer

	queue workqueue.RateLimitingInterface

	// allows overriding of StorageObjectInUseProtection feature Enabled/Disabled for testing
	storageObjectInUseProtectionEnabled bool

	// allows overriding of GenericEphemeralVolume feature Enabled/Disabled for testing
	genericEphemeralVolumeFeatureEnabled bool
}

// NewPVCProtectionController returns a new instance of PVCProtectionController.
func NewPVCProtectionController(pvcInformer coreinformers.PersistentVolumeClaimInformer, podInformer coreinformers.PodInformer, cl clientset.Interface, storageObjectInUseProtectionFeatureEnabled, genericEphemeralVolumeFeatureEnabled bool) (*Controller, error) {
	e := &Controller{
		client:                               cl,
		queue:                                workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "pvcprotection"),
		storageObjectInUseProtectionEnabled:  storageObjectInUseProtectionFeatureEnabled,
		genericEphemeralVolumeFeatureEnabled: genericEphemeralVolumeFeatureEnabled,
	}
	if cl != nil && cl.CoreV1().RESTClient().GetRateLimiter() != nil {
		ratelimiter.RegisterMetricAndTrackRateLimiterUsage("persistentvolumeclaim_protection_controller", cl.CoreV1().RESTClient().GetRateLimiter())
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
	e.podIndexer = podInformer.Informer().GetIndexer()
	if err := common.AddIndexerIfNotPresent(e.podIndexer, common.PodPVCIndex, common.PodPVCIndexFunc(genericEphemeralVolumeFeatureEnabled)); err != nil {
		return nil, fmt.Errorf("Could not initialize pvc protection controller: %v", err)
	}
	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			e.podAddedDeletedUpdated(nil, obj, false)
		},
		DeleteFunc: func(obj interface{}) {
			e.podAddedDeletedUpdated(nil, obj, true)
		},
		UpdateFunc: func(old, new interface{}) {
			e.podAddedDeletedUpdated(old, new, false)
		},
	})

	return e, nil
}

// Run runs the controller goroutines.
func (c *Controller) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.InfoS("Starting PVC protection controller")
	defer klog.InfoS("Shutting down PVC protection controller")

	if !cache.WaitForNamedCacheSync("PVC protection", stopCh, c.pvcListerSynced, c.podListerSynced) {
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
		utilruntime.HandleError(fmt.Errorf("error parsing PVC key %q: %v", pvcKey, err))
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
	klog.V(4).InfoS("Processing PVC", "PVC", klog.KRef(pvcNamespace, pvcName))
	startTime := time.Now()
	defer func() {
		klog.V(4).InfoS("Finished processing PVC", "PVC", klog.KRef(pvcNamespace, pvcName), fmt.Sprintf("(%v)", time.Since(startTime)))
	}()

	pvc, err := c.pvcLister.PersistentVolumeClaims(pvcNamespace).Get(pvcName)
	if apierrors.IsNotFound(err) {
		klog.V(4).InfoS("PVC not found, ignoring", "PVC", klog.KRef(pvcNamespace, pvcName))
		return nil
	}
	if err != nil {
		return err
	}

	if protectionutil.IsDeletionCandidate(pvc, volumeutil.PVCProtectionFinalizer) {
		// PVC should be deleted. Check if it's used and remove finalizer if
		// it's not.
		isUsed, err := c.isBeingUsed(pvc)
		if err != nil {
			return err
		}
		if !isUsed {
			return c.removeFinalizer(pvc)
		}
		klog.V(2).InfoS("Keeping PVC", "PVC", klog.KRef(pvcNamespace, pvcName), " because it is still being used")
	}

	if protectionutil.NeedToAddFinalizer(pvc, volumeutil.PVCProtectionFinalizer) {
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
	_, err := c.client.CoreV1().PersistentVolumeClaims(claimClone.Namespace).Update(context.TODO(), claimClone, metav1.UpdateOptions{})
	if err != nil {
		klog.ErrorS(err, "Error adding protection finalizer to PVC", "PVC", klog.KObj(pvc))
		return err
	}
	klog.V(3).InfoS("Added protection finalizer to PVC", "PVC", klog.KObj(pvc))
	return nil
}

func (c *Controller) removeFinalizer(pvc *v1.PersistentVolumeClaim) error {
	claimClone := pvc.DeepCopy()
	claimClone.ObjectMeta.Finalizers = slice.RemoveString(claimClone.ObjectMeta.Finalizers, volumeutil.PVCProtectionFinalizer, nil)
	_, err := c.client.CoreV1().PersistentVolumeClaims(claimClone.Namespace).Update(context.TODO(), claimClone, metav1.UpdateOptions{})
	if err != nil {
		klog.ErrorS(err, "Error removing protection finalizer from PVC", "PVC", klog.KObj(pvc))
		return err
	}
	klog.V(3).InfoS("Removed protection finalizer from PVC", "PVC", klog.KObj(pvc))
	return nil
}

func (c *Controller) isBeingUsed(pvc *v1.PersistentVolumeClaim) (bool, error) {
	// Look for a Pod using pvc in the Informer's cache. If one is found the
	// correct decision to keep pvc is taken without doing an expensive live
	// list.
	if inUse, err := c.askInformer(pvc); err != nil {
		// No need to return because a live list will follow.
		klog.Error(err)
	} else if inUse {
		return true, nil
	}

	// Even if no Pod using pvc was found in the Informer's cache it doesn't
	// mean such a Pod doesn't exist: it might just not be in the cache yet. To
	// be 100% confident that it is safe to delete pvc make sure no Pod is using
	// it among those returned by a live list.
	return c.askAPIServer(pvc)
}

func (c *Controller) askInformer(pvc *v1.PersistentVolumeClaim) (bool, error) {
	klog.V(4).InfoS("Looking for Pods using PVC", "PVC", klog.KObj(pvc), "in the Informer's cache")

	// The indexer is used to find pods which might use the PVC.
	objs, err := c.podIndexer.ByIndex(common.PodPVCIndex, fmt.Sprintf("%s/%s", pvc.Namespace, pvc.Name))
	if err != nil {
		return false, fmt.Errorf("cache-based list of pods failed while processing %s/%s: %s", pvc.Namespace, pvc.Name, err.Error())
	}
	for _, obj := range objs {
		pod, ok := obj.(*v1.Pod)
		if !ok {
			continue
		}

		if c.genericEphemeralVolumeFeatureEnabled {
			// We still need to look at each volume: that's redundant for volume.PersistentVolumeClaim,
			// but for volume.Ephemeral we need to be sure that this particular PVC is the one
			// created for the ephemeral volume.
			if c.podUsesPVC(pod, pvc) {
				return true, nil
			}
			continue

		}

		// This is the traditional behavior without GenericEphemeralVolume enabled.
		if pod.Spec.NodeName == "" {
			continue
		}
		// found a pod using this PVC
		return true, nil
	}

	klog.V(4).InfoS("No Pod using PVC", "PVC", klog.KObj(pvc), "was found in the Informer's cache")
	return false, nil
}

func (c *Controller) askAPIServer(pvc *v1.PersistentVolumeClaim) (bool, error) {
	klog.V(4).InfoS("Looking for Pods using PVC", "PVC", klog.KObj(pvc), "with a live list")

	podsList, err := c.client.CoreV1().Pods(pvc.Namespace).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("live list of pods failed: %s", err.Error())
	}

	for _, pod := range podsList.Items {
		if c.podUsesPVC(&pod, pvc) {
			return true, nil
		}
	}

	klog.V(2).InfoS("PVC is unused", "PVC", klog.KObj(pvc))
	return false, nil
}

func (c *Controller) podUsesPVC(pod *v1.Pod, pvc *v1.PersistentVolumeClaim) bool {
	// Check whether pvc is used by pod only if pod is scheduled, because
	// kubelet sees pods after they have been scheduled and it won't allow
	// starting a pod referencing a PVC with a non-nil deletionTimestamp.
	if pod.Spec.NodeName != "" {
		for _, volume := range pod.Spec.Volumes {
			if volume.PersistentVolumeClaim != nil && volume.PersistentVolumeClaim.ClaimName == pvc.Name ||
				c.genericEphemeralVolumeFeatureEnabled && !podIsShutDown(pod) && volume.Ephemeral != nil && pod.Name+"-"+volume.Name == pvc.Name && metav1.IsControlledBy(pvc, pod) {
				klog.V(2).InfoS("Pod", "pod", klog.KObj(pod), "uses PVC", "PVC", klog.KObj(pvc))
				return true
			}
		}
	}
	return false
}

// podIsShutDown returns true if kubelet is done with the pod or
// it was force-deleted.
func podIsShutDown(pod *v1.Pod) bool {
	// The following text is based on how pod shutdown was
	// initially described to me. During PR review, it was pointed out
	// that this is not correct: "deleteGracePeriodSeconds tells
	// kubelet when it can start force terminating the
	// containers. Volume teardown only starts after containers
	// are termianted. So there is an additional time period after
	// the grace period where volume teardown is happening."
	//
	// TODO (https://github.com/kubernetes/enhancements/issues/1698#issuecomment-655344680):
	// investigate what kubelet really does and if necessary,
	// add some other signal for "kubelet is done". For now the check
	// is used only for ephemeral volumes, because it
	// is needed to avoid the deadlock.
	//
	// A pod that has a deletionTimestamp and a zero
	// deletionGracePeriodSeconds
	// a) has been processed by kubelet and is ready for deletion or
	// b) was force-deleted.
	//
	// It's now just waiting for garbage collection. We could wait
	// for it to actually get removed, but that may be blocked by
	// finalizers for the pod and thus get delayed.
	//
	// Worse, it is possible that there is a cyclic dependency
	// (pod finalizer waits for PVC to get removed, PVC protection
	// controller waits for pod to get removed).  By considering
	// the PVC unused in this case, we allow the PVC to get
	// removed and break such a cycle.
	//
	// Therefore it is better to proceed with PVC removal,
	// which is safe (case a) and/or desirable (case b).
	return pod.DeletionTimestamp != nil && pod.DeletionGracePeriodSeconds != nil && *pod.DeletionGracePeriodSeconds == 0
}

// pvcAddedUpdated reacts to pvc added/updated events
func (c *Controller) pvcAddedUpdated(obj interface{}) {
	pvc, ok := obj.(*v1.PersistentVolumeClaim)
	if !ok {
		utilruntime.HandleError(fmt.Errorf("PVC informer returned non-PVC object: %#v", obj))
		return
	}
	key, err := cache.MetaNamespaceKeyFunc(pvc)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for Persistent Volume Claim %#v: %v", pvc, err))
		return
	}
	klog.V(4).InfoS("Got event on PVC", key)

	if protectionutil.NeedToAddFinalizer(pvc, volumeutil.PVCProtectionFinalizer) || protectionutil.IsDeletionCandidate(pvc, volumeutil.PVCProtectionFinalizer) {
		c.queue.Add(key)
	}
}

// podAddedDeletedUpdated reacts to Pod events
func (c *Controller) podAddedDeletedUpdated(old, new interface{}, deleted bool) {
	if pod := c.parsePod(new); pod != nil {
		c.enqueuePVCs(pod, deleted)

		// An update notification might mask the deletion of a pod X and the
		// following creation of a pod Y with the same namespaced name as X. If
		// that's the case X needs to be processed as well to handle the case
		// where it is blocking deletion of a PVC not referenced by Y, otherwise
		// such PVC will never be deleted.
		if oldPod := c.parsePod(old); oldPod != nil && oldPod.UID != pod.UID {
			c.enqueuePVCs(oldPod, true)
		}
	}
}

func (*Controller) parsePod(obj interface{}) *v1.Pod {
	if obj == nil {
		return nil
	}
	pod, ok := obj.(*v1.Pod)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %#v", obj))
			return nil
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a Pod %#v", obj))
			return nil
		}
	}
	return pod
}

func (c *Controller) enqueuePVCs(pod *v1.Pod, deleted bool) {
	// Filter out pods that can't help us to remove a finalizer on PVC
	if !deleted && !volumeutil.IsPodTerminated(pod, pod.Status) && pod.Spec.NodeName != "" {
		return
	}

	klog.V(4).InfoS("Enqueuing PVCs for Pod", "pod", klog.KObj(pod), "(UID=", pod.UID, ")")

	// Enqueue all PVCs that the pod uses
	for _, volume := range pod.Spec.Volumes {
		switch {
		case volume.PersistentVolumeClaim != nil:
			c.queue.Add(pod.Namespace + "/" + volume.PersistentVolumeClaim.ClaimName)
		case c.genericEphemeralVolumeFeatureEnabled && volume.Ephemeral != nil:
			c.queue.Add(pod.Namespace + "/" + pod.Name + "-" + volume.Name)
		}
	}
}
