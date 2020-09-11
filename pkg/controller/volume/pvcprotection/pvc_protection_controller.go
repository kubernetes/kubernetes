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
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
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

	nodeLister       corelisters.NodeLister
	nodeListerSynced cache.InformerSynced

	pvLister       corelisters.PersistentVolumeLister
	pvListerSynced cache.InformerSynced

	queue workqueue.RateLimitingInterface

	// allows overriding of StorageObjectInUseProtection feature Enabled/Disabled for testing
	storageObjectInUseProtectionEnabled bool

	// allows overriding of GenericEphemeralVolume feature Enabled/Disabled for testing
	genericEphemeralVolumeFeatureEnabled bool
}

const (
	volumeHandleCSIPrefix = "kubernetes.io/csi/"
	volumeHandleSep       = "^"
)

// NewPVCProtectionController returns a new instance of PVCProtectionController.
func NewPVCProtectionController(pvcInformer coreinformers.PersistentVolumeClaimInformer, podInformer coreinformers.PodInformer, nodeInformer coreinformers.NodeInformer, pvInformer coreinformers.PersistentVolumeInformer, cl clientset.Interface, storageObjectInUseProtectionFeatureEnabled, genericEphemeralVolumeFeatureEnabled bool) (*Controller, error) {
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

	e.nodeLister = nodeInformer.Lister()
	e.nodeListerSynced = nodeInformer.Informer().HasSynced

	e.pvLister = pvInformer.Lister()
	e.pvListerSynced = pvInformer.Informer().HasSynced
	return e, nil
}

// Run runs the controller goroutines.
func (c *Controller) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting PVC protection controller")
	defer klog.Infof("Shutting down PVC protection controller")

	if !cache.WaitForNamedCacheSync("PVC protection", stopCh, c.pvcListerSynced, c.podListerSynced, c.nodeListerSynced, c.pvListerSynced) {
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

	// 'needsReque' is set to true in two cases:
	// 1. Errors (other than NotFound error) while retrieving PVC from cache.
	// 2. if PV is detected to be in use by a node, and there is no Pod referencing the
	// PVC(which is marked for delete). We expect the condition to be short lived as the
	// volume should be detached from the node soon.
	// For the above cases, we retry without any exponential backup. Even if we do not use
	// exponential backoff, default ratelimiter still uses the BucketRateLimiter to control
	// the overall rate at which keys are added to the queue.
	needsReque, err := c.processPVC(pvcNamespace, pvcName)
	if err == nil && !needsReque {
		c.queue.Forget(pvcKey)
		return true
	} else if needsReque {
		c.queue.Forget(pvcKey)
	}

	if err != nil {
		utilruntime.HandleError(fmt.Errorf("PVC %v failed with : %v", pvcKey, err))
	}
	klog.V(4).Infof("Reque PVC key %v", pvcKey)
	c.queue.AddRateLimited(pvcKey)
	return true
}

// This function processes a PVC, and returns error encountered (if any), and a bool indicating if the PVC needs to be requeued.
func (c *Controller) processPVC(pvcNamespace, pvcName string) (bool, error) {
	klog.V(4).Infof("Processing PVC %s/%s", pvcNamespace, pvcName)
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished processing PVC %s/%s (%v)", pvcNamespace, pvcName, time.Since(startTime))
	}()

	pvc, err := c.pvcLister.PersistentVolumeClaims(pvcNamespace).Get(pvcName)
	if apierrors.IsNotFound(err) {
		klog.V(4).Infof("PVC %s/%s not found, ignoring", pvcNamespace, pvcName)
		return false, nil
	}
	if err != nil {
		// We expect this error to be resolved soon, hence requeue the PVC key.
		return true, err
	}

	if protectionutil.IsDeletionCandidate(pvc, volumeutil.PVCProtectionFinalizer) {
		// PVC should be deleted. Check if it's used and remove finalizer if
		// it's not.
		isUsed, needsRequeue, err := c.isBeingUsed(pvc)
		if err != nil {
			return needsRequeue, err
		}
		if !isUsed {
			return needsRequeue, c.removeFinalizer(pvc)
		}
		klog.V(2).Infof("Keeping PVC %s/%s because it is still being used", pvc.Namespace, pvc.Name)
		return needsRequeue, nil
	}

	if protectionutil.NeedToAddFinalizer(pvc, volumeutil.PVCProtectionFinalizer) {
		// PVC is not being deleted -> it should have the finalizer. The
		// finalizer should be added by admission plugin, this is just to add
		// the finalizer to old PVCs that were created before the admission
		// plugin was enabled.
		return false, c.addFinalizer(pvc)
	}
	return false, nil
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
		klog.V(3).Infof("Error adding protection finalizer to PVC %s/%s: %v", pvc.Namespace, pvc.Name, err)
		return err
	}
	klog.V(3).Infof("Added protection finalizer to PVC %s/%s", pvc.Namespace, pvc.Name)
	return nil
}

func (c *Controller) removeFinalizer(pvc *v1.PersistentVolumeClaim) error {
	claimClone := pvc.DeepCopy()
	claimClone.ObjectMeta.Finalizers = slice.RemoveString(claimClone.ObjectMeta.Finalizers, volumeutil.PVCProtectionFinalizer, nil)
	_, err := c.client.CoreV1().PersistentVolumeClaims(claimClone.Namespace).Update(context.TODO(), claimClone, metav1.UpdateOptions{})
	if err != nil {
		klog.V(3).Infof("Error removing protection finalizer from PVC %s/%s: %v", pvc.Namespace, pvc.Name, err)
		return err
	}
	klog.V(3).Infof("Removed protection finalizer from PVC %s/%s", pvc.Namespace, pvc.Name)
	return nil
}

// This function checks if a PVC is in use and returns values inUse,
// needsRequeue and error.
// isUse: This value indicates if a PVC is in use by any pod or node
// needsRequeue: This value indicates that the controller loop should
// requeue the PVC in the controller, instead of waiting for a new event
// handler call on the PVC. This is an optimization to react faster to a
// change in state where no pod references the PVC.
func (c *Controller) isBeingUsed(pvc *v1.PersistentVolumeClaim) (bool, bool, error) {
	// If PVC is marked for delete, before the PVC.Spec.Volume is populated,
	// we should not block PVC deletion. By the time PVC is deleted, if PV
	// creation was already triggered and the volume plugin is in the process
	// of creation of the volume, then after the PV is created, the PV controller
	// will trigger a delete for the PV (if it has a reclaim policy of delete),
	// since the claim will not be found.
	var pv *v1.PersistentVolume
	if pvc.Spec.VolumeName != "" {
		var err error
		pv, err = c.getPV(pvc)
		// If PV is not found, then node check for volumes in use will be skipped.
		if err != nil && !apierrors.IsNotFound(err) {
			return false, false, err
		}
	}

	// Look for a Pod using pvc in the Informer's cache. If one is found the
	// correct decision to keep pvc is taken without doing an expensive live
	// list. No requeue is needed as we will wait for a Pod event.
	if inUse, err := c.askPodInformer(pvc); err != nil {
		// No need to return because a live list will follow.
		klog.Error(err)
	} else if inUse {
		// If we reach here, this means, the the PVC is still in use by a Pod, hence no
		// reque is needed. We will wait for an event to process the PVC.
		return true, false, nil
	}

	// Check for volumes in use by nodes for CSI volume plugins.
	// In-tree volume plugins need investigation on how to map a pvc.Spec.VolumeName
	// to node.Status.VolumesInUse.
	if pv != nil && pv.Spec.CSI != nil {
		uniqueVolHandle, err := c.generateUniqueCSIVolumeHandle(pv.Spec.CSI)
		if err != nil {
			return false, false, err
		}
		if inUse, err := c.askNodeInformer(uniqueVolHandle, pvc); err != nil {
			// No need to return because a live list will follow.
			klog.Error(err)
		} else if inUse {
			// If we reach here, this means we have not found any Pod , referencing this PVC,
			// in the informer cache, and the volume is still in use by a node. So we reque
			// the PVC again to process it faster.
			return true, true, nil
		}
	}

	// Even if no Pod using pvc was found in the Informer's cache it doesn't
	// mean such a Pod doesn't exist: it might just not be in the cache yet. To
	// be 100% confident that it is safe to delete pvc make sure no Pod is using
	// it among those returned by a live list.
	if inUse, err := c.askAPIServerForPod(pvc); err != nil {
		return false, false, err
	} else if inUse {
		// If we reach here, this means, the the PVC is still in use by a Pod, hence no
		// reque is needed. We will wait for an event to process the PVC.
		return true, false, nil
	}

	if pv != nil && pv.Spec.CSI != nil {
		uniqueVolHandle, err := c.generateUniqueCSIVolumeHandle(pv.Spec.CSI)
		if err != nil {
			return false, false, err
		}
		if inUse, err := c.askAPIServerForNode(uniqueVolHandle, pvc); err != nil {
			return false, false, err
		} else if inUse {
			// If we reach here, this means we have not found any Pod , referencing this PVC,
			// and the volume is still in use by a node. So we reque the PVC again to process
			// it faster.
			return true, true, nil
		}
	}

	return false, false, nil
}

func (c *Controller) askPodInformer(pvc *v1.PersistentVolumeClaim) (bool, error) {
	klog.V(4).Infof("Looking for Pods using PVC %s/%s in the Informer's cache", pvc.Namespace, pvc.Name)

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

	klog.V(4).Infof("No Pod using PVC %s/%s was found in the Informer's cache", pvc.Namespace, pvc.Name)
	return false, nil
}

func (c *Controller) askAPIServerForPod(pvc *v1.PersistentVolumeClaim) (bool, error) {
	klog.V(4).Infof("Looking for Pods using PVC %s/%s with a live list", pvc.Namespace, pvc.Name)

	podsList, err := c.client.CoreV1().Pods(pvc.Namespace).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("live list of pods failed: %s", err.Error())
	}

	for _, pod := range podsList.Items {
		if c.podUsesPVC(&pod, pvc) {
			return true, nil
		}
	}

	klog.V(2).Infof("PVC %s/%s is unused", pvc.Namespace, pvc.Name)
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
				klog.V(2).Infof("Pod %s/%s uses PVC %s", pod.Namespace, pod.Name, pvc)
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
	klog.V(4).Infof("Got event on PVC %s", key)

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

	klog.V(4).Infof("Enqueuing PVCs for Pod %s/%s (UID=%s)", pod.Namespace, pod.Name, pod.UID)

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

func (c *Controller) getPV(pvc *v1.PersistentVolumeClaim) (*v1.PersistentVolume, error) {
	pv, err := c.pvLister.Get(pvc.Spec.VolumeName)
	if err != nil {
		klog.V(3).Infof("unexpected error getting persistent volume %q from cache for claim %s/%s: %v", pvc.Spec.VolumeName, pvc.Namespace, pvc.Name, err)
		pv, err = c.client.CoreV1().PersistentVolumes().Get(context.TODO(), pvc.Spec.VolumeName, metav1.GetOptions{})
	}
	return pv, err
}

func (c *Controller) askNodeInformer(volumeHandle string, pvc *v1.PersistentVolumeClaim) (bool, error) {
	klog.V(3).Infof("For claim %s/%s checking for volume handle %s in node informer cache", pvc.Namespace, pvc.Name, volumeHandle)
	if volumeHandle == "" {
		return false, fmt.Errorf("Empty volume handle for claim %s/%s", pvc.Namespace, pvc.Name)
	}

	nodes, err := c.nodeLister.List(labels.Everything())
	if err != nil {
		return false, fmt.Errorf("unexpected error listing nodes from node informer: %v", err)
	}
	return c.isVolumeInUse(volumeHandle, nodes), nil
}

func (c *Controller) askAPIServerForNode(volumeHandle string, pvc *v1.PersistentVolumeClaim) (bool, error) {
	klog.V(3).Infof("For claim %s/%s, volume %s, list nodes from API server", pvc.Namespace, pvc.Name, volumeHandle)
	nodes, err := c.client.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("List of nodes from API server failed: %s", err.Error())
	}
	nodesPtr := []*v1.Node{}
	for i := range nodes.Items {
		nodesPtr = append(nodesPtr, &nodes.Items[i])
	}

	return c.isVolumeInUse(volumeHandle, nodesPtr), nil
}

// Helper function to determine if a given volume is present in the node's volumesInUse list.
func (c *Controller) isVolumeInUse(volHandle string, nodes []*v1.Node) bool {
	for _, node := range nodes {
		for _, volInUse := range node.Status.VolumesInUse {
			volInUseStr := strings.TrimPrefix(string(volInUse), volumeHandleCSIPrefix)
			if volInUseStr == volHandle {
				klog.Infof("Found volume %s in use at node %s", volInUseStr, node.Name)
				return true
			}
		}
	}
	return false
}

func (c *Controller) generateUniqueCSIVolumeHandle(source *v1.CSIPersistentVolumeSource) (string, error) {
	if source.Driver == "" || source.VolumeHandle == "" {
		return "", fmt.Errorf("Failed to fetch CSI driver plugin or volume for CSI source %+v", source)
	}
	return fmt.Sprintf("%s%s%s", source.Driver, volumeHandleSep, source.VolumeHandle), nil
}
