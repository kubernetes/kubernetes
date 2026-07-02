/*
Copyright The Kubernetes Authors.

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

package podcheckpoint

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
	nodeapply "k8s.io/client-go/applyconfigurations/node/v1alpha1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	nodeinformers "k8s.io/client-go/informers/node/v1alpha1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	nodelisters "k8s.io/client-go/listers/node/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
	"k8s.io/kubernetes/pkg/controller"
)

// RestoreLockFinalizer is installed at creation, before a restore Pod can
// reference the checkpoint. It is removed only during deletion, once no Pod
// restores from the checkpoint.
const RestoreLockFinalizer = nodeapi.PodCheckpointRestoreProtectionFinalizer

// Controller manages the lifecycle of PodCheckpoint objects.
//
// The controller assigns each checkpoint to the node running its source Pod,
// so kubelets can watch only their own checkpoints. The kubelet executes the
// checkpoint and finalizes its status (see Kubelet.startPodCheckpointWatch).
// The controller also keeps the restore-lock finalizer while a Pod is restoring
// from a checkpoint, as the PVC protection controller does for Pods that use a
// claim. Garbage collection of checkpoint objects/archives is a follow-up.
type Controller struct {
	controllerName    string
	kubeClient        clientset.Interface
	eventBroadcaster  record.EventBroadcaster
	recorder          record.EventRecorder
	checkpointLister  nodelisters.PodCheckpointLister
	checkpointsSynced cache.InformerSynced
	podLister         corelisters.PodLister
	// podIndexer belongs to the shared Pod informer, whose indexes other
	// controllers also use; see controller.AddPodRestoreFromIndexer.
	podIndexer      cache.Indexer
	podsSynced      cache.InformerSynced
	checkpointQueue workqueue.TypedRateLimitingInterface[cache.ObjectName]
	syncHandler     func(context.Context, cache.ObjectName) error
}

// NewController creates a new PodCheckpoint lifecycle controller. It reads
// PodCheckpoints and Pods from the shared informers so it can assign
// checkpoints and maintain the restore-lock finalizer.
func NewController(ctx context.Context, controllerName string, checkpointInformer nodeinformers.TypedPodCheckpointInformer, podInformer coreinformers.TypedPodInformer, kubeClient clientset.Interface) (*Controller, error) {
	logger := klog.FromContext(ctx)
	eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
	c := &Controller{
		controllerName:    controllerName,
		kubeClient:        kubeClient,
		eventBroadcaster:  eventBroadcaster,
		recorder:          eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: controllerName}),
		checkpointLister:  checkpointInformer.Lister(),
		checkpointsSynced: checkpointInformer.Informer().HasSynced,
		podLister:         podInformer.Lister(),
		podIndexer:        podInformer.Informer().GetIndexer(),
		podsSynced:        podInformer.Informer().HasSynced,
		checkpointQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[cache.ObjectName](),
			workqueue.TypedRateLimitingQueueConfig[cache.ObjectName]{
				Name:   controllerName,
				Logger: &logger,
			},
		),
	}
	c.syncHandler = c.syncCheckpoint

	if err := controller.AddPodRestoreFromIndexer(podInformer.Informer()); err != nil {
		return nil, fmt.Errorf("failed to add Pod indexer: %w", err)
	}

	if _, err := checkpointInformer.TypedInformer().AddTypedEventHandler(nodeinformers.PodCheckpointHandlerFuncs{
		AddFunc:    c.enqueueCheckpoint,
		UpdateFunc: c.updateCheckpoint,
	}, cache.HandlerOptions{Logger: &logger}); err != nil {
		return nil, fmt.Errorf("failed to add PodCheckpoint event handler: %w", err)
	}

	// The state of a restore Pod decides whether its checkpoint stays locked.
	if _, err := podInformer.TypedInformer().AddTypedEventHandler(coreinformers.PodHandlerFuncs{
		AddFunc:    c.addPod,
		UpdateFunc: c.updatePod,
		DeleteFunc: func(deleted coreinformers.DeletedPod) {
			if deleted.OptionalObj != nil {
				c.enqueueRestoredCheckpoint(deleted.OptionalObj)
			}
		},
	}, cache.HandlerOptions{Logger: &logger}); err != nil {
		return nil, fmt.Errorf("failed to add Pod event handler: %w", err)
	}

	return c, nil
}

func (c *Controller) enqueueCheckpoint(checkpoint *nodev1alpha1.PodCheckpoint) {
	c.checkpointQueue.Add(cache.MetaObjectToName(checkpoint))
}

// updateCheckpoint ignores updates that this controller does not act on. After
// assignment the kubelet writes status several times per checkpoint, and only
// the start of deletion and changes to RestoreLockFinalizer matter then.
func (c *Controller) updateCheckpoint(old, checkpoint *nodev1alpha1.PodCheckpoint) {
	if needsAssignment(checkpoint) || restoreLockChanged(old, checkpoint) {
		c.enqueueCheckpoint(checkpoint)
	}
}

func restoreLockChanged(old, checkpoint *nodev1alpha1.PodCheckpoint) bool {
	return old.UID != checkpoint.UID ||
		(old.DeletionTimestamp == nil) != (checkpoint.DeletionTimestamp == nil) ||
		slices.Contains(old.Finalizers, RestoreLockFinalizer) != slices.Contains(checkpoint.Finalizers, RestoreLockFinalizer)
}

// enqueueRestoredCheckpoint enqueues the checkpoint that the Pod restores from,
// whose restore lock may now be released.
func (c *Controller) enqueueRestoredCheckpoint(pod *v1.Pod) {
	if pod.Spec.RestoreFrom == nil || pod.Spec.RestoreFrom.Name == "" {
		return
	}
	c.checkpointQueue.Add(cache.NewObjectName(pod.Namespace, pod.Spec.RestoreFrom.Name))
}

// addPod also covers a Pod that the cache first sees after a relist, which may
// have finished restoring already.
func (c *Controller) addPod(pod *v1.Pod) {
	c.enqueueRestoredCheckpoint(pod)
}

// updatePod skips Pods that do not restore from a checkpoint, and updates that
// cannot change restoreActive, so a restore Pod is not reconciled for every
// status write while it runs.
func (c *Controller) updatePod(old, pod *v1.Pod) {
	if pod.Spec.RestoreFrom == nil || !restorePodChanged(old, pod) {
		return
	}
	c.enqueueRestoredCheckpoint(pod)
}

func restorePodChanged(old, pod *v1.Pod) bool {
	if !old.DeletionTimestamp.Equal(pod.DeletionTimestamp) || old.Status.Phase != pod.Status.Phase {
		return true
	}
	_, oldCondition := podutil.GetPodCondition(&old.Status, v1.PodRestored)
	_, condition := podutil.GetPodCondition(&pod.Status, v1.PodRestored)
	if oldCondition == nil || condition == nil {
		return oldCondition != condition
	}
	return oldCondition.Status != condition.Status
}

// Run starts workers to reconcile PodCheckpoints.
func (c *Controller) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrashWithContext(ctx)

	logger := klog.FromContext(ctx)
	logger.Info("Starting controller", "controller", c.controllerName)

	c.eventBroadcaster.StartStructuredLogging(3)
	c.eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: c.kubeClient.CoreV1().Events("")})

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down controller", "controller", c.controllerName)
		c.checkpointQueue.ShutDown()
		wg.Wait()
		c.eventBroadcaster.Shutdown()
	}()

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.checkpointsSynced, c.podsSynced) {
		return
	}

	for range workers {
		wg.Go(func() {
			wait.UntilWithContext(ctx, c.worker, time.Second)
		})
	}

	<-ctx.Done()
}

func (c *Controller) worker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	key, quit := c.checkpointQueue.Get()
	if quit {
		return false
	}
	defer c.checkpointQueue.Done(key)

	logger := klog.FromContext(ctx)
	if err := c.syncHandler(ctx, key); err != nil {
		logger.V(4).Info("Failed to sync PodCheckpoint", "key", key, "err", err)
		c.checkpointQueue.AddRateLimited(key)
		return true
	}
	c.checkpointQueue.Forget(key)
	return true
}

// syncHandler assigns pending checkpoints and keeps protection until deletion,
// then releases it once no Pod restores from the checkpoint.
// Decisions start from the informer caches; API reads happen only before a
// write that the caches cannot justify on their own.
func (c *Controller) syncCheckpoint(ctx context.Context, key cache.ObjectName) error {
	namespace, name := key.Parts()
	checkpoint, err := c.checkpointLister.PodCheckpoints(namespace).Get(name)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}

	has := slices.Contains(checkpoint.Finalizers, RestoreLockFinalizer)
	if checkpoint.DeletionTimestamp == nil {
		if !has {
			return c.setRestoreLock(ctx, checkpoint, true)
		}
		return c.assignCheckpoint(ctx, checkpoint)
	}
	if !has {
		return nil
	}

	active, err := c.hasActiveRestore(checkpoint.Namespace, checkpoint.Name)
	if err != nil {
		return err
	}
	if active {
		return nil
	}

	// The cache can miss a restore Pod that admission accepted shortly before
	// the deletion. Pod field selectors do not support spec.restoreFrom, so list
	// the namespace live, as askAPIServer in the PVC protection controller does.
	// This narrows the window without closing it: admission can read the
	// checkpoint before the deletion starts while the Pod is stored only after
	// this list. The kubelet must then fail that restore because the checkpoint
	// is gone.
	pods, err := c.kubeClient.CoreV1().Pods(checkpoint.Namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return err
	}
	for i := range pods.Items {
		if restoreActive(&pods.Items[i], checkpoint.Name) {
			return nil
		}
	}
	return c.setRestoreLock(ctx, checkpoint, false)
}

// needsAssignment reports whether the checkpoint still waits for this
// controller. An assignment is kept even if the source Pod later disappears or
// its name is reused, because the kubelet on the assigned node reports the
// outcome. A deleting checkpoint is never captured, and Ready=True or a
// failure reason means that the outcome is final.
func needsAssignment(checkpoint *nodev1alpha1.PodCheckpoint) bool {
	if checkpoint.DeletionTimestamp != nil {
		return false
	}
	if checkpoint.Status == nil {
		return true
	}
	if checkpoint.Status.NodeName != nil || checkpoint.Status.SourcePodUID != nil {
		return false
	}
	condition := apimeta.FindStatusCondition(checkpoint.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
	if condition == nil {
		return true
	}
	terminal := condition.Status == metav1.ConditionTrue ||
		condition.Reason == nodev1alpha1.PodCheckpointReasonFailed ||
		condition.Reason == nodev1alpha1.PodCheckpointReasonSourcePodReplaced
	return !terminal
}

// assignCheckpoint publishes routing before the kubelet can observe the object
// through its node-scoped watch, or fails the checkpoint when its source Pod
// cannot be captured.
//
// REST storage accepts a checkpoint only while the source Pod exists, matches
// spec.sourcePod.uid, is bound and Running, and is not being deleted. The Pod
// can change before assignment, so the controller checks again. If it has
// reached Failed or Succeeded, it can no longer be checkpointed; the controller
// marks the checkpoint Ready=False with reason CheckpointFailed instead of
// assigning a node. These source failures are final.
func (c *Controller) assignCheckpoint(ctx context.Context, checkpoint *nodev1alpha1.PodCheckpoint) error {
	ref := &checkpoint.Spec.SourcePod
	if !needsAssignment(checkpoint) || ref.Name == "" {
		return nil
	}
	pod, err := c.getSourcePod(ctx, checkpoint.Namespace, ref)
	if err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("get source Pod %s/%s for checkpoint assignment: %w", checkpoint.Namespace, ref.Name, err)
	}

	reason := nodev1alpha1.PodCheckpointReasonFailed
	var message string
	switch {
	case apierrors.IsNotFound(err):
		message = fmt.Sprintf("source Pod %q no longer exists", ref.Name)
	case ref.UID != nil && *ref.UID != pod.UID:
		reason = nodev1alpha1.PodCheckpointReasonSourcePodReplaced
		message = fmt.Sprintf("source Pod %q has UID %q, not requested UID %q", ref.Name, pod.UID, *ref.UID)
	case pod.DeletionTimestamp != nil:
		message = fmt.Sprintf("source Pod %q is being deleted", ref.Name)
	case pod.Spec.NodeName == "":
		message = fmt.Sprintf("source Pod %q is not bound to a node", ref.Name)
	case podutil.IsPodTerminal(pod):
		message = fmt.Sprintf("source Pod %q has terminated with phase %s", ref.Name, pod.Status.Phase)
	default:
		_, err := c.applyStatus(ctx, checkpoint, nodeapply.PodCheckpointStatus().
			WithNodeName(pod.Spec.NodeName).
			WithSourcePodUID(pod.UID))
		return err
	}

	// Set completionTime for failures too, so every final outcome has one.
	now := metav1.Now()
	updated, err := c.applyStatus(ctx, checkpoint, nodeapply.PodCheckpointStatus().
		WithConditions(metav1ac.Condition().
			WithType(nodev1alpha1.PodCheckpointConditionReady).
			WithStatus(metav1.ConditionFalse).
			WithReason(reason).
			WithMessage(message).
			WithObservedGeneration(checkpoint.Generation).
			WithLastTransitionTime(now)).
		WithCompletionTime(now))
	if err != nil {
		return err
	}
	c.recorder.Event(&v1.ObjectReference{
		APIVersion: nodev1alpha1.SchemeGroupVersion.String(), Kind: "PodCheckpoint",
		Namespace: updated.Namespace, Name: updated.Name, UID: updated.UID, ResourceVersion: updated.ResourceVersion,
	}, v1.EventTypeWarning, nodev1alpha1.PodCheckpointReasonFailed, message)
	return nil
}

// applyStatus writes the status fields that this controller owns with
// server-side apply. The decision was made from the cached checkpoint, so the
// request carries its resourceVersion. If the object changed since, the
// request fails with a conflict and the key returns to the queue; otherwise an
// outdated copy could replace a final failure with an assignment.
func (c *Controller) applyStatus(ctx context.Context, checkpoint *nodev1alpha1.PodCheckpoint, status *nodeapply.PodCheckpointStatusApplyConfiguration) (*nodev1alpha1.PodCheckpoint, error) {
	apply := nodeapply.PodCheckpoint(checkpoint.Name, checkpoint.Namespace).
		WithResourceVersion(checkpoint.ResourceVersion).
		WithStatus(status)
	return c.kubeClient.NodeV1alpha1().PodCheckpoints(checkpoint.Namespace).ApplyStatus(ctx, apply, metav1.ApplyOptions{FieldManager: c.controllerName, Force: true})
}

// getSourcePod returns the cached source Pod when it can be assigned. The cache
// can lag behind a Pod that was just created, scheduled, or replaced, so any
// other outcome is decided from a live read. This keeps a stale cache from
// failing the checkpoint because its source appears missing, replaced, or
// unbound.
func (c *Controller) getSourcePod(ctx context.Context, namespace string, ref *nodev1alpha1.PodReference) (*v1.Pod, error) {
	pod, err := c.podLister.Pods(namespace).Get(ref.Name)
	if err != nil && !apierrors.IsNotFound(err) {
		return nil, err
	}
	if err == nil && (ref.UID == nil || *ref.UID == pod.UID) &&
		pod.Spec.NodeName != "" && pod.DeletionTimestamp == nil && !podutil.IsPodTerminal(pod) {
		return pod, nil
	}
	return c.kubeClient.CoreV1().Pods(namespace).Get(ctx, ref.Name, metav1.GetOptions{})
}

// hasActiveRestore reports whether any cached Pod is currently restoring from
// the named checkpoint.
func (c *Controller) hasActiveRestore(namespace, name string) (bool, error) {
	objs, err := c.podIndexer.ByIndex(controller.PodRestoreFromIndex, controller.PodRestoreFromIndexKey(namespace, name))
	if err != nil {
		return false, err
	}
	for _, obj := range objs {
		pod, ok := obj.(*v1.Pod)
		if ok && restoreActive(pod, name) {
			return true, nil
		}
	}
	return false, nil
}

// restoreActive reports whether the Pod can still read the checkpoint's data,
// so the checkpoint keeps RestoreLockFinalizer. A Pod is active until it
// terminates or reports a final PodRestored condition, because until then a
// kubelet can still start or run the restore. A deleted Pod no longer counts,
// as in PVC protection; the kubelet keeps node-local data while its restore
// still reads it.
func restoreActive(pod *v1.Pod, checkpointName string) bool {
	if pod.Spec.RestoreFrom == nil || pod.Spec.RestoreFrom.Name != checkpointName || podutil.IsPodTerminal(pod) {
		return false
	}
	_, condition := podutil.GetPodCondition(&pod.Status, v1.PodRestored)
	return condition == nil || condition.Status == v1.ConditionUnknown
}

// setRestoreLock adds or removes the restore-lock finalizer on the PodCheckpoint.
// A failed patch returns the key to the queue, which decides again from the
// next cache state.
func (c *Controller) setRestoreLock(ctx context.Context, checkpoint *nodev1alpha1.PodCheckpoint, present bool) error {
	patch, err := finalizerPatch(checkpoint.UID, checkpoint.ResourceVersion, RestoreLockFinalizer, present)
	if err != nil {
		return err
	}
	_, err = c.kubeClient.NodeV1alpha1().PodCheckpoints(checkpoint.Namespace).Patch(ctx, checkpoint.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
	return err
}

// finalizerPatch adds or removes one finalizer with a strategic merge patch,
// like removeTrackingFinalizerPatch in the Job controller, so finalizers of
// other components are left alone. Checkpoint retention also checks the observed
// resourceVersion so a newly registered consumer cannot be missed. The UID
// makes the patch fail instead of changing an object that reused the name.
func finalizerPatch(uid types.UID, resourceVersion, finalizer string, add bool) ([]byte, error) {
	list := "$deleteFromPrimitiveList/finalizers"
	if add {
		list = "finalizers"
	}
	metadata := map[string]any{"uid": uid, list: []string{finalizer}}
	if resourceVersion != "" {
		metadata["resourceVersion"] = resourceVersion
	}
	return json.Marshal(map[string]any{"metadata": metadata})
}
