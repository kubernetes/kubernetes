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
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/taints"
)

// RestoreLockFinalizer is installed at creation, before a restore Pod can
// reference the checkpoint. It is removed only during deletion, after checking
// the persisted restore reservations.
const RestoreLockFinalizer = nodeapi.PodCheckpointRestoreProtectionFinalizer

// Controller manages the lifecycle of PodCheckpoint objects.
//
// The controller assigns each checkpoint to the node running its source Pod,
// so kubelets can watch only their own checkpoints. The kubelet executes the
// checkpoint and finalizes its status (see Kubelet.startPodCheckpointWatch).
// The controller also keeps the restore-lock finalizer while a Pod is restoring
// from a checkpoint, and releases the restore protection of Pods once the
// kubelet no longer needs them. Garbage collection of checkpoint
// objects/archives is a follow-up.
type Controller struct {
	controllerName    string
	kubeClient        clientset.Interface
	eventBroadcaster  record.EventBroadcaster
	recorder          record.EventRecorder
	checkpointLister  nodelisters.PodCheckpointLister
	checkpointsSynced cache.InformerSynced
	podLister         corelisters.PodLister
	// podIndexer belongs to the shared Pod informer, whose indexes other
	// controllers also use; see controller.AddPodNodeNameIndexer.
	podIndexer  cache.Indexer
	podsSynced  cache.InformerSynced
	nodeLister  corelisters.NodeLister
	nodesSynced cache.InformerSynced
	// checkpointQueue holds PodCheckpoints to assign or whose restore lock may
	// change.
	checkpointQueue workqueue.TypedRateLimitingInterface[string]
	// restorePodQueue holds restore Pods that may release their protection.
	restorePodQueue workqueue.TypedRateLimitingInterface[string]
}

// NewController creates a new PodCheckpoint lifecycle controller. It reads
// PodCheckpoints, Pods and Nodes from the shared informers so it can assign
// checkpoints, maintain the restore-lock finalizer and release restore Pods.
func NewController(ctx context.Context, controllerName string, checkpointInformer nodeinformers.TypedPodCheckpointInformer, podInformer coreinformers.TypedPodInformer, nodeInformer coreinformers.TypedNodeInformer, kubeClient clientset.Interface) (*Controller, error) {
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
		nodeLister:        nodeInformer.Lister(),
		nodesSynced:       nodeInformer.Informer().HasSynced,
		checkpointQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name:   controllerName + "_pod_checkpoint",
				Logger: &logger,
			},
		),
		restorePodQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name:   controllerName + "_restore_pod",
				Logger: &logger,
			},
		),
	}
	var checkpointHandler, podHandler cache.ResourceEventHandlerRegistration
	success := false
	defer func() {
		if success {
			return
		}
		if podHandler != nil {
			_ = cache.ShutDownEventHandler(podInformer.Informer(), podHandler)
		}
		if checkpointHandler != nil {
			_ = cache.ShutDownEventHandler(checkpointInformer.Informer(), checkpointHandler)
		}
		c.checkpointQueue.ShutDown()
		c.restorePodQueue.ShutDown()
		c.eventBroadcaster.Shutdown()
	}()

	if err := controller.AddPodNodeNameIndexer(podInformer.Informer()); err != nil {
		return nil, fmt.Errorf("failed to add Pod indexer: %w", err)
	}

	checkpointHandler, err := checkpointInformer.TypedInformer().AddTypedEventHandler(nodeinformers.PodCheckpointHandlerFuncs{
		AddFunc:    c.enqueueCheckpoint,
		UpdateFunc: c.updateCheckpoint,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to add PodCheckpoint event handler: %w", err)
	}

	// Pod status decides whether the Pod keeps its restore protection.
	// Checkpoint reservations are released separately by the kubelet.
	podHandler, err = podInformer.TypedInformer().AddTypedEventHandler(coreinformers.PodHandlerFuncs{
		AddFunc:    c.addPod,
		UpdateFunc: c.updatePod,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to add Pod event handler: %w", err)
	}

	// No kubelet reports for the Pods of a deleted or out-of-service Node.
	if _, err := nodeInformer.TypedInformer().AddTypedEventHandler(coreinformers.NodeHandlerFuncs{
		UpdateFunc: func(old, node *v1.Node) {
			c.updateNode(logger, old, node)
		},
		DeleteFunc: func(deleted coreinformers.DeletedNode) {
			c.enqueueNodeRestorePods(logger, deleted.GetName())
		},
	}); err != nil {
		return nil, fmt.Errorf("failed to add Node event handler: %w", err)
	}

	success = true
	return c, nil
}

func (c *Controller) enqueueCheckpoint(checkpoint *nodev1alpha1.PodCheckpoint) {
	c.checkpointQueue.Add(cache.MetaObjectToName(checkpoint).String())
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
		!slices.Equal(old.Status.ActiveRestores, checkpoint.Status.ActiveRestores) ||
		(old.DeletionTimestamp == nil) != (checkpoint.DeletionTimestamp == nil) ||
		slices.Contains(old.Finalizers, RestoreLockFinalizer) != slices.Contains(checkpoint.Finalizers, RestoreLockFinalizer)
}

func (c *Controller) enqueueRestorePod(pod *v1.Pod) {
	c.restorePodQueue.Add(cache.MetaObjectToName(pod).String())
}

// addPod also covers a Pod that the cache first sees after a relist, which may
// have finished restoring already.
func (c *Controller) addPod(pod *v1.Pod) {
	if pod.Spec.RestoreFrom == nil {
		return
	}
	if restoreProtected(pod) {
		c.enqueueRestorePod(pod)
	}
}

// updatePod skips Pods that do not restore from a checkpoint, and updates that
// cannot change restoreProtectionReleasable, so a restore Pod
// is not reconciled for every status write while it runs.
func (c *Controller) updatePod(old, pod *v1.Pod) {
	if pod.Spec.RestoreFrom == nil || !restorePodChanged(old, pod) {
		return
	}
	if restoreProtected(pod) {
		c.enqueueRestorePod(pod)
	}
}

func restorePodChanged(old, pod *v1.Pod) bool {
	if !slices.Equal(old.Finalizers, pod.Finalizers) || !old.DeletionTimestamp.Equal(pod.DeletionTimestamp) ||
		old.Status.Phase != pod.Status.Phase {
		return true
	}
	_, oldCondition := podutil.GetPodCondition(&old.Status, v1.PodRestored)
	_, condition := podutil.GetPodCondition(&pod.Status, v1.PodRestored)
	if oldCondition == nil || condition == nil {
		return oldCondition != condition
	}
	return oldCondition.Status != condition.Status || oldCondition.Reason != condition.Reason
}

// updateNode skips Node updates other than the one that marks it out of
// service, so restore Pods are not listed for every Node status write.
func (c *Controller) updateNode(logger klog.Logger, old, node *v1.Node) {
	if !nodeOutOfService(old) && nodeOutOfService(node) {
		c.enqueueNodeRestorePods(logger, node.Name)
	}
}

// enqueueNodeRestorePods enqueues the deleting restore Pods that were bound to
// a deleted or out-of-service Node, because their protection no longer waits
// for a kubelet report.
func (c *Controller) enqueueNodeRestorePods(logger klog.Logger, nodeName string) {
	objs, err := c.podIndexer.ByIndex(controller.PodNodeNameKeyIndex, nodeName)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to list restore Pods of Node", "node", nodeName)
		return
	}
	for _, obj := range objs {
		pod, ok := obj.(*v1.Pod)
		if ok && pod.DeletionTimestamp != nil && restoreProtected(pod) {
			c.enqueueRestorePod(pod)
		}
	}
}

// Run starts one worker for each queue. Each item needs at most a few small API
// requests, and checkpoints are created explicitly, so one worker per queue
// keeps up.
func (c *Controller) Run(ctx context.Context) {
	defer utilruntime.HandleCrashWithContext(ctx)

	logger := klog.FromContext(ctx)
	logger.Info("Starting controller", "controller", c.controllerName)

	c.eventBroadcaster.StartStructuredLogging(3)
	c.eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: c.kubeClient.CoreV1().Events("")})

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down controller", "controller", c.controllerName)
		c.checkpointQueue.ShutDown()
		c.restorePodQueue.ShutDown()
		wg.Wait()
		c.eventBroadcaster.Shutdown()
	}()

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.checkpointsSynced, c.podsSynced, c.nodesSynced) {
		return
	}

	wg.Go(func() {
		wait.UntilWithContext(ctx, c.worker, time.Second)
	})
	wg.Go(func() {
		wait.UntilWithContext(ctx, c.restorePodWorker, time.Second)
	})

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

	if err := c.syncHandler(ctx, key); err != nil {
		utilruntime.HandleErrorWithLogger(klog.FromContext(ctx), err, "Failed to sync PodCheckpoint", "key", key)
		c.checkpointQueue.AddRateLimited(key)
		return true
	}
	c.checkpointQueue.Forget(key)
	return true
}

func (c *Controller) restorePodWorker(ctx context.Context) {
	for c.processNextRestorePod(ctx) {
	}
}

func (c *Controller) processNextRestorePod(ctx context.Context) bool {
	key, quit := c.restorePodQueue.Get()
	if quit {
		return false
	}
	defer c.restorePodQueue.Done(key)

	if err := c.syncRestorePod(ctx, key); err != nil {
		utilruntime.HandleErrorWithLogger(klog.FromContext(ctx), err, "Failed to sync restore Pod", "key", key)
		c.restorePodQueue.AddRateLimited(key)
		return true
	}
	c.restorePodQueue.Forget(key)
	return true
}

// syncHandler assigns pending checkpoints and keeps protection until deletion,
// then releases it once the persisted restore reservations are empty.
// Decisions start from the informer caches; API reads happen only before a
// write that the caches cannot justify on their own.
func (c *Controller) syncHandler(ctx context.Context, key string) error {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
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

	if len(checkpoint.Status.ActiveRestores) != 0 {
		return nil
	}
	return c.setRestoreLock(ctx, checkpoint, false)
}

// needsAssignment reports whether the checkpoint still waits for this
// controller. An assignment is kept even if the source Pod later disappears or
// its name is reused, because the kubelet on the assigned node reports the
// outcome. A deleting checkpoint is never captured, and Ready=True or a
// failure reason means that the outcome is final.
func needsAssignment(checkpoint *nodev1alpha1.PodCheckpoint) bool {
	if checkpoint.DeletionTimestamp != nil || checkpoint.Status.NodeName != nil || checkpoint.Status.SourcePodUID != nil {
		return false
	}
	condition := apimeta.FindStatusCondition(checkpoint.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady)
	if condition == nil {
		return true
	}
	return condition.Status != metav1.ConditionTrue &&
		condition.Reason != nodev1alpha1.PodCheckpointReasonFailed &&
		condition.Reason != nodev1alpha1.PodCheckpointReasonSourcePodReplaced
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
		if apierrors.IsNotFound(err) {
			return nil
		}
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
	if apierrors.IsNotFound(err) {
		return nil
	}
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

func restoreProtected(pod *v1.Pod) bool {
	return pod.Spec.RestoreFrom != nil && slices.Contains(pod.Finalizers, nodeapi.PodRestoreProtectionFinalizer)
}

// restoreProtectionReleasable reports whether the kubelet no longer needs the
// Pod object to finish restore work. The kubelet reports in the PodRestored
// condition:
//   - True: the restore completed, so the runtime no longer reads the
//     checkpoint.
//   - False with RestoreCleanupCompleted or RestoreNotStarted: the runtime
//     confirmed cleanup of a failed restore, or no restore work reached it.
//   - False with RestoreCleanupUnconfirmed: the Pod was deleted before the
//     runtime confirmed cleanup. The kubelet keeps retrying cleanup for the
//     node-local operation, which does not need the Pod object.
//
// Other reasons mean that the kubelet can still call the runtime for this Pod.
// A deleted Pod that was never bound needs no report, because no kubelet
// received it and binding rejects a Pod that is being deleted.
func restoreProtectionReleasable(pod *v1.Pod) bool {
	if pod.DeletionTimestamp != nil && pod.Spec.NodeName == "" {
		return true
	}
	_, condition := podutil.GetPodCondition(&pod.Status, v1.PodRestored)
	if condition == nil {
		return false
	}
	switch condition.Status {
	case v1.ConditionTrue:
		return true
	case v1.ConditionFalse:
		switch condition.Reason {
		case nodeapi.PodRestoreReasonCleanupCompleted, nodeapi.PodRestoreReasonNotStarted, nodeapi.PodRestoreReasonCleanupUnconfirmed:
			return true
		}
	}
	return false
}

// syncRestorePod releases PodRestoreProtectionFinalizer once the Pod no longer
// needs it. The Pod is reconciled independently of its checkpoint, which may
// have disappeared through an administrator's explicit removal of checkpoint
// protection.
//
// Besides this controller, only a client with update permission on
// pods/finalizers can remove the finalizer: Pod storage checks that permission
// on every update that removes it. An administrator can use it to release a Pod
// that no kubelet will report for.
func (c *Controller) syncRestorePod(ctx context.Context, key string) error {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	pod, err := c.podLister.Pods(namespace).Get(name)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if !restoreProtected(pod) {
		return nil
	}
	if !restoreProtectionReleasable(pod) {
		// No kubelet reports for a Pod of a deleted or out-of-service Node,
		// and PodGC cannot remove the deleting Pod while the finalizer
		// remains. Cleanup on the node, if any, continues without the Pod
		// object.
		if pod.DeletionTimestamp == nil {
			return nil
		}
		unavailable, err := c.nodeUnavailable(ctx, pod.Spec.NodeName)
		if err != nil {
			return err
		}
		if !unavailable {
			return nil
		}
		klog.FromContext(ctx).V(2).Info("Releasing restore protection of a Pod whose Node was deleted or marked out of service", "pod", klog.KObj(pod), "node", pod.Spec.NodeName)
	}
	patch, err := finalizerPatch(pod.UID, "", nodeapi.PodRestoreProtectionFinalizer, false)
	if err != nil {
		return err
	}
	_, err = c.kubeClient.CoreV1().Pods(namespace).Patch(ctx, name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
	if apierrors.IsNotFound(err) {
		return nil
	}
	return err
}

// nodeUnavailable reports whether the Node no longer exists or is out of
// service. The cache can lag behind a Node that was just registered, so a miss
// is confirmed with a live read, as PodGC does before it deletes the Pods of a
// deleted Node.
func (c *Controller) nodeUnavailable(ctx context.Context, name string) (bool, error) {
	node, err := c.nodeLister.Get(name)
	if err == nil {
		return nodeOutOfService(node), nil
	}
	if !apierrors.IsNotFound(err) {
		return false, err
	}
	node, err = c.kubeClient.CoreV1().Nodes().Get(ctx, name, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return true, nil
	}
	if err != nil {
		return false, err
	}
	return nodeOutOfService(node), nil
}

// nodeOutOfService reports whether an administrator has marked a NotReady Node
// with the node.kubernetes.io/out-of-service taint, which non-graceful node
// shutdown requires only after the node is confirmed to be powered off. PodGC
// uses the same condition to force-delete the Node's terminating Pods. A
// timeout would not establish that the kubelet stopped using the checkpoint,
// and the taint, unlike a zero deletion grace period, needs Node update
// permission.
func nodeOutOfService(node *v1.Node) bool {
	return !nodeutil.IsNodeReady(node) && taints.TaintKeyExists(node.Spec.Taints, v1.TaintNodeOutOfService)
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
	if apierrors.IsNotFound(err) {
		return nil
	}
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
