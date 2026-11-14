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
	"fmt"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
)

// Controller manages the lifecycle of PodCheckpoint objects.
//
// The controller assigns each checkpoint to the node running its source Pod,
// so kubelets can watch only their own checkpoints. The kubelet executes the
// checkpoint and finalizes its status (see Kubelet.startPodCheckpointWatch).
// Garbage collection of checkpoint objects/archives is a follow-up.
type Controller struct {
	controllerName    string
	kubeClient        clientset.Interface
	eventBroadcaster  record.EventBroadcaster
	recorder          record.EventRecorder
	checkpointLister  nodelisters.PodCheckpointLister
	checkpointsSynced cache.InformerSynced
	podLister         corelisters.PodLister
	podsSynced        cache.InformerSynced
	queue             workqueue.TypedRateLimitingInterface[string]
}

// NewController creates a new PodCheckpoint lifecycle controller. It reads
// PodCheckpoints and their source Pods from the shared informers so it can
// assign checkpoints.
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
		podsSynced:        podInformer.Informer().HasSynced,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name:   controllerName,
				Logger: &logger,
			},
		),
	}

	if _, err := checkpointInformer.TypedInformer().AddTypedEventHandler(nodeinformers.PodCheckpointHandlerFuncs{
		AddFunc:    c.enqueueCheckpoint,
		UpdateFunc: c.updateCheckpoint,
	}); err != nil {
		c.queue.ShutDown()
		c.eventBroadcaster.Shutdown()
		return nil, fmt.Errorf("failed to add PodCheckpoint event handler: %w", err)
	}

	return c, nil
}

func (c *Controller) enqueueCheckpoint(checkpoint *nodev1alpha1.PodCheckpoint) {
	c.queue.Add(cache.MetaObjectToName(checkpoint).String())
}

// updateCheckpoint ignores updates once the checkpoint no longer needs
// assignment. The kubelet then writes status several times per checkpoint,
// and none of those writes concerns this controller.
func (c *Controller) updateCheckpoint(_, checkpoint *nodev1alpha1.PodCheckpoint) {
	if needsAssignment(checkpoint) {
		c.enqueueCheckpoint(checkpoint)
	}
}

// Run starts one worker. Each item needs one or two small API writes, and
// checkpoints are created explicitly, so a single worker keeps up.
func (c *Controller) Run(ctx context.Context) {
	defer utilruntime.HandleCrashWithContext(ctx)

	logger := klog.FromContext(ctx)
	logger.Info("Starting controller", "controller", c.controllerName)

	c.eventBroadcaster.StartStructuredLogging(3)
	c.eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: c.kubeClient.CoreV1().Events("")})

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down controller", "controller", c.controllerName)
		c.queue.ShutDown()
		wg.Wait()
		c.eventBroadcaster.Shutdown()
	}()

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.checkpointsSynced, c.podsSynced) {
		return
	}

	wg.Go(func() {
		wait.UntilWithContext(ctx, c.worker, time.Second)
	})

	<-ctx.Done()
}

func (c *Controller) worker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *Controller) processNextWorkItem(ctx context.Context) bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	if err := c.syncHandler(ctx, key); err != nil {
		utilruntime.HandleErrorWithLogger(klog.FromContext(ctx), err, "Failed to sync PodCheckpoint", "key", key)
		c.queue.AddRateLimited(key)
		return true
	}
	c.queue.Forget(key)
	return true
}

// syncHandler assigns a pending checkpoint. Decisions start from the informer
// caches; API reads happen only before a write that the caches cannot justify
// on their own.
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
	return c.assignCheckpoint(ctx, checkpoint)
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
