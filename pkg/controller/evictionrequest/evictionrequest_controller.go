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

// Package evictionrequest implements the eviction request controller.
// The eviction request controller watches EvictionRequest objects and coordinates
// the graceful eviction of pods by managing responders.
//
// See KEP-4563 for more details: https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/4563-eviction-request-api
package evictionrequest

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"sync"
	"time"

	coordinationv1alpha1 "k8s.io/api/coordination/v1alpha1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coordinationapply "k8s.io/client-go/applyconfigurations/coordination/v1alpha1"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
	coordinationinformers "k8s.io/client-go/informers/coordination/v1alpha1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	coordinationlisters "k8s.io/client-go/listers/coordination/v1alpha1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/evictionrequest/metrics"
	"k8s.io/utils/clock"
)

const (
	// ResponderHeartbeatTimeout is the timeout for responder heartbeat.
	// If an responder doesn't update its heartbeat within this duration, it's considered timed out.
	ResponderHeartbeatTimeout = 20 * time.Minute

	// GracefulCompletionDelay is the delay before setting the Evicted condition when
	// a pod is deleted or terminal while an active responder hasn't reported completion.
	// This gives the responder time to report its final status.
	GracefulCompletionDelay = 5 * time.Second
)

// EvictionRequestController is the eviction request controller implementation.
// It watches EvictionRequest objects and coordinates the graceful eviction of pods
// by managing responders.
//
// The controller uses two separate reconciliation loops:
// 1. Main loop (queue) - handles validation and responder selection
// 2. Label sync loop (labelSyncQueue) - synchronizes pod labels to EvictionRequest
//
// Responsibilities (per KEP-4563):
// 1. Validation - verify target pod exists, reject invalid requests
// 2. Responder management - select active responders, handle timeouts, advance through list
// 3. Label synchronization - sync pod labels to EvictionRequest
// 4. Observation of target lifecycle (Pod) and status reporting
type EvictionRequestController struct {
	controllerName string
	kubeClient     clientset.Interface

	evictionRequestLister       coordinationlisters.EvictionRequestLister
	evictionRequestListerSynced cache.InformerSynced

	podLister       corelisters.PodLister
	podListerSynced cache.InformerSynced

	// queue is the main work queue for EvictionRequest reconciliation.
	// Handles validation and responder selection.
	queue workqueue.TypedRateLimitingInterface[string]

	// labelSyncQueue handles synchronization of pod labels to EvictionRequest.
	// This is kept separate because UpdateStatus() blocks label updates (see
	// evictionRequestStatusStrategy.GetResetFields) to prevent responders from
	// mutating labels. We use Server-Side Apply in a separate queue to work around this.
	labelSyncQueue workqueue.TypedRateLimitingInterface[string]

	// syncHandler is the function called to sync an EvictionRequest.
	// It may be replaced during tests.
	syncHandler func(ctx context.Context, key string) error

	// syncLabelHandler is the function called to sync labels for an EvictionRequest.
	// It may be replaced during tests.
	syncLabelHandler func(ctx context.Context, key string) error

	// clock is used for time-based operations (e.g., checking heartbeat timeouts).
	// It may be replaced during tests with a fake clock.
	clock clock.PassiveClock
}

// NewController creates a new eviction request controller.
func NewController(
	ctx context.Context,
	evictionRequestInformer coordinationinformers.EvictionRequestInformer,
	podInformer coreinformers.PodInformer,
	kubeClient clientset.Interface,
	controllerName string,
) (*EvictionRequestController, error) {
	logger := klog.FromContext(ctx)

	metrics.Register()

	c := &EvictionRequestController{
		controllerName:              controllerName,
		kubeClient:                  kubeClient,
		evictionRequestLister:       evictionRequestInformer.Lister(),
		evictionRequestListerSynced: evictionRequestInformer.Informer().HasSynced,
		podLister:                   podInformer.Lister(),
		podListerSynced:             podInformer.Informer().HasSynced,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: controllerName,
			},
		),
		labelSyncQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: controllerName + "_labelsync",
			},
		),
		clock: clock.RealClock{},
	}

	c.syncHandler = c.sync
	c.syncLabelHandler = c.syncLabels

	// Watch EvictionRequest changes
	if _, err := evictionRequestInformer.Informer().AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			c.enqueue(logger, obj)
			// Initial label sync when EvictionRequest is created
			evictionRequest := obj.(*coordinationv1alpha1.EvictionRequest)
			c.enqueueLabelSyncForEvictionRequest(evictionRequest)
		},
		UpdateFunc: func(old, new any) { c.enqueue(logger, new) },
		DeleteFunc: func(obj any) { c.enqueue(logger, obj) },
	}, cache.HandlerOptions{Logger: &logger}); err != nil {
		return nil, err
	}

	// Watch Pod changes to trigger reconciliation for associated EvictionRequests.
	// - Label changes trigger label sync
	// - Terminal phase transitions trigger main reconciliation to detect eviction completion
	// - Deletions trigger main reconciliation to detect eviction completion
	if _, err := podInformer.Informer().AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(old, new any) {
			oldPod := old.(*v1.Pod)
			newPod := new.(*v1.Pod)
			if !maps.Equal(oldPod.Labels, newPod.Labels) {
				c.enqueueLabelSync(logger, newPod)
			}
			if !podutil.IsPodTerminal(oldPod) && podutil.IsPodTerminal(newPod) {
				c.queue.Add(evictionRequestKeyForPod(newPod))
			}
		},
		DeleteFunc: func(obj any) {
			c.deletePod(logger, obj)
		},
	}, cache.HandlerOptions{Logger: &logger}); err != nil {
		return nil, err
	}

	return c, nil
}

// Run starts the eviction request controller.
func (c *EvictionRequestController) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()

	logger := klog.FromContext(ctx)
	logger.Info("Starting controller", "controller", c.controllerName)

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down controller", "controller", c.controllerName)
		c.queue.ShutDown()
		c.labelSyncQueue.ShutDown()
		wg.Wait()
	}()

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.podListerSynced, c.evictionRequestListerSynced) {
		return
	}

	for range workers {
		wg.Go(func() {
			wait.UntilWithContext(ctx, c.runWorker, time.Second)
		})
	}

	wg.Go(func() {
		wait.UntilWithContext(ctx, c.runLabelSyncWorker, time.Second)
	})

	<-ctx.Done()
}

// enqueue adds an EvictionRequest to the main work queue.
func (c *EvictionRequestController) enqueue(logger klog.Logger, obj any) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to get key for object")
		return
	}
	c.queue.Add(key)
}

// deletePod enqueues the EvictionRequest for a deleted Pod.
func (c *EvictionRequestController) deletePod(logger klog.Logger, obj any) {
	pod, ok := obj.(*v1.Pod)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleErrorWithLogger(logger, nil, "Couldn't get object from tombstone", "object", obj)
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleErrorWithLogger(logger, nil, "Tombstone contained object that is not a Pod", "object", obj)
			return
		}
	}
	evictionRequestKey := evictionRequestKeyForPod(pod)
	c.queue.Add(evictionRequestKey)
}

// enqueueLabelSync adds a Pod to the label sync queue.
func (c *EvictionRequestController) enqueueLabelSync(logger klog.Logger, pod *v1.Pod) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(pod)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to get key for pod")
		return
	}
	c.labelSyncQueue.Add(key)
}

// enqueueLabelSyncForEvictionRequest queues the target pod of an EvictionRequest for label sync.
func (c *EvictionRequestController) enqueueLabelSyncForEvictionRequest(evictionRequest *coordinationv1alpha1.EvictionRequest) {
	if evictionRequest.Spec.Target.Pod != nil {
		podKey := evictionRequest.Namespace + "/" + evictionRequest.Spec.Target.Pod.Name
		c.labelSyncQueue.Add(podKey)
	}
}

// runWorker processes items from the main queue.
func (c *EvictionRequestController) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

// runLabelSyncWorker processes items from the label sync queue.
func (c *EvictionRequestController) runLabelSyncWorker(ctx context.Context) {
	for c.processNextLabelSyncWorkItem(ctx) {
	}
}

// processNextWorkItem processes the next item from the main queue.
func (c *EvictionRequestController) processNextWorkItem(ctx context.Context) bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	err := c.syncHandler(ctx, key)
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleErrorWithContext(ctx, err, "Failed to sync EvictionRequest", "key", key)
	c.queue.AddRateLimited(key)
	return true
}

// processNextLabelSyncWorkItem processes the next item from the label sync queue.
func (c *EvictionRequestController) processNextLabelSyncWorkItem(ctx context.Context) bool {
	key, quit := c.labelSyncQueue.Get()
	if quit {
		return false
	}
	defer c.labelSyncQueue.Done(key)

	err := c.syncLabelHandler(ctx, key)
	if err == nil {
		c.labelSyncQueue.Forget(key)
		return true
	}

	utilruntime.HandleErrorWithContext(ctx, err, "Failed to sync EvictionRequest labels", "key", key)
	c.labelSyncQueue.AddRateLimited(key)
	return true
}

// sync reconciles an EvictionRequest for the main loop.
// Handles validation and responder management.
func (c *EvictionRequestController) sync(ctx context.Context, key string) error {
	logger := klog.FromContext(ctx)
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	evictionRequest, err := c.evictionRequestLister.EvictionRequests(namespace).Get(name)
	if errors.IsNotFound(err) {
		logger.V(4).Info("EvictionRequest not found, skipping", "evictionRequest", klog.KRef(namespace, name))
		return nil
	}
	if err != nil {
		return err
	}
	logger.V(4).Info("Syncing EvictionRequest", "evictionRequest", klog.KObj(evictionRequest))

	// Terminal EvictionRequests (Canceled or Evicted) don't need further processing.
	// Without this check, a re-sync after setting Canceled (e.g., pod not found on
	// first sync) would fall through to computeCompletionCondition, which would
	// overwrite the Canceled condition with Evicted.
	if hasCompleted(evictionRequest) {
		return nil
	}

	// Resolve target pod
	var targetPod *v1.Pod
	switch {
	case evictionRequest.Spec.Target.Pod != nil:
		pod, err := c.podLister.Pods(namespace).Get(evictionRequest.Spec.Target.Pod.Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
		targetPod = pod
	}

	target := newTargetInfo(evictionRequest.Spec.Target, targetPod)
	logger.V(4).Info("Target info", "evictionRequest", klog.KObj(evictionRequest), "targetName", target.targetName(), "desiredTargetUID", target.targetUID())

	if evictionRequest.Status.ObservedGeneration == nil {
		failed, evicted := validate(c.clock, evictionRequest, target)
		if failed != nil || evicted != nil {
			return c.applyStatus(ctx, evictionRequest,
				coordinationapply.EvictionRequestStatus().
					WithObservedGeneration(evictionRequest.Generation).
					WithConditions(failed, evicted),
			)
		}
	}

	statusApply, resyncAfter := c.computeStatus(ctx, evictionRequest, target)

	if resyncAfter > 0 {
		c.queue.AddAfter(key, resyncAfter)
	}

	return c.applyStatus(ctx, evictionRequest, statusApply)
}

// syncLabels reconciles EvictionRequests for label synchronization based on a Pod key.
// Syncs target pod's .metadata.labels to EvictionRequest's .metadata.labels.
// This overwrites any conflicting labels in the EvictionRequest, allowing
// responders to use custom label selectors when watching eviction requests.
func (c *EvictionRequestController) syncLabels(ctx context.Context, key string) error {
	logger := klog.FromContext(ctx)
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	pod, err := c.podLister.Pods(namespace).Get(name)
	if errors.IsNotFound(err) {
		logger.V(4).Info("Pod not found, skipping label sync", "pod", klog.KRef(namespace, name)) // TODO(@johankj): Remove this verbose log
		return nil
	}
	if err != nil {
		return err
	}

	if len(pod.Labels) == 0 {
		return nil
	}

	// EvictionRequest name must match pod UID
	evictionRequest, err := c.evictionRequestLister.EvictionRequests(namespace).Get(string(pod.UID))
	if errors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}

	logger.V(4).Info("Syncing EvictionRequest labels", "evictionRequest", klog.KObj(evictionRequest), "pod", klog.KObj(pod))

	// Pod labels take precedence over EvictionRequest labels
	desired := maps.Clone(evictionRequest.Labels)
	if desired == nil {
		desired = make(map[string]string, len(pod.Labels))
	}
	maps.Copy(desired, pod.Labels)

	if maps.Equal(desired, evictionRequest.Labels) {
		return nil
	}

	applyConfig := coordinationapply.EvictionRequest(evictionRequest.Name, namespace).WithLabels(desired)

	_, err = c.kubeClient.CoordinationV1alpha1().
		EvictionRequests(namespace).
		Apply(ctx, applyConfig, metav1.ApplyOptions{FieldManager: c.controllerName, Force: true})
	if err != nil {
		logger.Error(err, "Failed to update EvictionRequest labels", "evictionRequest", klog.KObj(evictionRequest), "pod", klog.KObj(pod))
		return err
	}

	logger.V(4).Info("Synced labels from pod to EvictionRequest", "evictionRequest", klog.KObj(evictionRequest), "pod", klog.KObj(pod))
	return nil
}

// computeStatus computes the desired status for an EvictionRequest and builds the status apply configuration.
// It returns the status apply configuration and an optional resync duration.
func (c *EvictionRequestController) computeStatus(
	ctx context.Context,
	evictionRequest *coordinationv1alpha1.EvictionRequest,
	target targetInfo,
) (*coordinationapply.EvictionRequestStatusApplyConfiguration, time.Duration) {
	logger := klog.FromContext(ctx)

	statusApply := coordinationapply.EvictionRequestStatus().
		WithObservedGeneration(evictionRequest.Generation)

	completionResync := shouldDeferCompletion(evictionRequest, target, c.clock)
	failed, evicted := computeConditions(c.clock, evictionRequest, target, completionResync > 0)
	statusApply.WithConditions(failed, evicted)

	isTerminal := *failed.Status == metav1.ConditionTrue || *evicted.Status == metav1.ConditionTrue
	if isTerminal {
		logger.V(4).Info("Terminal condition reached", "evictionRequest", klog.KObj(evictionRequest))
	}

	targetResponders := computeTargetResponders(evictionRequest, target)
	active, processed, progressionResync := computeResponderProgression(evictionRequest, targetResponders, isTerminal, c.clock)

	// Report metrics per KEP-4563
	targetEntityType := target.targetType()
	if len(active) > 0 {
		metrics.ActiveResponder.WithLabelValues(evictionRequest.Namespace, evictionRequest.Name, targetEntityType.String(), active[0]).Set(1)
	}
	for _, p := range processed {
		metrics.ProcessedResponder.WithLabelValues(evictionRequest.Namespace, evictionRequest.Name, targetEntityType.String(), p).Set(1)
	}
	for _, r := range evictionRequest.Spec.Requesters {
		metrics.ActiveRequester.WithLabelValues(evictionRequest.Namespace, evictionRequest.Name, targetEntityType.String(), r.Name).Set(1)
	}
	metrics.PodResponders.WithLabelValues(evictionRequest.Namespace, evictionRequest.Name, targetEntityType.String()).Set(float64(len(targetResponders)))

	resyncAfter := progressionResync
	if completionResync > 0 {
		resyncAfter = completionResync
	}

	for _, ti := range targetResponders {
		statusApply.WithTargetResponders(coordinationapply.TargetResponder().WithName(ti))
	}

	statusApply.WithActiveResponders(active...)
	statusApply.WithProcessedResponders(processed...)

	// Include responder entries for all target responders so SSA doesn't
	// remove them. Only set Name and StartTime — other fields (HeartbeatTime,
	// CompletionTime, Message) are owned by the responders via their own
	// field manager.
	for _, ti := range targetResponders {
		isApply := coordinationapply.ResponderStatus().WithName(ti)

		existing := findResponderStatus(evictionRequest.Status.Responders, ti)
		if existing != nil && existing.StartTime != nil {
			isApply.WithStartTime(*existing.StartTime)
		} else if len(active) > 0 && ti == active[0] {
			isApply.WithStartTime(metav1.NewTime(c.clock.Now()))
		}

		statusApply.WithResponders(isApply)
	}

	return statusApply, resyncAfter
}

// applyStatus applies the status to the EvictionRequest using Server-Side Apply.
func (c *EvictionRequestController) applyStatus(
	ctx context.Context,
	evictionRequest *coordinationv1alpha1.EvictionRequest,
	statusApply *coordinationapply.EvictionRequestStatusApplyConfiguration,
) error {
	applyConfig := coordinationapply.EvictionRequest(evictionRequest.Name, evictionRequest.Namespace).
		WithStatus(statusApply)

	_, err := c.kubeClient.CoordinationV1alpha1().
		EvictionRequests(evictionRequest.Namespace).
		ApplyStatus(ctx, applyConfig, metav1.ApplyOptions{
			FieldManager: c.controllerName,
			Force:        true,
		})
	return err
}

// validate returns the Canceled and Evicted conditions if the validation fails.
func validate(
	clock clock.PassiveClock,
	evictionRequest *coordinationv1alpha1.EvictionRequest,
	target targetInfo,
) (failed, evicted *metav1ac.ConditionApplyConfiguration) {
	failureMsg := ""
	switch {
	case target.targetType() == noTarget:
		// Non-pod targets are not supported, so cancel the eviction request.
		// This is covered by API validation. We will end up here only if the apiserver serves a new type which the
		// controller does not recognize.
		failureMsg = "Unsupported target type"
	case !target.exists():
		failureMsg = fmt.Sprintf("Target %s not found", target.targetType())
	case target.targetUID() != target.GetObjectMeta().GetUID():
		failureMsg = fmt.Sprintf("Target %s UID mismatch: expected %s, got %s", target.targetType(), target.targetUID(), target.GetObjectMeta().GetUID())
	case target.hasSchedulingGroup():
		failureMsg = fmt.Sprintf("Target %s references a SchedulingGroup. Eviction is currently not supported.", target.targetType())
	}

	if len(failureMsg) > 0 {
		failed = setCondition(clock, evictionRequest.Status.Conditions, coordinationv1alpha1.EvictionRequestConditionFailed,
			metav1.ConditionTrue, coordinationv1alpha1.EvictionRequestConditionReasonEvictionRequestInvalid, failureMsg)
		evicted = setCondition(clock, evictionRequest.Status.Conditions, coordinationv1alpha1.EvictionRequestConditionEvicted,
			metav1.ConditionFalse, "UnableToPerform", "")
		return failed, evicted
	}
	return nil, nil
}

// computeConditions returns the full pair of Canceled and Evicted conditions.
// Both are always set — defaulting to False until flipped to True.
// Precondition checks run only on first sync (ObservedGeneration == nil).
// Completion checks are skipped when deferCompletion is true.
func computeConditions(
	clock clock.PassiveClock,
	evictionRequest *coordinationv1alpha1.EvictionRequest,
	target targetInfo,
	deferCompletion bool,
) (failed, evicted *metav1ac.ConditionApplyConfiguration) {
	existing := evictionRequest.Status.Conditions
	failed = setCondition(clock, existing, coordinationv1alpha1.EvictionRequestConditionFailed,
		metav1.ConditionFalse, "Pending", "")
	evicted = setCondition(clock, existing, coordinationv1alpha1.EvictionRequestConditionEvicted,
		metav1.ConditionFalse, "Pending", "")

	if deferCompletion {
		return failed, evicted
	}

	// Completion checks
	if target.isGone() {
		evicted = setCondition(clock, existing, coordinationv1alpha1.EvictionRequestConditionEvicted,
			metav1.ConditionTrue, coordinationv1alpha1.EvictionRequestConditionReasonPodDeleted, "Target pod has been deleted")
	} else if target.isTerminal() {
		evicted = setCondition(clock, existing, coordinationv1alpha1.EvictionRequestConditionEvicted,
			metav1.ConditionTrue, coordinationv1alpha1.EvictionRequestConditionReasonPodTerminal, "Pod has reached terminal state")
	} else if len(evictionRequest.Spec.Requesters) == 0 {
		failed = setCondition(clock, existing, coordinationv1alpha1.EvictionRequestConditionFailed,
			metav1.ConditionTrue, coordinationv1alpha1.EvictionRequestConditionReasonCanceledDueToNoRequesters, "Requesters list is empty")
	}

	return failed, evicted
}

// shouldDeferCompletion returns how long to wait before setting the completion
// condition, giving the active responder time to report its final status
// before the eviction is finalized. Returns 0 when no deferral is needed.
func shouldDeferCompletion(
	evictionRequest *coordinationv1alpha1.EvictionRequest,
	target targetInfo,
	clock clock.PassiveClock,
) time.Duration {
	if len(evictionRequest.Status.ActiveResponders) == 0 {
		return 0
	}

	activeResponder := evictionRequest.Status.ActiveResponders[0]
	responderStatus := findResponderStatus(evictionRequest.Status.Responders, activeResponder)
	if responderStatus != nil && responderStatus.CompletionTime != nil {
		return 0
	}

	meta := target.GetObjectMeta()
	if meta == nil || meta.GetDeletionTimestamp() == nil {
		return 0
	}

	if remaining := GracefulCompletionDelay - clock.Since(meta.GetDeletionTimestamp().Time); remaining > 0 {
		return remaining
	}
	return 0
}

// computeTargetResponders computes the target responders list.
// Returns the existing list if already initialized, or a new list from the target's eviction responders.
func computeTargetResponders(
	evictionRequest *coordinationv1alpha1.EvictionRequest,
	target targetInfo,
) []string {
	// TargetResponders is immutable after first initialization
	if len(evictionRequest.Status.TargetResponders) > 0 {
		targets := make([]string, len(evictionRequest.Status.TargetResponders))
		for i, ti := range evictionRequest.Status.TargetResponders {
			targets[i] = ti.Name
		}
		return targets
	}

	// Target may be unavailable if it was deleted before TargetResponders were initialized
	// (e.g., precondition failure on first sync, then re-queued after status update).
	if target.isGone() {
		return nil
	}

	responders := target.evictionResponders()
	targets := make([]string, 0, len(responders)+1)
	for _, ei := range responders {
		targets = append(targets, ei.Name)
	}
	// Default imperative-eviction responder triggers actual pod eviction
	targets = append(targets, string(coordinationv1alpha1.EvictionResponderImperativeEviction))
	return targets
}

// computeResponderProgression computes the active and processed responders.
// When isComplete is true, moves any active responder to processed and clears active.
func computeResponderProgression(evictionRequest *coordinationv1alpha1.EvictionRequest, targetResponders []string, isComplete bool, clock clock.PassiveClock) (active []string, processed []string, resyncAfter time.Duration) {
	processed = slices.Clone(evictionRequest.Status.ProcessedResponders)

	activeResponder := ""
	if len(evictionRequest.Status.ActiveResponders) > 0 {
		activeResponder = evictionRequest.Status.ActiveResponders[0]
	}

	// Completion: move active responder to processed (if any) and clear active.
	if isComplete {
		if activeResponder != "" && !slices.Contains(processed, activeResponder) {
			processed = append(processed, activeResponder)
		}
		return nil, processed, 0
	}

	// No target responders: nothing to activate
	if len(targetResponders) == 0 {
		return nil, processed, 0
	}

	activeResponderStatus := findResponderStatus(evictionRequest.Status.Responders, activeResponder)
	shouldAdvance, resyncAfter := shouldAdvanceResponder(activeResponderStatus, clock)

	// Keep current active responder: not yet complete and hasn't timed out
	if !shouldAdvance {
		return []string{activeResponder}, processed, resyncAfter
	}

	// Advance to next responder: mark current as processed, then select next unprocessed
	if activeResponder != "" && !slices.Contains(processed, activeResponder) {
		processed = append(processed, activeResponder)
	}
	active = selectNextUnprocessed(targetResponders, processed)
	return active, processed, 0
}

// selectNextUnprocessed finds the first unprocessed responder from the target list.
func selectNextUnprocessed(targetResponders, processed []string) []string {
	for _, ti := range targetResponders {
		if !slices.Contains(processed, ti) {
			return []string{ti}
		}
	}
	return nil
}

// findResponderStatus finds the status for a given responder name.
func findResponderStatus(statuses []coordinationv1alpha1.ResponderStatus, name string) *coordinationv1alpha1.ResponderStatus {
	for i := range statuses {
		if statuses[i].Name == name {
			return &statuses[i]
		}
	}
	return nil
}

// shouldAdvanceResponder determines if we should advance from the current responder.
// Returns (shouldAdvance, resyncAfter). If not advancing, resyncAfter indicates when to check again.
func shouldAdvanceResponder(status *coordinationv1alpha1.ResponderStatus, clock clock.PassiveClock) (bool, time.Duration) {
	// Advance as there is no current active responder
	if status == nil {
		return true, 0
	}

	// Advance as responder has completed
	if status.CompletionTime != nil {
		return true, 0
	}

	t := status.StartTime
	if status.HeartbeatTime != nil {
		t = status.HeartbeatTime
	}

	if t != nil {
		elapsed := clock.Since(t.Time)
		// Advance as heartbeat timeout has been reached
		if elapsed >= ResponderHeartbeatTimeout {
			return true, 0
		}
		// Schedule resync when timeout would occur
		return false, ResponderHeartbeatTimeout - elapsed
	}

	// Should not be reached, as an active responder is initialized with a StartTime
	return false, 0
}

// evictionRequestKeyForPod returns the work queue key for an EvictionRequest
// that targets the given pod. EvictionRequests are named after their target pod's UID.
func evictionRequestKeyForPod(pod *v1.Pod) string {
	return pod.Namespace + "/" + string(pod.UID)
}
