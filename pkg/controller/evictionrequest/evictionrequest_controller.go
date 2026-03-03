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
// the graceful eviction of pods by managing interceptors.
//
// See KEP-4563 for more details: https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/4563-eviction-request-api
package evictionrequest

import (
	"context"
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
	coreapply "k8s.io/client-go/applyconfigurations/core/v1"
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
	// InterceptorHeartbeatTimeout is the timeout for interceptor heartbeat.
	// If an interceptor doesn't update its heartbeat within this duration, it's considered timed out.
	InterceptorHeartbeatTimeout = 20 * time.Minute

	// GracefulCompletionDelay is the delay before setting the Evicted condition when
	// a pod is deleted or terminal while an active interceptor hasn't reported completion.
	// This gives the interceptor time to report its final status.
	GracefulCompletionDelay = 5 * time.Second
)

// EvictionRequestController is the eviction request controller implementation.
// It watches EvictionRequest objects and coordinates the graceful eviction of pods
// by managing interceptors.
//
// The controller uses two separate reconciliation loops:
// 1. Main loop (queue) - handles validation and interceptor selection
// 2. Label sync loop (labelSyncQueue) - synchronizes pod labels to EvictionRequest
//
// Responsibilities (per KEP-4563):
// 1. Validation - verify target pod exists, reject invalid requests
// 2. Interceptor management - select active interceptors, handle timeouts, advance through list
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
	// Handles validation and interceptor selection.
	queue workqueue.TypedRateLimitingInterface[string]

	// labelSyncQueue handles synchronization of pod labels to EvictionRequest.
	// This is kept separate because UpdateStatus() blocks label updates (see
	// evictionRequestStatusStrategy.GetResetFields) to prevent interceptors from
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
// Handles validation and interceptor management.
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

	// Terminal EvictionRequests (Canceled or Evicted) don't need further processing.
	// Without this check, a re-sync after setting Canceled (e.g., pod not found on
	// first sync) would fall through to computeCompletionCondition, which would
	// overwrite the Canceled condition with Evicted.
	if isTerminal(evictionRequest) {
		return nil
	}

	// Resolve target pod
	var targetPod *v1.Pod
	if evictionRequest.Spec.Target.Pod != nil {
		pod, err := c.podLister.Pods(namespace).Get(evictionRequest.Spec.Target.Pod.Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
		targetPod = pod
	} else {
		// Non-pod targets are not supported yet, so cancel the eviction request
		statusApply := coordinationapply.EvictionRequestStatus().
			WithObservedGeneration(evictionRequest.Generation).
			WithConditions(newCondition(coordinationv1alpha1.EvictionRequestConditionCanceled,
				metav1.ConditionTrue, string(coordinationv1alpha1.EvictionRequestConditionReasonValidationFailed),
				"Unsupported target type"))
		return c.applyStatus(ctx, evictionRequest, statusApply)
	}

	target := newTargetInfo(evictionRequest.Spec.Target, targetPod)
	logger.V(4).Info("Syncing EvictionRequest", "evictionRequest", klog.KObj(evictionRequest), "target", target.name())

	statusApply, resyncAfter := c.computeStatus(ctx, evictionRequest, target)

	if resyncAfter > 0 {
		c.queue.AddAfter(key, resyncAfter)
	}

	return c.applyStatus(ctx, evictionRequest, statusApply)
}

// syncLabels reconciles EvictionRequests for label synchronization based on a Pod key.
// Syncs target pod's .metadata.labels to EvictionRequest's .metadata.labels.
// This overwrites any conflicting labels in the EvictionRequest, allowing
// interceptors to use custom label selectors when watching eviction requests.
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
		logger.Error(err, "Failed to patch EvictionRequest labels", "evictionRequest", klog.KObj(evictionRequest), "pod", klog.KObj(pod))
		return err
	}

	logger.V(4).Info("Synced labels from pod to EvictionRequest", "evictionRequest", klog.KObj(evictionRequest), "pod", klog.KObj(pod))
	return nil
}

// computeStatus first computes the desired status for an EvictionRequest and then builds the status apply configuration
// It returns the status apply configuration and an optional resync duration.
func (c *EvictionRequestController) computeStatus(
	ctx context.Context,
	evictionRequest *coordinationv1alpha1.EvictionRequest,
	target targetInfo,
) (*coordinationapply.EvictionRequestStatusApplyConfiguration, time.Duration) {
	logger := klog.FromContext(ctx)

	statusApply := coordinationapply.EvictionRequestStatus().
		WithObservedGeneration(evictionRequest.Generation)

	if condition := computePreconditionFailure(evictionRequest, target); condition != nil {
		logger.V(4).Info("Setting precondition failure condition", "evictionRequest", klog.KObj(evictionRequest),
			"type", *condition.Type, "reason", *condition.Reason)
		return statusApply.WithConditions(condition), 0
	}

	targetInterceptors := computeTargetInterceptors(evictionRequest, target)

	completionResync := shouldDeferCompletion(evictionRequest, target, c.clock)
	var completionCondition *metav1ac.ConditionApplyConfiguration
	if completionResync == 0 {
		completionCondition = computeCompletionCondition(evictionRequest, target)
	}

	active, processed, progressionResync := computeInterceptorProgression(evictionRequest, targetInterceptors, completionCondition != nil, c.clock)

	// Report metrics per KEP-4563
	if len(active) > 0 {
		metrics.ActiveInterceptor.WithLabelValues(evictionRequest.Namespace, evictionRequest.Name, active[0]).Set(1)
	}
	metrics.ProcessedInterceptor.WithLabelValues(evictionRequest.Namespace, evictionRequest.Name).Set(float64(len(processed)))
	metrics.ActiveRequester.WithLabelValues(evictionRequest.Namespace, evictionRequest.Name).Set(float64(len(evictionRequest.Spec.Requesters)))
	metrics.PodInterceptors.WithLabelValues(evictionRequest.Namespace, evictionRequest.Name).Set(float64(len(targetInterceptors)))

	resyncAfter := progressionResync
	if completionResync > 0 {
		resyncAfter = completionResync
	}

	for _, ti := range targetInterceptors {
		statusApply.WithTargetInterceptors(coreapply.EvictionInterceptor().WithName(ti))
	}

	statusApply.WithActiveInterceptors(active...)
	statusApply.WithProcessedInterceptors(processed...)

	if completionCondition != nil {
		logger.V(4).Info("Setting completion condition", "evictionRequest", klog.KObj(evictionRequest),
			"type", *completionCondition.Type, "reason", *completionCondition.Reason)
		statusApply.WithConditions(completionCondition)
	} else if completionResync > 0 {
		logger.V(4).Info("Deferring completion to allow active interceptor to report",
			"evictionRequest", klog.KObj(evictionRequest), "resyncAfter", completionResync)
	}

	// Include interceptor entries for all target interceptors so SSA doesn't
	// remove them. Only set Name and StartTime — other fields (HeartbeatTime,
	// CompletionTime, Message) are owned by the interceptors via their own
	// field manager.
	for _, ti := range targetInterceptors {
		isApply := coordinationapply.InterceptorStatus().WithName(ti)

		existing := findInterceptorStatus(evictionRequest.Status.Interceptors, ti)
		if existing != nil && existing.StartTime != nil {
			isApply.WithStartTime(*existing.StartTime)
		} else if len(active) > 0 && ti == active[0] {
			isApply.WithStartTime(metav1.NewTime(c.clock.Now()))
		}

		statusApply.WithInterceptors(isApply)
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

// computePreconditionFailure checks for precondition failures that prevent processing from starting.
// Returns a Canceled condition if validation fails before we've ever observed this request.
// Returns nil if preconditions pass and we can proceed with processing.
func computePreconditionFailure(
	evictionRequest *coordinationv1alpha1.EvictionRequest,
	target targetInfo,
) *metav1ac.ConditionApplyConfiguration {
	if evictionRequest.Status.ObservedGeneration > 0 {
		return nil
	}

	if valid, message := target.isValidTarget(); !valid {
		return newCondition(coordinationv1alpha1.EvictionRequestConditionCanceled,
			metav1.ConditionTrue, string(coordinationv1alpha1.EvictionRequestConditionReasonValidationFailed), message)
	}

	return nil
}

// computeCompletionCondition checks for eviction completion after processing has started.
// Returns a condition if the eviction has reached a terminal state or nil if eviction is still in progress.
func computeCompletionCondition(evictionRequest *coordinationv1alpha1.EvictionRequest, target targetInfo) *metav1ac.ConditionApplyConfiguration {
	if target.isGone() {
		return newCondition(coordinationv1alpha1.EvictionRequestConditionEvicted,
			metav1.ConditionTrue, string(coordinationv1alpha1.EvictionRequestConditionReasonPodDeleted), "Target pod has been deleted")
	}

	if target.isTerminal() {
		return newCondition(coordinationv1alpha1.EvictionRequestConditionEvicted,
			metav1.ConditionTrue, string(coordinationv1alpha1.EvictionRequestConditionReasonPodTerminal), "Pod has reached terminal state")
	}

	if len(evictionRequest.Spec.Requesters) == 0 {
		return newCondition(coordinationv1alpha1.EvictionRequestConditionCanceled,
			metav1.ConditionTrue, string(coordinationv1alpha1.EvictionRequestConditionReasonNoRequesters), "Requesters list is empty")
	}

	return nil
}

// shouldDeferCompletion returns how long to wait before setting the completion
// condition, giving the active interceptor time to report its final status
// before the eviction is finalized. Returns 0 when no deferral is needed.
func shouldDeferCompletion(
	evictionRequest *coordinationv1alpha1.EvictionRequest,
	target targetInfo,
	clock clock.PassiveClock,
) time.Duration {
	if len(evictionRequest.Status.ActiveInterceptors) == 0 {
		return 0
	}

	activeInterceptor := evictionRequest.Status.ActiveInterceptors[0]
	interceptorStatus := findInterceptorStatus(evictionRequest.Status.Interceptors, activeInterceptor)
	if interceptorStatus != nil && interceptorStatus.CompletionTime != nil {
		return 0
	}

	deletionTimestamp := target.deletionTimestamp()
	if deletionTimestamp.IsZero() {
		return 0
	}

	if remaining := GracefulCompletionDelay - clock.Since(deletionTimestamp); remaining > 0 {
		return remaining
	}
	return 0
}

// computeTargetInterceptors computes the target interceptors list.
// Returns the existing list if already initialized, or a new list from the target's eviction interceptors.
func computeTargetInterceptors(
	evictionRequest *coordinationv1alpha1.EvictionRequest,
	target targetInfo,
) []string {
	// TargetInterceptors is immutable after first initialization
	if len(evictionRequest.Status.TargetInterceptors) > 0 {
		targets := make([]string, len(evictionRequest.Status.TargetInterceptors))
		for i, ti := range evictionRequest.Status.TargetInterceptors {
			targets[i] = ti.Name
		}
		return targets
	}

	// Target may be unavailable if it was deleted before TargetInterceptors were initialized
	// (e.g., precondition failure on first sync, then re-queued after status update).
	if target.isGone() {
		return nil
	}

	interceptors := target.evictionInterceptors()
	targets := make([]string, 0, len(interceptors)+1)
	for _, ei := range interceptors {
		targets = append(targets, ei.Name)
	}
	// Default imperative-eviction interceptor triggers actual pod eviction
	targets = append(targets, string(coordinationv1alpha1.EvictionInterceptorImperativeEviction))
	return targets
}

// computeInterceptorProgression computes the active and processed interceptors.
// When isComplete is true, moves any active interceptor to processed and clears active.
func computeInterceptorProgression(evictionRequest *coordinationv1alpha1.EvictionRequest, targetInterceptors []string, isComplete bool, clock clock.PassiveClock) (active []string, processed []string, resyncAfter time.Duration) {
	processed = slices.Clone(evictionRequest.Status.ProcessedInterceptors)

	activeInterceptor := ""
	if len(evictionRequest.Status.ActiveInterceptors) > 0 {
		activeInterceptor = evictionRequest.Status.ActiveInterceptors[0]
	}

	// Completion: move active interceptor to processed (if any) and clear active.
	if isComplete {
		if activeInterceptor != "" && !slices.Contains(processed, activeInterceptor) {
			processed = append(processed, activeInterceptor)
		}
		return nil, processed, 0
	}

	// No target interceptors: nothing to activate
	if len(targetInterceptors) == 0 {
		return nil, processed, 0
	}

	activeInterceptorStatus := findInterceptorStatus(evictionRequest.Status.Interceptors, activeInterceptor)
	shouldAdvance, resyncAfter := shouldAdvanceInterceptor(activeInterceptorStatus, clock)

	// Keep current active interceptor: not yet complete and hasn't timed out
	if !shouldAdvance {
		return []string{activeInterceptor}, processed, resyncAfter
	}

	// Advance to next interceptor: mark current as processed, then select next unprocessed
	if activeInterceptor != "" && !slices.Contains(processed, activeInterceptor) {
		processed = append(processed, activeInterceptor)
	}
	active = selectNextUnprocessed(targetInterceptors, processed)
	return active, processed, 0
}

// selectNextUnprocessed finds the first unprocessed interceptor from the target list.
func selectNextUnprocessed(targetInterceptors, processed []string) []string {
	for _, ti := range targetInterceptors {
		if !slices.Contains(processed, ti) {
			return []string{ti}
		}
	}
	return nil
}

// findInterceptorStatus finds the status for a given interceptor name.
func findInterceptorStatus(statuses []coordinationv1alpha1.InterceptorStatus, name string) *coordinationv1alpha1.InterceptorStatus {
	for i := range statuses {
		if statuses[i].Name == name {
			return &statuses[i]
		}
	}
	return nil
}

// shouldAdvanceInterceptor determines if we should advance from the current interceptor.
// Returns (shouldAdvance, resyncAfter). If not advancing, resyncAfter indicates when to check again.
func shouldAdvanceInterceptor(status *coordinationv1alpha1.InterceptorStatus, clock clock.PassiveClock) (bool, time.Duration) {
	// Advance as there is no current active interceptor
	if status == nil {
		return true, 0
	}

	// Advance as interceptor has completed
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
		if elapsed >= InterceptorHeartbeatTimeout {
			return true, 0
		}
		// Schedule resync when timeout would occur
		return false, InterceptorHeartbeatTimeout - elapsed
	}

	// Should not be reached, as an active interceptor is initialized with a StartTime
	return false, 0
}

// evictionRequestKeyForPod returns the work queue key for an EvictionRequest
// that targets the given pod. EvictionRequests are named after their target pod's UID.
func evictionRequestKeyForPod(pod *v1.Pod) string {
	return pod.Namespace + "/" + string(pod.UID)
}
