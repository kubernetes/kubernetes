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

package evictionrequest

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apimachinerylabels "k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	lifecycleapply "k8s.io/client-go/applyconfigurations/lifecycle/v1alpha1"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	lifecycleinformers "k8s.io/client-go/informers/lifecycle/v1alpha1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	lifecyclelisters "k8s.io/client-go/listers/lifecycle/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/utils/clock"
	"k8s.io/utils/ptr"
)

const (
	// ResponderHeartbeatTimeout is the timeout for responder heartbeat.
	// If a responder doesn't update its heartbeat within this duration, it's considered timed out.
	ResponderHeartbeatTimeout = 20 * time.Minute

	// GracefulCompletionDelay is the delay before setting the Evicted condition when
	// a pod is deleted or terminal while an active responder hasn't reported completion.
	// This gives the responder time to report its final status.
	GracefulCompletionDelay = 5 * time.Second

	// specified in the API
	maxEvictionStatusRequesters = 100

	// We need to know which evictions we have to sync for an EvictionRequest and a pod.
	evictionByTargetUIDIndexKey = "evictionByTargetUID"
	// We need to know which evictionRequests we have to sync for an Eviction.
	evictionRequestByTargetUID = "evictionRequestByTargetUID"
)

// EvictionRequestController is the eviction request controller implementation.
// It watches EvictionRequest, creates Eviction objects and coordinates the graceful
// eviction of pods by managing responders.
//
// The controller uses two separate reconciliation loops:
// 1. Eviction loop (evictionQueue)
//   - Performs partial validation (invariants).
//   - Looks up requesters and handles cancellations.
//   - Selects and coordinates responders. Updates status and conditions
//
// 2. EvictionRequest loop (evictionRequestQueue)
//   - Performs early validation.
//   - Creates Evictions.
//   - Synchronize Eviction conditions back to EvictionRequests.
//
// 2. Refresh Eviction loop (evictionRefreshQueue)
//   - Computes responder and requester labels and adds them to Evictions.
//   - Adds EvictionRequest owners references to Evictions.
//
// Responsibilities (per KEP-4563):
// 1. Validation - verify that target pod exists, reject invalid requests.
// 2. Registration - ensure an active Eviction if there is at least one EvictionRequest with an Eviction intent.
// 3. Cancellation and GC - respond to cancellations according to EvictionRequests intents. Set owner references to Evictions.
// 4. Responder management - select active responders, handle timeouts, advance through the list.
// 5. Label generation - set "responder", "requester" and "requesterresponder" label values on Evictions.
// 6. Condition synchronization - synchronize relevant Eviction conditions to EvictionRequests.
// 7. Observation of target lifecycle (Pod) and status reporting.
type EvictionRequestController struct {
	controllerName string
	kubeClient     clientset.Interface

	evictionLister       lifecyclelisters.EvictionLister
	evictionListerSynced cache.InformerSynced

	evictionRequestLister       lifecyclelisters.EvictionRequestLister
	evictionRequestListerSynced cache.InformerSynced

	podLister       corelisters.PodLister
	podListerSynced cache.InformerSynced

	// evictionIndexer allows looking up evictions by target UID
	evictionIndexer cache.Indexer

	// evictionRequestIndexer allows looking up eviction requests by target UID
	evictionRequestIndexer cache.Indexer

	// evictionQueue is the work queue for Eviction reconciliation.
	// Handles responder selection.
	evictionQueue workqueue.TypedRateLimitingInterface[string]

	// evictionRequestQueue is the work queue for EvictionRequest reconciliation.
	// Handles Eviction creation.
	evictionRequestQueue workqueue.TypedRateLimitingInterface[string]

	// evictionMetaRefreshQueue generates responder and requester labels, as well as owner references,
	// and adds them to Evictions.
	// 1. This is kept separate because UpdateStatus() blocks metadata updates (see
	//    evictionStatusStrategy.GetResetFields). We use Server-Side Apply in a separate
	//    evictionQueue to work around this.
	// 2. Not every evictionQueue sync requires an evictionMetaRefreshQueue sync.
	evictionMetaRefreshQueue workqueue.TypedRateLimitingInterface[string]

	// syncEvictionHandler is the function called to sync an Eviction.
	// It may be replaced during tests.
	syncEvictionHandler func(ctx context.Context, key string) error

	// syncEvictionRequestHandler is the function called to sync an EvictionRequest.
	// It may be replaced during tests.
	syncEvictionRequestHandler func(ctx context.Context, key string) error

	// syncEvictionMetaRefreshHandler is the function called to refresh an Eviction metadata.
	// It may be replaced during tests.
	syncEvictionMetaRefreshHandler func(ctx context.Context, key string) error

	// clock is used for time-based operations (e.g., checking heartbeat timeouts).
	// It may be replaced during tests with a fake clock.
	clock clock.PassiveClock
}

// NewController creates a new eviction request controller.
func NewController(
	ctx context.Context,
	controllerName string,
	evictionInformer lifecycleinformers.TypedEvictionInformer,
	evictionRequestInformer lifecycleinformers.TypedEvictionRequestInformer,
	podInformer coreinformers.TypedPodInformer,
	kubeClient clientset.Interface,
) (*EvictionRequestController, error) {
	logger := klog.FromContext(ctx)

	c := &EvictionRequestController{
		controllerName:              controllerName,
		kubeClient:                  kubeClient,
		evictionLister:              evictionInformer.Lister(),
		evictionListerSynced:        evictionInformer.Informer().HasSynced,
		evictionRequestLister:       evictionRequestInformer.Lister(),
		evictionRequestListerSynced: evictionRequestInformer.Informer().HasSynced,
		podLister:                   podInformer.Lister(),
		podListerSynced:             podInformer.Informer().HasSynced,
		evictionQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: controllerName + "_eviction",
			},
		),
		evictionRequestQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: controllerName + "_eviction_request",
			},
		),
		evictionMetaRefreshQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: controllerName + "_eviction_metadata_refresh",
			},
		),
		clock: clock.RealClock{},
	}

	c.syncEvictionHandler = c.syncEviction
	c.syncEvictionRequestHandler = c.syncEvictionRequest
	c.syncEvictionMetaRefreshHandler = c.syncEvictionMetaRefresh

	// Watch Eviction changes
	// - Reconcile Eviction status
	// - Reconcile Eviction labels and ownerReferences
	// - Trigger EvictionRequest back updates
	if _, err := evictionInformer.TypedInformer().AddTypedEventHandler(lifecycleinformers.EvictionHandlerFuncs{
		AddFunc: func(eviction *lifecyclev1alpha1.Eviction) {
			c.enqueueEviction(logger, eviction)
			c.enqueueEvictionRefresh(logger, eviction)
			// No need for EvictionRequest sync: the Eviction has been just created by it
		},
		UpdateFunc: func(oldEviction, newEviction *lifecyclev1alpha1.Eviction) {
			c.enqueueEviction(logger, newEviction)
			// Reconcile labels and ownerReferences
			if !reflect.DeepEqual(oldEviction.OwnerReferences, newEviction.OwnerReferences) ||
				!reflect.DeepEqual(oldEviction.Labels, newEviction.Labels) {
				c.enqueueEvictionRefresh(logger, newEviction)
			}
			// Sync conditions to EvictionRequests and ensure there is an active Eviction
			if !reflect.DeepEqual(
				meta.FindStatusCondition(oldEviction.Status.Conditions, string(lifecyclev1alpha1.EvictionConditionTargetEvicted)),
				meta.FindStatusCondition(newEviction.Status.Conditions, string(lifecyclev1alpha1.EvictionConditionTargetEvicted))) ||
				!reflect.DeepEqual(
					meta.FindStatusCondition(oldEviction.Status.Conditions, string(lifecyclev1alpha1.EvictionConditionFailed)),
					meta.FindStatusCondition(newEviction.Status.Conditions, string(lifecyclev1alpha1.EvictionConditionFailed))) {
				c.enqueueEvictionRequestsForEviction(logger, newEviction)
			}
		},
		DeleteFunc: func(deletedEviction lifecycleinformers.DeletedEviction) {
			// Ensure there is an active Eviction on deletion for existing EvictionRequests.
			// This sync should usually be not needed, since we do not delete evictions. Deletions should be done by
			// GC and in that case there is no EvictionRequest to sync. This gets used only when a 3rd party
			// manually deletes Evictions.
			if deletedEviction.OptionalObj != nil {
				c.enqueueEvictionRequestsForEviction(logger, deletedEviction.OptionalObj)
			}

		},
	}, cache.HandlerOptions{Logger: &logger}); err != nil {
		return nil, err
	}

	// Watch EvictionRequest changes
	// - Create and Cancel Evictions
	// - Synchronize Eviction conditions
	// - Update Eviction status.requesters
	if _, err := evictionRequestInformer.TypedInformer().AddTypedEventHandler(lifecycleinformers.EvictionRequestHandlerFuncs{
		AddFunc: func(evictionRequest *lifecyclev1alpha1.EvictionRequest) {
			c.enqueueEvictionRequest(logger, evictionRequest)
			// update requesters
			c.enqueueEvictionForEvictionRequest(logger, evictionRequest)
			// add requester label and owner reference, refresh is not needed during update since these values are immutable
			c.enqueueEvictionRefreshesForEvictionRequest(logger, evictionRequest)
		},
		UpdateFunc: func(oldEvictionRequest, newEvictionRequest *lifecyclev1alpha1.EvictionRequest) {
			c.enqueueEvictionRequest(logger, newEvictionRequest)
			// update requesters or cancel eviction
			if oldEvictionRequest.Spec.Intent != newEvictionRequest.Spec.Intent ||
				(oldEvictionRequest.DeletionTimestamp == nil && newEvictionRequest.DeletionTimestamp != nil) {
				c.enqueueEvictionForEvictionRequest(logger, newEvictionRequest)
			}
		},
		DeleteFunc: func(deletedEvictionRequest lifecycleinformers.DeletedEvictionRequest) {
			// refresh eviction requesters (intents) and labels
			if deletedEvictionRequest.OptionalObj != nil {
				c.enqueueEvictionForEvictionRequest(logger, deletedEvictionRequest.OptionalObj)
				c.enqueueEvictionRefreshesForEvictionRequest(logger, deletedEvictionRequest.OptionalObj)
			}
		},
	}, cache.HandlerOptions{Logger: &logger}); err != nil {
		return nil, err
	}

	// Watch Pod changes to trigger reconciliation for associated Evictions.
	// - Terminal phase transitions trigger main reconciliation to detect eviction completion
	// - Deletions trigger main reconciliation to detect eviction completion
	if _, err := podInformer.TypedInformer().AddTypedEventHandler(coreinformers.PodHandlerFuncs{
		UpdateFunc: func(oldPod, newPod *v1.Pod) {
			if !oldPod.DeletionTimestamp.Equal(newPod.DeletionTimestamp) ||
				(oldPod.Status.Phase != newPod.Status.Phase && podutil.IsPodTerminal(newPod)) {
				evictions, err := c.listEvictionsForTarget(targetIndexValueForPod(newPod))
				if err != nil {
					utilruntime.HandleError(fmt.Errorf("failed to list eviction for pod %s/%s: %w", newPod.Namespace, newPod.Name, err))
				}
				for _, eviction := range evictions {
					c.enqueueEviction(logger, eviction)
				}
			}
		},
		DeleteFunc: func(deletedPod coreinformers.DeletedPod) {
			if deletedPod.OptionalObj != nil {
				c.deletePod(logger, deletedPod.OptionalObj)
			}
		},
	}, cache.HandlerOptions{Logger: &logger}); err != nil {
		return nil, err
	}

	if err := evictionInformer.Informer().AddIndexers(cache.Indexers{
		evictionByTargetUIDIndexKey: func(obj interface{}) ([]string, error) {
			eviction, ok := obj.(*lifecyclev1alpha1.Eviction)
			if !ok {
				return nil, fmt.Errorf("unexpected object type %T", obj)
			}
			if indexValue := targetIndexValueForEviction(eviction); len(indexValue) > 0 {
				return []string{indexValue}, nil
			}
			return nil, nil
		},
	}); err != nil {
		return nil, fmt.Errorf("adding eviction indexer: %w", err)
	}
	c.evictionIndexer = evictionInformer.Informer().GetIndexer()

	if err := evictionRequestInformer.Informer().AddIndexers(cache.Indexers{
		evictionRequestByTargetUID: func(obj interface{}) ([]string, error) {
			evictionRequest, ok := obj.(*lifecyclev1alpha1.EvictionRequest)
			if !ok {
				return nil, fmt.Errorf("unexpected object type %T", obj)
			}
			if indexValue := targetIndexValueForEvictionRequest(evictionRequest); len(indexValue) > 0 {
				return []string{indexValue}, nil
			}
			return nil, nil
		},
	}); err != nil {
		return nil, fmt.Errorf("adding eviction request indexer: %w", err)
	}
	c.evictionRequestIndexer = evictionRequestInformer.Informer().GetIndexer()

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
		c.evictionQueue.ShutDown()
		c.evictionRequestQueue.ShutDown()
		c.evictionMetaRefreshQueue.ShutDown()
		wg.Wait()
	}()

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.podListerSynced, c.evictionListerSynced, c.evictionRequestListerSynced) {
		return
	}

	for range workers {
		wg.Go(func() {
			wait.UntilWithContext(ctx, c.runEvictionRequestWorker, time.Second)
		})
		wg.Go(func() {
			wait.UntilWithContext(ctx, c.runEvictionWorker, time.Second)
		})
	}

	wg.Go(func() {
		wait.UntilWithContext(ctx, c.runEvictionRefreshWorker, time.Second)
	})

	<-ctx.Done()
}

// enqueueEviction adds an Eviction to the evictionQueue.
func (c *EvictionRequestController) enqueueEviction(logger klog.Logger, eviction *lifecyclev1alpha1.Eviction) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(eviction)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to get key for object")
		return
	}
	c.evictionQueue.Add(key)
}

// enqueueEvictionForEvictionRequest adds all Evictions matching evictionRequest to the evictionQueue.
func (c *EvictionRequestController) enqueueEvictionForEvictionRequest(logger klog.Logger, evictionRequest *lifecyclev1alpha1.EvictionRequest) {
	evictions, err := c.listEvictionsForTarget(targetIndexValueForEvictionRequest(evictionRequest))
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "failed to list evictions for eviction request", "evictionRequest", klog.KObj(evictionRequest))
	}
	// Evictions should react to intents.
	for _, eviction := range evictions {
		c.enqueueEviction(logger, eviction)
	}
}

// enqueueEvictionRequest adds an EvictionRequest to the evictionRequestQueue.
func (c *EvictionRequestController) enqueueEvictionRequest(logger klog.Logger, evictionRequest *lifecyclev1alpha1.EvictionRequest) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(evictionRequest)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to get key for object")
		return
	}
	c.evictionRequestQueue.Add(key)
}

// enqueueEvictionRequestsForEviction adds all EvictionRequests matching eviction to the evictionRequestQueue.
func (c *EvictionRequestController) enqueueEvictionRequestsForEviction(logger klog.Logger, eviction *lifecyclev1alpha1.Eviction) {
	requests, err := c.listEvictionRequestsForTarget(targetIndexValueForEviction(eviction))
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "failed to list eviction requests for eviction", "eviction", klog.KObj(eviction))
	}
	// Sync conditions to EvictionRequests and ensure there is an active Eviction.
	for _, request := range requests {
		c.enqueueEvictionRequest(logger, request)
	}
}

// enqueueEvictionRefresh adds an Eviction to the evictionMetaRefreshQueue.
func (c *EvictionRequestController) enqueueEvictionRefresh(logger klog.Logger, eviction *lifecyclev1alpha1.Eviction) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(eviction)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to get key for object")
		return
	}
	// refresh metadata
	c.evictionMetaRefreshQueue.Add(key)
}

// enqueueEvictionRefreshesForEvictionRequest adds all Evictions matching evictionRequest to the evictionMetaRefreshQueue.
func (c *EvictionRequestController) enqueueEvictionRefreshesForEvictionRequest(logger klog.Logger, evictionRequest *lifecyclev1alpha1.EvictionRequest) {
	evictions, err := c.listEvictionsForTarget(targetIndexValueForEvictionRequest(evictionRequest))
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "failed to list evictions for eviction request", "evictionRequest", klog.KObj(evictionRequest))
	}
	// Update labels and ownerRefs for matching evictions.
	for _, eviction := range evictions {
		c.enqueueEvictionRefresh(logger, eviction)
	}
}

// deletePod adds all Evictions matching deleted pod to the evictionQueue with a short delay.
func (c *EvictionRequestController) deletePod(logger klog.Logger, pod *v1.Pod) {
	evictions, err := c.listEvictionsForTarget(targetIndexValueForPod(pod))
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "failed to list evictions for pod", "pod", klog.KObj(pod))
	}
	// Find and sync the active or unsynced evictions.
	for _, eviction := range evictions {
		if hasEvictionCompleted(eviction) {
			continue
		}
		key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(eviction)
		if err != nil {
			utilruntime.HandleErrorWithLogger(logger, err, "Failed to get key for eviction")
			return
		}
		c.evictionQueue.AddAfter(key, GracefulCompletionDelay)
	}
}

// runEvictionWorker processes items from the evictionQueue.
func (c *EvictionRequestController) runEvictionWorker(ctx context.Context) {
	for c.processNextEvictionWorkItem(ctx) {
	}
}

// runEvictionRefreshWorker processes items from the evictionMetaRefreshQueue.
func (c *EvictionRequestController) runEvictionRefreshWorker(ctx context.Context) {
	for c.processEvictionRefreshWorkItem(ctx) {
	}
}

// runEvictionProcessingWorker processes items from the evictionRequestQueue.
func (c *EvictionRequestController) runEvictionRequestWorker(ctx context.Context) {
	for c.processNextEvictionRequestWorkItem(ctx) {
	}
}

// processNextEvictionWorkItem processes the next item from the evictionQueue.
func (c *EvictionRequestController) processNextEvictionWorkItem(ctx context.Context) bool {
	key, quit := c.evictionQueue.Get()
	if quit {
		return false
	}
	defer c.evictionQueue.Done(key)

	err := c.syncEvictionHandler(ctx, key)
	if err == nil {
		c.evictionQueue.Forget(key)
		return true
	}

	utilruntime.HandleErrorWithContext(ctx, err, "Failed to sync Eviction", "key", key)
	c.evictionQueue.AddRateLimited(key)
	return true
}

// processEvictionRefreshWorkItem processes the next item from the evictionMetaRefreshQueue.
func (c *EvictionRequestController) processEvictionRefreshWorkItem(ctx context.Context) bool {
	key, quit := c.evictionMetaRefreshQueue.Get()
	if quit {
		return false
	}
	defer c.evictionMetaRefreshQueue.Done(key)

	err := c.syncEvictionMetaRefreshHandler(ctx, key)
	if err == nil {
		c.evictionMetaRefreshQueue.Forget(key)
		return true
	}

	utilruntime.HandleErrorWithContext(ctx, err, "Failed to sync Eviction metadata refresh", "key", key)
	c.evictionMetaRefreshQueue.AddRateLimited(key)
	return true
}

// processEvictionRefreshWorkItem processes the next item from the evictionRequestQueue.
func (c *EvictionRequestController) processNextEvictionRequestWorkItem(ctx context.Context) bool {
	key, quit := c.evictionRequestQueue.Get()
	if quit {
		return false
	}
	defer c.evictionRequestQueue.Done(key)

	err := c.syncEvictionRequestHandler(ctx, key)
	if err == nil {
		c.evictionRequestQueue.Forget(key)
		return true
	}

	utilruntime.HandleErrorWithContext(ctx, err, "Failed to sync EvictionRequest", "key", key)
	c.evictionRequestQueue.AddRateLimited(key)
	return true
}

// syncEviction reconciles an Eviction status
func (c *EvictionRequestController) syncEviction(ctx context.Context, key string) error {
	logger := klog.FromContext(ctx)
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	eviction, err := c.evictionLister.Evictions(namespace).Get(name)
	if errors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	logger.V(4).Info("Syncing Eviction", "eviction", klog.KObj(eviction))

	// Terminal Evictions (Canceled or Evicted) don't need further processing.
	// Without this check, a re-sync after setting Canceled (e.g., pod not found on
	// first sync) would fall through to computeCompletionCondition, which would
	// overwrite the Canceled condition with Evicted.
	if hasEvictionCompleted(eviction) {
		return nil
	}

	// Resolve target pod
	var targetPod *v1.Pod
	if eviction.Spec.Target.Pod != nil {
		pod, err := c.podLister.Pods(namespace).Get(eviction.Spec.Target.Pod.Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
		targetPod = pod
	}

	target := newTargetInfoForEviction(eviction.Spec.Target, targetPod)
	logger.V(4).Info("Target info for Eviction", "eviction", klog.KObj(eviction), "targetType", target.targetType(), "targetName", target.targetName(), "desiredTargetUID", target.targetUID())
	now := c.clock.Now()

	targetEvictions, err := c.listEvictionsForTarget(targetIndexValueForEviction(eviction))
	if err != nil {
		return err
	}
	mostRelevantEviction, _ := findRelevantEviction(targetEvictions)
	isDuplicate := !hasEvictionCompleted(mostRelevantEviction) && mostRelevantEviction.UID != eviction.UID
	// Spec is immutable for now, so this should be false after the first sync.
	generationChanged := !ptr.Equal(eviction.Status.ObservedGeneration, new(eviction.Generation))

	// early validation
	if failed, evicted, conditionsChanged := validateEviction(now, eviction, target, isDuplicate); failed != nil {
		// immutable spec so there shouldn't be any generation update
		if conditionsChanged || generationChanged {
			return c.applyEvictionStatus(ctx, eviction,
				lifecycleapply.EvictionStatus().
					WithObservedGeneration(eviction.Generation).
					WithConditions(failed, evicted),
			)
		}
		return nil
	}

	targetEvictionRequests, err := c.listEvictionRequestsForTarget(targetIndexValueForEviction(eviction))
	if err != nil {
		return err
	}

	statusApply, resyncAfter, changed := c.computeEvictionStatus(ctx, eviction, target, targetEvictionRequests)

	if resyncAfter != nil {
		c.evictionQueue.AddAfter(key, *resyncAfter)
	}
	if changed || generationChanged {
		return c.applyEvictionStatus(ctx, eviction, statusApply.WithObservedGeneration(eviction.Generation))
	}
	return nil
}

// validateEviction returns the Canceled and Evicted conditions if the validation fails.
func validateEviction(now time.Time, eviction *lifecyclev1alpha1.Eviction, target targetInfo, isDuplicate bool) (*metav1ac.ConditionApplyConfiguration, *metav1ac.ConditionApplyConfiguration, bool) {
	failureMsg := ""
	switch {
	case target.targetType() == noTarget:
		// Non-pod targets are not supported, so cancel the eviction request.
		// This is covered by API validation. We will end up here only if the apiserver serves a new type which the
		// controller does not recognize.
		failureMsg = "Unsupported target type."
	case isDuplicate:
		failureMsg = "Active Eviction already exists for the same target."
	case target.hasSchedulingGroup():
		failureMsg = fmt.Sprintf("Target %s references a SchedulingGroup. Eviction is currently not supported.", target.targetType())
	}
	// target match is handled by EvictionRequests

	if len(failureMsg) > 0 {
		failed, failedChanged := setCondition(now, eviction.Status.Conditions, lifecyclev1alpha1.EvictionConditionFailed,
			metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid, failureMsg)
		evicted, evictedChanged := setCondition(now, eviction.Status.Conditions, lifecyclev1alpha1.EvictionConditionTargetEvicted,
			metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, "")
		return failed, evicted, failedChanged || evictedChanged
	}
	return nil, nil, false
}

// computeEvictionStatus computes the desired status for an Eviction and builds the status apply configuration.
// It returns the status apply configuration and an optional resync duration.
func (c *EvictionRequestController) computeEvictionStatus(ctx context.Context, eviction *lifecyclev1alpha1.Eviction, target targetInfo, requests []*lifecyclev1alpha1.EvictionRequest) (*lifecycleapply.EvictionStatusApplyConfiguration, *time.Duration, bool) {
	var resyncAfter *time.Duration
	// All operations should reference the same time instant.
	now := c.clock.Now()
	logger := klog.FromContext(ctx)

	statusApply := lifecycleapply.EvictionStatus()

	changed := updateRequestersForEvictionStatusApply(eviction, requests, maxEvictionStatusRequesters, statusApply)

	var targetIsGone, targetIsTerminal, evictionIsCanceled bool
	switch {
	case target.isGone():
		targetIsGone = true
	case target.isTerminal():
		targetIsTerminal = true
	case !hasEvictionIntent(requests):
		evictionIsCanceled = true
	}

	targetResponders, changedBecauseInitialization := getOrInitializeTargetResponders(eviction, target)
	// Sort responders first, then use ordered progression to simplify the processing.
	// This means lower indexes are Activated first, and the last index is processed last.
	targetResponders = sortTargetResponders(targetResponders)
	logger.V(4).Info("Found responders", "targetResponders", targetResponders)

	beforeProgressionTargetResponders := make([]lifecyclev1alpha1.TargetResponder, len(targetResponders))
	copy(beforeProgressionTargetResponders, targetResponders)
	isProgressionDone, progressionResync := computeResponderProgression(now, eviction, targetResponders, target, targetIsGone, targetIsTerminal, evictionIsCanceled)
	changed = changed || changedBecauseInitialization || !reflect.DeepEqual(beforeProgressionTargetResponders, targetResponders)

	for _, responder := range targetResponders {
		targetRespondersApply := lifecycleapply.TargetResponder().
			WithName(responder.Name).
			WithState(responder.State)
		if responder.Priority != nil {
			targetRespondersApply = targetRespondersApply.WithPriority(*responder.Priority)
		}
		statusApply.WithTargetResponders(targetRespondersApply)
	}

	// Include ResponderStatus entries for all target responders so SSA doesn't
	// remove them. Only set Name and StartTime — other fields (HeartbeatTime,
	// CompletionTime, Message) are owned by the responders via their own
	// field manager.
	for _, targetResponder := range targetResponders {
		// first sync is triggered together with targetResponders so we don't have to set changed variable, except for the startTime
		isApply := lifecycleapply.ResponderStatus().WithName(targetResponder.Name)

		existing := findResponderStatus(eviction.Status.Responders, targetResponder.Name)
		if existing != nil && existing.StartTime != nil {
			isApply.WithStartTime(*existing.StartTime)
		} else if targetResponder.State == lifecyclev1alpha1.ResponderStateActive {
			isApply.WithStartTime(metav1.NewTime(now))
			changed = true
		}
		statusApply.WithResponders(isApply)
	}

	activeIdx := findTargetResponderIdx(targetResponders, lifecyclev1alpha1.ResponderStateActive)
	var activeResponderStatus *lifecyclev1alpha1.ResponderStatus
	if activeIdx != -1 {
		activeResponderStatus = findResponderStatus(eviction.Status.Responders, targetResponders[activeIdx].Name)
	}

	isWaitingForResponderUpdate := false
	switch {
	case targetIsGone || targetIsTerminal:
		resyncAfter = shouldDeferCompletion(now, activeResponderStatus, target)
		isWaitingForResponderUpdate = resyncAfter != nil
	default:
		resyncAfter = progressionResync
	}

	failed, evicted, conditionsChanged := computeEvictionConditions(now, eviction, isWaitingForResponderUpdate, targetIsGone, targetIsTerminal, evictionIsCanceled, isProgressionDone)
	changed = changed || conditionsChanged
	isFinal := *failed.Status == metav1.ConditionTrue || *evicted.Status == metav1.ConditionTrue
	if isFinal {
		logger.V(4).Info("Terminal condition reached", "eviction", klog.KObj(eviction), "isGone", targetIsGone, "isTerminal", targetIsTerminal, "isCanceled", evictionIsCanceled)
	}
	statusApply.WithConditions(failed, evicted)

	return statusApply, resyncAfter, changed
}

func updateRequestersForEvictionStatusApply(eviction *lifecyclev1alpha1.Eviction, requests []*lifecyclev1alpha1.EvictionRequest, limit int, statusApply *lifecycleapply.EvictionStatusApplyConfiguration) bool {
	originalRequesters := map[string]*lifecyclev1alpha1.Requester{}
	sortedRequests := append([]*lifecyclev1alpha1.EvictionRequest{}, requests...)
	// add existing
	for _, oldRequester := range eviction.Status.Requesters {
		originalRequesters[oldRequester.Name] = &oldRequester
		sortedRequests = append(sortedRequests, &lifecyclev1alpha1.EvictionRequest{
			Spec: lifecyclev1alpha1.EvictionRequestSpec{
				Requester: oldRequester.Name,
				Intent:    lifecyclev1alpha1.EvictionRequestIntent(oldRequester.Intent),
			},
		})
	}

	sortEvictionRequestsByRelevance(sortedRequests)

	processedRequesters := map[string]*lifecyclev1alpha1.EvictionRequest{}
	changed := false
	// update old requesters
	for i, request := range sortedRequests {
		if i >= limit {
			break
		}
		if _, ok := processedRequesters[request.Spec.Requester]; ok {
			// duplicate - already processed
			continue
		}
		intent := lifecyclev1alpha1.RequesterIntentWithdrawn
		if !request.CreationTimestamp.IsZero() && request.DeletionTimestamp == nil {
			// We can use intent from active and existing EvictionRequest
			intent = lifecyclev1alpha1.RequesterIntent(request.Spec.Intent) // 1:1 mapping for now
		}
		statusApply.WithRequesters(lifecycleapply.Requester().WithName(request.Spec.Requester).WithIntent(intent))
		processedRequesters[request.Spec.Requester] = request
		if !changed {
			if oldRequester := originalRequesters[request.Spec.Requester]; oldRequester == nil || oldRequester.Intent != intent {
				changed = true
			}
		}
	}
	return changed
}

// computeEvictionConditions returns the full pair of Canceled and Evicted conditions.
// Both are always set — defaulting to False until flipped to True.
func computeEvictionConditions(
	now time.Time,
	eviction *lifecyclev1alpha1.Eviction,
	isWaitingForResponderUpdate bool,
	isGone, isTerminal, isCanceled, isProgressionDone bool,
) (failed, evicted *metav1ac.ConditionApplyConfiguration, changed bool) {
	existing := eviction.Status.Conditions
	var failedChanged, evictedChanged bool
	switch {
	// Completion checks
	case isGone && !isWaitingForResponderUpdate:
		evicted, evictedChanged = setCondition(now, existing, lifecyclev1alpha1.EvictionConditionTargetEvicted,
			metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonPodDeleted, "Target pod has been deleted.")
		failed, failedChanged = setCondition(now, existing, lifecyclev1alpha1.EvictionConditionFailed,
			metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonSucceeded, "")
	case isTerminal && !isWaitingForResponderUpdate:
		evicted, evictedChanged = setCondition(now, existing, lifecyclev1alpha1.EvictionConditionTargetEvicted,
			metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonPodTerminal, "Pod has reached terminal state.")
		failed, failedChanged = setCondition(now, existing, lifecyclev1alpha1.EvictionConditionFailed,
			metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonSucceeded, "")
	case isCanceled:
		failed, failedChanged = setCondition(now, existing, lifecyclev1alpha1.EvictionConditionFailed,
			metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonCanceledDueToNoRequesters, "No active requesters with eviction intent.")
		evicted, evictedChanged = setCondition(now, existing, lifecyclev1alpha1.EvictionConditionTargetEvicted,
			metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, "")
	case isProgressionDone:
		failed, failedChanged = setCondition(now, existing, lifecyclev1alpha1.EvictionConditionFailed,
			metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonNoFurtherResponder, "All responders have completed without evicting the target.")
		evicted, evictedChanged = setCondition(now, existing, lifecyclev1alpha1.EvictionConditionTargetEvicted,
			metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, "")
	default:
		failed, failedChanged = setCondition(now, existing, lifecyclev1alpha1.EvictionConditionFailed,
			metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction, "")
		evicted, evictedChanged = setCondition(now, existing, lifecyclev1alpha1.EvictionConditionTargetEvicted,
			metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction, "")
	}
	return failed, evicted, failedChanged || evictedChanged
}

// syncEvictionRequest reconciles an EvictionRequest status and creates new Evictions
func (c *EvictionRequestController) syncEvictionRequest(ctx context.Context, key string) error {
	logger := klog.FromContext(ctx)
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	evictionRequest, err := c.evictionRequestLister.EvictionRequests(namespace).Get(name)
	if errors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	logger.V(4).Info("Syncing EvictionRequest", "evictionRequest", klog.KObj(evictionRequest), "requester", evictionRequest.Spec.Requester, "intent", evictionRequest.Spec.Intent)

	// Resolve target pod - for early validation.
	var targetPod *v1.Pod
	if evictionRequest.Spec.Target.Pod != nil {
		pod, err := c.podLister.Pods(namespace).Get(evictionRequest.Spec.Target.Pod.Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
		targetPod = pod
	}

	target := newTargetInfoForEvictionRequest(evictionRequest.Spec.Target, targetPod)

	var targetEvictions []*lifecyclev1alpha1.Eviction
	if target.targetType() != noTarget {
		targetEvictions, err = c.listEvictionsForTarget(targetIndexValueForEvictionRequest(evictionRequest))
		if err != nil {
			return err
		}
	}
	generationChanged := !ptr.Equal(evictionRequest.Status.ObservedGeneration, new(evictionRequest.Generation))

	// Terminal Eviction Requests (Evicted or Failed with EvictionInvalid reason) don't need further processing.
	// Without this check, a re-sync after setting Failed (e.g., pod not found on
	// first sync) would fall through to computeCompletionCondition, which would
	// overwrite the Canceled condition with Evicted.
	// But we have to ensure that .status.observedGeneration gets updated in case the spec changes.
	if hasEvictionRequestCompleted(evictionRequest) && !generationChanged {
		return nil
	}

	logger.V(4).Info("Target info for EvictionRequest", "evictionRequest", klog.KObj(evictionRequest), "targetType", target.targetType(), "targetName", target.targetName(), "desiredTargetUID", target.targetUID(), "activeEvictions", len(targetEvictions))
	now := c.clock.Now()
	statusApplyConfiguration := lifecycleapply.EvictionRequestStatus().WithObservedGeneration(evictionRequest.Generation)

	// early validation
	if failed, evicted, conditionsChanged := validateEvictionRequest(now, evictionRequest, target, len(targetEvictions) > 0); failed != nil {
		if conditionsChanged || generationChanged {
			return c.applyEvictionRequestStatus(ctx, evictionRequest, statusApplyConfiguration.WithConditions(failed, evicted))
		}
		return nil
	}

	relevantEviction, shouldCreate := findRelevantEviction(targetEvictions)
	// sync conditions with the relevant eviction
	evicted, evictedChanged := convertToEvictionRequestConditions(now, evictionRequest, relevantEviction, lifecyclev1alpha1.EvictionConditionTargetEvicted)
	failed, failedChanged := convertToEvictionRequestConditions(now, evictionRequest, relevantEviction, lifecyclev1alpha1.EvictionConditionFailed)
	changed := generationChanged || evictedChanged || failedChanged

	targetEvictionRequests, err := c.listEvictionRequestsForTarget(targetIndexValueForEvictionRequest(evictionRequest))
	if err != nil {
		return err
	}

	// create new eviction if needed
	if hasEvictionIntent(targetEvictionRequests) && shouldCreate {
		allEvictions, err := c.evictionLister.Evictions(namespace).List(apimachinerylabels.Everything())
		if err != nil {
			return err
		}
		newEvictionApplyConfig := lifecycleapply.Eviction(newEvictionName(target, allEvictions), evictionRequest.Namespace).
			WithOwnerReferences(evictionRequestAsOwnerReference(evictionRequest)).
			WithLabels(map[string]string{
				// labels are reconciled by the refreshEviction loop from now on according to the roles
				evictionRequest.Spec.Requester: string(lifecyclev1alpha1.EvictionParticipantRoleRequester),
			}).
			WithSpec(lifecycleapply.EvictionSpec().
				WithTarget(target.toEvictionTargetApply()),
			)
		if err := c.applyEviction(ctx, newEvictionApplyConfig); err != nil {
			return err
		}
	}

	if changed {
		return c.applyEvictionRequestStatus(ctx, evictionRequest, statusApplyConfiguration.WithConditions(failed, evicted))
	}
	return nil
}

// validate returns the Canceled and Evicted conditions if the validation fails.
func validateEvictionRequest(now time.Time, evictionRequest *lifecyclev1alpha1.EvictionRequest, target targetInfo, hasEvictions bool) (failed, evicted *metav1ac.ConditionApplyConfiguration, changed bool) {
	failureMsg := ""
	switch {
	case target.targetType() == noTarget:
		// Non-pod targets are not supported, so cancel the eviction request.
		// This is covered by API validation. We will end up here only if the apiserver serves a new type which the
		// controller does not recognize.
		failureMsg = "Unsupported target type."
	case !hasEvictions && !target.targetFoundByName():
		failureMsg = fmt.Sprintf("Target %s not found.", target.targetType())
	case !hasEvictions && target.targetUID() != target.GetObjectMeta().GetUID():
		failureMsg = fmt.Sprintf("Target %s UID mismatch: expected %s, got %s.", target.targetType(), target.targetUID(), target.GetObjectMeta().GetUID())
	case target.hasSchedulingGroup():
		failureMsg = fmt.Sprintf("Target %s references a SchedulingGroup. Eviction is currently not supported.", target.targetType())
	}

	if len(failureMsg) > 0 {
		failed, failedChanged := setCondition(now, evictionRequest.Status.Conditions, lifecyclev1alpha1.EvictionConditionFailed,
			metav1.ConditionTrue, lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid, failureMsg)
		evicted, evictedChanged := setCondition(now, evictionRequest.Status.Conditions, lifecyclev1alpha1.EvictionConditionTargetEvicted,
			metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonEvictionFailed, "")
		return failed, evicted, failedChanged || evictedChanged
	}
	return nil, nil, false
}

// syncEvictionMetaRefresh reconciles an EvictionRequest metadata
//   - Computes responder and requester labels and adds them to Evictions allowing
//     responders to use their names as label selectors when watching evictions.
//   - Adds EvictionRequest owners references to Evictions for GC
func (c *EvictionRequestController) syncEvictionMetaRefresh(ctx context.Context, key string) error {
	logger := klog.FromContext(ctx)
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	eviction, err := c.evictionLister.Evictions(namespace).Get(name)
	if errors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}

	logger.V(4).Info("Refreshing Eviction labels and owner references", "eviction", klog.KObj(eviction))

	// Resolve target pod
	var targetPod *v1.Pod
	switch {
	case eviction.Spec.Target.Pod != nil:
		pod, err := c.podLister.Pods(namespace).Get(eviction.Spec.Target.Pod.Name)
		if err != nil && !errors.IsNotFound(err) {
			return err
		}
		targetPod = pod
	default:
		logger.Error(nil, "Unrecognized target", "eviction", klog.KObj(eviction))
		return nil
	}

	target := newTargetInfoForEviction(eviction.Spec.Target, targetPod)

	targetEvictionRequests, err := c.listEvictionRequestsForTarget(targetIndexValueForEviction(eviction))
	if err != nil {
		return err
	}

	applyConfig := lifecycleapply.Eviction(eviction.Name, namespace).
		// ensure we do not write nil to an immutable field
		WithSpec(lifecycleapply.EvictionSpec().WithTarget(target.toEvictionTargetApply()))

	labels := computeEvictionLabels(eviction, target, targetEvictionRequests)

	needsUpdate := evictionLabelsNeedSSAUpdate(eviction.Labels, labels)
	applyConfig.WithLabels(labels)

	// Repair owner references even if target is gone.
	type requestOwnerRefKey struct {
		name string
		uid  types.UID
	}

	desiredOwnerRefs := map[requestOwnerRefKey]*metav1ac.OwnerReferenceApplyConfiguration{}
	for _, targetEvictionRequest := range targetEvictionRequests {
		// This ensures we do not reintroduce owner ref during orphaning.
		if targetEvictionRequest.DeletionTimestamp == nil {
			desiredOwnerRefs[requestOwnerRefKey{
				name: targetEvictionRequest.Name,
				uid:  targetEvictionRequest.UID,
			}] = evictionRequestAsOwnerReference(targetEvictionRequest)
		}
	}

	evictionReqguestGVK := lifecyclev1alpha1.SchemeGroupVersion.WithKind("EvictionRequest")
	// Never remove owner refs managed by us, this is a job for GC.
	for i, ref := range eviction.OwnerReferences {
		if ref.Kind == evictionReqguestGVK.Kind && ref.APIVersion == evictionReqguestGVK.GroupVersion().String() {
			// Preserve ref even if the targetEvictionRequests has just been removed from the cache
			// to prevent races with GC.
			applyConfig.WithOwnerReferences(metav1ac.OwnerReference().
				WithKind(ref.Kind).
				WithAPIVersion(ref.APIVersion).
				WithName(ref.Name).
				WithUID(ref.UID))
			// Mark processed for existing EvictionRequests.
			delete(desiredOwnerRefs, requestOwnerRefKey{
				name: eviction.OwnerReferences[i].Name,
				uid:  eviction.OwnerReferences[i].UID,
			})
		}
	}

	// Add new owner references.
	for _, desiredRef := range desiredOwnerRefs {
		applyConfig.WithOwnerReferences(desiredRef)
		needsUpdate = true
	}

	if !needsUpdate {
		return nil
	}

	return c.applyEviction(ctx, applyConfig)
}

// applyEviction applies the spec and metadata to the Eviction using Server-Side Apply.
func (c *EvictionRequestController) applyEviction(ctx context.Context, evictionApply *lifecycleapply.EvictionApplyConfiguration) error {
	_, err := c.kubeClient.LifecycleV1alpha1().
		Evictions(*evictionApply.Namespace).
		Apply(ctx, evictionApply, metav1.ApplyOptions{
			FieldManager: c.controllerName,
			Force:        true,
		})
	return err
}

// applyEvictionStatus applies the status to the Eviction using Server-Side Apply.
func (c *EvictionRequestController) applyEvictionStatus(
	ctx context.Context,
	eviction *lifecyclev1alpha1.Eviction,
	statusApply *lifecycleapply.EvictionStatusApplyConfiguration,
) error {
	applyConfig := lifecycleapply.Eviction(eviction.Name, eviction.Namespace).
		WithStatus(statusApply)

	_, err := c.kubeClient.LifecycleV1alpha1().
		Evictions(eviction.Namespace).
		ApplyStatus(ctx, applyConfig, metav1.ApplyOptions{
			FieldManager: c.controllerName,
			Force:        true,
		})
	return err
}

// applyEvictionStatus applies the status to the EvictionRequest using Server-Side Apply.
func (c *EvictionRequestController) applyEvictionRequestStatus(
	ctx context.Context,
	evictionRequest *lifecyclev1alpha1.EvictionRequest,
	statusApply *lifecycleapply.EvictionRequestStatusApplyConfiguration,
) error {
	applyConfig := lifecycleapply.EvictionRequest(evictionRequest.Name, evictionRequest.Namespace).
		WithStatus(statusApply)

	_, err := c.kubeClient.LifecycleV1alpha1().
		EvictionRequests(evictionRequest.Namespace).
		ApplyStatus(ctx, applyConfig, metav1.ApplyOptions{
			FieldManager: c.controllerName,
			Force:        true,
		})
	return err
}

// listEvictionsForPod returns all Evictions whose eviction.Spec.Target.Pod.UID equal to the given Pod UID.
func (c *EvictionRequestController) listEvictionsForTarget(evictionTargetIndexValue string) ([]*lifecyclev1alpha1.Eviction, error) {
	if len(evictionTargetIndexValue) == 0 {
		return nil, fmt.Errorf("evictionTargetIndexValue is empty")
	}
	all, err := c.evictionIndexer.ByIndex(evictionByTargetUIDIndexKey, evictionTargetIndexValue)
	if err != nil {
		return nil, err
	}
	matched := make([]*lifecyclev1alpha1.Eviction, 0, len(all))
	for _, ev := range all {
		eviction, ok := ev.(*lifecyclev1alpha1.Eviction)
		if !ok {
			continue
		}
		matched = append(matched, eviction)
	}
	return matched, nil
}

// listEvictionsForPod returns all EvictionRequests whose evictionRequest.Spec.Target.Pod.UID equal to the given Pod UID.
func (c *EvictionRequestController) listEvictionRequestsForTarget(evictionRequestTargetIndexValue string) ([]*lifecyclev1alpha1.EvictionRequest, error) {
	if len(evictionRequestTargetIndexValue) == 0 {
		return nil, fmt.Errorf("evictionRequestTargetIndexValue is empty")
	}
	all, err := c.evictionRequestIndexer.ByIndex(evictionRequestByTargetUID, evictionRequestTargetIndexValue)
	if err != nil {
		return nil, err
	}
	matched := make([]*lifecyclev1alpha1.EvictionRequest, 0, len(all))
	for _, er := range all {
		evictionRequest, ok := er.(*lifecyclev1alpha1.EvictionRequest)
		if !ok {
			continue
		}
		matched = append(matched, evictionRequest)
	}
	return matched, nil
}

func targetIndexValueForPod(pod *v1.Pod) string {
	if pod != nil {
		return pod.Namespace + "/pod/" + string(pod.UID)
	}
	return ""
}

func targetIndexValueForEvictionRequest(evictionRequest *lifecyclev1alpha1.EvictionRequest) string {
	if evictionRequest.Spec.Target.Pod != nil {
		return evictionRequest.Namespace + "/pod/" + string(evictionRequest.Spec.Target.Pod.UID)
	}
	return ""
}

func targetIndexValueForEviction(eviction *lifecyclev1alpha1.Eviction) string {
	if eviction.Spec.Target.Pod != nil {
		return eviction.Namespace + "/pod/" + string(eviction.Spec.Target.Pod.UID)
	}
	return ""
}
