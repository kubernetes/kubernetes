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

package imperativeevictionresponder

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	coordinationv1alpha1 "k8s.io/api/coordination/v1alpha1"
	v1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/validate"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coordinationapplyv1alpha1 "k8s.io/client-go/applyconfigurations/coordination/v1alpha1"
	coordinationinformers "k8s.io/client-go/informers/coordination/v1alpha1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	coordinationlisters "k8s.io/client-go/listers/coordination/v1alpha1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	apiv1pod "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/utils/clock"
	"k8s.io/utils/ptr"
)

type ImperativeEvictionResponderController struct {
	controllerName string

	kubeClient clientset.Interface

	evictionRequestLister       coordinationlisters.EvictionRequestLister
	evictionRequestListerSynced cache.InformerSynced

	podLister       corelisters.PodLister
	podListerSynced cache.InformerSynced

	// queue tracks EvictionRequest keys
	queue workqueue.TypedRateLimitingInterface[string]

	// syncHandler is the function called to sync an EvictionRequest.
	// It may be replaced during tests.
	syncHandler func(ctx context.Context, key string) (*time.Duration, error)

	heartbeatMinDurationBetweenUpdates time.Duration
	heartbeatMaxDurationBetweenUpdates time.Duration

	maxImperativeEvictionBackoff time.Duration

	lastEvictionAttempts *lastEvictionAttempts

	// clock is used for time-based operations.
	// It may be replaced during tests with a fake clock.
	clock clock.PassiveClock
}

// NewController creates a new imperative eviction responder controller.
func NewController(
	ctx context.Context,
	controllerName string,
	evictionRequestInformer coordinationinformers.EvictionRequestInformer,
	podInformer coreinformers.PodInformer,
	kubeClient clientset.Interface,
) (*ImperativeEvictionResponderController, error) {
	logger := klog.FromContext(ctx)

	c := &ImperativeEvictionResponderController{
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
		heartbeatMinDurationBetweenUpdates: 5 * time.Second,
		heartbeatMaxDurationBetweenUpdates: 3 * time.Minute,
		maxImperativeEvictionBackoff:       10 * time.Minute,
		lastEvictionAttempts:               NewLastEvictionAttempts(),
		clock:                              clock.RealClock{},
	}

	c.syncHandler = c.sync

	if _, err := evictionRequestInformer.Informer().AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			evictionRequest := obj.(*coordinationv1alpha1.EvictionRequest)
			if evictionRequest.Spec.Target.Pod == nil {
				return
			}
			c.enqueue(logger, evictionRequest)
		},
		UpdateFunc: func(old, new any) {
			oldEvictionRequest := old.(*coordinationv1alpha1.EvictionRequest)
			newEvictionRequest := new.(*coordinationv1alpha1.EvictionRequest)

			if newEvictionRequest.Spec.Target.Pod == nil {
				return
			}
			oldTargetInterceptor := findTargetResponderStatus(oldEvictionRequest)
			newTargetInterceptor := findTargetResponderStatus(oldEvictionRequest)
			if !validate.SemanticDeepEqual(oldTargetInterceptor, newTargetInterceptor) || // observe own .state that is controlled by the evictionrequest-controller
				len(oldEvictionRequest.Status.Responders) != len(newEvictionRequest.Status.Responders) ||
				meta.IsStatusConditionTrue(oldEvictionRequest.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionFailed)) != meta.IsStatusConditionTrue(newEvictionRequest.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionFailed)) ||
				meta.IsStatusConditionTrue(oldEvictionRequest.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionEvicted)) != meta.IsStatusConditionTrue(newEvictionRequest.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionEvicted)) {
				c.enqueue(logger, newEvictionRequest)
			}
		},
		DeleteFunc: func(obj interface{}) {
			// clean up lastEvictionAttempts
			c.deleteEvictionRequest(logger, obj)
		},
	}, cache.HandlerOptions{Logger: &logger}); err != nil {
		return nil, err
	}

	// call back when the pod is deleted, terminated or removed from etcd
	if _, err := podInformer.Informer().AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(old, new any) {
			oldPod := old.(*v1.Pod)
			newPod := new.(*v1.Pod)
			if !oldPod.DeletionTimestamp.Equal(newPod.DeletionTimestamp) ||
				(oldPod.Status.Phase != newPod.Status.Phase && apiv1pod.IsPodTerminal(newPod)) {
				// pod has been evicted
				c.lastEvictionAttempts.remove(newPod.UID)
				c.queue.Add(evictionRequestKey(newPod))
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

func (c *ImperativeEvictionResponderController) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()

	logger := klog.FromContext(ctx)
	logger.Info("Starting", "controller", c.controllerName)

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down", "controller", c.controllerName)
		c.queue.ShutDown()
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

	<-ctx.Done()
}

// enqueue adds an EvictionRequest to the queue.
func (c *ImperativeEvictionResponderController) enqueue(logger klog.Logger, obj any) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to get key for object")
		return
	}
	c.queue.Add(key)
}

// deleteEvictionRequest clears lastEvictionAttempts store for a deleted EvictionRequest.
func (c *ImperativeEvictionResponderController) deleteEvictionRequest(logger klog.Logger, obj any) {
	evictionRequest, ok := obj.(*coordinationv1alpha1.EvictionRequest)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleErrorWithLogger(logger, nil, "Couldn't get object from tombstone", "object", obj)
			return
		}
		evictionRequest, ok = tombstone.Obj.(*coordinationv1alpha1.EvictionRequest)
		if !ok {
			utilruntime.HandleErrorWithLogger(logger, nil, "Tombstone contained object that is not a EvictionRequest", "object", obj)
			return
		}
	}
	if podTarget := evictionRequest.Spec.Target.Pod; podTarget != nil {
		c.lastEvictionAttempts.remove(podTarget.UID)
	}
}

// deletePod enqueues the EvictionRequest for a deleted Pod.
func (c *ImperativeEvictionResponderController) deletePod(logger klog.Logger, obj any) {
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
	c.lastEvictionAttempts.remove(pod.UID)
	c.queue.Add(evictionRequestKey(pod))
}

func (c *ImperativeEvictionResponderController) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *ImperativeEvictionResponderController) processNextWorkItem(ctx context.Context) bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	addAfter, err := c.syncHandler(ctx, key)
	if err == nil {
		c.queue.Forget(key)
		// /eviction subresource call failures are handled through addAfter and not through the rate limiter.
		// These calls do not return an error from syncHandler, but write an EvictionRequest status instead.
		// This ensures that default rate limiting for other errors (e.g. status updates) works correctly.
		if addAfter != nil {
			c.queue.AddAfter(key, *addAfter)
		}
		return true
	}

	utilruntime.HandleErrorWithContext(ctx, err, "Failed to sync EvictionRequest", "key", key)
	c.queue.AddRateLimited(key)
	return true
}

func (c *ImperativeEvictionResponderController) sync(ctx context.Context, key string) (*time.Duration, error) {
	logger := klog.FromContext(ctx)
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return nil, err
	}

	evictionRequest, err := c.evictionRequestLister.EvictionRequests(namespace).Get(name)
	if errors.IsNotFound(err) {
		// no work
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	podTarget := evictionRequest.Spec.Target.Pod
	if podTarget == nil {
		// irrelevant request for the controller
		return nil, nil
	}
	logger.V(4).Info("Syncing EvictionRequest", "evictionRequest", klog.KObj(evictionRequest))
	if evictionRequest.Status.ObservedGeneration == nil {
		// wait for the eviction request controller first
		return nil, nil
	}
	if meta.IsStatusConditionTrue(evictionRequest.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionFailed)) ||
		meta.IsStatusConditionTrue(evictionRequest.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionEvicted)) {
		// Eviction request final state reached: failed/canceled or done.
		// This will forget the queue item in case we are still retrying the eviction, but the eviction request has been failed/canceled in the meantime.
		return nil, nil
	}

	targetResponderStatus := findTargetResponderStatus(evictionRequest)
	if targetResponderStatus == nil || targetResponderStatus.State != coordinationv1alpha1.ResponderStateActive {
		// TODO, update status when state is Canceled?
		// must be designated active by the evictionrequest-controller first
		return nil, nil
	}
	lastResponderStatus := findResponderStatus(evictionRequest)
	if lastResponderStatus == nil {
		// status have to be computed by the evictionrequest-controller first
		return nil, nil
	}
	if lastResponderStatus.CompletionTime != nil {
		// work done
		return nil, nil
	}

	pod, err := c.podLister.Pods(namespace).Get(podTarget.Name)
	isNotFound := false
	if errors.IsNotFound(err) {
		isNotFound = true
	} else if err != nil {
		return nil, err
	}

	evictionCompleted := false
	evictionMessage := ""
	var expectedCompletionTime *time.Time
	var addAfter *time.Duration
	switch {
	case isNotFound || pod.UID != podTarget.UID:
		evictionCompleted = true
		evictionMessage = fmt.Sprintf("%q pod has been deleted and fully terminated", podTarget.Name)
	case pod.DeletionTimestamp != nil && apiv1pod.IsPodTerminal(pod):
		evictionCompleted = true
		evictionMessage = fmt.Sprintf("%q pod has been deleted and fully terminated (pod phase=%q)", pod.Name, pod.Status.Phase)
	case pod.DeletionTimestamp != nil:
		reason := ""
		if strings.Contains(lastResponderStatus.Message, "pod has been deleted via the /eviction subresource") {
			reason = " via the /eviction subresource"
		}
		evictionMessage = fmt.Sprintf("%q pod has been deleted%v and is being terminated gracefully", pod.Name, reason)
		expectedCompletionTime = ptr.To(pod.DeletionTimestamp.Time)
		// not complete - schedule heartbeat
		addAfter = ptr.To(c.heartbeatMaxDurationBetweenUpdates)
	case apiv1pod.IsPodTerminal(pod):
		evictionCompleted = true
		evictionMessage = fmt.Sprintf("%q pod has been fully terminated (pod phase=%q)", pod.Name, pod.Status.Phase)
	default:
		evictionMessage, addAfter = c.evict(ctx, pod, lastResponderStatus.Message)
	}

	// Use the same time tick to report time in the status fields/
	now := c.clock.Now()

	// The controller has to report a heartbeat every three minutes to indicate that it is running properly,
	// even if the evict backoff period is longer.
	if addAfter != nil && *addAfter > c.heartbeatMaxDurationBetweenUpdates {
		addAfter = ptr.To(c.heartbeatMaxDurationBetweenUpdates)
	}

	// API validation constraints
	if expectedCompletionTime != nil && expectedCompletionTime.Before(now) {
		expectedCompletionTime = ptr.To(now)
	}
	if len(evictionMessage) > 4000 {
		evictionMessage = evictionMessage[:4000]
	}

	// Update status/

	shouldUpdateHeartbeat := lastResponderStatus.HeartbeatTime == nil || lastResponderStatus.HeartbeatTime.Time.Before(now.Add(-c.heartbeatMaxDurationBetweenUpdates+time.Second))
	shouldUpdateExpectedCompletionTime := expectedCompletionTime != nil && !ptr.To(metav1.Time{Time: *expectedCompletionTime}).Equal(lastResponderStatus.ExpectedCompletionTime)
	shouldUpdateCompletionTime := evictionCompleted && lastResponderStatus.CompletionTime == nil

	if shouldUpdateHeartbeat || shouldUpdateExpectedCompletionTime || shouldUpdateCompletionTime || lastResponderStatus.Message != evictionMessage {
		newResponderStatus := toResponderStatusApplyConfiguration(*lastResponderStatus).WithMessage(evictionMessage)
		// Piggyback a heartbeat update if there is any update.
		newResponderStatus.WithHeartbeatTime(metav1.Time{Time: now})

		if shouldUpdateExpectedCompletionTime {
			newResponderStatus.WithExpectedCompletionTime(metav1.Time{Time: *expectedCompletionTime})
		}
		if shouldUpdateCompletionTime {
			newResponderStatus.WithExpectedCompletionTime(metav1.Time{Time: now})
			newResponderStatus.WithCompletionTime(metav1.Time{Time: now})
		}
		statusApplyUpdate := coordinationapplyv1alpha1.EvictionRequest(evictionRequest.Name, evictionRequest.Namespace).
			WithStatus(coordinationapplyv1alpha1.EvictionRequestStatus().WithResponders(newResponderStatus))
		_, err = c.kubeClient.CoordinationV1alpha1().EvictionRequests(evictionRequest.Namespace).ApplyStatus(ctx, statusApplyUpdate, metav1.ApplyOptions{
			FieldManager: c.controllerName,
			Force:        true,
		})
		if err != nil {
			return nil, err
		}
	}
	return addAfter, nil
}

func (c *ImperativeEvictionResponderController) evict(ctx context.Context, pod *v1.Pod, lastEvictionMessage string) (string, *time.Duration) {
	lastAttemptTime, ok := c.lastEvictionAttempts.get(pod.UID)
	// Apply backoff to /eviction calls, skip if we are too early since the last failed eviction.
	if ok {
		attempts := getRecordedAttempts(lastEvictionMessage)
		lastAttemptAddAfter := exponentialBackoff(c.heartbeatMinDurationBetweenUpdates, c.maxImperativeEvictionBackoff, attempts)
		addAfter := lastAttemptTime.Add(lastAttemptAddAfter).Sub(c.clock.Now())
		if addAfter > 0 {
			return lastEvictionMessage, &addAfter
		}
	}

	err := c.kubeClient.PolicyV1().Evictions(pod.Namespace).Evict(ctx, &policyv1.Eviction{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
		DeleteOptions: &metav1.DeleteOptions{
			Preconditions: &metav1.Preconditions{
				UID: ptr.To(pod.UID),
			},
		},
	})

	if err != nil {
		attempts := getRecordedAttempts(lastEvictionMessage) + 1
		c.lastEvictionAttempts.set(pod.UID, c.clock.Now())
		// 10 minutes max backoff
		addAfter := exponentialBackoff(c.heartbeatMinDurationBetweenUpdates, c.maxImperativeEvictionBackoff, attempts)
		// use suggested server delay if present
		if delay, shouldDelay := errors.SuggestsClientDelay(err); shouldDelay && (addAfter < time.Second*time.Duration(delay)) {
			addAfter = time.Second * time.Duration(delay)
		}
		evictionMessage := fmt.Sprintf("%q pod deletion via the /eviction subresource failed (attempts=%d): %v", pod.Name, attempts, err)
		return evictionMessage, &addAfter
	}

	c.lastEvictionAttempts.remove(pod.UID)
	evictionMessage := fmt.Sprintf("%q pod has been deleted via the /eviction subresource and will be terminated gracefully", pod.Name)
	return evictionMessage, nil
}

func evictionRequestKey(pod metav1.ObjectMetaAccessor) string {
	return pod.GetObjectMeta().GetNamespace() + "/" + string(pod.GetObjectMeta().GetUID())
}
