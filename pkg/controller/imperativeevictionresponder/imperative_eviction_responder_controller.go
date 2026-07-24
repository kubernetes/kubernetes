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

	v1 "k8s.io/api/core/v1"
	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	policyv1 "k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/validate"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	lifecycleapplyv1alpha1 "k8s.io/client-go/applyconfigurations/lifecycle/v1alpha1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	lifecycleinformers "k8s.io/client-go/informers/lifecycle/v1alpha1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	lifecyclelisters "k8s.io/client-go/listers/lifecycle/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	apiv1pod "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/utils/clock"
	"k8s.io/utils/ptr"
)

const (
	// We need to know which evictions we have to sync for a pod.
	evictionByPodUIDIndexKey = "evictionByPodUID"
)

// ImperativeEvictionResponderController is a default imperative-eviction.k8s.io/evictor responder
// controller for pods. This responder is executed last (priority 100), after all other responders
// have run, and attempts to evict a pod using the imperative eviction API
// (pods/<name>/eviction subresource). It fulfills  the declarative Eviction API contract.
type ImperativeEvictionResponderController struct {
	controllerName string

	kubeClient clientset.Interface

	evictionLister       lifecyclelisters.EvictionLister
	evictionListerSynced cache.InformerSynced

	podLister       corelisters.PodLister
	podListerSynced cache.InformerSynced

	// evictionIndexer allows looking up evictions by Pod UID
	evictionIndexer cache.Indexer

	// queue tracks Eviction keys
	queue workqueue.TypedRateLimitingInterface[string]

	// syncHandler is the function called to sync an Eviction.
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
	evictionInformer lifecycleinformers.TypedEvictionInformer,
	podInformer coreinformers.TypedPodInformer,
	kubeClient clientset.Interface,
) (*ImperativeEvictionResponderController, error) {
	logger := klog.FromContext(ctx)

	c := &ImperativeEvictionResponderController{
		controllerName:       controllerName,
		kubeClient:           kubeClient,
		evictionLister:       evictionInformer.Lister(),
		evictionListerSynced: evictionInformer.Informer().HasSynced,
		evictionIndexer:      evictionInformer.Informer().GetIndexer(),
		podLister:            podInformer.Lister(),
		podListerSynced:      podInformer.Informer().HasSynced,
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

	if _, err := evictionInformer.TypedInformer().AddTypedEventHandler(lifecycleinformers.EvictionHandlerFuncs{
		AddFunc: func(eviction *lifecyclev1alpha1.Eviction) {
			if !shouldHandleEviction(eviction) {
				return
			}
			c.enqueue(logger, eviction)
		},
		UpdateFunc: func(oldEviction, newEviction *lifecyclev1alpha1.Eviction) {
			if !shouldHandleEviction(newEviction) {
				return
			}
			// We don't need to check condition.status changes, because once they become True, they cannot be reverted,
			// and we can just simply stop responding.
			oldTargetInterceptor := findTargetResponderStatus(oldEviction)
			newTargetInterceptor := findTargetResponderStatus(newEviction)
			if !validate.SemanticDeepEqual(oldTargetInterceptor, newTargetInterceptor) || // observe own .state that is controlled by the eviction-controller
				hasLabelChanged(oldEviction, newEviction, lifecyclev1alpha1.EvictionResponderImperativeEviction) || // detect changes in the responder role - can potentially flip over time
				len(oldEviction.Status.Responders) != len(newEviction.Status.Responders) { // detect responder initialization
				c.enqueue(logger, newEviction)
			}
		},
		DeleteFunc: func(deletedEviction lifecycleinformers.DeletedEviction) {
			// clean up lastEvictionAttempts
			if deletedEviction.OptionalObj != nil && deletedEviction.OptionalObj.Spec.Target.Pod != nil {
				c.lastEvictionAttempts.remove(deletedEviction.OptionalObj.Spec.Target.Pod.UID)
			}
		},
	}, cache.HandlerOptions{Logger: &logger}); err != nil {
		return nil, err
	}

	if err := evictionInformer.Informer().AddIndexers(cache.Indexers{
		evictionByPodUIDIndexKey: func(obj interface{}) ([]string, error) {
			eviction, ok := obj.(*lifecyclev1alpha1.Eviction)
			if !ok {
				return nil, fmt.Errorf("unexpected object type %T", obj)
			}
			if eviction.Spec.Target.Pod != nil {
				return []string{string(eviction.Spec.Target.Pod.UID)}, nil
			}
			return nil, nil
		},
	}); err != nil {
		return nil, fmt.Errorf("adding eviction indexer: %w", err)
	}

	// call back when the pod is deleted, terminated or removed from etcd
	if _, err := podInformer.TypedInformer().AddTypedEventHandler(coreinformers.PodHandlerFuncs{
		UpdateFunc: func(oldPod, newPod *v1.Pod) {
			if !oldPod.DeletionTimestamp.Equal(newPod.DeletionTimestamp) ||
				(oldPod.Status.Phase != newPod.Status.Phase && apiv1pod.IsPodTerminal(newPod)) {
				// pod has been evicted
				c.lastEvictionAttempts.remove(newPod.UID)
				// returns a single eviction or an error if more
				evictions, err := c.listActiveEvictionsForPod(newPod)
				if err != nil {
					utilruntime.HandleErrorWithLogger(logger, err, "failed to list evictions for pod", "pod", klog.KObj(newPod))
				}
				for _, eviction := range evictions {
					c.queue.Add(evictionKey(eviction))
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

	return c, nil
}

// enqueue adds an Eviction to the queue.
func (c *ImperativeEvictionResponderController) enqueue(logger klog.Logger, obj any) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to get key for object")
		return
	}
	c.queue.Add(key)
}

// deletePod enqueues the Eviction for a deleted Pod.
func (c *ImperativeEvictionResponderController) deletePod(logger klog.Logger, pod *v1.Pod) {
	c.lastEvictionAttempts.remove(pod.UID)
	// returns a single eviction or an error if more
	evictions, err := c.listActiveEvictionsForPod(pod)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "failed to list evictions for pod", "pod", klog.KObj(pod))
	}
	for _, eviction := range evictions {
		c.queue.Add(evictionKey(eviction))
	}
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

	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.podListerSynced, c.evictionListerSynced) {
		return
	}

	for range workers {
		wg.Go(func() {
			wait.UntilWithContext(ctx, c.runWorker, time.Second)
		})
	}

	<-ctx.Done()
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
		// These calls do not return an error from syncHandler, but write an Eviction status instead.
		// This ensures that default rate limiting for other errors (e.g. status updates) works correctly.
		if addAfter != nil {
			c.queue.AddAfter(key, *addAfter)
		}
		return true
	}

	utilruntime.HandleErrorWithContext(ctx, err, "Failed to sync Eviction", "key", key)
	c.queue.AddRateLimited(key)
	return true
}

func (c *ImperativeEvictionResponderController) sync(ctx context.Context, key string) (*time.Duration, error) {
	logger := klog.FromContext(ctx)
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return nil, err
	}

	eviction, err := c.evictionLister.Evictions(namespace).Get(name)
	if errors.IsNotFound(err) {
		// no work
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	if !shouldHandleEviction(eviction) {
		// irrelevant or completed eviction for the controller; also filtered in ResourceEventHandler
		return nil, nil
	}

	logger.V(4).Info("Syncing Eviction", "eviction", klog.KObj(eviction))
	if eviction.Status.ObservedGeneration == nil {
		// wait for the evictionrequest-controller first
		return nil, nil
	}

	// check that this responder is designated active by the evictionrequest-controller
	targetResponderStatus := findTargetResponderStatus(eviction)
	if targetResponderStatus == nil {
		return nil, nil
	}
	switch targetResponderStatus.State {
	case lifecyclev1alpha1.ResponderStateActive:
	default:
		return nil, nil
	}

	lastResponderStatus := findResponderStatus(eviction)
	if lastResponderStatus == nil {
		// status have to be computed by the evictionrequest-controller first
		return nil, nil
	}
	if lastResponderStatus.CompletionTime != nil {
		// work done
		return nil, nil
	}

	podTarget := eviction.Spec.Target.Pod
	pod, err := c.podLister.Pods(namespace).Get(podTarget.Name)
	isNotFound := false
	if errors.IsNotFound(err) {
		isNotFound = true
	} else if err != nil {
		return nil, err
	}

	evictionCompleted := false
	evictionMessage := ""
	lastEvictionMessage := ptr.Deref(lastResponderStatus.Message, "")
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
		if strings.Contains(lastEvictionMessage, "pod has been been marked for deletion via the /eviction subresource") {
			reason = " via the /eviction subresource"
		}
		evictionMessage = fmt.Sprintf("%q pod has been been marked for deletion%v and is being terminated gracefully", pod.Name, reason)
		expectedCompletionTime = new(pod.DeletionTimestamp.Time)
		// not complete - schedule heartbeat
		addAfter = new(c.heartbeatMaxDurationBetweenUpdates)
	case apiv1pod.IsPodTerminal(pod):
		evictionCompleted = true
		evictionMessage = fmt.Sprintf("%q pod has been fully terminated (pod phase=%q)", pod.Name, pod.Status.Phase)
	default:
		evictionMessage, addAfter = c.evict(ctx, pod, lastEvictionMessage)
	}

	// Use the same time tick to report time in the status fields
	now := c.clock.Now()

	// The controller has to report a heartbeat every three minutes to indicate that it is running properly,
	// even if the evict backoff period is longer.
	if addAfter != nil && *addAfter > c.heartbeatMaxDurationBetweenUpdates {
		addAfter = new(c.heartbeatMaxDurationBetweenUpdates)
	}

	// API validation constraints
	if expectedCompletionTime != nil && expectedCompletionTime.Before(now) {
		expectedCompletionTime = new(now)
	}
	if len(evictionMessage) > 4000 {
		evictionMessage = evictionMessage[:4000]
	}

	// Update status/

	shouldUpdateHeartbeat := lastResponderStatus.HeartbeatTime == nil || lastResponderStatus.HeartbeatTime.Time.Before(now.Add(-c.heartbeatMaxDurationBetweenUpdates+time.Second))
	shouldUpdateExpectedCompletionTime := expectedCompletionTime != nil && !new(metav1.Time{Time: *expectedCompletionTime}).Equal(lastResponderStatus.ExpectedCompletionTime)
	shouldUpdateCompletionTime := evictionCompleted && lastResponderStatus.CompletionTime == nil

	if shouldUpdateHeartbeat || shouldUpdateExpectedCompletionTime || shouldUpdateCompletionTime || lastEvictionMessage != evictionMessage {
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
		statusApplyUpdate := lifecycleapplyv1alpha1.Eviction(eviction.Name, eviction.Namespace).
			WithStatus(lifecycleapplyv1alpha1.EvictionStatus().WithResponders(newResponderStatus))
		_, err = c.kubeClient.LifecycleV1alpha1().Evictions(eviction.Namespace).ApplyStatus(ctx, statusApplyUpdate, metav1.ApplyOptions{
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
		if attempts > 0 {
			// the first backoff starts with 0
			attempts--
		}
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
				UID: new(pod.UID),
			},
		},
	})

	if err != nil {
		oldAttempts := getRecordedAttempts(lastEvictionMessage)
		c.lastEvictionAttempts.set(pod.UID, c.clock.Now())
		// 10 minutes max backoff
		addAfter := exponentialBackoff(c.heartbeatMinDurationBetweenUpdates, c.maxImperativeEvictionBackoff, oldAttempts)
		// use suggested server delay if present
		if delay, shouldDelay := errors.SuggestsClientDelay(err); shouldDelay && (addAfter < time.Second*time.Duration(delay)) {
			addAfter = time.Second * time.Duration(delay)
		}
		attempts := oldAttempts + 1
		evictionMessage := fmt.Sprintf("%q pod deletion via the /eviction subresource failed (attempts=%d): %v", pod.Name, attempts, err)
		return evictionMessage, &addAfter
	}

	c.lastEvictionAttempts.remove(pod.UID)
	evictionMessage := fmt.Sprintf("%q pod has been been marked for deletion via the /eviction subresource and will be terminated gracefully", pod.Name)
	return evictionMessage, nil
}

// listActiveEvictionsForPod returns a list with Evictions whose eviction.Spec.Target.Pod.UID equal to the given Pod UID
// and are active.
// Returns an error if there is more than 1 active Eviction in the returned list, but also the list with evictions.
func (c *ImperativeEvictionResponderController) listActiveEvictionsForPod(pod *v1.Pod) ([]*lifecyclev1alpha1.Eviction, error) {
	all, err := c.evictionIndexer.ByIndex(evictionByPodUIDIndexKey, string(pod.UID))
	if err != nil {
		return nil, err
	}
	evictions := make([]*lifecyclev1alpha1.Eviction, 0, len(all))
	for _, ev := range all {
		eviction, ok := ev.(*lifecyclev1alpha1.Eviction)
		if !ok {
			continue
		}
		if eviction.Status.ObservedGeneration == nil {
			// wait for the evictionrequest-controller first - will get queued by the Eviction EventHandler
			return nil, nil
		}
		if !shouldHandleEviction(eviction) {
			continue
		}
		// Usually, there should be only a single or no Eviction per Pod. If there are more, only one should be active
		// and have an active responder.
		evictions = append(evictions, eviction)
	}
	if len(evictions) > 1 {
		return evictions, fmt.Errorf("found more than 1 active eviction matching pod %s, evictions %d", klog.KObj(pod).String(), len(evictions))
	}
	return evictions, nil
}

func evictionKey(eviction metav1.ObjectMetaAccessor) string {
	return eviction.GetObjectMeta().GetNamespace() + "/" + eviction.GetObjectMeta().GetName()
}
