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

package tainteviction

import (
	"context"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	corev1informers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	apipod "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/controller/tainteviction/metrics"
	controllerutil "k8s.io/kubernetes/pkg/controller/util/node"
	utilpod "k8s.io/kubernetes/pkg/util/pod"
)

const (
	// TODO (k82cn): Figure out a reasonable number of workers/channels and propagate
	// the number of workers up making it a parameter of Run() function.

	// NodeUpdateChannelSize defines the size of channel for node update events.
	NodeUpdateChannelSize = 10
	// UpdateWorkerSize defines the size of workers for node update or/and pod update.
	UpdateWorkerSize     = 8
	podUpdateChannelSize = 1
	retries              = 5
)

type nodeUpdateItem struct {
	nodeName string
}

type podUpdateItem struct {
	podName      string
	podNamespace string
	nodeName     string
}

// podEvictionItem represents a single pod eviction that is waiting in the
// durable rate-limited retry queue (podEvictionQueue). It carries the
// original createdAt/fireAt timestamps from the timed worker that produced
// it so that processPodEvictionRetry can determine whether the original
// toleration window has already expired.
type podEvictionItem struct {
	podRef    NamespacedObject
	createdAt time.Time // when the timed worker was originally created
	fireAt    time.Time // when the timed worker was scheduled to fire
	nodeName  string    // node whose taints produced the original decision
	taintSet  string    // NoExecute taints that produced the original decision
}

type podEvictionState struct {
	item       podEvictionItem
	generation uint64
}

// podEvictionDecisionKind is the outcome of evaluating a pod against the
// current set of NoExecute taints on its node.
type podEvictionDecisionKind int

const (
	// podEvictionNone means the pod tolerates all taints indefinitely; no
	// eviction action is needed.
	podEvictionNone podEvictionDecisionKind = iota
	// podEvictionNow means the pod does not tolerate at least one taint and
	// must be deleted immediately.
	podEvictionNow
	// podEvictionLater means the pod tolerates all taints but only for a
	// finite TolerationSeconds window; deletion should be scheduled for when
	// that window expires.
	podEvictionLater
)

type podEvictionDecision struct {
	kind            podEvictionDecisionKind
	startTime       time.Time
	triggerTime     time.Time
	tolerationTime  time.Duration
	taintSet        string
	keepExisting    bool
	cancelScheduled bool
}

func taintSetKey(taints []v1.Taint) string {
	identities := make([]string, 0, len(taints))
	for i := range taints {
		timeAdded := "<nil>"
		if taints[i].TimeAdded != nil {
			timeAdded = taints[i].TimeAdded.Time.UTC().Format(time.RFC3339Nano)
		}
		identities = append(identities, fmt.Sprintf("%q|%q|%q|%q", taints[i].Key, taints[i].Value, taints[i].Effect, timeAdded))
	}
	sort.Strings(identities)
	return strings.Join(identities, "\n")
}

func finiteTaintSetKey(taints []v1.Taint, tolerations []v1.Toleration) string {
	finiteTaints := make([]v1.Taint, 0, len(taints))
	for i := range taints {
		if tolerations[i].TolerationSeconds != nil && *tolerations[i].TolerationSeconds > 0 {
			finiteTaints = append(finiteTaints, taints[i])
		}
	}
	return taintSetKey(finiteTaints)
}

func taintSetsOverlap(left, right string) bool {
	// Empty values are produced only by callers that predate eviction-context
	// tracking. Treat them conservatively as the same decision.
	if left == "" || right == "" || left == right {
		return true
	}
	leftTaints := strings.Split(left, "\n")
	rightTaints := strings.Split(right, "\n")
	i, j := 0, 0
	for i < len(leftTaints) && j < len(rightTaints) {
		switch strings.Compare(leftTaints[i], rightTaints[j]) {
		case 0:
			return true
		case -1:
			i++
		default:
			j++
		}
	}
	return false
}

func hash(val string, max int) int {
	hasher := fnv.New32a()
	io.WriteString(hasher, val)
	return int(hasher.Sum32() % uint32(max))
}

// GetPodsByNodeNameFunc returns the list of pods assigned to the specified node.
type GetPodsByNodeNameFunc func(nodeName string) ([]*v1.Pod, error)

// Controller listens to Taint/Toleration changes and is responsible for removing Pods
// from Nodes tainted with NoExecute Taints.
type Controller struct {
	name string

	client                clientset.Interface
	broadcaster           record.EventBroadcaster
	recorder              record.EventRecorder
	podLister             corelisters.PodLister
	podListerSynced       cache.InformerSynced
	nodeLister            corelisters.NodeLister
	nodeListerSynced      cache.InformerSynced
	getPodsAssignedToNode GetPodsByNodeNameFunc

	taintEvictionQueue *TimedWorkerQueue
	// keeps a map from nodeName to all noExecute taints on that Node
	taintedNodesLock sync.Mutex
	taintedNodes     map[string][]v1.Taint

	nodeUpdateChannels []chan nodeUpdateItem
	podUpdateChannels  []chan podUpdateItem

	nodeUpdateQueue  workqueue.TypedInterface[nodeUpdateItem]
	podUpdateQueue   workqueue.TypedInterface[podUpdateItem]
	podEvictionQueue workqueue.TypedRateLimitingInterface[podEvictionItem]
	podEvictionLock  sync.Mutex
	// podEvictionTokens is the authoritative set of pod eviction retries
	// currently in flight. It is keyed by NamespacedName.String() and
	// guarded by podEvictionLock. The generation prevents a worker from
	// relinquishing ownership after an update requested reconciliation.
	podEvictionTokens map[string]podEvictionState
}

func (tc *Controller) deletePodHandler() func(ctx context.Context, fireAt time.Time, args *WorkArgs) error {
	return func(ctx context.Context, fireAt time.Time, args *WorkArgs) error {
		klog.FromContext(ctx).Info("Deleting pod", "controller", tc.name, "pod", args.Object)
		tc.emitPodDeletionEvent(args.Object.NamespacedName)

		var err error
		for i := 0; i < retries; i++ {
			var deleted bool
			deleted, err = tc.addConditionAndDeletePod(ctx, args.Object)
			if err == nil {
				if deleted {
					metrics.PodDeletionsTotal.Inc()
					metrics.PodDeletionsLatency.Observe(time.Since(fireAt).Seconds())
				}
				return nil
			}
			time.Sleep(10 * time.Millisecond)
		}
		tc.addPodEvictionRetry(args.Object, args.CreatedAt, fireAt, args.nodeName, args.taintSet)
		return err
	}
}

// addConditionAndDeletePod sets the DisruptionTarget pod condition and then
// deletes the pod identified by podRef. It returns (true, nil) when the pod
// was successfully deleted, (false, nil) when the pod was already gone or
// otherwise does not need deletion (wrong UID, already terminating), and
// (false, err) on a retryable failure. A NotFound error on the status Patch or
// Delete call is treated as a clean success — a concurrent deletion raced us
// to it.
func (tc *Controller) addConditionAndDeletePod(ctx context.Context, podRef NamespacedObject) (bool, error) {
	pod, err := tc.client.CoreV1().Pods(podRef.Namespace).Get(ctx, podRef.Name, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if podRef.UID != "" && pod.UID != podRef.UID {
		return false, nil
	}
	if pod.DeletionTimestamp != nil {
		return false, nil
	}

	newStatus := pod.Status.DeepCopy()
	updated := apipod.UpdatePodCondition(newStatus, &v1.PodCondition{
		Type:               v1.DisruptionTarget,
		ObservedGeneration: apipod.CalculatePodConditionObservedGeneration(&pod.Status, pod.Generation, v1.DisruptionTarget),
		Status:             v1.ConditionTrue,
		Reason:             "DeletionByTaintManager",
		Message:            "Taint manager: deleting due to NoExecute taint",
	})
	if updated {
		if _, _, _, err := utilpod.PatchPodStatus(ctx, tc.client, pod.Namespace, pod.Name, pod.UID, pod.Status, *newStatus); err != nil {
			if apierrors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
	}
	deleteOptions := metav1.DeleteOptions{}
	if podRef.UID != "" {
		deleteOptions.Preconditions = &metav1.Preconditions{UID: &podRef.UID}
	}
	if err := tc.client.CoreV1().Pods(podRef.Namespace).Delete(ctx, podRef.Name, deleteOptions); err != nil {
		// A concurrent deletion between our Get and Delete means the pod is already
		// gone; treat it as a success rather than a retryable failure.
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// addPodEvictionRetry registers a durable eviction retry for podRef and
// enqueues it in the rate-limited podEvictionQueue. createdAt and fireAt
// are the original timestamps from the timed worker that exhausted its
// burst attempts; they are stored so that the retry path can determine
// whether the pod's toleration window has already expired.
//
// If a newer retry is already registered for the same pod (higher createdAt),
// the call is a no-op and returns (item, false). Otherwise the new item
// replaces any stale entry and returns (item, true).
func (tc *Controller) addPodEvictionRetry(podRef NamespacedObject, createdAt, fireAt time.Time, nodeName, taintSet string) (podEvictionItem, bool) {
	key := podRef.NamespacedName.String()
	item := podEvictionItem{podRef: podRef, createdAt: createdAt, fireAt: fireAt, nodeName: nodeName, taintSet: taintSet}
	tc.podEvictionLock.Lock()
	current, exists := tc.podEvictionTokens[key]
	if exists && item.createdAt.Before(current.item.createdAt) {
		tc.podEvictionLock.Unlock()
		return item, false
	}
	generation := uint64(1)
	if exists {
		generation = current.generation + 1
	}
	tc.podEvictionTokens[key] = podEvictionState{item: item, generation: generation}
	tc.podEvictionLock.Unlock()

	tc.podEvictionQueue.AddRateLimited(item)
	return item, true
}

// podEvictionRetryMatches reports whether item is still the current,
// authoritative retry token for its pod. An item that has been superseded
// by a newer producer or explicitly cancelled returns false and must be
// discarded by the worker without performing any deletion.
func (tc *Controller) podEvictionRetryMatches(item podEvictionItem) bool {
	_, matches := tc.podEvictionRetryGeneration(item)
	return matches
}

func (tc *Controller) podEvictionRetryGeneration(item podEvictionItem) (uint64, bool) {
	key := item.podRef.NamespacedName.String()
	tc.podEvictionLock.Lock()
	defer tc.podEvictionLock.Unlock()
	current, ok := tc.podEvictionTokens[key]
	return current.generation, ok && current.item == item
}

// requeuePodEvictionRetryForPod records that an update needs the current retry
// to evaluate the pod again. Incrementing the generation prevents a worker
// which already captured older state from relinquishing the retry.
func (tc *Controller) requeuePodEvictionRetryForPod(podRef NamespacedObject) bool {
	key := podRef.NamespacedName.String()
	tc.podEvictionLock.Lock()
	current, ok := tc.podEvictionTokens[key]
	if !ok || current.item.podRef != podRef {
		tc.podEvictionLock.Unlock()
		return false
	}
	current.generation++
	tc.podEvictionTokens[key] = current
	tc.podEvictionLock.Unlock()
	tc.podEvictionQueue.Add(current.item)
	return true
}

// forgetPodEvictionRetry removes item from podEvictionTokens if and only if
// it is still the current token and no update requested another reconciliation.
func (tc *Controller) forgetPodEvictionRetry(item podEvictionItem, generation uint64) bool {
	key := item.podRef.NamespacedName.String()
	tc.podEvictionLock.Lock()
	defer tc.podEvictionLock.Unlock()
	current, ok := tc.podEvictionTokens[key]
	if ok && current.item == item && current.generation == generation {
		delete(tc.podEvictionTokens, key)
		return true
	}
	return false
}

func getNoExecuteTaints(taints []v1.Taint) []v1.Taint {
	result := []v1.Taint{}
	for i := range taints {
		if taints[i].Effect == v1.TaintEffectNoExecute {
			result = append(result, taints[i])
		}
	}
	return result
}

// getMinTolerationTime returns minimal toleration time from the given slice, or -1 if it's infinite.
func getMinTolerationTime(tolerations []v1.Toleration) time.Duration {
	minTolerationTime := int64(math.MaxInt64)
	if len(tolerations) == 0 {
		return 0
	}

	for i := range tolerations {
		if tolerations[i].TolerationSeconds != nil {
			tolerationSeconds := *(tolerations[i].TolerationSeconds)
			if tolerationSeconds <= 0 {
				return 0
			} else if tolerationSeconds < minTolerationTime {
				minTolerationTime = tolerationSeconds
			}
		}
	}

	if minTolerationTime == int64(math.MaxInt64) {
		return -1
	}
	return time.Duration(minTolerationTime) * time.Second
}

// New creates a new Controller that will use passed clientset to communicate with the API server.
func New(ctx context.Context, c clientset.Interface, podInformer corev1informers.PodInformer, nodeInformer corev1informers.NodeInformer, controllerName string) (*Controller, error) {
	logger := klog.FromContext(ctx)
	metrics.Register()
	eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: controllerName})

	podIndexer := podInformer.Informer().GetIndexer()

	tm := &Controller{
		name: controllerName,

		client:           c,
		broadcaster:      eventBroadcaster,
		recorder:         recorder,
		podLister:        podInformer.Lister(),
		podListerSynced:  podInformer.Informer().HasSynced,
		nodeLister:       nodeInformer.Lister(),
		nodeListerSynced: nodeInformer.Informer().HasSynced,
		getPodsAssignedToNode: func(nodeName string) ([]*v1.Pod, error) {
			objs, err := podIndexer.ByIndex("spec.nodeName", nodeName)
			if err != nil {
				return nil, err
			}
			pods := make([]*v1.Pod, 0, len(objs))
			for _, obj := range objs {
				pod, ok := obj.(*v1.Pod)
				if !ok {
					continue
				}
				pods = append(pods, pod)
			}
			return pods, nil
		},
		taintedNodes:      make(map[string][]v1.Taint),
		podEvictionTokens: make(map[string]podEvictionState),

		nodeUpdateQueue: workqueue.NewTypedWithConfig(workqueue.TypedQueueConfig[nodeUpdateItem]{Name: "noexec_taint_node"}),
		podUpdateQueue:  workqueue.NewTypedWithConfig(workqueue.TypedQueueConfig[podUpdateItem]{Name: "noexec_taint_pod"}),
		podEvictionQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[podEvictionItem](),
			workqueue.TypedRateLimitingQueueConfig[podEvictionItem]{Name: "noexec_taint_pod_eviction"},
		),
	}
	tm.taintEvictionQueue = CreateWorkerQueue(tm.deletePodHandler())

	_, err := podInformer.Informer().AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			pod := obj.(*v1.Pod)
			tm.PodUpdated(nil, pod)
		},
		UpdateFunc: func(prev, obj interface{}) {
			prevPod := prev.(*v1.Pod)
			newPod := obj.(*v1.Pod)
			tm.PodUpdated(prevPod, newPod)
		},
		DeleteFunc: func(obj interface{}) {
			pod, isPod := obj.(*v1.Pod)
			// We can get DeletedFinalStateUnknown instead of *v1.Pod here and we need to handle that correctly.
			if !isPod {
				deletedState, ok := obj.(cache.DeletedFinalStateUnknown)
				if !ok {
					logger.Error(nil, "Received unexpected object", "object", obj)
					return
				}
				pod, ok = deletedState.Obj.(*v1.Pod)
				if !ok {
					logger.Error(nil, "DeletedFinalStateUnknown contained non-Pod object", "object", deletedState.Obj)
					return
				}
			}
			tm.PodUpdated(pod, nil)
		},
	}, cache.HandlerOptions{Logger: &logger})
	if err != nil {
		return nil, fmt.Errorf("unable to add pod event handler: %w", err)
	}

	_, err = nodeInformer.Informer().AddEventHandlerWithOptions(cache.ResourceEventHandlerFuncs{
		AddFunc: controllerutil.CreateAddNodeHandler(func(node *v1.Node) error {
			tm.NodeUpdated(nil, node)
			return nil
		}),
		UpdateFunc: controllerutil.CreateUpdateNodeHandler(func(oldNode, newNode *v1.Node) error {
			tm.NodeUpdated(oldNode, newNode)
			return nil
		}),
		DeleteFunc: controllerutil.CreateDeleteNodeHandler(logger, func(node *v1.Node) error {
			tm.NodeUpdated(node, nil)
			return nil
		}),
	}, cache.HandlerOptions{Logger: &logger})
	if err != nil {
		return nil, fmt.Errorf("unable to add node event handler: %w", err)
	}

	return tm, nil
}

// Run starts the controller which will run in loop until `stopCh` is closed.
func (tc *Controller) Run(ctx context.Context) {
	defer utilruntime.HandleCrashWithContext(ctx)

	logger := klog.FromContext(ctx)
	logger.Info("Starting", "controller", tc.name)

	// Start events processing pipeline.
	tc.broadcaster.StartStructuredLogging(3)
	tc.broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: tc.client.CoreV1().Events("")})
	logger.Info("Sending events to API server")
	defer tc.broadcaster.Shutdown()

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down controller", "controller", tc.name)
		tc.nodeUpdateQueue.ShutDown()
		tc.podUpdateQueue.ShutDown()
		tc.podEvictionQueue.ShutDown()
		tc.taintEvictionQueue.CancelAndWait()
		wg.Wait()
	}()

	// wait for the cache to be synced
	if !cache.WaitForNamedCacheSyncWithContext(ctx, tc.podListerSynced, tc.nodeListerSynced) {
		return
	}

	for i := 0; i < UpdateWorkerSize; i++ {
		tc.nodeUpdateChannels = append(tc.nodeUpdateChannels, make(chan nodeUpdateItem, NodeUpdateChannelSize))
		tc.podUpdateChannels = append(tc.podUpdateChannels, make(chan podUpdateItem, podUpdateChannelSize))
	}

	// Functions that are responsible for taking work items out of the workqueues and putting them into channels.
	wg.Go(func() {
		for {
			nodeUpdate, shutdown := tc.nodeUpdateQueue.Get()
			if shutdown {
				break
			}
			hash := hash(nodeUpdate.nodeName, UpdateWorkerSize)
			select {
			case <-ctx.Done():
				tc.nodeUpdateQueue.Done(nodeUpdate)
				return
			case tc.nodeUpdateChannels[hash] <- nodeUpdate:
				// tc.nodeUpdateQueue.Done is called by the nodeUpdateChannels worker
			}
		}
	})

	wg.Go(func() {
		for {
			podUpdate, shutdown := tc.podUpdateQueue.Get()
			if shutdown {
				break
			}
			// The fact that pods are processed by the same worker as nodes is used to avoid races
			// between node worker setting tc.taintedNodes and pod worker reading this to decide
			// whether to delete pod.
			// It's possible that even without this assumption this code is still correct.
			hash := hash(podUpdate.nodeName, UpdateWorkerSize)
			select {
			case <-ctx.Done():
				tc.podUpdateQueue.Done(podUpdate)
				return
			case tc.podUpdateChannels[hash] <- podUpdate:
				// tc.podUpdateQueue.Done is called by the podUpdateChannels worker
			}
		}
	})

	for i := 0; i < UpdateWorkerSize; i++ {
		wg.Go(func() {
			tc.worker(ctx, i)
		})
	}
	for range UpdateWorkerSize {
		wg.Go(func() {
			tc.podEvictionWorker(ctx)
		})
	}
	<-ctx.Done()
}

// podEvictionWorker drains podEvictionQueue. For each item it verifies that
// the retry token is still current (via podEvictionRetryMatches) before
// delegating to processPodEvictionRetry. Stale items are discarded without
// any deletion attempt. Failures are re-queued with rate limiting so that
// the retry respects the configured back-off.
func (tc *Controller) podEvictionWorker(ctx context.Context) {
	logger := klog.FromContext(ctx)
	for {
		item, shutdown := tc.podEvictionQueue.Get()
		if shutdown {
			return
		}

		func() {
			defer tc.podEvictionQueue.Done(item)

			generation, matches := tc.podEvictionRetryGeneration(item)
			if !matches {
				tc.podEvictionQueue.Forget(item)
				return
			}

			if err := tc.processPodEvictionRetry(ctx, item); err != nil {
				logger.V(3).Info("Pod eviction failed, will retry", "pod", item.podRef, "err", err)
				currentGeneration, current := tc.podEvictionRetryGeneration(item)
				switch {
				case !current:
					tc.podEvictionQueue.Forget(item)
				case currentGeneration == generation:
					tc.podEvictionQueue.AddRateLimited(item)
					// A concurrent update already added this item for immediate
					// reconciliation. Keep its rate-limiter history intact.
				}
				return
			}
			if tc.forgetPodEvictionRetry(item, generation) {
				tc.podEvictionQueue.Forget(item)
			}
		}()
	}
}

// processPodEvictionRetry handles one retry attempt for item. It re-evaluates
// the pod's current state and toleration against the node's taints and takes
// one of the following actions:
//
//   - podEvictionNone / taint gone / pod vanished: complete the retry (no-op).
//   - podEvictionNow (pod does not tolerate taint): delete the pod directly.
//   - podEvictionLater, keepExisting (timed worker already scheduled): complete
//     the retry and let the timed worker handle the deadline.
//   - podEvictionLater, timed eviction caused by an original finite taint:
//     retain the original start time, deleting if the current deadline has
//     expired or scheduling only the remaining window.
//   - podEvictionLater, immediate-burst retry (item.fireAt == item.createdAt):
//     the pod may have gained a finite toleration since the burst failure.
//     Schedule a fresh timed worker via taintEvictionQueue.
//
// Returns a non-nil error only for transient failures that should be retried.
func (tc *Controller) processPodEvictionRetry(ctx context.Context, item podEvictionItem) error {
	logger := klog.FromContext(ctx)
	podRef := item.podRef
	pod, err := tc.podLister.Pods(podRef.Namespace).Get(podRef.Name)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if pod.UID != podRef.UID || pod.DeletionTimestamp != nil || pod.Spec.NodeName == "" {
		return nil
	}
	taints, ok := func() ([]v1.Taint, bool) {
		tc.taintedNodesLock.Lock()
		defer tc.taintedNodesLock.Unlock()
		taints, ok := tc.taintedNodes[pod.Spec.NodeName]
		return taints, ok
	}()
	if !ok || len(taints) == 0 {
		return nil
	}

	now := time.Now()
	decision := tc.getPodEvictionDecision(logger, podRef, pod.Spec.NodeName, pod.Spec.Tolerations, taints, now)
	switch decision.kind {
	case podEvictionNone:
		return nil
	case podEvictionLater:
		// A retry from a timed eviction retains its original start time while at
		// least one finite taint from that decision remains. A replacement taint
		// gets a fresh window; a changed duration for the original taint does not.
		// Immediate-burst retries fall through with the current decision's start.
		if item.fireAt.After(item.createdAt) && (item.nodeName == "" || item.nodeName == pod.Spec.NodeName) && taintSetsOverlap(item.taintSet, decision.taintSet) {
			decision.startTime = item.createdAt
			decision.triggerTime = item.createdAt.Add(decision.tolerationTime)
			if !decision.triggerTime.After(item.fireAt) || !decision.triggerTime.After(now) {
				deleted, err := tc.addConditionAndDeletePod(ctx, podRef)
				if err != nil {
					return err
				}
				if deleted {
					metrics.PodDeletionsTotal.Inc()
					metrics.PodDeletionsLatency.Observe(time.Since(item.fireAt).Seconds())
				}
				return nil
			}
			decision.keepExisting = false
		}
		if decision.keepExisting {
			return nil
		}
		if tc.taintEvictionQueue.AddWork(ctx, newPodEvictionWorkArgs(podRef.Name, podRef.Namespace, podRef.UID, pod.Spec.NodeName, decision.taintSet), decision.startTime, decision.triggerTime) {
			return nil
		}
		return fmt.Errorf("pod eviction retry for %s found conflicting scheduled work", podRef)
	case podEvictionNow:
		deleted, err := tc.addConditionAndDeletePod(ctx, podRef)
		if err != nil {
			return err
		}
		if deleted {
			metrics.PodDeletionsTotal.Inc()
			metrics.PodDeletionsLatency.Observe(time.Since(item.fireAt).Seconds())
		}
		return nil
	default:
		utilruntime.HandleError(fmt.Errorf("unexpected pod eviction decision for %s: %d", podRef, decision.kind))
		return nil
	}
}

func (tc *Controller) worker(ctx context.Context, worker int) {
	// When processing events we want to prioritize Node updates over Pod updates,
	// as NodeUpdates that interest the controller should be handled as soon as possible -
	// we don't want user (or system) to wait until PodUpdate queue is drained before it can
	// start evicting Pods from tainted Nodes.
	for {
		select {
		case <-ctx.Done():
			return
		case nodeUpdate := <-tc.nodeUpdateChannels[worker]:
			tc.handleNodeUpdate(ctx, nodeUpdate)
			tc.nodeUpdateQueue.Done(nodeUpdate)
		case podUpdate := <-tc.podUpdateChannels[worker]:
			// If we found a Pod update we need to empty Node queue first.
		priority:
			for {
				select {
				case nodeUpdate := <-tc.nodeUpdateChannels[worker]:
					tc.handleNodeUpdate(ctx, nodeUpdate)
					tc.nodeUpdateQueue.Done(nodeUpdate)
				default:
					break priority
				}
			}
			// After Node queue is emptied we process podUpdate.
			tc.handlePodUpdate(ctx, podUpdate)
			tc.podUpdateQueue.Done(podUpdate)
		}
	}
}

// PodUpdated is used to notify the controller about Pod changes.
// oldPod is nil for pod additions; newPod is nil for pod deletions.
// The call is a no-op when neither the pod's tolerations nor its node name
// have changed, to avoid spurious queue entries on unrelated updates.
func (tc *Controller) PodUpdated(oldPod *v1.Pod, newPod *v1.Pod) {
	podName := ""
	podNamespace := ""
	nodeName := ""
	oldTolerations := []v1.Toleration{}
	if oldPod != nil {
		podName = oldPod.Name
		podNamespace = oldPod.Namespace
		nodeName = oldPod.Spec.NodeName
		oldTolerations = oldPod.Spec.Tolerations
	}
	newTolerations := []v1.Toleration{}
	if newPod != nil {
		podName = newPod.Name
		podNamespace = newPod.Namespace
		nodeName = newPod.Spec.NodeName
		newTolerations = newPod.Spec.Tolerations
	}

	if oldPod != nil && newPod != nil && helper.Semantic.DeepEqual(oldTolerations, newTolerations) && oldPod.Spec.NodeName == newPod.Spec.NodeName {
		return
	}
	updateItem := podUpdateItem{
		podName:      podName,
		podNamespace: podNamespace,
		nodeName:     nodeName,
	}

	tc.podUpdateQueue.Add(updateItem)
}

// NodeUpdated is used to notify the controller about Node changes.
// oldNode is nil for node additions; newNode is nil for node deletions.
// The call is a no-op when the set of NoExecute taints has not changed.
func (tc *Controller) NodeUpdated(oldNode *v1.Node, newNode *v1.Node) {
	nodeName := ""
	oldTaints := []v1.Taint{}
	if oldNode != nil {
		nodeName = oldNode.Name
		oldTaints = getNoExecuteTaints(oldNode.Spec.Taints)
	}

	newTaints := []v1.Taint{}
	if newNode != nil {
		nodeName = newNode.Name
		newTaints = getNoExecuteTaints(newNode.Spec.Taints)
	}

	if oldNode != nil && newNode != nil && helper.Semantic.DeepEqual(oldTaints, newTaints) {
		return
	}
	updateItem := nodeUpdateItem{
		nodeName: nodeName,
	}

	tc.nodeUpdateQueue.Add(updateItem)
}

func (tc *Controller) cancelWorkWithEvent(logger klog.Logger, nsName types.NamespacedName) {
	cancelledTimedWork, cancelledRetry := tc.cancelWork(logger, nsName)
	if cancelledTimedWork || cancelledRetry {
		tc.emitCancelPodDeletionEvent(nsName)
	}
}

func (tc *Controller) cancelWork(logger klog.Logger, nsName types.NamespacedName) (bool, bool) {
	tc.podEvictionLock.Lock()
	defer tc.podEvictionLock.Unlock()
	cancelledTimedWork := tc.taintEvictionQueue.CancelWork(logger, nsName.String())
	_, cancelledRetry := tc.podEvictionTokens[nsName.String()]
	delete(tc.podEvictionTokens, nsName.String())
	return cancelledTimedWork, cancelledRetry
}

// replacePodEvictionWork atomically transfers retry ownership to authoritative
// timed work produced by a Pod or Node update. A timed callback that fails
// cannot register its retry until the replacement is visible.
func (tc *Controller) replacePodEvictionWork(ctx context.Context, args *WorkArgs, createdAt, fireAt time.Time, force bool) (bool, bool) {
	key := args.Object.NamespacedName.String()
	tc.podEvictionLock.Lock()
	defer tc.podEvictionLock.Unlock()
	cancelledTimedWork := false
	if force {
		cancelledTimedWork = tc.taintEvictionQueue.CancelWork(klog.FromContext(ctx), key)
	}
	updatedTimedWork := tc.taintEvictionQueue.UpdateWork(ctx, args, createdAt, fireAt)
	_, cancelledRetry := tc.podEvictionTokens[key]
	delete(tc.podEvictionTokens, key)
	return cancelledTimedWork || updatedTimedWork, cancelledRetry
}

func (tc *Controller) getPodEvictionDecision(logger klog.Logger, podRef NamespacedObject, nodeName string, tolerations []v1.Toleration, taints []v1.Taint, now time.Time) podEvictionDecision {
	taintSet := taintSetKey(taints)
	if len(taints) == 0 {
		return podEvictionDecision{kind: podEvictionNone, taintSet: taintSet}
	}
	allTolerated, usedTolerations := v1helper.GetMatchingTolerations(logger, taints, tolerations)
	if !allTolerated {
		return podEvictionDecision{kind: podEvictionNow, taintSet: taintSet}
	}
	minTolerationTime := getMinTolerationTime(usedTolerations)
	// getMinTolerationTime returns negative value to denote infinite toleration.
	if minTolerationTime < 0 {
		return podEvictionDecision{kind: podEvictionNone, taintSet: taintSet}
	}
	if minTolerationTime == 0 {
		return podEvictionDecision{kind: podEvictionNow, taintSet: taintSet}
	}
	taintSet = finiteTaintSetKey(taints, usedTolerations)

	startTime := now
	triggerTime := startTime.Add(minTolerationTime)
	scheduledEviction := tc.taintEvictionQueue.GetWorkerUnsafe(podRef.NamespacedName.String())
	if scheduledEviction == nil {
		return podEvictionDecision{kind: podEvictionLater, startTime: startTime, triggerTime: triggerTime, tolerationTime: minTolerationTime, taintSet: taintSet}
	}
	if scheduledEviction.WorkItem.Object.UID != podRef.UID {
		return podEvictionDecision{kind: podEvictionLater, startTime: startTime, triggerTime: triggerTime, tolerationTime: minTolerationTime, taintSet: taintSet, cancelScheduled: true}
	}
	if scheduledEviction.WorkItem.nodeName != "" && scheduledEviction.WorkItem.nodeName != nodeName {
		return podEvictionDecision{kind: podEvictionLater, startTime: startTime, triggerTime: triggerTime, tolerationTime: minTolerationTime, taintSet: taintSet, cancelScheduled: true}
	}
	if !taintSetsOverlap(scheduledEviction.WorkItem.taintSet, taintSet) {
		return podEvictionDecision{kind: podEvictionLater, startTime: startTime, triggerTime: triggerTime, tolerationTime: minTolerationTime, taintSet: taintSet, cancelScheduled: true}
	}

	startTime = scheduledEviction.CreatedAt
	if startTime.Add(minTolerationTime).Before(triggerTime) {
		return podEvictionDecision{kind: podEvictionLater, startTime: startTime, triggerTime: scheduledEviction.FireAt, tolerationTime: minTolerationTime, taintSet: taintSet, keepExisting: true}
	}
	return podEvictionDecision{kind: podEvictionLater, startTime: startTime, triggerTime: triggerTime, tolerationTime: minTolerationTime, taintSet: taintSet, cancelScheduled: true}
}

func (tc *Controller) processPodOnNode(
	ctx context.Context,
	podRef NamespacedObject,
	nodeName string,
	tolerations []v1.Toleration,
	taints []v1.Taint,
	now time.Time,
) {
	logger := klog.FromContext(ctx)
	podNamespacedName := podRef.NamespacedName
	decision := tc.getPodEvictionDecision(logger, podRef, nodeName, tolerations, taints, now)
	switch decision.kind {
	case podEvictionNone:
		logger.V(4).Info("Current tolerations for pod tolerate forever or node has no taints, cancelling any scheduled deletion", "pod", podNamespacedName.String())
		tc.cancelWorkWithEvent(logger, podNamespacedName)
		return
	case podEvictionNow:
		logger.V(2).Info("Not all taints are tolerated after update for pod on node", "pod", podNamespacedName.String(), "node", klog.KRef("", nodeName))
		if tc.requeuePodEvictionRetryForPod(podRef) {
			return
		}
		updatedTimedWork, cancelledRetry := tc.replacePodEvictionWork(ctx, newPodEvictionWorkArgs(podNamespacedName.Name, podNamespacedName.Namespace, podRef.UID, nodeName, decision.taintSet), now, now, false)
		if updatedTimedWork || cancelledRetry {
			tc.emitCancelPodDeletionEvent(podNamespacedName)
		}
		return
	case podEvictionLater:
		// Let an existing retry reconcile the latest state. Its generation keeps
		// a worker evaluating an older snapshot from relinquishing ownership.
		if tc.requeuePodEvictionRetryForPod(podRef) {
			return
		}
		// A keepExisting decision may refer to a fired worker whose callback is
		// about to leave. Refresh it when no durable retry owns the pod.
		updatedTimedWork, cancelledRetry := tc.replacePodEvictionWork(ctx, newPodEvictionWorkArgs(podNamespacedName.Name, podNamespacedName.Namespace, podRef.UID, nodeName, decision.taintSet), decision.startTime, decision.triggerTime, decision.keepExisting)
		if decision.cancelScheduled && (updatedTimedWork || cancelledRetry) {
			tc.emitCancelPodDeletionEvent(podNamespacedName)
		}
	default:
		utilruntime.HandleError(fmt.Errorf("unexpected pod eviction decision for %s: %d", podRef, decision.kind))
	}
}

func (tc *Controller) handlePodUpdate(ctx context.Context, podUpdate podUpdateItem) {
	pod, err := tc.podLister.Pods(podUpdate.podNamespace).Get(podUpdate.podName)
	logger := klog.FromContext(ctx)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// Delete
			podNamespacedName := types.NamespacedName{Namespace: podUpdate.podNamespace, Name: podUpdate.podName}
			logger.V(4).Info("Noticed pod deletion", "pod", podNamespacedName)
			tc.cancelWorkWithEvent(logger, podNamespacedName)
			return
		}
		utilruntime.HandleError(fmt.Errorf("could not get pod %s/%s: %v", podUpdate.podName, podUpdate.podNamespace, err))
		return
	}

	// We key the workqueue and shard workers by nodeName. If we don't match the current state we should not be the one processing the current object.
	if pod.Spec.NodeName != podUpdate.nodeName {
		return
	}

	// Create or Update
	podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
	logger.V(4).Info("Noticed pod update", "pod", podNamespacedName)
	nodeName := pod.Spec.NodeName
	if nodeName == "" {
		tc.cancelWorkWithEvent(logger, podNamespacedName)
		return
	}
	taints, ok := func() ([]v1.Taint, bool) {
		tc.taintedNodesLock.Lock()
		defer tc.taintedNodesLock.Unlock()
		taints, ok := tc.taintedNodes[nodeName]
		return taints, ok
	}()
	// It's possible that Node was deleted, or Taints were removed before, which triggered
	// eviction cancelling if it was needed.
	if !ok {
		tc.cancelWorkWithEvent(logger, podNamespacedName)
		return
	}
	tc.processPodOnNode(ctx, NamespacedObject{NamespacedName: podNamespacedName, UID: pod.UID}, nodeName, pod.Spec.Tolerations, taints, time.Now())
}

func (tc *Controller) handleNodeUpdate(ctx context.Context, nodeUpdate nodeUpdateItem) {
	node, err := tc.nodeLister.Get(nodeUpdate.nodeName)
	logger := klog.FromContext(ctx)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// Delete
			logger.V(4).Info("Noticed node deletion", "node", klog.KRef("", nodeUpdate.nodeName))
			tc.taintedNodesLock.Lock()
			defer tc.taintedNodesLock.Unlock()
			delete(tc.taintedNodes, nodeUpdate.nodeName)
			return
		}
		utilruntime.HandleError(fmt.Errorf("cannot get node %s: %v", nodeUpdate.nodeName, err))
		return
	}

	// Create or Update
	logger.V(4).Info("Noticed node update", "node", klog.KObj(node))
	taints := getNoExecuteTaints(node.Spec.Taints)
	func() {
		tc.taintedNodesLock.Lock()
		defer tc.taintedNodesLock.Unlock()
		logger.V(4).Info("Updating known taints on node", "node", klog.KObj(node), "taints", taints)
		if len(taints) == 0 {
			delete(tc.taintedNodes, node.Name)
		} else {
			tc.taintedNodes[node.Name] = taints
		}
	}()

	// This is critical that we update tc.taintedNodes before we call getPodsAssignedToNode:
	// getPodsAssignedToNode can be delayed as long as all future updates to pods will call
	// tc.PodUpdated which will use tc.taintedNodes to potentially delete delayed pods.
	pods, err := tc.getPodsAssignedToNode(node.Name)
	if err != nil {
		logger.Error(err, "Failed to get pods assigned to node", "node", klog.KObj(node))
		return
	}
	if len(pods) == 0 {
		return
	}
	// Short circuit, to make this controller a bit faster.
	if len(taints) == 0 {
		logger.V(4).Info("All taints were removed from the node. Cancelling all evictions...", "node", klog.KObj(node))
		for i := range pods {
			tc.cancelWorkWithEvent(logger, types.NamespacedName{Namespace: pods[i].Namespace, Name: pods[i].Name})
		}
		return
	}

	now := time.Now()
	for _, pod := range pods {
		podNamespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
		tc.processPodOnNode(ctx, NamespacedObject{NamespacedName: podNamespacedName, UID: pod.UID}, node.Name, pod.Spec.Tolerations, taints, now)
	}
}

func (tc *Controller) emitPodDeletionEvent(nsName types.NamespacedName) {
	if tc.recorder == nil {
		return
	}
	ref := &v1.ObjectReference{
		APIVersion: "v1",
		Kind:       "Pod",
		Name:       nsName.Name,
		Namespace:  nsName.Namespace,
	}
	tc.recorder.Eventf(ref, v1.EventTypeNormal, "TaintManagerEviction", "Marking for deletion Pod %s", nsName.String())
}

func (tc *Controller) emitCancelPodDeletionEvent(nsName types.NamespacedName) {
	if tc.recorder == nil {
		return
	}
	ref := &v1.ObjectReference{
		APIVersion: "v1",
		Kind:       "Pod",
		Name:       nsName.Name,
		Namespace:  nsName.Namespace,
	}
	tc.recorder.Eventf(ref, v1.EventTypeNormal, "TaintManagerEviction", "Cancelling deletion of Pod %s", nsName.String())
}
