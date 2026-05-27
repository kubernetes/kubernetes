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

// This file contains structures that implement scheduling queue types.
// Scheduling queues hold entities (which can be pods or pod groups) waiting to be scheduled. This file implements a
// priority queue which has two sub queues and a additional data structure,
// namely: activeQ, backoffQ and unschedulableEntities.
// - activeQ holds entities that are being considered for scheduling.
// - backoffQ holds entities that moved from unschedulableEntities and will move to
//   activeQ when their backoff periods complete.
// - unschedulableEntities holds entities that were already attempted for scheduling and
//   are currently determined to be unschedulable.

package queue

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	listersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/backend/heap"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	apicalls "k8s.io/kubernetes/pkg/scheduler/framework/api_calls"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/utils/clock"
)

const (
	// DefaultPodMaxInUnschedulablePodsDuration is the default value for the maximum
	// time a pod can stay in unschedulableEntities. If a pod stays in unschedulableEntities
	// for longer than this value, the pod will be moved from unschedulableEntities to
	// backoffQ or activeQ. If this value is empty, the default value (5min)
	// will be used.
	DefaultPodMaxInUnschedulablePodsDuration time.Duration = 5 * time.Minute
	// Scheduling queue names
	activeQ        = "Active"
	backoffQ       = "Backoff"
	unschedulableQ = "Unschedulable"

	preEnqueue = "PreEnqueue"
)

const (
	// DefaultPodInitialBackoffDuration is the default value for the initial backoff duration
	// for unschedulable pods. To change the default podInitialBackoffDurationSeconds used by the
	// scheduler, update the ComponentConfig value in defaults.go
	DefaultPodInitialBackoffDuration time.Duration = 1 * time.Second
	// DefaultPodMaxBackoffDuration is the default value for the max backoff duration
	// for unschedulable pods. To change the default podMaxBackoffDurationSeconds used by the
	// scheduler, update the ComponentConfig value in defaults.go
	DefaultPodMaxBackoffDuration time.Duration = 10 * time.Second
)

// PreEnqueueCheck is a function type. It's used to build functions that
// run against a Pod and the caller can choose to enqueue or skip the Pod
// by the checking result.
type PreEnqueueCheck func(pod *v1.Pod) bool

// PodSigner creates a scheduling signature for a pod that represents its scheduling requirements.
// The signature is used by the opportunistic batching feature (KEP-5598) to reuse scheduling decisions.
// Returns nil if the pod is "unsignable" (some plugins cannot create a stable signature for it).
type PodSigner func(ctx context.Context, pod *v1.Pod) fwk.PodSignature

// SchedulingQueue is an interface for a queue to store pods waiting to be scheduled.
// The interface follows a pattern similar to cache.FIFO and cache.Heap and
// makes it easy to use those data structures as a SchedulingQueue.
type SchedulingQueue interface {
	fwk.PodNominator
	Add(ctx context.Context, pod *v1.Pod)
	// Activate moves the given pods to activeQ.
	// If a pod isn't found in unschedulableEntities or backoffQ and it's in-flight,
	// the wildcard event is registered so that the pod will be requeued when it comes back.
	// But, if a pod isn't found in unschedulableEntities or backoffQ and it's not in-flight (i.e., completely unknown pod),
	// Activate would ignore the pod.
	Activate(logger klog.Logger, pods map[string]*v1.Pod)
	// AddUnschedulablePodIfNotPresent adds an unschedulable pod back to scheduling queue.
	// The podSchedulingCycle represents the current scheduling cycle number which can be
	// returned by calling SchedulingCycle().
	AddUnschedulablePodIfNotPresent(logger klog.Logger, pInfo *framework.QueuedPodInfo, podSchedulingCycle int64) error
	// AddAttemptedPodGroupIfNeeded adds an attempted pod group back to scheduling queue.
	// If there are no pending pods, it will not add the pod group back to the queue.
	AddAttemptedPodGroupIfNeeded(logger klog.Logger, pgInfo *framework.QueuedPodGroupInfo, schedulingCycle int64) error
	// SchedulingCycle returns the current number of scheduling cycle which is
	// cached by scheduling queue. Normally, incrementing this number whenever
	// a pod is popped (e.g. called Pop()) is enough.
	SchedulingCycle() int64
	// Pop removes the head of the queue and returns it. It blocks if the
	// queue is empty and waits until a new item is added to the queue.
	Pop(logger klog.Logger) (framework.QueuedEntityInfo, error)
	// Done must be called for pod returned by Pop. This allows the queue to
	// keep track of which pods are currently being processed.
	Done(types.UID)
	Update(ctx context.Context, oldPod, newPod *v1.Pod)
	Delete(logger klog.Logger, pod *v1.Pod)
	// Important Note: preCheck shouldn't include anything that depends on the in-tree plugins' logic.
	// (e.g., filter Pods based on added/updated Node's capacity, etc.)
	MoveAllToActiveOrBackoffQueue(logger klog.Logger, event fwk.ClusterEvent, oldObj, newObj interface{}, preCheck PreEnqueueCheck)
	AssignedPodAdded(logger klog.Logger, pod *v1.Pod)
	AssignedPodUpdated(logger klog.Logger, oldPod, newPod *v1.Pod, event fwk.ClusterEvent)

	// Close closes the SchedulingQueue so that the goroutine which is
	// waiting to pop items can exit gracefully.
	Close()
	// Run starts the goroutines managing the queue.
	Run(logger klog.Logger)

	// PatchPodStatus handles the pod status update by sending an update API call through API dispatcher.
	// This method should be used only if the SchedulerAsyncAPICalls feature gate is enabled.
	PatchPodStatus(pod *v1.Pod, conditions []*v1.PodCondition, nominatingInfo *fwk.NominatingInfo) (<-chan error, error)

	// The following functions are supposed to be used only for testing or debugging.
	GetPod(name, namespace string, schedulingGroup *v1.PodSchedulingGroup) (*framework.QueuedPodInfo, bool)
	PendingPods() ([]*v1.Pod, string)
	InFlightPods() []*v1.Pod
	PodsInActiveQ() []*v1.Pod
	// PodsInBackoffQ returns all the Pods in the backoffQ.
	PodsInBackoffQ() []*v1.Pod
	UnschedulablePods() []*v1.Pod
	// PendingPodGroupPods returns all the pending pods waiting for their pod groups.
	PendingPodGroupPods() []*v1.Pod
}

// NewSchedulingQueue initializes a priority queue as a new scheduling queue.
func NewSchedulingQueue(
	lessFn fwk.LessFunc,
	informerFactory informers.SharedInformerFactory,
	opts ...Option) SchedulingQueue {
	return NewPriorityQueue(lessFn, informerFactory, opts...)
}

// PriorityQueue implements a scheduling queue.
// The head of PriorityQueue is the highest priority pending entity (pod or pod group). This structure
// has two sub queues and an additional data structure, namely: activeQ,
// backoffQ and unschedulableEntities.
//   - activeQ holds pods or pod groups that are being considered for scheduling.
//   - backoffQ holds pods or pod groups that moved from unschedulableEntities and will move to
//     activeQ when their backoff periods complete.
//   - unschedulableEntities holds pods or pod groups that were already attempted for scheduling and
//     are currently determined to be unschedulable.
type PriorityQueue struct {
	*nominator

	stop  chan struct{}
	clock clock.WithTicker

	// lock takes precedence and should be taken first,
	// before any other locks in the queue (activeQueue.lock or backoffQueue.lock or nominator.nLock).
	// Correct locking order is: lock > activeQueue.lock > backoffQueue.lock > nominator.nLock.
	lock sync.RWMutex

	// the maximum time a pod can stay in the unschedulableEntities.
	podMaxInUnschedulablePodsDuration time.Duration

	activeQ  activeQueuer
	backoffQ backoffQueuer
	// unschedulableEntities holds pods and pod groups that have been tried and determined unschedulable.
	unschedulableEntities *unschedulableEntities
	// pendingPodGroupPods stores all pending pods that wait for their corresponding pod group to be requeued.
	pendingPodGroupPods *pendingPodGroupMemberPods

	// preEnqueuePluginMap is keyed with profile and plugin name, valued with registered preEnqueue plugins.
	preEnqueuePluginMap map[string]map[string]fwk.PreEnqueuePlugin
	// queueingHintMap is keyed with profile name, valued with registered queueing hint functions.
	queueingHintMap QueueingHintMapPerProfile
	// pluginToEventsMap shows which plugin is interested in which events.
	pluginToEventsMap map[string][]fwk.ClusterEvent

	nsLister listersv1.NamespaceLister

	metricsRecorder *metrics.MetricAsyncRecorder
	// pluginMetricsSamplePercent is the percentage of plugin metrics to be sampled.
	pluginMetricsSamplePercent int

	// apiDispatcher is used for the methods that are expected to send API calls.
	// It's non-nil only if the SchedulerAsyncAPICalls feature gate is enabled.
	apiDispatcher fwk.APIDispatcher

	// podSigners maps a profile name to a signing function for that profile.
	podSigners map[string]PodSigner

	// isPopFromBackoffQEnabled indicates whether the feature gate SchedulerPopFromBackoffQ is enabled.
	isPopFromBackoffQEnabled bool
	// isGenericWorkloadEnabled indicates whether the feature gate GenericWorkload is enabled.
	isGenericWorkloadEnabled bool
	// isOpportunisticBatchingEnabled indicates whether the OpportunisticBatching feature gate is enabled.
	isOpportunisticBatchingEnabled bool
}

// QueueingHintFunction is the wrapper of QueueingHintFn that has PluginName.
type QueueingHintFunction struct {
	PluginName     string
	QueueingHintFn fwk.QueueingHintFn
}

// clusterEvent has the event and involved objects.
type clusterEvent struct {
	event fwk.ClusterEvent
	// oldObj is the object that involved this event.
	oldObj interface{}
	// newObj is the object that involved this event.
	newObj interface{}
}

type priorityQueueOptions struct {
	clock                             clock.WithTicker
	podInitialBackoffDuration         time.Duration
	podMaxBackoffDuration             time.Duration
	podMaxInUnschedulablePodsDuration time.Duration
	podLister                         listersv1.PodLister
	metricsRecorder                   *metrics.MetricAsyncRecorder
	pluginMetricsSamplePercent        int
	preEnqueuePluginMap               map[string]map[string]fwk.PreEnqueuePlugin
	queueingHintMap                   QueueingHintMapPerProfile
	apiDispatcher                     fwk.APIDispatcher
	podSigners                        map[string]PodSigner
}

// Option configures a PriorityQueue
type Option func(*priorityQueueOptions)

// WithClock sets clock for PriorityQueue, the default clock is clock.RealClock.
func WithClock(clock clock.WithTicker) Option {
	return func(o *priorityQueueOptions) {
		o.clock = clock
	}
}

// WithPodInitialBackoffDuration sets pod initial backoff duration for PriorityQueue.
func WithPodInitialBackoffDuration(duration time.Duration) Option {
	return func(o *priorityQueueOptions) {
		o.podInitialBackoffDuration = duration
	}
}

// WithPodMaxBackoffDuration sets pod max backoff duration for PriorityQueue.
func WithPodMaxBackoffDuration(duration time.Duration) Option {
	return func(o *priorityQueueOptions) {
		o.podMaxBackoffDuration = duration
	}
}

// WithPodLister sets pod lister for PriorityQueue.
func WithPodLister(pl listersv1.PodLister) Option {
	return func(o *priorityQueueOptions) {
		o.podLister = pl
	}
}

// WithPodMaxInUnschedulablePodsDuration sets podMaxInUnschedulablePodsDuration for PriorityQueue.
func WithPodMaxInUnschedulablePodsDuration(duration time.Duration) Option {
	return func(o *priorityQueueOptions) {
		o.podMaxInUnschedulablePodsDuration = duration
	}
}

// QueueingHintMapPerProfile is keyed with profile name, valued with queueing hint map registered for the profile.
type QueueingHintMapPerProfile map[string]QueueingHintMap

// QueueingHintMap is keyed with ClusterEvent, valued with queueing hint functions registered for the event.
type QueueingHintMap map[fwk.ClusterEvent][]*QueueingHintFunction

// WithQueueingHintMapPerProfile sets queueingHintMap for PriorityQueue.
func WithQueueingHintMapPerProfile(m QueueingHintMapPerProfile) Option {
	return func(o *priorityQueueOptions) {
		o.queueingHintMap = m
	}
}

// WithPreEnqueuePluginMap sets preEnqueuePluginMap for PriorityQueue.
func WithPreEnqueuePluginMap(m map[string]map[string]fwk.PreEnqueuePlugin) Option {
	return func(o *priorityQueueOptions) {
		o.preEnqueuePluginMap = m
	}
}

// WithMetricsRecorder sets metrics recorder.
func WithMetricsRecorder(recorder *metrics.MetricAsyncRecorder) Option {
	return func(o *priorityQueueOptions) {
		o.metricsRecorder = recorder
	}
}

// WithPluginMetricsSamplePercent sets the percentage of plugin metrics to be sampled.
func WithPluginMetricsSamplePercent(percent int) Option {
	return func(o *priorityQueueOptions) {
		o.pluginMetricsSamplePercent = percent
	}
}

// WithAPIDispatcher sets the API dispatcher.
func WithAPIDispatcher(apiDispatcher fwk.APIDispatcher) Option {
	return func(o *priorityQueueOptions) {
		o.apiDispatcher = apiDispatcher
	}
}

// WithPodSigners sets the pod signing functions per scheduler profile.
// Each profile can have its own signing function for computing pod signatures.
// Pod signatures enable opportunistic batching (KEP-5598) by allowing the scheduler
// to cache and reuse filtering/scoring results for identical pods.
func WithPodSigners(signers map[string]PodSigner) Option {
	return func(o *priorityQueueOptions) {
		o.podSigners = signers
	}
}

var defaultPriorityQueueOptions = priorityQueueOptions{
	clock:                             clock.RealClock{},
	podInitialBackoffDuration:         DefaultPodInitialBackoffDuration,
	podMaxBackoffDuration:             DefaultPodMaxBackoffDuration,
	podMaxInUnschedulablePodsDuration: DefaultPodMaxInUnschedulablePodsDuration,
}

// Making sure that PriorityQueue implements SchedulingQueue.
var _ SchedulingQueue = &PriorityQueue{}

// newQueuedPodInfoForLookup builds a QueuedPodInfo object for a lookup in the queue.
func newQueuedPodInfoForLookup(pod *v1.Pod, plugins ...string) *framework.QueuedPodInfo {
	// Since this is only used for a lookup in the queue, we only need to set the Pod,
	// and so we avoid creating a full PodInfo, which is expensive to instantiate frequently.
	return &framework.QueuedPodInfo{
		PodInfo: &framework.PodInfo{Pod: pod},
		QueueingParams: framework.QueueingParams{
			UnschedulablePlugins: sets.New(plugins...),
		},
	}
}

// newQueuedPodGroupInfoForLookup builds a QueuedPodGroupInfo object for a lookup in the queue.
func newQueuedPodGroupInfoForLookup(pod *v1.Pod) *framework.QueuedPodGroupInfo {
	// Since this is only used for a lookup in the queue, we only need to set the PodGroupInfo namespace and name,
	// and so we avoid creating a full QueuedPodGroupInfo, which is expensive to instantiate frequently.
	return &framework.QueuedPodGroupInfo{
		PodGroupInfo: &framework.PodGroupInfo{
			Namespace: pod.Namespace,
			Name:      *pod.Spec.SchedulingGroup.PodGroupName,
		},
	}
}

// NewPriorityQueue creates a PriorityQueue object.
func NewPriorityQueue(
	lessFn fwk.LessFunc,
	informerFactory informers.SharedInformerFactory,
	opts ...Option,
) *PriorityQueue {
	options := defaultPriorityQueueOptions
	if options.podLister == nil {
		options.podLister = informerFactory.Core().V1().Pods().Lister()
	}
	for _, opt := range opts {
		opt(&options)
	}

	isPopFromBackoffQEnabled := utilfeature.DefaultFeatureGate.Enabled(features.SchedulerPopFromBackoffQ)
	isGenericWorkloadEnabled := utilfeature.DefaultFeatureGate.Enabled(features.GenericWorkload)
	isOpportunisticBatchingEnabled := utilfeature.DefaultFeatureGate.Enabled(features.OpportunisticBatching)
	lessConverted := convertLessFn(lessFn)

	backoffQ := newBackoffQueue(options.clock, options.podInitialBackoffDuration, options.podMaxBackoffDuration, lessConverted, isPopFromBackoffQEnabled)
	pq := &PriorityQueue{
		clock:                             options.clock,
		stop:                              make(chan struct{}),
		podMaxInUnschedulablePodsDuration: options.podMaxInUnschedulablePodsDuration,
		backoffQ:                          backoffQ,
		unschedulableEntities:             newUnschedulableEntities(metrics.NewUnschedulablePodsRecorder(), metrics.NewGatedPodsRecorder()),
		pendingPodGroupPods:               newPendingPodGroupMemberPods(),
		preEnqueuePluginMap:               options.preEnqueuePluginMap,
		queueingHintMap:                   options.queueingHintMap,
		pluginToEventsMap:                 buildEventMap(options.queueingHintMap),
		metricsRecorder:                   options.metricsRecorder,
		pluginMetricsSamplePercent:        options.pluginMetricsSamplePercent,
		apiDispatcher:                     options.apiDispatcher,
		podSigners:                        options.podSigners,
		isPopFromBackoffQEnabled:          isPopFromBackoffQEnabled,
		isGenericWorkloadEnabled:          isGenericWorkloadEnabled,
		isOpportunisticBatchingEnabled:    isOpportunisticBatchingEnabled,
	}
	var backoffQPopper backoffQPopper
	if isPopFromBackoffQEnabled {
		backoffQPopper = backoffQ
	}
	pq.activeQ = newActiveQueue(heap.NewWithRecorder(queuedEntityKeyFunc, heap.LessFunc[framework.QueuedEntityInfo](lessConverted), metrics.NewActivePodsRecorder()), options.metricsRecorder, backoffQPopper)
	pq.nsLister = informerFactory.Core().V1().Namespaces().Lister()
	pq.nominator = newPodNominator(options.podLister)

	return pq
}

// ConvertLessFn wraps fwk.LessFunc and converts it to take framework.QueuedEntityInfo as arguments.
func convertLessFn(lessFn fwk.LessFunc) func(entity1, entity2 framework.QueuedEntityInfo) bool {
	return func(entity1, entity2 framework.QueuedEntityInfo) bool {
		return lessFn(entity1.(fwk.QueuedEntityInfo), entity2.(fwk.QueuedEntityInfo))
	}
}

func buildEventMap(qHintMap QueueingHintMapPerProfile) map[string][]fwk.ClusterEvent {
	eventMap := make(map[string][]fwk.ClusterEvent)

	for _, hintMap := range qHintMap {
		for event, qHints := range hintMap {
			for _, qHint := range qHints {
				eventMap[qHint.PluginName] = append(eventMap[qHint.PluginName], event)
			}
		}
	}

	return eventMap
}

// Run starts the goroutine to pump from backoffQ to activeQ
func (p *PriorityQueue) Run(logger klog.Logger) {
	go p.backoffQ.waitUntilAlignedWithOrderingWindow(func() {
		p.flushBackoffQCompleted(logger)
	}, p.stop)
	go wait.Until(func() {
		p.flushUnschedulableEntitiesLeftover(logger)
	}, 30*time.Second, p.stop)
}

// queueingStrategy indicates how the scheduling queue should enqueue the Pod from unschedulable pod pool.
type queueingStrategy int

const (
	// queueSkip indicates that the scheduling queue should skip requeuing the Pod to activeQ/backoffQ.
	queueSkip queueingStrategy = iota
	// queueAfterBackoff indicates that the scheduling queue should requeue the Pod after backoff is completed.
	queueAfterBackoff
	// queueImmediately indicates that the scheduling queue should skip backoff and requeue the Pod immediately to activeQ.
	queueImmediately
)

// isEventOfInterest returns true if the event is of interest by some plugins.
func (p *PriorityQueue) isEventOfInterest(logger klog.Logger, event fwk.ClusterEvent) bool {
	if framework.ClusterEventIsWildCard(event) {
		// Wildcard event moves Pods that failed with any plugins.
		return true
	}

	for _, hintMap := range p.queueingHintMap {
		for eventToMatch := range hintMap {
			if framework.MatchClusterEvents(eventToMatch, event) {
				// This event is interested by some plugins.
				return true
			}
		}
	}

	logger.V(6).Info("receive an event that isn't interested by any enabled plugins", "event", event)

	return false
}

// isPodGroupMember returns true if the pod is a member of a pod group.
func (p *PriorityQueue) isPodGroupMember(pod *v1.Pod) bool {
	return p.isGenericWorkloadEnabled && pod.Spec.SchedulingGroup != nil
}

// isEntityWorthRequeuing calls isPodWorthRequeuing for all pods belonging to the entity and returns the highest queueing strategy.
func (p *PriorityQueue) isEntityWorthRequeuing(logger klog.Logger, entity framework.QueuedEntityInfo, event fwk.ClusterEvent, oldObj, newObj interface{}) queueingStrategy {
	// For pod groups, if any pod is worth requeuing, the whole group is worth it.
	// But we should prioritize higher strategies.
	bestStrategy := queueSkip
	entity.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
		strategy := p.isPodWorthRequeuing(logger, pInfo, event, oldObj, newObj)
		if strategy > bestStrategy {
			bestStrategy = strategy
		}
		if bestStrategy == queueImmediately {
			return false
		}
		return true
	})
	return bestStrategy
}

// isPodWorthRequeuing calls QueueingHintFn of only plugins registered in pInfo.unschedulablePlugins and pInfo.PendingPlugins.
//
// If any of pInfo.PendingPlugins return Queue,
// the scheduling queue is supposed to enqueue this Pod to activeQ, skipping backoffQ.
// If any of pInfo.unschedulablePlugins return Queue,
// the scheduling queue is supposed to enqueue this Pod to activeQ/backoffQ depending on the remaining backoff time of the Pod.
func (p *PriorityQueue) isPodWorthRequeuing(logger klog.Logger, pInfo *framework.QueuedPodInfo, event fwk.ClusterEvent, oldObj, newObj interface{}) queueingStrategy {
	rejectorPlugins := pInfo.UnschedulablePlugins.Union(pInfo.PendingPlugins)
	if rejectorPlugins.Len() == 0 {
		logger.V(6).Info("Worth requeuing because no failed plugins", "pod", klog.KObj(pInfo.Pod))
		return queueAfterBackoff
	}

	if framework.ClusterEventIsWildCard(event) {
		// If the wildcard event has a Pod in newObj,
		// that indicates that the event wants to be effective for the Pod only.
		// Specifically, EventForceActivate could have a target Pod in newObj.
		if newObj != nil {
			if pod, ok := newObj.(*v1.Pod); !ok || pod.UID != pInfo.Pod.UID {
				// This wildcard event is not for this Pod.
				if ok {
					logger.V(6).Info("Not worth requeuing because the event is wildcard, but for another pod", "pod", klog.KObj(pInfo.Pod), "event", event.Label(), "newObj", klog.KObj(pod))
				}
				return queueSkip
			}
		}

		// If the wildcard event is special one as someone wants to force all Pods to move to activeQ/backoffQ.
		// We return queueAfterBackoff in this case, while resetting all blocked plugins.
		logger.V(6).Info("Worth requeuing because the event is wildcard", "pod", klog.KObj(pInfo.Pod), "event", event.Label())
		return queueAfterBackoff
	}

	hintMap, ok := p.queueingHintMap[pInfo.Pod.Spec.SchedulerName]
	if !ok {
		// shouldn't reach here unless bug.
		utilruntime.HandleErrorWithLogger(logger, nil, "No QueueingHintMap is registered for this profile", "profile", pInfo.Pod.Spec.SchedulerName, "pod", klog.KObj(pInfo.Pod))
		return queueAfterBackoff
	}

	pod := pInfo.Pod
	queueStrategy := queueSkip
	for eventToMatch, hintfns := range hintMap {
		if !framework.MatchClusterEvents(eventToMatch, event) {
			continue
		}

		for _, hintfn := range hintfns {
			if !rejectorPlugins.Has(hintfn.PluginName) {
				// skip if it's not hintfn from rejectorPlugins.
				continue
			}

			start := time.Now()
			hint, err := hintfn.QueueingHintFn(logger, pod, oldObj, newObj)
			if err != nil {
				// If the QueueingHintFn returned an error, we should treat the event as Queue so that we can prevent
				// the Pod from being stuck in the unschedulable pod pool.
				oldObjMeta, newObjMeta, asErr := util.As[klog.KMetadata](oldObj, newObj)
				if asErr != nil {
					utilruntime.HandleErrorWithLogger(logger, err, "QueueingHintFn returns error", "event", event, "plugin", hintfn.PluginName, "pod", klog.KObj(pod))
				} else {
					utilruntime.HandleErrorWithLogger(logger, err, "QueueingHintFn returns error", "event", event, "plugin", hintfn.PluginName, "pod", klog.KObj(pod), "oldObj", klog.KObj(oldObjMeta), "newObj", klog.KObj(newObjMeta))
				}
				hint = fwk.Queue
			}
			p.metricsRecorder.ObserveQueueingHintDurationAsync(hintfn.PluginName, event.Label(), queueingHintToLabel(hint, err), metrics.SinceInSeconds(start))

			if hint == fwk.QueueSkip {
				continue
			}

			if pInfo.PendingPlugins.Has(hintfn.PluginName) {
				// interprets Queue from the Pending plugin as queueImmediately.
				// We can return immediately because queueImmediately is the highest priority.
				return queueImmediately
			}

			// interprets Queue from the unschedulable plugin as queueAfterBackoff.

			if pInfo.PendingPlugins.Len() == 0 {
				// We can return immediately because no Pending plugins, which only can make queueImmediately, registered in this Pod,
				// and queueAfterBackoff is the second highest priority.
				return queueAfterBackoff
			}

			// We can't return immediately because there are some Pending plugins registered in this Pod.
			// We need to check if those plugins return Queue or not and if they do, we return queueImmediately.
			queueStrategy = queueAfterBackoff
		}
	}

	return queueStrategy
}

// queueingHintToLabel converts a hint and an error from QHint to a label string.
func queueingHintToLabel(hint fwk.QueueingHint, err error) string {
	if err != nil {
		return metrics.QueueingHintResultError
	}

	switch hint {
	case fwk.Queue:
		return metrics.QueueingHintResultQueue
	case fwk.QueueSkip:
		return metrics.QueueingHintResultQueueSkip
	}

	// Shouldn't reach here.
	return ""
}

// runPreEnqueuePlugins iterates PreEnqueue function in each registered PreEnqueuePlugin,
// and updates entity.GatingPlugin and entity.UnschedulablePlugins.
// Note: we need to associate the failed plugin to `entity`, so that the entity can be moved back
// to activeQ by related cluster event.
func (p *PriorityQueue) runPreEnqueuePlugins(ctx context.Context, entity framework.QueuedEntityInfo) {
	var anyGatedPodInfo *framework.QueuedPodInfo
	// Run PreEnqueue plugins for each pod, even if it could stop after the first being gated,
	// as we need to populate any per-pod metrics.
	entity.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
		p.runPreEnqueuePluginsForPod(ctx, pInfo)
		if pInfo.Gated() {
			// If any pod is gated, the whole entity is gated.
			// Otherwise, such gated pod would have to be put in a separate entity object
			// and tracked individually, complicating the flow.
			anyGatedPodInfo = pInfo
		}
		return true
	})
	if anyGatedPodInfo != nil {
		// Copying the gating plugin info only from a single pod is sufficient,
		// because if the entity is a pod group, all pods should be ungated before the entire pod group can be ungated,
		// including this pod.
		entity.SetGatingPlugin(anyGatedPodInfo.QueueingParams.GatingPlugin, anyGatedPodInfo.QueueingParams.GatingPluginEvents)
	} else {
		entity.SetGatingPlugin("", nil)
	}
}

func (p *PriorityQueue) runPreEnqueuePluginsForPod(ctx context.Context, pInfo *framework.QueuedPodInfo) {
	var s *fwk.Status
	pod := pInfo.Pod
	startTime := p.clock.Now()
	defer func() {
		metrics.FrameworkExtensionPointDuration.WithLabelValues(preEnqueue, s.Code().String(), pod.Spec.SchedulerName).Observe(metrics.SinceInSeconds(startTime))
	}()

	shouldRecordMetric := rand.Intn(100) < p.pluginMetricsSamplePercent
	gatingPlugin := pInfo.GatingPlugin
	if gatingPlugin != "" {
		// Run the gating plugin first
		s := p.runPreEnqueuePlugin(ctx, p.preEnqueuePluginMap[pod.Spec.SchedulerName][gatingPlugin], pInfo, shouldRecordMetric)
		if !s.IsSuccess() {
			// No need to iterate other plugins
			return
		}
	}

	for _, pl := range p.preEnqueuePluginMap[pod.Spec.SchedulerName] {
		if gatingPlugin != "" && pl.Name() == gatingPlugin {
			// should be run already above.
			continue
		}
		s := p.runPreEnqueuePlugin(ctx, pl, pInfo, shouldRecordMetric)
		if !s.IsSuccess() {
			// No need to iterate other plugins
			return
		}
	}
	// all plugins passed
	pInfo.GatingPlugin = ""
}

// runPreEnqueuePlugin runs the PreEnqueue plugin and update pInfo's fields accordingly if needed.
func (p *PriorityQueue) runPreEnqueuePlugin(ctx context.Context, pl fwk.PreEnqueuePlugin, pInfo *framework.QueuedPodInfo, shouldRecordMetric bool) *fwk.Status {
	pod := pInfo.Pod
	startTime := p.clock.Now()
	s := pl.PreEnqueue(ctx, pod)
	if shouldRecordMetric {
		p.metricsRecorder.ObservePluginDurationAsync(preEnqueue, pl.Name(), s.Code().String(), p.clock.Since(startTime).Seconds())
	}
	if s.IsSuccess() {
		// No need to change GatingPlugin; it's overwritten by the next PreEnqueue plugin if they gate this pod, or it's overwritten with an empty string if all PreEnqueue plugins pass.
		return s
	}
	// Only increment metric and insert if not already incremented for this plugin
	if !pInfo.UnschedulablePlugins.Has(pl.Name()) && !pInfo.PendingPlugins.Has(pl.Name()) {
		metrics.UnschedulableReason(pl.Name(), pod.Spec.SchedulerName).Inc()
	}
	pInfo.UnschedulablePlugins.Insert(pl.Name())
	pInfo.GatingPlugin = pl.Name()
	pInfo.GatingPluginEvents = p.pluginToEventsMap[pInfo.GatingPlugin]
	if s.Code() == fwk.Error {
		utilruntime.HandleErrorWithContext(ctx, s.AsError(), "Unexpected error running PreEnqueue plugin", "pod", klog.KObj(pod), "plugin", pl.Name())
	} else {
		klog.FromContext(ctx).V(4).Info("Status after running PreEnqueue plugin", "pod", klog.KObj(pod), "plugin", pl.Name(), "status", s)
	}

	return s
}

// AddNominatedPod adds the given pod to the nominator.
// It locks the PriorityQueue to make sure it won't race with any other method.
func (p *PriorityQueue) AddNominatedPod(logger klog.Logger, pi fwk.PodInfo, nominatingInfo *fwk.NominatingInfo) {
	p.lock.Lock()
	p.nominator.addNominatedPod(logger, pi, nominatingInfo)
	p.lock.Unlock()
}

// moveToActiveQ tries to add the entity to the active queue.
// If the entity doesn't pass PreEnqueue plugins, it gets added to unschedulableEntities instead.
// movesFromBackoffQ should be set to true, if the entity directly moves from the backoffQ, so the PreEnqueue call can be skipped.
// It returns a boolean flag to indicate whether the entity is added successfully.
// Entity should be removed from the backoffQ before calling moveToActiveQ.
// Note: it does not signal the Pop() method to wake up,
// so the caller is responsible for calling activeQ.broadcast() after executing this method.
func (p *PriorityQueue) moveToActiveQ(logger klog.Logger, entity framework.QueuedEntityInfo, event string, movesFromBackoffQ bool) bool {
	gatedBefore := entity.Gated()
	// If SchedulerPopFromBackoffQ feature gate is enabled,
	// PreEnqueue plugins were called when the entity was added to the backoffQ.
	// Don't need to repeat it here when the entity is directly moved from the backoffQ.
	skipPreEnqueue := p.isPopFromBackoffQEnabled && movesFromBackoffQ
	if !skipPreEnqueue {
		p.runPreEnqueuePlugins(context.Background(), entity)
	}

	if entity.Gated() {
		if p.unschedulableEntities.get(entity) == nil {
			logger.V(5).Info("Entity moved to an internal scheduling queue, because it is gated", "type", entity.Type(), "entity", klog.KObj(entity), "event", event, "queue", unschedulableQ)
		}
		// Clearing WasFlushedFromUnschedulable is typically done on scheduling failure, but in case the flushed pod was gated, it never attempts scheduling.
		// We clear it here to ensure it's not set the next time the pod is woken up by a non-flush event.
		entity.SetWasFlushedFromUnschedulable(false)
		p.unschedulableEntities.addOrUpdate(entity, gatedBefore, event)
		// Entity not moved to activeQ.
		return false
	}
	now := p.clock.Now()
	entity.SetInitialAttemptTimestamp(now)
	p.unschedulableEntities.delete(entity, gatedBefore)

	p.activeQ.add(logger, entity, event)
	// Pod successfully moved to activeQ.
	return true
}

// moveToBackoffQ tries to add the entity to the backoff queue.
// If SchedulerPopFromBackoffQ feature gate is enabled and the entity doesn't pass PreEnqueue plugins, it gets added to unschedulableEntities instead.
// It returns a boolean flag to indicate whether the entity is added successfully.
func (p *PriorityQueue) moveToBackoffQ(logger klog.Logger, entity framework.QueuedEntityInfo, event string) bool {
	gatedBefore := entity.Gated()
	// If SchedulerPopFromBackoffQ feature gate is enabled,
	// PreEnqueue plugins are called on inserting entities to the backoffQ,
	// not to call them again on popping out.
	if p.isPopFromBackoffQEnabled {
		p.runPreEnqueuePlugins(context.Background(), entity)
		if entity.Gated() {
			if uInfo := p.unschedulableEntities.get(entity); uInfo == nil {
				logger.V(5).Info("Entity moved to an internal scheduling queue, because it is gated", "type", entity.Type(), "entity", klog.KObj(entity), "event", event, "queue", unschedulableQ)
			}
			// Clearing WasFlushedFromUnschedulable is typically done on scheduling failure, but in case the flushed pod was gated, it never attempts scheduling.
			// We clear it here to ensure it's not set the next time the pod is woken up by a non-flush event.
			entity.SetWasFlushedFromUnschedulable(false)
			p.unschedulableEntities.addOrUpdate(entity, gatedBefore, event)
			return false
		}
	}
	p.unschedulableEntities.delete(entity, gatedBefore)

	p.backoffQ.add(logger, entity, event)
	return true
}

// Add adds a pod to the active queue. It should be called only when a new pod
// is added so there is no chance the pod is already in active/unschedulable/backoff queues
func (p *PriorityQueue) Add(ctx context.Context, pod *v1.Pod) {
	p.lock.Lock()
	defer p.lock.Unlock()

	p.addPod(ctx, pod)
}

// addPod adds a pod to the active queue, unless it's a pod group member,
// in which case it's added to its queued pod group.
func (p *PriorityQueue) addPod(ctx context.Context, pod *v1.Pod) {
	logger := klog.FromContext(ctx)
	pInfo := p.newQueuedPodInfo(ctx, pod)
	// addNominatedPod is called here unconditionally to ensure that the nomination of the added pod
	// (even if gated, and thus not entering activeQ) is properly recorded in the nominator.
	// Furthermore, this must be called before moveToActiveQ to prevent a potential data race,
	// where an active scheduler loop could pop and process the pod before its nomination is recorded.
	p.nominator.addNominatedPod(logger, pInfo.PodInfo, nil)
	if p.isPodGroupMember(pod) {
		p.addPodGroupMember(logger, pInfo)
		return
	}
	if added := p.moveToActiveQ(logger, pInfo, framework.EventUnscheduledPodAdd.Label(), false); added {
		p.activeQ.broadcast()
	}
}

// addPodGroupMember adds pInfo as a member of its pod group into the scheduling queue.
func (p *PriorityQueue) addPodGroupMember(logger klog.Logger, pInfo *framework.QueuedPodInfo) {
	if added := p.addToPodGroupIfExists(logger, pInfo); added {
		return
	}
	pgInfoLookup := newQueuedPodGroupInfoForLookup(pInfo.Pod)
	if p.activeQ.isLastPoppedEntity(pgInfoLookup) {
		// If the last popped entity is the matching pod group, add the pod to the pending pod group pods,
		// so it will be added to the pod group when it's requeued.
		p.pendingPodGroupPods.add(pgInfoLookup, pInfo)
		logger.V(5).Info("Pod added to pending pod group pods, waiting for its pod group to be requeued", "podGroup", klog.KObj(pgInfoLookup), "pod", klog.KObj(pInfo))
	} else {
		// Create a new group as it's the first member pod in the queue.
		pgInfo := p.newQueuedPodGroupInfo(pInfo)
		if added := p.moveToActiveQ(logger, pgInfo, framework.EventUnscheduledPodAdd.Label(), false); added {
			p.activeQ.broadcast()
		}
		logger.V(5).Info("Pod added to new pod group info", "podGroup", klog.KObj(pgInfoLookup), "pod", klog.KObj(pInfo))
	}
}

// deleteFromAnyQueue deletes an entity from any queue it may be in and returns
// the entity and the queueing strategy (matching the queue where the entity was found).
// If the entity is not found, it returns nil entity.
func (p *PriorityQueue) deleteFromAnyQueue(entityLookup framework.QueuedEntityInfo) (framework.QueuedEntityInfo, queueingStrategy) {
	if entity := p.activeQ.delete(entityLookup); entity != nil {
		return entity, queueImmediately
	}
	if entity := p.backoffQ.delete(entityLookup); entity != nil {
		return entity, queueAfterBackoff
	}
	if entity := p.unschedulableEntities.get(entityLookup); entity != nil {
		p.unschedulableEntities.delete(entity, entity.Gated())
		return entity, queueSkip
	}
	return nil, queueSkip
}

// addToPodGroupIfExists tries to add pInfo as a member of its pod group into the scheduling queue.
// It returns true if pInfo was added to an existing pod group.
func (p *PriorityQueue) addToPodGroupIfExists(logger klog.Logger, pInfo *framework.QueuedPodInfo) bool {
	pgInfoLookup := newQueuedPodGroupInfoForLookup(pInfo.Pod)

	entity, strategy := p.deleteFromAnyQueue(pgInfoLookup)
	if entity == nil {
		return false
	}
	pgInfo := entity.(*framework.QueuedPodGroupInfo)

	pgInfo.AddPod(pInfo)
	queue := p.requeueEntityWithQueueingStrategy(logger, pgInfo, strategy, framework.EventUnscheduledPodAdd.Label())
	if queue == activeQ || (p.isPopFromBackoffQEnabled && queue == backoffQ) {
		p.activeQ.broadcast()
	}
	logger.V(5).Info("Pod added to existing pod group info", "podGroup", klog.KObj(pgInfo), "pod", klog.KObj(pInfo), "queue", queue)
	return true
}

// Activate moves the given pods to activeQ.
// If a pod isn't found in unschedulableEntities or backoffQ and it's in-flight,
// the wildcard event is registered so that the pod will be requeued when it comes back.
// But, if a pod isn't found in unschedulableEntities or backoffQ and it's not in-flight (i.e., completely unknown pod),
// Activate would ignore the pod.
// If activating a pod that is a member of a pod group, the whole pod group is activated.
func (p *PriorityQueue) Activate(logger klog.Logger, pods map[string]*v1.Pod) {
	p.lock.Lock()
	defer p.lock.Unlock()

	activated := false
	for _, pod := range pods {
		var entityLookup framework.QueuedEntityInfo
		if p.isPodGroupMember(pod) {
			entityLookup = newQueuedPodGroupInfoForLookup(pod)
		} else {
			entityLookup = newQueuedPodInfoForLookup(pod)
		}
		if p.activate(logger, entityLookup) {
			activated = true
			continue
		}

		// If this pod is in-flight, register the activation event so that the pod will be requeued when it comes back.
		// Specifically in the in-tree plugins, this is for the scenario with the preemption plugin
		// where the async preemption API calls are all done or fail at some point before the Pod comes back to the queue.
		p.activeQ.addEventsIfPodInFlight(nil, pod, []fwk.ClusterEvent{framework.EventForceActivate})
	}

	if activated {
		p.activeQ.broadcast()
	}
}

func (p *PriorityQueue) activate(logger klog.Logger, entityLookup framework.QueuedEntityInfo) bool {
	var entity framework.QueuedEntityInfo
	var movesFromBackoffQ bool
	// Verify if the entity is present in unschedulableEntities or backoffQ.
	if entity = p.unschedulableEntities.get(entityLookup); entity == nil {
		// Entity may be present in the backoffQ. Try to delete the entity from the backoffQ now
		// to make sure it won't be popped from the backoffQ just before moving it to the activeQ.
		if entity = p.backoffQ.delete(entityLookup); entity == nil {
			// Entity is not present in the backoffQ. Don't activate it.
			return false
		}
		movesFromBackoffQ = true
	}

	if entity == nil {
		// Redundant safe check. We shouldn't reach here.
		utilruntime.HandleErrorWithLogger(logger, nil, "Internal error: cannot obtain entity")
		return false
	}

	return p.moveToActiveQ(logger, entity, framework.ForceActivate, movesFromBackoffQ)
}

// SchedulingCycle returns current scheduling cycle.
func (p *PriorityQueue) SchedulingCycle() int64 {
	return p.activeQ.schedulingCycle()
}

// determineSchedulingHintForInFlightPod looks at the unschedulable plugins of the given Pod
// and determines the scheduling hint for this Pod while checking the events that happened during in-flight.
func (p *PriorityQueue) determineSchedulingHintForInFlightPod(logger klog.Logger, pInfo *framework.QueuedPodInfo) queueingStrategy {
	if len(pInfo.UnschedulablePlugins) == 0 && len(pInfo.PendingPlugins) == 0 {
		// No failed plugins are associated with this Pod.
		// Meaning something unusual (a temporal failure on kube-apiserver, etc) happened and this Pod gets moved back to the queue.
		// In this case, we should retry scheduling it because this Pod may not be retried until the next flush.
		return queueAfterBackoff
	}

	events, err := p.activeQ.clusterEventsForPod(logger, pInfo)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Error getting cluster events for pod", "pod", klog.KObj(pInfo.Pod))
		return queueAfterBackoff
	}

	// check if there is an event that makes this Pod schedulable based on pInfo.UnschedulablePlugins.
	queueingStrategy := queueSkip
	for _, e := range events {
		logger.V(5).Info("Checking event for in-flight pod", "pod", klog.KObj(pInfo.Pod), "event", e.event.Label())

		switch p.isPodWorthRequeuing(logger, pInfo, e.event, e.oldObj, e.newObj) {
		case queueSkip:
			continue
		case queueImmediately:
			// queueImmediately is the highest priority.
			// No need to go through the rest of the events.
			return queueImmediately
		case queueAfterBackoff:
			// replace schedulingHint with queueAfterBackoff
			queueingStrategy = queueAfterBackoff
			if pInfo.PendingPlugins.Len() == 0 {
				// We can return immediately because no Pending plugins, which only can make queueImmediately, registered in this Pod,
				// and queueAfterBackoff is the second highest priority.
				return queueAfterBackoff
			}
		}
	}
	return queueingStrategy
}

// AddUnschedulablePodIfNotPresent inserts a pod that cannot be scheduled into
// the queue, unless it is already in the queue.
func (p *PriorityQueue) AddUnschedulablePodIfNotPresent(logger klog.Logger, pInfo *framework.QueuedPodInfo, podSchedulingCycle int64) error {
	p.lock.Lock()
	defer p.lock.Unlock()

	// In any case, this Pod will be moved back to the queue and we should call Done.
	calledDone := false
	defer func() {
		if !calledDone {
			p.Done(pInfo.Pod.UID)
		}
	}()

	pod := pInfo.Pod
	if p.unschedulableEntities.get(pInfo) != nil {
		return fmt.Errorf("Pod %v is already present in unschedulable queue", klog.KObj(pod))
	}

	if p.activeQ.has(pInfo) {
		return fmt.Errorf("Pod %v is already present in the active queue", klog.KObj(pod))
	}
	if p.backoffQ.has(pInfo) {
		return fmt.Errorf("Pod %v is already present in the backoff queue", klog.KObj(pod))
	}

	if len(pInfo.UnschedulablePlugins) == 0 && len(pInfo.PendingPlugins) == 0 {
		// This Pod came back because of some unexpected errors (e.g., a network issue).
		pInfo.ConsecutiveErrorsCount++
	} else {
		// This Pod is rejected by some plugins, not coming back due to unexpected errors (e.g., a network issue)
		pInfo.UnschedulableCount++
		// We should reset the error count because the error is gone.
		pInfo.ConsecutiveErrorsCount = 0
	}
	// Refresh the timestamp since the pod is re-added.
	pInfo.Timestamp = p.clock.Now()
	// We changed ConsecutiveErrorsCount or UnschedulableCount plus Timestamp, and now the calculated backoff time should be different,
	// removing the cached backoff time.
	pInfo.BackoffExpiration = time.Time{}
	// Clear the flush flag since the pod is returning to the queue after a scheduling attempt.
	pInfo.WasFlushedFromUnschedulable = false

	if p.isPodGroupMember(pod) {
		// Done has to be called before adding the unschedulable pod group member.
		// Otherwise, a race can happen when the pod is added to an active pod group,
		// and the Pop sees the pod being not Done'd.
		p.Done(pInfo.Pod.UID)
		calledDone = true
		// When the pod is a pod group member, process it in that context.
		p.addUnschedulablePodGroupMember(logger, pInfo)
		return nil
	}

	p.activeQ.clearPoppedEntity()
	rejectorPlugins := pInfo.UnschedulablePlugins.Union(pInfo.PendingPlugins)
	for plugin := range rejectorPlugins {
		metrics.UnschedulableReason(plugin, pInfo.Pod.Spec.SchedulerName).Inc()
	}

	// We check whether this Pod may change its scheduling result by any of events that happened during scheduling.
	schedulingHint := p.determineSchedulingHintForInFlightPod(logger, pInfo)

	// In this case, we try to requeue this Pod to activeQ/backoffQ.
	queue := p.requeueEntityWithQueueingStrategy(logger, pInfo, schedulingHint, framework.ScheduleAttemptFailure)
	logger.V(3).Info("Pod moved to an internal scheduling queue", "pod", klog.KObj(pod), "event", framework.ScheduleAttemptFailure, "queue", queue, "schedulingCycle", podSchedulingCycle, "hint", schedulingHint, "unschedulable plugins", rejectorPlugins)
	if queue == activeQ || (p.isPopFromBackoffQEnabled && queue == backoffQ) {
		// When the Pod is moved to activeQ, need to let p.cond know so that the Pod will be pop()ed out.
		p.activeQ.broadcast()
	}

	return nil
}

// addUnschedulablePodGroupMember adds pInfo as a member of its pod group into the scheduling queue.
func (p *PriorityQueue) addUnschedulablePodGroupMember(logger klog.Logger, pInfo *framework.QueuedPodInfo) {
	rejectorPlugins := pInfo.UnschedulablePlugins.Union(pInfo.PendingPlugins)
	for plugin := range rejectorPlugins {
		metrics.UnschedulableReason(plugin, pInfo.Pod.Spec.SchedulerName).Inc()
	}

	// Pod should be just re-added to the scheduling queue, because the queued pod group
	// can be in any state. addPodGroupMember handles all possible cases.
	p.addPodGroupMember(logger, pInfo)
}

// AddAttemptedPodGroupIfNeeded adds an attempted pod group back to scheduling queue.
// If there are no pending pods, it will not add the pod group back to the queue.
// Should be called synchronously to the pod group scheduling cycle.
func (p *PriorityQueue) AddAttemptedPodGroupIfNeeded(logger klog.Logger, pgInfo *framework.QueuedPodGroupInfo, schedulingCycle int64) error {
	p.lock.Lock()
	defer p.lock.Unlock()

	if p.unschedulableEntities.get(pgInfo) != nil {
		return fmt.Errorf("pod group %v is already present in unschedulable queue", klog.KObj(pgInfo))
	}

	if p.activeQ.has(pgInfo) {
		return fmt.Errorf("pod group %v is already present in the active queue", klog.KObj(pgInfo))
	}
	if p.backoffQ.has(pgInfo) {
		return fmt.Errorf("pod group %v is already present in the backoff queue", klog.KObj(pgInfo))
	}

	p.activeQ.clearPoppedEntity()
	// Get the pending pods and put them into the pod group.
	pendingPods := p.pendingPodGroupPods.get(pgInfo)
	if len(pendingPods) == 0 {
		// No pending pods, nothing to requeue.
		return nil
	}
	pgInfo.SetPods(pendingPods)
	p.pendingPodGroupPods.clear(pgInfo)

	hasErrorPods := false
	pgInfo.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
		if pInfo.UnschedulablePlugins.Len() == 0 && pInfo.PendingPlugins.Len() == 0 {
			hasErrorPods = true
			return false
		}
		return true
	})
	if hasErrorPods {
		// This Pod group came back because of some unexpected errors (e.g., a network issue).
		pgInfo.ConsecutiveErrorsCount++
	} else {
		// This Pod group is rejected by some plugins, not coming back due to unexpected errors (e.g., a network issue)
		pgInfo.UnschedulableCount++
		// We should reset the error count because the error is gone.
		pgInfo.ConsecutiveErrorsCount = 0
	}
	// Refresh the timestamp since the pod is re-added.
	pgInfo.Timestamp = p.clock.Now()
	// We changed ConsecutiveErrorsCount or UnschedulableCount plus Timestamp, and now the calculated backoff time should be different,
	// removing the cached backoff time.
	pgInfo.BackoffExpiration = time.Time{}
	// Clear the flush flag since the pod is returning to the queue after a scheduling attempt.
	pgInfo.WasFlushedFromUnschedulable = false

	rejectorPlugins := pgInfo.UnschedulablePlugins.Union(pgInfo.PendingPlugins)

	// Try to requeue this pod group to activeQ or backoffQ.
	// If the PreEnqueue fails for this pod group, it will be moved to the unschedulableEntities.
	queue := unschedulableQ
	if p.backoffQ.isEntityBackingoff(pgInfo) {
		if added := p.moveToBackoffQ(logger, pgInfo, framework.ScheduleAttemptFailure); added {
			queue = backoffQ
		}
	} else {
		if added := p.moveToActiveQ(logger, pgInfo, framework.ScheduleAttemptFailure, false); added {
			queue = activeQ
		}
	}
	logger.V(3).Info("Pod group moved to an internal scheduling queue", "podGroup", klog.KObj(pgInfo), "event", framework.ScheduleAttemptFailure, "queue", queue, "schedulingCycle", schedulingCycle, "unschedulable plugins", rejectorPlugins)
	if queue == activeQ || (p.isPopFromBackoffQEnabled && queue == backoffQ) {
		// When the Pod group is moved to activeQ, need to let p.cond know so that the Pod will be pop()ed out.
		p.activeQ.broadcast()
	}

	return nil
}

// flushBackoffQCompleted Moves all pods from backoffQ which have completed backoff in to activeQ
func (p *PriorityQueue) flushBackoffQCompleted(logger klog.Logger) {
	p.lock.Lock()
	defer p.lock.Unlock()
	activated := false
	podsCompletedBackoff := p.backoffQ.popAllBackoffCompleted(logger)
	for _, pInfo := range podsCompletedBackoff {
		if added := p.moveToActiveQ(logger, pInfo, framework.BackoffComplete, true); added {
			activated = true
		}
	}
	if activated {
		p.activeQ.broadcast()
	}
}

// flushUnschedulableEntitiesLeftover moves entities which stay in unschedulableEntities
// longer than podMaxInUnschedulablePodsDuration to backoffQ or activeQ.
func (p *PriorityQueue) flushUnschedulableEntitiesLeftover(logger klog.Logger) {
	p.lock.Lock()
	defer p.lock.Unlock()

	var entitiesToMove []framework.QueuedEntityInfo
	currentTime := p.clock.Now()
	for _, entity := range p.unschedulableEntities.entityInfoMap {
		lastScheduleTime := entity.GetTimestamp()
		// TODO(macsko): Once the PodGroups support queueing hints, we may want to extend the max duration for PodGroups.
		if currentTime.Sub(lastScheduleTime) > p.podMaxInUnschedulablePodsDuration {
			entity.SetWasFlushedFromUnschedulable(true)
			entitiesToMove = append(entitiesToMove, entity)
		}
	}

	if len(entitiesToMove) > 0 {
		p.moveEntitiesToActiveOrBackoffQueue(logger, entitiesToMove, framework.EventUnschedulableTimeout, nil, nil)
	}
}

// Pop removes the head of the active queue and returns it. It blocks if the
// activeQ is empty and waits until a new item is added to the queue. It
// increments scheduling cycle when a pod is popped.
// Note: This method should NOT be locked by the p.lock at any moment,
// as it would lead to scheduling throughput degradation.
func (p *PriorityQueue) Pop(logger klog.Logger) (framework.QueuedEntityInfo, error) {
	return p.activeQ.pop(logger)
}

// Done must be called for pod returned by Pop. This allows the queue to
// keep track of which pods are currently being processed.
func (p *PriorityQueue) Done(pod types.UID) {
	p.activeQ.done(pod)
}

func (p *PriorityQueue) InFlightPods() []*v1.Pod {
	return p.activeQ.listInFlightPods()
}

// Update updates a pod in the active or backoff queue if present. Otherwise, it removes
// the entity from the unschedulable queue if pod is updated in a way that it may
// become schedulable and adds the updated one to the active queue.
// If pod is not present in any of the queues, it is added to the active queue.
func (p *PriorityQueue) Update(ctx context.Context, oldPod, newPod *v1.Pod) {
	p.lock.Lock()
	defer p.lock.Unlock()
	logger := klog.FromContext(ctx)

	// Because the pod itself is being updated here, events are associated with the resource TargetPod.
	events := framework.PodSchedulingPropertiesChange(newPod, oldPod, true)

	var entityLookup framework.QueuedEntityInfo
	if p.isPodGroupMember(oldPod) {
		entityLookup = newQueuedPodGroupInfoForLookup(oldPod)
	} else {
		entityLookup = newQueuedPodInfoForLookup(oldPod)
	}

	updated := false
	// Run the following code under the activeQ lock to make sure that in the meantime entity is not popped from either activeQ or backoffQ.
	// This way, the event will be registered or the entity will be updated consistently.
	// Locking only the part of Update method is sufficient, because in the other part the entity is in the unschedulableEntities,
	// which is protected by p.lock anyway.
	p.activeQ.underLock(func(unlockedActiveQ unlockedActiveQueuer) {
		// The inflight pod will be requeued using the latest version from the informer cache, which matches what the event delivers.
		// Record this Pod update because
		// this update may make the Pod schedulable in case it gets rejected and comes back to the queue.
		// We can clean it up once we change updatePodInSchedulingQueue to call MoveAllToActiveOrBackoffQueue.
		// See https://github.com/kubernetes/kubernetes/pull/125578#discussion_r1648338033 for more context.
		if exists := unlockedActiveQ.addEventsIfPodInFlight(oldPod, newPod, events); exists {
			logger.V(6).Info("The pod doesn't need to be queued for now because it's being scheduled and will be queued back if necessary", "pod", klog.KObj(newPod))
			updated = true
			return
		}
		// If the entity is already in the active queue, just update the pod there.
		if pInfo := unlockedActiveQ.update(newPod, entityLookup); pInfo != nil {
			p.UpdateNominatedPod(logger, oldPod, pInfo.PodInfo)
			pInfo.PodSignature = p.signPod(ctx, newPod)
			updated = true
			return
		}

		// If the entity is in the backoff queue, update the pod there.
		if pInfo := p.backoffQ.update(newPod, entityLookup); pInfo != nil {
			p.UpdateNominatedPod(logger, oldPod, pInfo.PodInfo)
			pInfo.PodSignature = p.signPod(ctx, newPod)
			updated = true
			return
		}
	})

	if updated {
		return
	}

	// If the pod is in the unschedulable queue, updating it may make it schedulable.
	if entity := p.unschedulableEntities.get(entityLookup); entity != nil {
		pInfo, err := entity.Update(newPod)
		if err != nil {
			logger.Error(err, "Failed to update pod in an entity", "type", entity.Type(), "entity", klog.KObj(entity), "pod", klog.KObj(newPod))
			// Handle this case gracefully by adding the newPod to the queue.
			p.addPod(ctx, newPod)
			return
		}
		p.UpdateNominatedPod(logger, oldPod, pInfo.PodInfo)
		pInfo.PodSignature = p.signPod(ctx, newPod)

		// When unscheduled Pods are updated, we check with QueueingHint
		// whether the update may make the entities schedulable.
		// Plugins have to implement a QueueingHint for Pod/Update event
		// if the rejection from them could be resolved by updating unscheduled Pods itself.
		for _, evt := range events {
			hint := p.isEntityWorthRequeuing(logger, entity, evt, oldPod, newPod)
			queue := p.requeueEntityWithQueueingStrategy(logger, entity, hint, evt.Label())
			if queue != unschedulableQ {
				logger.V(5).Info("Entity moved to an internal scheduling queue because the Pod is updated", "type", entity.Type(), "entity", klog.KObj(entity), "pod", klog.KObj(newPod), "event", evt.Label(), "queue", queue)
			}
			if queue == activeQ || (p.isPopFromBackoffQEnabled && queue == backoffQ) {
				p.activeQ.broadcast()
				break
			}
		}
		return
	} else if p.isPodGroupMember(newPod) {
		pgInfoLookup := entityLookup.(*framework.QueuedPodGroupInfo)
		if p.pendingPodGroupPods.has(pgInfoLookup) {
			if updated := p.pendingPodGroupPods.update(pgInfoLookup, newPod); updated {
				return
			}
		}
	}
	// If the entity is not in any of the queues, we add it.
	p.addPod(ctx, newPod)
}

// decreaseUnschedulableReasonMetric decreases the metrics for the rejector plugins
// which are both UnschedulablePlugins and PendingPlugins.
func decreaseUnschedulableReasonMetric(entity framework.QueuedEntityInfo) {
	entity.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
		for plugin := range pInfo.UnschedulablePlugins.Union(pInfo.PendingPlugins) {
			metrics.UnschedulableReason(plugin, pInfo.Pod.Spec.SchedulerName).Dec()
		}
		return true
	})
}

// Delete deletes the entity from activeQ, backoffQ or unschedulableEntities.
// It assumes the pod is only in one structure.
func (p *PriorityQueue) Delete(logger klog.Logger, pod *v1.Pod) {
	p.lock.Lock()
	defer p.lock.Unlock()

	p.DeleteNominatedPodIfExists(pod)
	if p.isPodGroupMember(pod) {
		p.deletePodGroupMember(logger, pod)
	} else {
		p.deletePod(pod)
	}
}

// deletePodGroupMember removes a pod from its pod group in the queue.
// If the pod group is empty after removal, the pod group is removed from the queue.
func (p *PriorityQueue) deletePodGroupMember(logger klog.Logger, pod *v1.Pod) {
	pgInfoLookup := newQueuedPodGroupInfoForLookup(pod)

	entity, strategy := p.deleteFromAnyQueue(pgInfoLookup)
	if entity == nil {
		pgInfoLookup := newQueuedPodGroupInfoForLookup(pod)
		pInfo := p.pendingPodGroupPods.delete(pgInfoLookup, pod)
		if pInfo == nil {
			return
		}
		// Drop metric for deleted pod.
		for plugin := range pInfo.UnschedulablePlugins.Union(pInfo.PendingPlugins) {
			metrics.UnschedulableReason(plugin, pInfo.Pod.Spec.SchedulerName).Dec()
		}
		logger.V(5).Info("Pod deleted from pending pod group info", "podGroup", klog.KObj(pgInfoLookup), "pod", klog.KObj(pInfo))
		return
	}
	pgInfo := entity.(*framework.QueuedPodGroupInfo)

	pInfo := pgInfo.RemovePod(pod)
	if pInfo != nil {
		// Drop metric for deleted pod.
		for plugin := range pInfo.UnschedulablePlugins.Union(pInfo.PendingPlugins) {
			metrics.UnschedulableReason(plugin, pInfo.Pod.Spec.SchedulerName).Dec()
		}
		logger.V(5).Info("Pod deleted from existing pod group info", "podGroup", klog.KObj(pgInfo), "pod", klog.KObj(pInfo))
	}
	if len(pgInfo.QueuedPodInfos) == 0 {
		// The pod group is empty, so don't add it back to any queue.
		return
	}

	queue := p.requeueEntityWithQueueingStrategy(logger, pgInfo, strategy, framework.EventUnscheduledPodAdd.Label())
	if queue == activeQ || (p.isPopFromBackoffQEnabled && queue == backoffQ) {
		p.activeQ.broadcast()
	}
}

// deletePod removes an individual pod from the queue.
func (p *PriorityQueue) deletePod(pod *v1.Pod) {
	pInfoLookup := newQueuedPodInfoForLookup(pod)

	// Check activeQ
	if entity := p.activeQ.delete(pInfoLookup); entity != nil {
		// Drop metric for deleted pod.
		decreaseUnschedulableReasonMetric(entity)
		return
	}

	// Check backoffQ
	if entity := p.backoffQ.delete(pInfoLookup); entity != nil {
		// Drop metric for deleted pod.
		decreaseUnschedulableReasonMetric(entity)
		return
	}

	// Check unschedulableEntities
	if entity := p.unschedulableEntities.get(pInfoLookup); entity != nil {
		p.unschedulableEntities.delete(pInfoLookup, entity.Gated())
		// Drop metric for deleted pod.
		decreaseUnschedulableReasonMetric(entity)
	}
}

// AssignedPodAdded is called when a bound pod is added. Creation of this pod
// may make pending pods with matching affinity terms schedulable.
func (p *PriorityQueue) AssignedPodAdded(logger klog.Logger, pod *v1.Pod) {
	p.lock.Lock()

	// Pre-filter Pods to move by getUnschedulablePodsWithCrossTopologyTerm
	// because Pod related events shouldn't make Pods that rejected by single-node scheduling requirement schedulable.
	p.moveEntitiesToActiveOrBackoffQueue(logger, p.getUnschedulablePodsWithCrossTopologyTerm(logger, pod), framework.EventAssignedPodAdd, nil, pod)
	p.lock.Unlock()
}

// AssignedPodUpdated is called when a bound pod is updated. Change of labels
// may make pending pods with matching affinity terms schedulable.
func (p *PriorityQueue) AssignedPodUpdated(logger klog.Logger, oldPod, newPod *v1.Pod, event fwk.ClusterEvent) {
	p.lock.Lock()
	if (framework.MatchClusterEvents(fwk.ClusterEvent{Resource: fwk.AssignedPod, ActionType: fwk.UpdatePodScaleDown}, event)) {
		// In this case, we don't want to pre-filter Pods by getUnschedulablePodsWithCrossTopologyTerm
		// because Pod related events may make Pods that were rejected by NodeResourceFit schedulable.
		p.moveAllToActiveOrBackoffQueue(logger, event, oldPod, newPod, nil)
	} else {
		// Pre-filter Pods to move by getUnschedulablePodsWithCrossTopologyTerm
		// because Pod related events only make Pods rejected by cross topology term schedulable.
		p.moveEntitiesToActiveOrBackoffQueue(logger, p.getUnschedulablePodsWithCrossTopologyTerm(logger, newPod), event, oldPod, newPod)
	}
	p.lock.Unlock()
}

// NOTE: this function assumes a lock has been acquired in the caller.
// moveAllToActiveOrBackoffQueue moves all pods from unschedulableEntities to activeQ or backoffQ.
// This function adds all pods and then signals the condition variable to ensure that
// if Pop() is waiting for an entity, it receives the signal after all the pods are in the
// queue and the head is the highest priority pod.
func (p *PriorityQueue) moveAllToActiveOrBackoffQueue(logger klog.Logger, event fwk.ClusterEvent, oldObj, newObj interface{}, preCheck PreEnqueueCheck) {
	if !p.isEventOfInterest(logger, event) {
		// No plugin is interested in this event.
		// Return early before iterating all entities in unschedulableEntities for preCheck.
		return
	}

	unschedulableEntities := make([]framework.QueuedEntityInfo, 0, len(p.unschedulableEntities.entityInfoMap))
	for _, entity := range p.unschedulableEntities.entityInfoMap {
		entity.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
			if preCheck == nil || preCheck(pInfo.Pod) {
				unschedulableEntities = append(unschedulableEntities, entity)
				return false
			}
			return true
		})
	}
	p.moveEntitiesToActiveOrBackoffQueue(logger, unschedulableEntities, event, oldObj, newObj)
}

// MoveAllToActiveOrBackoffQueue moves all pods from unschedulableEntities to activeQ or backoffQ.
// This function adds all pods and then signals the condition variable to ensure that
// if Pop() is waiting for an item, it receives the signal after all the pods are in the
// queue and the head is the highest priority pod.
func (p *PriorityQueue) MoveAllToActiveOrBackoffQueue(logger klog.Logger, event fwk.ClusterEvent, oldObj, newObj interface{}, preCheck PreEnqueueCheck) {
	p.lock.Lock()
	defer p.lock.Unlock()
	p.moveAllToActiveOrBackoffQueue(logger, event, oldObj, newObj, preCheck)
}

// requeueEntityWithQueueingStrategy tries to requeue entity to activeQ, backoffQ or unschedulable pod pool based on schedulingHint.
// It returns the queue name entity goes.
//
// NOTE: this function assumes lock has been acquired in caller
func (p *PriorityQueue) requeueEntityWithQueueingStrategy(logger klog.Logger, entity framework.QueuedEntityInfo, strategy queueingStrategy, event string) string {
	if strategy == queueSkip {
		// Current Gate status is required for already exisiting entities. For new entities, this parameter is ignored/unused by addPod/addPodGroup.
		p.unschedulableEntities.addOrUpdate(entity, entity.Gated(), event)
		return unschedulableQ
	}

	// Entity might have completed its backoff time while being in unschedulableEntities,
	// so we should check isEntityBackingoff before moving the entity to backoffQ.
	if strategy == queueAfterBackoff && p.backoffQ.isEntityBackingoff(entity) {
		if added := p.moveToBackoffQ(logger, entity, event); added {
			return backoffQ
		}
		return unschedulableQ
	}

	// Reach here if schedulingHint is QueueImmediately, or schedulingHint is Queue but the entity is not backing off.
	if added := p.moveToActiveQ(logger, entity, event, false); added {
		return activeQ
	}
	// Entity is gated. We don't have to push it back to unschedulable queue, because moveToActiveQ should already have done that.
	return unschedulableQ
}

// NOTE: this function assumes lock has been acquired in caller
func (p *PriorityQueue) moveEntitiesToActiveOrBackoffQueue(logger klog.Logger, entityInfoList []framework.QueuedEntityInfo, event fwk.ClusterEvent, oldObj, newObj interface{}) {
	if !p.isEventOfInterest(logger, event) {
		// No plugin is interested in this event.
		return
	}

	activated := false
	for _, entity := range entityInfoList {
		// As an optimization, we avoid re-evaluating gated entities for events unrelated to their gating plugin.
		// However, wildcard events (e.g., periodic flushes) always trigger re-evaluation to entities pods don't
		// get stuck due to incomplete or incorrect queueing hints.
		if entity.Gated() && !framework.ClusterEventIsWildCard(event) && !framework.MatchAnyClusterEvent(event, entity.GetGatingPluginEvents()) {
			// This event doesn't interest the gating plugin of this Pod,
			// which means this event never moves this Pod to activeQ.
			continue
		}

		schedulingHint := p.isEntityWorthRequeuing(logger, entity, event, oldObj, newObj)
		if schedulingHint == queueSkip {
			// QueueingHintFn determined that this entity isn't worth putting to activeQ or backoffQ by this event.
			logger.V(5).Info("Event is not making entity schedulable", "type", entity.Type(), "entity", entity, "event", event.Label())
			continue
		}

		p.unschedulableEntities.delete(entity, entity.Gated())
		queue := p.requeueEntityWithQueueingStrategy(logger, entity, schedulingHint, event.Label())
		if queue == activeQ || (p.isPopFromBackoffQEnabled && queue == backoffQ) {
			activated = true
		}
	}

	// AddUnschedulablePodIfNotPresent might get called for in-flight Pods later, and in
	// AddUnschedulablePodIfNotPresent we need to know whether events were
	// observed while scheduling them.
	if added := p.activeQ.addEventIfAnyInFlight(oldObj, newObj, event); added {
		logger.V(5).Info("Event received while entities are in flight", "event", event.Label())
	}

	if activated {
		p.activeQ.broadcast()
	}
}

// getUnschedulablePodsWithCrossTopologyTerm returns unschedulable pods which either of following conditions is met:
// - have any affinity term that matches "pod".
// - rejected by PodTopologySpread plugin.
// NOTE: this function assumes lock has been acquired in caller.
func (p *PriorityQueue) getUnschedulablePodsWithCrossTopologyTerm(logger klog.Logger, pod *v1.Pod) []framework.QueuedEntityInfo {
	nsLabels := interpodaffinity.GetNamespaceLabelsSnapshot(logger, pod.Namespace, p.nsLister)

	var entitiesToMove []framework.QueuedEntityInfo
	for _, entity := range p.unschedulableEntities.entityInfoMap {
		entity.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
			if pInfo.UnschedulablePlugins.Has(podtopologyspread.Name) && pod.Namespace == pInfo.Pod.Namespace {
				// This Pod may be schedulable now by this Pod event.
				// Any pod matches for an entity, can move to the next entity.
				entitiesToMove = append(entitiesToMove, entity)
				return false
			}

			for _, term := range pInfo.RequiredAffinityTerms {
				if term.Matches(pod, nsLabels) {
					// Any pod matches for an entity, can move to the next entity.
					entitiesToMove = append(entitiesToMove, entity)
					return false
				}
			}
			return true
		})
	}

	return entitiesToMove
}

// PodsInActiveQ returns all the Pods in the activeQ.
func (p *PriorityQueue) PodsInActiveQ() []*v1.Pod {
	return p.activeQ.list()
}

// PodsInBackoffQ returns all the Pods in the backoffQ.
func (p *PriorityQueue) PodsInBackoffQ() []*v1.Pod {
	return p.backoffQ.list()
}

// UnschedulablePods returns all the pods in unschedulable state.
func (p *PriorityQueue) UnschedulablePods() []*v1.Pod {
	p.lock.RLock()
	defer p.lock.RUnlock()
	var result []*v1.Pod
	for _, entity := range p.unschedulableEntities.entityInfoMap {
		entity.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
			result = append(result, pInfo.Pod)
			return true
		})
	}
	return result
}

// PendingPodGroupPods returns all the pending pods waiting for their pod groups.
func (p *PriorityQueue) PendingPodGroupPods() []*v1.Pod {
	p.lock.RLock()
	defer p.lock.RUnlock()
	var result []*v1.Pod
	for _, pInfos := range p.pendingPodGroupPods.podGroupToPods {
		for _, pInfo := range pInfos {
			result = append(result, pInfo.Pod)
		}
	}
	return result
}

// GetPod searches for a pod in the activeQ, backoffQ, and unschedulableEntities.
func (p *PriorityQueue) GetPod(name, namespace string, schedulingGroup *v1.PodSchedulingGroup) (*framework.QueuedPodInfo, bool) {
	p.lock.RLock()
	defer p.lock.RUnlock()

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1.PodSpec{
			SchedulingGroup: schedulingGroup,
		},
	}
	var pInfo *framework.QueuedPodInfo
	p.activeQ.underRLock(func(unlockedActiveQ unlockedActiveQueueReader) {
		pInfo = p.getPod(pod, unlockedActiveQ)
	})
	return pInfo, pInfo != nil
}

func (p *PriorityQueue) getPod(podLookup *v1.Pod, unlockedActiveQ unlockedActiveQueueReader) *framework.QueuedPodInfo {
	var entityLookup framework.QueuedEntityInfo
	if p.isPodGroupMember(podLookup) {
		entityLookup = newQueuedPodGroupInfoForLookup(podLookup)
	} else {
		entityLookup = newQueuedPodInfoForLookup(podLookup)
	}

	entity := p.getEntityFromAnyQueue(unlockedActiveQ, entityLookup)
	if entity == nil {
		if !p.isPodGroupMember(podLookup) {
			return nil
		}
		return p.pendingPodGroupPods.getPod(entityLookup, podLookup)
	}
	var foundPodInfo *framework.QueuedPodInfo
	entity.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
		if pInfo.Pod.Name == podLookup.Name && pInfo.Pod.Namespace == podLookup.Namespace {
			foundPodInfo = pInfo
			return false
		}
		return true
	})
	return foundPodInfo
}

var pendingPodsSummary = "activeQ:%v; backoffQ:%v; unschedulableEntities:%v"
var pendingPodsExtendedSummary = "activeQ:%v; backoffQ:%v; unschedulableEntities:%v; pendingPodGroupMemberPods:%v"

// PendingPods returns all the pending pods in the queue; accompanied by a debugging string
// recording showing the number of pods in each queue respectively.
// This function is used for debugging purposes in the scheduler cache dumper and comparer.
func (p *PriorityQueue) PendingPods() ([]*v1.Pod, string) {
	p.lock.RLock()
	defer p.lock.RUnlock()
	result := p.PodsInActiveQ()
	activeQLen := len(result)
	backoffQPods := p.PodsInBackoffQ()
	backoffQLen := len(backoffQPods)
	result = append(result, backoffQPods...)
	unschedulablePodsLen := 0
	for _, entity := range p.unschedulableEntities.entityInfoMap {
		entity.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
			result = append(result, pInfo.Pod)
			unschedulablePodsLen++
			return true
		})
	}
	if !p.isGenericWorkloadEnabled {
		return result, fmt.Sprintf(pendingPodsSummary, activeQLen, backoffQLen, unschedulablePodsLen)
	}
	pendingPodGroupPodsLen := p.pendingPodGroupPods.len()
	return result, fmt.Sprintf(pendingPodsExtendedSummary, activeQLen, backoffQLen, unschedulablePodsLen, pendingPodGroupPodsLen)
}

// PatchPodStatus handles the pod status update by sending an update API call through API dispatcher.
// This method should be used only if the SchedulerAsyncAPICalls feature gate is enabled.
func (p *PriorityQueue) PatchPodStatus(pod *v1.Pod, conditions []*v1.PodCondition, nominatingInfo *fwk.NominatingInfo) (<-chan error, error) {
	// Don't store anything in the cache. This might be extended in the next releases.
	onFinish := make(chan error, 1)
	err := p.apiDispatcher.Add(apicalls.Implementations.PodStatusPatch(pod, conditions, nominatingInfo), fwk.APICallOptions{
		OnFinish: onFinish,
	})
	if fwk.IsUnexpectedError(err) {
		return onFinish, err
	}
	return onFinish, nil
}

// Note: this function assumes the caller locks both p.lock.RLock and p.activeQ.getLock().RLock.
func (p *PriorityQueue) nominatedPodToInfo(np podRef, unlockedActiveQ unlockedActiveQueueReader) *framework.PodInfo {
	pod := np.toPod()
	pInfo := p.getPod(pod, unlockedActiveQ)
	if pInfo != nil {
		return pInfo.PodInfo
	}
	return &framework.PodInfo{Pod: pod}
}

// getEntityFromAnyQueue returns an entity from the activeQ, backoffQ, or unschedulableEntities.
// Returns nil if the entity is not found in any queue.
// The appropriate metric for the entity is also returned.
// Note: In case the caller modifies the size of the entity, it should update the queue metric itself to match the new size.
func (p *PriorityQueue) getEntityFromAnyQueue(unlockedActiveQ unlockedActiveQueueReader, entityLookup framework.QueuedEntityInfo) framework.QueuedEntityInfo {
	existing, ok := unlockedActiveQ.get(entityLookup)
	if ok {
		return existing
	}
	existing, ok = p.backoffQ.get(entityLookup)
	if ok {
		return existing
	}
	if existing = p.unschedulableEntities.get(entityLookup); existing != nil {
		if existing.Gated() {
			return existing
		}
		return existing
	}
	return nil
}

// Close closes the priority queue.
func (p *PriorityQueue) Close() {
	p.lock.Lock()
	defer p.lock.Unlock()
	close(p.stop)
	p.activeQ.close()
	p.activeQ.broadcast()
}

// NominatedPodsForNode returns a copy of pods that are nominated to run on the given node,
// but they are waiting for other pods to be removed from the node.
// CAUTION: Make sure you don't call this function while taking any queue's lock in any scenario.
func (p *PriorityQueue) NominatedPodsForNode(nodeName string) []fwk.PodInfo {
	p.lock.RLock()
	defer p.lock.RUnlock()
	nominatedPods := p.nominator.nominatedPodsForNode(nodeName)
	if len(nominatedPods) == 0 {
		return []fwk.PodInfo{}
	}
	pods := make([]fwk.PodInfo, len(nominatedPods))
	p.activeQ.underRLock(func(unlockedActiveQ unlockedActiveQueueReader) {
		for i, np := range nominatedPods {
			pods[i] = p.nominatedPodToInfo(np, unlockedActiveQ).DeepCopy()
		}
	})
	return pods
}

// signPod computes the scheduling signature for the given pod when OpportunisticBatching is enabled.
// The signature is used to cache and reuse scheduling results for identical pods.
func (p *PriorityQueue) signPod(ctx context.Context, pod *v1.Pod) fwk.PodSignature {
	if !p.isOpportunisticBatchingEnabled {
		return nil
	}

	if p.podSigners == nil {
		return nil
	}

	signer, ok := p.podSigners[pod.Spec.SchedulerName]
	if !ok {
		utilruntime.HandleErrorWithContext(ctx, nil, "No signer registered for scheduler profile", "pod", klog.KObj(pod), "schedulerName", pod.Spec.SchedulerName)
		return nil
	}

	return signer(ctx, pod)
}

// newQueuedPodInfo builds a QueuedPodInfo object.
func (p *PriorityQueue) newQueuedPodInfo(ctx context.Context, pod *v1.Pod, plugins ...string) *framework.QueuedPodInfo {
	now := p.clock.Now()
	// ignore this err since apiserver doesn't properly validate affinity terms
	// and we can't fix the validation for backwards compatibility.
	podInfo, _ := framework.NewPodInfo(pod)
	return &framework.QueuedPodInfo{
		PodInfo: podInfo,
		QueueingParams: framework.QueueingParams{
			Timestamp:               now,
			InitialAttemptTimestamp: nil,
			UnschedulablePlugins:    sets.New(plugins...),
		},
		PodSignature: p.signPod(ctx, pod),
	}
}

// newQueuedPodGroupInfo builds a QueuedPodGroupInfo object.
func (p *PriorityQueue) newQueuedPodGroupInfo(podInfo *framework.QueuedPodInfo) *framework.QueuedPodGroupInfo {
	return &framework.QueuedPodGroupInfo{
		PodGroupInfo: &framework.PodGroupInfo{
			Namespace:       podInfo.Pod.Namespace,
			Name:            *podInfo.Pod.Spec.SchedulingGroup.PodGroupName,
			UnscheduledPods: []*v1.Pod{podInfo.Pod},
		},
		QueuedPodInfos: []*framework.QueuedPodInfo{podInfo},
		QueueingParams: framework.QueueingParams{
			Timestamp:               podInfo.Timestamp,
			InitialAttemptTimestamp: podInfo.InitialAttemptTimestamp,
		},
	}
}

// queuedEntityKeyFunc returns a unique key for a queued entity based on its type, namespace, and name.
func queuedEntityKeyFunc(obj framework.QueuedEntityInfo) string {
	return fmt.Sprintf("%s/%s/%s", obj.Type(), obj.GetNamespace(), obj.GetName())
}
