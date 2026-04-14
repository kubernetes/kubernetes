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
// Scheduling queues hold pods waiting to be scheduled. This file implements a
// priority queue which has two sub queues and a additional data structure,
// namely: activeQ, backoffQ and unschedulablePods.
// - activeQ holds pods that are being considered for scheduling.
// - backoffQ holds pods that moved from unschedulablePods and will move to
//   activeQ when their backoff periods complete.
// - unschedulablePods holds pods that were already attempted for scheduling and
//   are currently determined to be unschedulable.

package queue

import (
	"context"
	"fmt"
	"math/rand"
	"reflect"
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
	"k8s.io/client-go/tools/cache"
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
	// time a pod can stay in unschedulablePods. If a pod stays in unschedulablePods
	// for longer than this value, the pod will be moved from unschedulablePods to
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

// SchedulingQueue is an interface for a queue to store pods waiting to be scheduled.
// The interface follows a pattern similar to cache.FIFO and cache.Heap and
// makes it easy to use those data structures as a SchedulingQueue.
type SchedulingQueue interface {
	fwk.PodNominator
	Add(logger klog.Logger, pod *v1.Pod)
	// Activate moves the given pods to activeQ.
	// If a pod isn't found in unschedulablePods or backoffQ and it's in-flight,
	// the wildcard event is registered so that the pod will be requeued when it comes back.
	// But, if a pod isn't found in unschedulablePods or backoffQ and it's not in-flight (i.e., completely unknown pod),
	// Activate would ignore the pod.
	Activate(logger klog.Logger, pods map[string]*v1.Pod)
	// AddUnschedulableIfNotPresent adds an unschedulable pod back to scheduling queue.
	// The podSchedulingCycle represents the current scheduling cycle number which can be
	// returned by calling SchedulingCycle().
	AddUnschedulableIfNotPresent(logger klog.Logger, pod *framework.QueuedPodInfo, podSchedulingCycle int64) error
	// SchedulingCycle returns the current number of scheduling cycle which is
	// cached by scheduling queue. Normally, incrementing this number whenever
	// a pod is popped (e.g. called Pop()) is enough.
	SchedulingCycle() int64
	// Pop removes the head of the queue and returns it. It blocks if the
	// queue is empty and waits until a new item is added to the queue.
	Pop(logger klog.Logger) (*framework.QueuedPodInfo, error)
	// Done must be called for pod returned by Pop. This allows the queue to
	// keep track of which pods are currently being processed.
	Done(types.UID)
	Update(logger klog.Logger, oldPod, newPod *v1.Pod)
	Delete(pod *v1.Pod)
	// Important Note: preCheck shouldn't include anything that depends on the in-tree plugins' logic.
	// (e.g., filter Pods based on added/updated Node's capacity, etc.)
	// We know currently some do, but we'll eventually remove them in favor of the scheduling queue hint.
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
	PatchPodStatus(pod *v1.Pod, condition *v1.PodCondition, nominatingInfo *fwk.NominatingInfo) (<-chan error, error)

	// The following functions are supposed to be used only for testing or debugging.
	GetPod(name, namespace string) (*framework.QueuedPodInfo, bool)
	PendingPods() ([]*v1.Pod, string)
	InFlightPods() []*v1.Pod
	PodsInActiveQ() []*v1.Pod
	// PodsInBackoffQ returns all the Pods in the backoffQ.
	PodsInBackoffQ() []*v1.Pod
	UnschedulablePods() []*v1.Pod
}

// NewSchedulingQueue initializes a priority queue as a new scheduling queue.
func NewSchedulingQueue(
	lessFn fwk.LessFunc,
	informerFactory informers.SharedInformerFactory,
	opts ...Option) SchedulingQueue {
	return NewPriorityQueue(lessFn, informerFactory, opts...)
}

// PriorityQueue implements a scheduling queue.
// The head of PriorityQueue is the highest priority pending pod. This structure
// has two sub queues and a additional data structure, namely: activeQ,
// backoffQ and unschedulablePods.
//   - activeQ holds pods that are being considered for scheduling.
//   - backoffQ holds pods that moved from unschedulablePods and will move to
//     activeQ when their backoff periods complete.
//   - unschedulablePods holds pods that were already attempted for scheduling and
//     are currently determined to be unschedulable.
type PriorityQueue struct {
	*nominator

	stop  chan struct{}
	clock clock.WithTicker

	// lock takes precedence and should be taken first,
	// before any other locks in the queue (activeQueue.lock or backoffQueue.lock or nominator.nLock).
	// Correct locking order is: lock > activeQueue.lock > backoffQueue.lock > nominator.nLock.
	lock sync.RWMutex

	// the maximum time a pod can stay in the unschedulablePods.
	podMaxInUnschedulablePodsDuration time.Duration

	activeQ  activeQueuer
	backoffQ backoffQueuer
	// unschedulablePods holds pods that have been tried and determined unschedulable.
	unschedulablePods *unschedulablePods
	// moveRequestCycle caches the sequence number of scheduling cycle when we
	// received a move request. Unschedulable pods in and before this scheduling
	// cycle will be put back to activeQueue if we were trying to schedule them
	// when we received move request.
	// TODO: this will be removed after SchedulingQueueHint goes to stable and the feature gate is removed.
	moveRequestCycle int64

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

	// isSchedulingQueueHintEnabled indicates whether the feature gate for the scheduling queue is enabled.
	isSchedulingQueueHintEnabled bool
	// isPopFromBackoffQEnabled indicates whether the feature gate SchedulerPopFromBackoffQ is enabled.
	isPopFromBackoffQEnabled bool
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
		PodInfo:              &framework.PodInfo{Pod: pod},
		UnschedulablePlugins: sets.New(plugins...),
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

	isSchedulingQueueHintEnabled := utilfeature.DefaultFeatureGate.Enabled(features.SchedulerQueueingHints)
	isPopFromBackoffQEnabled := utilfeature.DefaultFeatureGate.Enabled(features.SchedulerPopFromBackoffQ)
	lessConverted := convertLessFn(lessFn)

	backoffQ := newBackoffQueue(options.clock, options.podInitialBackoffDuration, options.podMaxBackoffDuration, lessFn, isPopFromBackoffQEnabled)
	pq := &PriorityQueue{
		clock:                             options.clock,
		stop:                              make(chan struct{}),
		podMaxInUnschedulablePodsDuration: options.podMaxInUnschedulablePodsDuration,
		backoffQ:                          backoffQ,
		unschedulablePods:                 newUnschedulablePods(metrics.NewUnschedulablePodsRecorder(), metrics.NewGatedPodsRecorder()),
		preEnqueuePluginMap:               options.preEnqueuePluginMap,
		queueingHintMap:                   options.queueingHintMap,
		pluginToEventsMap:                 buildEventMap(options.queueingHintMap),
		metricsRecorder:                   options.metricsRecorder,
		pluginMetricsSamplePercent:        options.pluginMetricsSamplePercent,
		moveRequestCycle:                  -1,
		apiDispatcher:                     options.apiDispatcher,
		isSchedulingQueueHintEnabled:      isSchedulingQueueHintEnabled,
		isPopFromBackoffQEnabled:          isPopFromBackoffQEnabled,
	}
	var backoffQPopper backoffQPopper
	if isPopFromBackoffQEnabled {
		backoffQPopper = backoffQ
	}
	pq.activeQ = newActiveQueue(heap.NewWithRecorder(podInfoKeyFunc, heap.LessFunc[*framework.QueuedPodInfo](lessConverted), metrics.NewActivePodsRecorder()), isSchedulingQueueHintEnabled, options.metricsRecorder, backoffQPopper)
	pq.nsLister = informerFactory.Core().V1().Namespaces().Lister()
	pq.nominator = newPodNominator(options.podLister)

	return pq
}

// Helper function that wraps fwk.LessFunc and converts it to take *framework.QueuedPodInfo as arguments.
func convertLessFn(lessFn fwk.LessFunc) func(podInfo1, podInfo2 *framework.QueuedPodInfo) bool {
	return func(podInfo1, podInfo2 *framework.QueuedPodInfo) bool {
		return lessFn(podInfo1, podInfo2)
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
		p.flushUnschedulablePodsLeftover(logger)
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

// isPodWorthRequeuing calls QueueingHintFn of only plugins registered in pInfo.unschedulablePlugins and pInfo.PendingPlugins.
//
// If any of pInfo.PendingPlugins return Queue,
// the scheduling queue is supposed to enqueue this Pod to activeQ, skipping backoffQ.
// If any of pInfo.unschedulablePlugins return Queue,
// the scheduling queue is supposed to enqueue this Pod to activeQ/backoffQ depending on the remaining backoff time of the Pod.
// If all QueueingHintFns returns Skip, the scheduling queue enqueues the Pod back to unschedulable Pod pool
// because no plugin changes the scheduling result via the event.
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
// and updates pInfo.GatingPlugin and pInfo.UnschedulablePlugins.
// Note: we need to associate the failed plugin to `pInfo`, so that the pod can be moved back
// to activeQ by related cluster event.
func (p *PriorityQueue) runPreEnqueuePlugins(ctx context.Context, pInfo *framework.QueuedPodInfo) {
	var s *fwk.Status
	pod := pInfo.Pod
	startTime := p.clock.Now()
	defer func() {
		metrics.FrameworkExtensionPointDuration.WithLabelValues(preEnqueue, s.Code().String(), pod.Spec.SchedulerName).Observe(metrics.SinceInSeconds(startTime))
	}()

	shouldRecordMetric := rand.Intn(100) < p.pluginMetricsSamplePercent
	logger := klog.FromContext(ctx)
	gatingPlugin := pInfo.GatingPlugin
	if gatingPlugin != "" {
		// Run the gating plugin first
		s := p.runPreEnqueuePlugin(ctx, logger, p.preEnqueuePluginMap[pod.Spec.SchedulerName][gatingPlugin], pInfo, shouldRecordMetric)
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
		s := p.runPreEnqueuePlugin(ctx, logger, pl, pInfo, shouldRecordMetric)
		if !s.IsSuccess() {
			// No need to iterate other plugins
			return
		}
	}
	// all plugins passed
	pInfo.GatingPlugin = ""
}

// runPreEnqueuePlugin runs the PreEnqueue plugin and update pInfo's fields accordingly if needed.
func (p *PriorityQueue) runPreEnqueuePlugin(ctx context.Context, logger klog.Logger, pl fwk.PreEnqueuePlugin, pInfo *framework.QueuedPodInfo, shouldRecordMetric bool) *fwk.Status {
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
	pInfo.UnschedulablePlugins.Insert(pl.Name())
	metrics.UnschedulableReason(pl.Name(), pod.Spec.SchedulerName).Inc()
	pInfo.GatingPlugin = pl.Name()
	pInfo.GatingPluginEvents = p.pluginToEventsMap[pInfo.GatingPlugin]
	if s.Code() == fwk.Error {
		utilruntime.HandleErrorWithContext(ctx, s.AsError(), "Unexpected error running PreEnqueue plugin", "pod", klog.KObj(pod), "plugin", pl.Name())
	} else {
		logger.V(4).Info("Status after running PreEnqueue plugin", "pod", klog.KObj(pod), "plugin", pl.Name(), "status", s)
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

// moveToActiveQ tries to add the pod to the active queue.
// If the pod doesn't pass PreEnqueue plugins, it gets added to unschedulablePods instead.
// movesFromBackoffQ should be set to true, if the pod directly moves from the backoffQ, so the PreEnqueue call can be skipped.
// It returns a boolean flag to indicate whether the pod is added successfully.
func (p *PriorityQueue) moveToActiveQ(logger klog.Logger, pInfo *framework.QueuedPodInfo, event string, movesFromBackoffQ bool) bool {
	gatedBefore := pInfo.Gated()
	// If SchedulerPopFromBackoffQ feature gate is enabled,
	// PreEnqueue plugins were called when the pod was added to the backoffQ.
	// Don't need to repeat it here when the pod is directly moved from the backoffQ.
	skipPreEnqueue := p.isPopFromBackoffQEnabled && movesFromBackoffQ
	if !skipPreEnqueue {
		p.runPreEnqueuePlugins(context.Background(), pInfo)
	}

	added := false
	p.activeQ.underLock(func(unlockedActiveQ unlockedActiveQueuer) {
		if pInfo.Gated() {
			// Add the Pod to unschedulablePods if it's not passing PreEnqueuePlugins.
			if unlockedActiveQ.has(pInfo) {
				return
			}
			if p.backoffQ.has(pInfo) {
				return
			}
			if p.unschedulablePods.get(pInfo.Pod) != nil {
				return
			}
			p.unschedulablePods.addOrUpdate(pInfo, event)
			logger.V(5).Info("Pod moved to an internal scheduling queue, because the pod is gated", "pod", klog.KObj(pInfo.Pod), "event", event, "queue", unschedulableQ)
			return
		}
		if pInfo.InitialAttemptTimestamp == nil {
			now := p.clock.Now()
			pInfo.InitialAttemptTimestamp = &now
		}
		p.unschedulablePods.delete(pInfo.Pod, gatedBefore)
		p.backoffQ.delete(pInfo)

		unlockedActiveQ.add(logger, pInfo, event)
		added = true
		if event == framework.EventUnscheduledPodAdd.Label() || event == framework.EventUnscheduledPodUpdate.Label() {
			p.nominator.addNominatedPod(logger, pInfo.PodInfo, nil)
		}
	})
	return added
}

// moveToBackoffQ tries to add the pod to the backoff queue.
// If SchedulerPopFromBackoffQ feature gate is enabled and the pod doesn't pass PreEnqueue plugins, it gets added to unschedulablePods instead.
// It returns a boolean flag to indicate whether the pod is added successfully.
func (p *PriorityQueue) moveToBackoffQ(logger klog.Logger, pInfo *framework.QueuedPodInfo, event string) bool {
	gatedBefore := pInfo.Gated()
	// If SchedulerPopFromBackoffQ feature gate is enabled,
	// PreEnqueue plugins are called on inserting pods to the backoffQ,
	// not to call them again on popping out.
	if p.isPopFromBackoffQEnabled {
		p.runPreEnqueuePlugins(context.Background(), pInfo)
		if pInfo.Gated() {
			if p.unschedulablePods.get(pInfo.Pod) == nil {
				p.unschedulablePods.addOrUpdate(pInfo, event)
				logger.V(5).Info("Pod moved to an internal scheduling queue", "pod", klog.KObj(pInfo.Pod), "event", event, "queue", unschedulableQ)
			}
			return false
		}
	}
	p.unschedulablePods.delete(pInfo.Pod, gatedBefore)

	p.backoffQ.add(logger, pInfo, event)
	return true
}

// Add adds a pod to the active queue. It should be called only when a new pod
// is added so there is no chance the pod is already in active/unschedulable/backoff queues
func (p *PriorityQueue) Add(logger klog.Logger, pod *v1.Pod) {
	p.lock.Lock()
	defer p.lock.Unlock()

	pInfo := p.newQueuedPodInfo(pod)
	if added := p.moveToActiveQ(logger, pInfo, framework.EventUnscheduledPodAdd.Label(), false); added {
		p.activeQ.broadcast()
	}
}

// Activate moves the given pods to activeQ.
// If a pod isn't found in unschedulablePods or backoffQ and it's in-flight,
// the wildcard event is registered so that the pod will be requeued when it comes back.
// But, if a pod isn't found in unschedulablePods or backoffQ and it's not in-flight (i.e., completely unknown pod),
// Activate would ignore the pod.
func (p *PriorityQueue) Activate(logger klog.Logger, pods map[string]*v1.Pod) {
	p.lock.Lock()
	defer p.lock.Unlock()

	activated := false
	for _, pod := range pods {
		if p.activate(logger, pod) {
			activated = true
			continue
		}

		// If this pod is in-flight, register the activation event (for when QHint is enabled) or update moveRequestCycle (for when QHints is disabled)
		// so that the pod will be requeued when it comes back.
		// Specifically in the in-tree plugins, this is for the scenario with the preemption plugin
		// where the async preemption API calls are all done or fail at some point before the Pod comes back to the queue.
		p.activeQ.addEventsIfPodInFlight(nil, pod, []fwk.ClusterEvent{framework.EventForceActivate})
		p.moveRequestCycle = p.activeQ.schedulingCycle()
	}

	if activated {
		p.activeQ.broadcast()
	}
}

func (p *PriorityQueue) activate(logger klog.Logger, pod *v1.Pod) bool {
	var pInfo *framework.QueuedPodInfo
	var movesFromBackoffQ bool
	// Verify if the pod is present in unschedulablePods or backoffQ.
	if pInfo = p.unschedulablePods.get(pod); pInfo == nil {
		// If the pod doesn't belong to unschedulablePods or backoffQ, don't activate it.
		// The pod can be already in activeQ.
		var exists bool
		pInfo, exists = p.backoffQ.get(newQueuedPodInfoForLookup(pod))
		if !exists {
			return false
		}
		// Delete pod from the backoffQ now to make sure it won't be popped from the backoffQ
		// just before moving it to the activeQ
		if deleted := p.backoffQ.delete(pInfo); !deleted {
			// Pod was popped from the backoffQ in the meantime. Don't activate it.
			return false
		}
		movesFromBackoffQ = true
	}

	if pInfo == nil {
		// Redundant safe check. We shouldn't reach here.
		utilruntime.HandleErrorWithLogger(logger, nil, "Internal error: cannot obtain pInfo")
		return false
	}

	return p.moveToActiveQ(logger, pInfo, framework.ForceActivate, movesFromBackoffQ)
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

// addUnschedulableWithoutQueueingHint inserts a pod that cannot be scheduled into
// the queue, unless it is already in the queue. Normally, PriorityQueue puts
// unschedulable pods in `unschedulablePods`. But if there has been a recent move
// request, then the pod is put in `backoffQ`.
// TODO: This function is called only when p.isSchedulingQueueHintEnabled is false,
// and this will be removed after SchedulingQueueHint goes to stable and the feature gate is removed.
func (p *PriorityQueue) addUnschedulableWithoutQueueingHint(logger klog.Logger, pInfo *framework.QueuedPodInfo, podSchedulingCycle int64) error {
	pod := pInfo.Pod

	// When the queueing hint is enabled, they are used differently.
	// But, we use all of them as UnschedulablePlugins when the queueing hint isn't enabled so that we don't break the old behaviour.
	rejectorPlugins := pInfo.UnschedulablePlugins.Union(pInfo.PendingPlugins)

	// If a move request has been received, move it to the BackoffQ, otherwise move
	// it to unschedulablePods.
	for plugin := range rejectorPlugins {
		metrics.UnschedulableReason(plugin, pInfo.Pod.Spec.SchedulerName).Inc()
	}
	if p.moveRequestCycle >= podSchedulingCycle || len(rejectorPlugins) == 0 {
		// Two cases to move a Pod to the active/backoff queue:
		// - The Pod is rejected by some plugins, but a move request is received after this Pod's scheduling cycle is started.
		//   In this case, the received event may be make Pod schedulable and we should retry scheduling it.
		// - No unschedulable plugins are associated with this Pod,
		//   meaning something unusual (a temporal failure on kube-apiserver, etc) happened and this Pod gets moved back to the queue.
		//   In this case, we should retry scheduling it because this Pod may not be retried until the next flush.
		if added := p.moveToBackoffQ(logger, pInfo, framework.ScheduleAttemptFailure); added {
			if p.isPopFromBackoffQEnabled {
				p.activeQ.broadcast()
			}
		}
	} else {
		p.unschedulablePods.addOrUpdate(pInfo, framework.ScheduleAttemptFailure)
		logger.V(5).Info("Pod moved to an internal scheduling queue", "pod", klog.KObj(pod), "event", framework.ScheduleAttemptFailure, "queue", unschedulableQ)
	}

	return nil
}

// AddUnschedulableIfNotPresent inserts a pod that cannot be scheduled into
// the queue, unless it is already in the queue. Normally, PriorityQueue puts
// unschedulable pods in `unschedulablePods`. But if there has been a recent move
// request, then the pod is put in `backoffQ`.
func (p *PriorityQueue) AddUnschedulableIfNotPresent(logger klog.Logger, pInfo *framework.QueuedPodInfo, podSchedulingCycle int64) error {
	p.lock.Lock()
	defer p.lock.Unlock()

	// In any case, this Pod will be moved back to the queue and we should call Done.
	defer p.Done(pInfo.Pod.UID)

	pod := pInfo.Pod
	if p.unschedulablePods.get(pod) != nil {
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

	if !p.isSchedulingQueueHintEnabled {
		// fall back to the old behavior which doesn't depend on the queueing hint.
		return p.addUnschedulableWithoutQueueingHint(logger, pInfo, podSchedulingCycle)
	}

	// If a move request has been received, move it to the BackoffQ, otherwise move
	// it to unschedulablePods.
	rejectorPlugins := pInfo.UnschedulablePlugins.Union(pInfo.PendingPlugins)
	for plugin := range rejectorPlugins {
		metrics.UnschedulableReason(plugin, pInfo.Pod.Spec.SchedulerName).Inc()
	}

	// We check whether this Pod may change its scheduling result by any of events that happened during scheduling.
	schedulingHint := p.determineSchedulingHintForInFlightPod(logger, pInfo)

	// In this case, we try to requeue this Pod to activeQ/backoffQ.
	queue := p.requeuePodWithQueueingStrategy(logger, pInfo, schedulingHint, framework.ScheduleAttemptFailure)
	logger.V(3).Info("Pod moved to an internal scheduling queue", "pod", klog.KObj(pod), "event", framework.ScheduleAttemptFailure, "queue", queue, "schedulingCycle", podSchedulingCycle, "hint", schedulingHint, "unschedulable plugins", rejectorPlugins)
	if queue == activeQ || (p.isPopFromBackoffQEnabled && queue == backoffQ) {
		// When the Pod is moved to activeQ, need to let p.cond know so that the Pod will be pop()ed out.
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

// flushUnschedulablePodsLeftover moves pods which stay in unschedulablePods
// longer than podMaxInUnschedulablePodsDuration to backoffQ or activeQ.
func (p *PriorityQueue) flushUnschedulablePodsLeftover(logger klog.Logger) {
	p.lock.Lock()
	defer p.lock.Unlock()

	var podsToMove []*framework.QueuedPodInfo
	currentTime := p.clock.Now()
	for _, pInfo := range p.unschedulablePods.podInfoMap {
		lastScheduleTime := pInfo.Timestamp
		if currentTime.Sub(lastScheduleTime) > p.podMaxInUnschedulablePodsDuration {
			podsToMove = append(podsToMove, pInfo)
		}
	}

	if len(podsToMove) > 0 {
		p.movePodsToActiveOrBackoffQueue(logger, podsToMove, framework.EventUnschedulableTimeout, nil, nil)
	}
}

// Pop removes the head of the active queue and returns it. It blocks if the
// activeQ is empty and waits until a new item is added to the queue. It
// increments scheduling cycle when a pod is popped.
// Note: This method should NOT be locked by the p.lock at any moment,
// as it would lead to scheduling throughput degradation.
func (p *PriorityQueue) Pop(logger klog.Logger) (*framework.QueuedPodInfo, error) {
	return p.activeQ.pop(logger)
}

// Done must be called for pod returned by Pop. This allows the queue to
// keep track of which pods are currently being processed.
func (p *PriorityQueue) Done(pod types.UID) {
	if !p.isSchedulingQueueHintEnabled {
		// do nothing if schedulingQueueHint is disabled.
		// In that case, we don't have inFlightPods and inFlightEvents.
		return
	}
	p.activeQ.done(pod)
}

func (p *PriorityQueue) InFlightPods() []*v1.Pod {
	if !p.isSchedulingQueueHintEnabled {
		// do nothing if schedulingQueueHint is disabled.
		// In that case, we don't have inFlightPods and inFlightEvents.
		return nil
	}
	return p.activeQ.listInFlightPods()
}

// isPodUpdated checks if the pod is updated in a way that it may have become
// schedulable. It drops status of the pod and compares it with old version,
// except for pod.status.resourceClaimStatuses and
// pod.status.extendedResourceClaimStatus: changing that may have an
// effect on scheduling.
func isPodUpdated(oldPod, newPod *v1.Pod) bool {
	strip := func(pod *v1.Pod) *v1.Pod {
		p := pod.DeepCopy()
		p.ResourceVersion = ""
		p.Generation = 0
		p.Status = v1.PodStatus{
			ResourceClaimStatuses:       pod.Status.ResourceClaimStatuses,
			ExtendedResourceClaimStatus: pod.Status.ExtendedResourceClaimStatus,
		}
		p.ManagedFields = nil
		p.Finalizers = nil
		return p
	}
	return !reflect.DeepEqual(strip(oldPod), strip(newPod))
}

// Update updates a pod in the active or backoff queue if present. Otherwise, it removes
// the item from the unschedulable queue if pod is updated in a way that it may
// become schedulable and adds the updated one to the active queue.
// If pod is not present in any of the queues, it is added to the active queue.
func (p *PriorityQueue) Update(logger klog.Logger, oldPod, newPod *v1.Pod) {
	p.lock.Lock()
	defer p.lock.Unlock()

	var events []fwk.ClusterEvent
	if p.isSchedulingQueueHintEnabled {
		events = framework.PodSchedulingPropertiesChange(newPod, oldPod)
	}

	updated := false
	// Run the following code under the activeQ lock to make sure that in the meantime pod is not popped from either activeQ or backoffQ.
	// This way, the event will be registered or the pod will be updated consistently.
	// Locking only the part of Update method is sufficient, because in the other part the pod is in the unscheduledPods
	// which is protected by p.lock anyway.
	p.activeQ.underLock(func(unlockedActiveQ unlockedActiveQueuer) {
		if p.isSchedulingQueueHintEnabled {
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
		}
		if oldPod != nil {
			oldPodInfo := newQueuedPodInfoForLookup(oldPod)
			// If the pod is already in the active queue, just update it there.
			if pInfo := unlockedActiveQ.update(newPod, oldPodInfo); pInfo != nil {
				p.UpdateNominatedPod(logger, oldPod, pInfo.PodInfo)
				updated = true
				return
			}

			// If the pod is in the backoff queue, update it there.
			if pInfo := p.backoffQ.update(newPod, oldPodInfo); pInfo != nil {
				p.UpdateNominatedPod(logger, oldPod, pInfo.PodInfo)
				updated = true
				return
			}
		}
	})
	if updated {
		return
	}

	// If the pod is in the unschedulable queue, updating it may make it schedulable.
	if pInfo := p.unschedulablePods.get(newPod); pInfo != nil {
		_ = pInfo.Update(newPod)
		p.UpdateNominatedPod(logger, oldPod, pInfo.PodInfo)
		if p.isSchedulingQueueHintEnabled {
			// When unscheduled Pods are updated, we check with QueueingHint
			// whether the update may make the pods schedulable.
			// Plugins have to implement a QueueingHint for Pod/Update event
			// if the rejection from them could be resolved by updating unscheduled Pods itself.
			for _, evt := range events {
				hint := p.isPodWorthRequeuing(logger, pInfo, evt, oldPod, newPod)
				queue := p.requeuePodWithQueueingStrategy(logger, pInfo, hint, evt.Label())
				if queue != unschedulableQ {
					logger.V(5).Info("Pod moved to an internal scheduling queue because the Pod is updated", "pod", klog.KObj(newPod), "event", evt.Label(), "queue", queue)
				}
				if queue == activeQ || (p.isPopFromBackoffQEnabled && queue == backoffQ) {
					p.activeQ.broadcast()
					break
				}
			}
			return
		}
		if isPodUpdated(oldPod, newPod) {
			// Pod might have completed its backoff time while being in unschedulablePods,
			// so we should check isPodBackingoff before moving the pod to backoffQ.
			if p.backoffQ.isPodBackingoff(pInfo) {
				if added := p.moveToBackoffQ(logger, pInfo, framework.EventUnscheduledPodUpdate.Label()); added {
					if p.isPopFromBackoffQEnabled {
						p.activeQ.broadcast()
					}
				}
				return
			}

			if added := p.moveToActiveQ(logger, pInfo, framework.EventUnscheduledPodUpdate.Label(), false); added {
				p.activeQ.broadcast()
			}
			return
		}

		// Pod update didn't make it schedulable, keep it in the unschedulable queue.
		p.unschedulablePods.addOrUpdate(pInfo, framework.EventUnscheduledPodUpdate.Label())
		return
	}
	// If pod is not in any of the queues, we put it in the active queue.
	pInfo := p.newQueuedPodInfo(newPod)
	if added := p.moveToActiveQ(logger, pInfo, framework.EventUnscheduledPodUpdate.Label(), false); added {
		p.activeQ.broadcast()
	}
}

// Delete deletes the item from either of the two queues. It assumes the pod is
// only in one queue.
func (p *PriorityQueue) Delete(pod *v1.Pod) {
	p.lock.Lock()
	defer p.lock.Unlock()
	p.DeleteNominatedPodIfExists(pod)
	pInfo := newQueuedPodInfoForLookup(pod)
	if err := p.activeQ.delete(pInfo); err == nil {
		return
	}
	if deleted := p.backoffQ.delete(pInfo); deleted {
		return
	}
	if pInfo = p.unschedulablePods.get(pod); pInfo != nil {
		p.unschedulablePods.delete(pod, pInfo.Gated())
	}
}

// AssignedPodAdded is called when a bound pod is added. Creation of this pod
// may make pending pods with matching affinity terms schedulable.
func (p *PriorityQueue) AssignedPodAdded(logger klog.Logger, pod *v1.Pod) {
	p.lock.Lock()

	// Pre-filter Pods to move by getUnschedulablePodsWithCrossTopologyTerm
	// because Pod related events shouldn't make Pods that rejected by single-node scheduling requirement schedulable.
	p.movePodsToActiveOrBackoffQueue(logger, p.getUnschedulablePodsWithCrossTopologyTerm(logger, pod), framework.EventAssignedPodAdd, nil, pod)
	p.lock.Unlock()
}

// AssignedPodUpdated is called when a bound pod is updated. Change of labels
// may make pending pods with matching affinity terms schedulable.
func (p *PriorityQueue) AssignedPodUpdated(logger klog.Logger, oldPod, newPod *v1.Pod, event fwk.ClusterEvent) {
	p.lock.Lock()
	if (framework.MatchClusterEvents(fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdatePodScaleDown}, event)) {
		// In this case, we don't want to pre-filter Pods by getUnschedulablePodsWithCrossTopologyTerm
		// because Pod related events may make Pods that were rejected by NodeResourceFit schedulable.
		p.moveAllToActiveOrBackoffQueue(logger, event, oldPod, newPod, nil)
	} else {
		// Pre-filter Pods to move by getUnschedulablePodsWithCrossTopologyTerm
		// because Pod related events only make Pods rejected by cross topology term schedulable.
		p.movePodsToActiveOrBackoffQueue(logger, p.getUnschedulablePodsWithCrossTopologyTerm(logger, newPod), event, oldPod, newPod)
	}
	p.lock.Unlock()
}

// NOTE: this function assumes a lock has been acquired in the caller.
// moveAllToActiveOrBackoffQueue moves all pods from unschedulablePods to activeQ or backoffQ.
// This function adds all pods and then signals the condition variable to ensure that
// if Pop() is waiting for an item, it receives the signal after all the pods are in the
// queue and the head is the highest priority pod.
func (p *PriorityQueue) moveAllToActiveOrBackoffQueue(logger klog.Logger, event fwk.ClusterEvent, oldObj, newObj interface{}, preCheck PreEnqueueCheck) {
	if !p.isEventOfInterest(logger, event) {
		// No plugin is interested in this event.
		// Return early before iterating all pods in unschedulablePods for preCheck.
		return
	}

	unschedulablePods := make([]*framework.QueuedPodInfo, 0, len(p.unschedulablePods.podInfoMap))
	for _, pInfo := range p.unschedulablePods.podInfoMap {
		if preCheck == nil || preCheck(pInfo.Pod) {
			unschedulablePods = append(unschedulablePods, pInfo)
		}
	}
	p.movePodsToActiveOrBackoffQueue(logger, unschedulablePods, event, oldObj, newObj)
}

// MoveAllToActiveOrBackoffQueue moves all pods from unschedulablePods to activeQ or backoffQ.
// This function adds all pods and then signals the condition variable to ensure that
// if Pop() is waiting for an item, it receives the signal after all the pods are in the
// queue and the head is the highest priority pod.
func (p *PriorityQueue) MoveAllToActiveOrBackoffQueue(logger klog.Logger, event fwk.ClusterEvent, oldObj, newObj interface{}, preCheck PreEnqueueCheck) {
	p.lock.Lock()
	defer p.lock.Unlock()
	p.moveAllToActiveOrBackoffQueue(logger, event, oldObj, newObj, preCheck)
}

// requeuePodWithQueueingStrategy tries to requeue Pod to activeQ, backoffQ or unschedulable pod pool based on schedulingHint.
// It returns the queue name Pod goes.
//
// NOTE: this function assumes lock has been acquired in caller
func (p *PriorityQueue) requeuePodWithQueueingStrategy(logger klog.Logger, pInfo *framework.QueuedPodInfo, strategy queueingStrategy, event string) string {
	if strategy == queueSkip {
		p.unschedulablePods.addOrUpdate(pInfo, event)
		return unschedulableQ
	}

	// Pod might have completed its backoff time while being in unschedulablePods,
	// so we should check isPodBackingoff before moving the pod to backoffQ.
	if strategy == queueAfterBackoff && p.backoffQ.isPodBackingoff(pInfo) {
		if added := p.moveToBackoffQ(logger, pInfo, event); added {
			return backoffQ
		}
		return unschedulableQ
	}

	// Reach here if schedulingHint is QueueImmediately, or schedulingHint is Queue but the pod is not backing off.
	if added := p.moveToActiveQ(logger, pInfo, event, false); added {
		return activeQ
	}
	// Pod is gated. We don't have to push it back to unschedulable queue, because moveToActiveQ should already have done that.
	return unschedulableQ
}

// NOTE: this function assumes lock has been acquired in caller
func (p *PriorityQueue) movePodsToActiveOrBackoffQueue(logger klog.Logger, podInfoList []*framework.QueuedPodInfo, event fwk.ClusterEvent, oldObj, newObj interface{}) {
	if !p.isEventOfInterest(logger, event) {
		// No plugin is interested in this event.
		return
	}

	activated := false
	for _, pInfo := range podInfoList {
		if pInfo.Gated() && !framework.MatchAnyClusterEvent(event, pInfo.GatingPluginEvents) {
			// This event doesn't interest the gating plugin of this Pod,
			// which means this event never moves this Pod to activeQ.
			continue
		}

		schedulingHint := p.isPodWorthRequeuing(logger, pInfo, event, oldObj, newObj)
		if schedulingHint == queueSkip {
			// QueueingHintFn determined that this Pod isn't worth putting to activeQ or backoffQ by this event.
			logger.V(5).Info("Event is not making pod schedulable", "pod", klog.KObj(pInfo.Pod), "event", event.Label())
			continue
		}

		p.unschedulablePods.delete(pInfo.Pod, pInfo.Gated())
		queue := p.requeuePodWithQueueingStrategy(logger, pInfo, schedulingHint, event.Label())
		if queue == activeQ || (p.isPopFromBackoffQEnabled && queue == backoffQ) {
			activated = true
		}
	}

	p.moveRequestCycle = p.activeQ.schedulingCycle()

	if p.isSchedulingQueueHintEnabled {
		// AddUnschedulableIfNotPresent might get called for in-flight Pods later, and in
		// AddUnschedulableIfNotPresent we need to know whether events were
		// observed while scheduling them.
		if added := p.activeQ.addEventIfAnyInFlight(oldObj, newObj, event); added {
			logger.V(5).Info("Event received while pods are in flight", "event", event.Label())
		}
	}

	if activated {
		p.activeQ.broadcast()
	}
}

// getUnschedulablePodsWithCrossTopologyTerm returns unschedulable pods which either of following conditions is met:
// - have any affinity term that matches "pod".
// - rejected by PodTopologySpread plugin.
// NOTE: this function assumes lock has been acquired in caller.
func (p *PriorityQueue) getUnschedulablePodsWithCrossTopologyTerm(logger klog.Logger, pod *v1.Pod) []*framework.QueuedPodInfo {
	nsLabels := interpodaffinity.GetNamespaceLabelsSnapshot(logger, pod.Namespace, p.nsLister)

	var podsToMove []*framework.QueuedPodInfo
	for _, pInfo := range p.unschedulablePods.podInfoMap {
		if pInfo.UnschedulablePlugins.Has(podtopologyspread.Name) && pod.Namespace == pInfo.Pod.Namespace {
			// This Pod may be schedulable now by this Pod event.
			podsToMove = append(podsToMove, pInfo)
			continue
		}

		for _, term := range pInfo.RequiredAffinityTerms {
			if term.Matches(pod, nsLabels) {
				podsToMove = append(podsToMove, pInfo)
				break
			}
		}
	}

	return podsToMove
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
	var result []*v1.Pod
	for _, pInfo := range p.unschedulablePods.podInfoMap {
		result = append(result, pInfo.Pod)
	}
	return result
}

var pendingPodsSummary = "activeQ:%v; backoffQ:%v; unschedulablePods:%v"

// GetPod searches for a pod in the activeQ, backoffQ, and unschedulablePods.
func (p *PriorityQueue) GetPod(name, namespace string) (pInfo *framework.QueuedPodInfo, ok bool) {
	p.lock.RLock()
	defer p.lock.RUnlock()

	pInfoLookup := &framework.QueuedPodInfo{
		PodInfo: &framework.PodInfo{
			Pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      name,
					Namespace: namespace,
				},
			},
		},
	}
	if pInfo, ok = p.backoffQ.get(pInfoLookup); ok {
		return pInfo, true
	}
	if pInfo = p.unschedulablePods.get(pInfoLookup.Pod); pInfo != nil {
		return pInfo, true
	}

	p.activeQ.underRLock(func(unlockedActiveQ unlockedActiveQueueReader) {
		pInfo, ok = unlockedActiveQ.get(pInfoLookup)
	})
	return
}

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
	for _, pInfo := range p.unschedulablePods.podInfoMap {
		result = append(result, pInfo.Pod)
	}
	return result, fmt.Sprintf(pendingPodsSummary, activeQLen, backoffQLen, len(p.unschedulablePods.podInfoMap))
}

// PatchPodStatus handles the pod status update by sending an update API call through API dispatcher.
// This method should be used only if the SchedulerAsyncAPICalls feature gate is enabled.
func (p *PriorityQueue) PatchPodStatus(pod *v1.Pod, condition *v1.PodCondition, nominatingInfo *fwk.NominatingInfo) (<-chan error, error) {
	// Don't store anything in the cache. This might be extended in the next releases.
	onFinish := make(chan error, 1)
	err := p.apiDispatcher.Add(apicalls.Implementations.PodStatusPatch(pod, condition, nominatingInfo), fwk.APICallOptions{
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
	pInfoLookup := newQueuedPodInfoForLookup(pod)

	queuedPodInfo, exists := unlockedActiveQ.get(pInfoLookup)
	if exists {
		return queuedPodInfo.PodInfo
	}

	queuedPodInfo = p.unschedulablePods.get(pod)
	if queuedPodInfo != nil {
		return queuedPodInfo.PodInfo
	}

	queuedPodInfo, exists = p.backoffQ.get(pInfoLookup)
	if exists {
		return queuedPodInfo.PodInfo
	}

	return &framework.PodInfo{Pod: pod}
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

	pods := make([]fwk.PodInfo, len(nominatedPods))
	p.activeQ.underRLock(func(unlockedActiveQ unlockedActiveQueueReader) {
		for i, np := range nominatedPods {
			pods[i] = p.nominatedPodToInfo(np, unlockedActiveQ).DeepCopy()
		}
	})
	return pods
}

// newQueuedPodInfo builds a QueuedPodInfo object.
func (p *PriorityQueue) newQueuedPodInfo(pod *v1.Pod, plugins ...string) *framework.QueuedPodInfo {
	now := p.clock.Now()
	// ignore this err since apiserver doesn't properly validate affinity terms
	// and we can't fix the validation for backwards compatibility.
	podInfo, _ := framework.NewPodInfo(pod)
	return &framework.QueuedPodInfo{
		PodInfo:                 podInfo,
		Timestamp:               now,
		InitialAttemptTimestamp: nil,
		UnschedulablePlugins:    sets.New(plugins...),
	}
}

func podInfoKeyFunc(pInfo *framework.QueuedPodInfo) string {
	return cache.NewObjectName(pInfo.Pod.Namespace, pInfo.Pod.Name).String()
}
