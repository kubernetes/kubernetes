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
	"container/list"
	"context"
	"fmt"
	"math/rand"
	"reflect"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	listersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/interpodaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/internal/heap"
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
	activeQ           = "Active"
	backoffQ          = "Backoff"
	unschedulablePods = "Unschedulable"

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
	framework.PodNominator
	Add(logger klog.Logger, pod *v1.Pod) error
	// Activate moves the given pods to activeQ iff they're in unschedulablePods or backoffQ.
	// The passed-in pods are originally compiled from plugins that want to activate Pods,
	// by injecting the pods through a reserved CycleState struct (PodsToActivate).
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
	Update(logger klog.Logger, oldPod, newPod *v1.Pod) error
	Delete(pod *v1.Pod) error
	// TODO(sanposhiho): move all PreEnqueueCkeck to Requeue and delete it from this parameter eventually.
	// Some PreEnqueueCheck include event filtering logic based on some in-tree plugins
	// and it affect badly to other plugins.
	// See https://github.com/kubernetes/kubernetes/issues/110175
	MoveAllToActiveOrBackoffQueue(logger klog.Logger, event framework.ClusterEvent, oldObj, newObj interface{}, preCheck PreEnqueueCheck)
	AssignedPodAdded(logger klog.Logger, pod *v1.Pod)
	AssignedPodUpdated(logger klog.Logger, oldPod, newPod *v1.Pod)
	PendingPods() ([]*v1.Pod, string)
	PodsInActiveQ() []*v1.Pod
	// Close closes the SchedulingQueue so that the goroutine which is
	// waiting to pop items can exit gracefully.
	Close()
	// Run starts the goroutines managing the queue.
	Run(logger klog.Logger)
}

// NewSchedulingQueue initializes a priority queue as a new scheduling queue.
func NewSchedulingQueue(
	lessFn framework.LessFunc,
	informerFactory informers.SharedInformerFactory,
	opts ...Option) SchedulingQueue {
	return NewPriorityQueue(lessFn, informerFactory, opts...)
}

// NominatedNodeName returns nominated node name of a Pod.
func NominatedNodeName(pod *v1.Pod) string {
	return pod.Status.NominatedNodeName
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
	clock clock.Clock

	// pod initial backoff duration.
	podInitialBackoffDuration time.Duration
	// pod maximum backoff duration.
	podMaxBackoffDuration time.Duration
	// the maximum time a pod can stay in the unschedulablePods.
	podMaxInUnschedulablePodsDuration time.Duration

	cond sync.Cond

	// inFlightPods holds the UID of all pods which have been popped out for which Done
	// hasn't been called yet - in other words, all pods that are currently being
	// processed (being scheduled, in permit, or in the binding cycle).
	//
	// The values in the map are the entry of each pod in the inFlightEvents list.
	// The value of that entry is the *v1.Pod at the time that scheduling of that
	// pod started, which can be useful for logging or debugging.
	inFlightPods map[types.UID]*list.Element

	// inFlightEvents holds the events received by the scheduling queue
	// (entry value is clusterEvent) together with in-flight pods (entry
	// value is *v1.Pod). Entries get added at the end while the mutex is
	// locked, so they get serialized.
	//
	// The pod entries are added in Pop and used to track which events
	// occurred after the pod scheduling attempt for that pod started.
	// They get removed when the scheduling attempt is done, at which
	// point all events that occurred in the meantime are processed.
	//
	// After removal of a pod, events at the start of the list are no
	// longer needed because all of the other in-flight pods started
	// later. Those events can be removed.
	inFlightEvents *list.List

	// activeQ is heap structure that scheduler actively looks at to find pods to
	// schedule. Head of heap is the highest priority pod.
	activeQ *heap.Heap
	// podBackoffQ is a heap ordered by backoff expiry. Pods which have completed backoff
	// are popped from this heap before the scheduler looks at activeQ
	podBackoffQ *heap.Heap
	// unschedulablePods holds pods that have been tried and determined unschedulable.
	unschedulablePods *UnschedulablePods
	// schedulingCycle represents sequence number of scheduling cycle and is incremented
	// when a pod is popped.
	schedulingCycle int64
	// moveRequestCycle caches the sequence number of scheduling cycle when we
	// received a move request. Unschedulable pods in and before this scheduling
	// cycle will be put back to activeQueue if we were trying to schedule them
	// when we received move request.
	// TODO: this will be removed after SchedulingQueueHint goes to stable and the feature gate is removed.
	moveRequestCycle int64

	// preEnqueuePluginMap is keyed with profile name, valued with registered preEnqueue plugins.
	preEnqueuePluginMap map[string][]framework.PreEnqueuePlugin
	// queueingHintMap is keyed with profile name, valued with registered queueing hint functions.
	queueingHintMap QueueingHintMapPerProfile

	// closed indicates that the queue is closed.
	// It is mainly used to let Pop() exit its control loop while waiting for an item.
	closed bool

	nsLister listersv1.NamespaceLister

	metricsRecorder metrics.MetricAsyncRecorder
	// pluginMetricsSamplePercent is the percentage of plugin metrics to be sampled.
	pluginMetricsSamplePercent int

	// isSchedulingQueueHintEnabled indicates whether the feature gate for the scheduling queue is enabled.
	isSchedulingQueueHintEnabled bool
}

// QueueingHintFunction is the wrapper of QueueingHintFn that has PluginName.
type QueueingHintFunction struct {
	PluginName     string
	QueueingHintFn framework.QueueingHintFn
}

// clusterEvent has the event and involved objects.
type clusterEvent struct {
	event framework.ClusterEvent
	// oldObj is the object that involved this event.
	oldObj interface{}
	// newObj is the object that involved this event.
	newObj interface{}
}

type priorityQueueOptions struct {
	clock                             clock.Clock
	podInitialBackoffDuration         time.Duration
	podMaxBackoffDuration             time.Duration
	podMaxInUnschedulablePodsDuration time.Duration
	podLister                         listersv1.PodLister
	metricsRecorder                   metrics.MetricAsyncRecorder
	pluginMetricsSamplePercent        int
	preEnqueuePluginMap               map[string][]framework.PreEnqueuePlugin
	queueingHintMap                   QueueingHintMapPerProfile
}

// Option configures a PriorityQueue
type Option func(*priorityQueueOptions)

// WithClock sets clock for PriorityQueue, the default clock is clock.RealClock.
func WithClock(clock clock.Clock) Option {
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
type QueueingHintMap map[framework.ClusterEvent][]*QueueingHintFunction

// WithQueueingHintMapPerProfile sets queueingHintMap for PriorityQueue.
func WithQueueingHintMapPerProfile(m QueueingHintMapPerProfile) Option {
	return func(o *priorityQueueOptions) {
		o.queueingHintMap = m
	}
}

// WithPreEnqueuePluginMap sets preEnqueuePluginMap for PriorityQueue.
func WithPreEnqueuePluginMap(m map[string][]framework.PreEnqueuePlugin) Option {
	return func(o *priorityQueueOptions) {
		o.preEnqueuePluginMap = m
	}
}

// WithMetricsRecorder sets metrics recorder.
func WithMetricsRecorder(recorder metrics.MetricAsyncRecorder) Option {
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
	lessFn framework.LessFunc,
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

	comp := func(podInfo1, podInfo2 interface{}) bool {
		pInfo1 := podInfo1.(*framework.QueuedPodInfo)
		pInfo2 := podInfo2.(*framework.QueuedPodInfo)
		return lessFn(pInfo1, pInfo2)
	}

	pq := &PriorityQueue{
		nominator:                         newPodNominator(options.podLister),
		clock:                             options.clock,
		stop:                              make(chan struct{}),
		podInitialBackoffDuration:         options.podInitialBackoffDuration,
		podMaxBackoffDuration:             options.podMaxBackoffDuration,
		podMaxInUnschedulablePodsDuration: options.podMaxInUnschedulablePodsDuration,
		activeQ:                           heap.NewWithRecorder(podInfoKeyFunc, comp, metrics.NewActivePodsRecorder()),
		unschedulablePods:                 newUnschedulablePods(metrics.NewUnschedulablePodsRecorder(), metrics.NewGatedPodsRecorder()),
		inFlightPods:                      make(map[types.UID]*list.Element),
		inFlightEvents:                    list.New(),
		preEnqueuePluginMap:               options.preEnqueuePluginMap,
		queueingHintMap:                   options.queueingHintMap,
		metricsRecorder:                   options.metricsRecorder,
		pluginMetricsSamplePercent:        options.pluginMetricsSamplePercent,
		moveRequestCycle:                  -1,
		isSchedulingQueueHintEnabled:      utilfeature.DefaultFeatureGate.Enabled(features.SchedulerQueueingHints),
	}
	pq.cond.L = &pq.lock
	pq.podBackoffQ = heap.NewWithRecorder(podInfoKeyFunc, pq.podsCompareBackoffCompleted, metrics.NewBackoffPodsRecorder())
	pq.nsLister = informerFactory.Core().V1().Namespaces().Lister()

	return pq
}

// Run starts the goroutine to pump from podBackoffQ to activeQ
func (p *PriorityQueue) Run(logger klog.Logger) {
	go wait.Until(func() {
		p.flushBackoffQCompleted(logger)
	}, 1.0*time.Second, p.stop)
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
func (p *PriorityQueue) isEventOfInterest(logger klog.Logger, event framework.ClusterEvent) bool {
	if event.IsWildCard() {
		return true
	}

	for _, hintMap := range p.queueingHintMap {
		for eventToMatch := range hintMap {
			if eventToMatch.Match(event) {
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
func (p *PriorityQueue) isPodWorthRequeuing(logger klog.Logger, pInfo *framework.QueuedPodInfo, event framework.ClusterEvent, oldObj, newObj interface{}) queueingStrategy {
	rejectorPlugins := pInfo.UnschedulablePlugins.Union(pInfo.PendingPlugins)
	if rejectorPlugins.Len() == 0 {
		logger.V(6).Info("Worth requeuing because no failed plugins", "pod", klog.KObj(pInfo.Pod))
		return queueAfterBackoff
	}

	if event.IsWildCard() {
		// If the wildcard event is special one as someone wants to force all Pods to move to activeQ/backoffQ.
		// We return queueAfterBackoff in this case, while resetting all blocked plugins.
		logger.V(6).Info("Worth requeuing because the event is wildcard", "pod", klog.KObj(pInfo.Pod))
		return queueAfterBackoff
	}

	hintMap, ok := p.queueingHintMap[pInfo.Pod.Spec.SchedulerName]
	if !ok {
		// shouldn't reach here unless bug.
		logger.Error(nil, "No QueueingHintMap is registered for this profile", "profile", pInfo.Pod.Spec.SchedulerName, "pod", klog.KObj(pInfo.Pod))
		return queueAfterBackoff
	}

	pod := pInfo.Pod
	queueStrategy := queueSkip
	for eventToMatch, hintfns := range hintMap {
		if !eventToMatch.Match(event) {
			continue
		}

		for _, hintfn := range hintfns {
			if !rejectorPlugins.Has(hintfn.PluginName) {
				// skip if it's not hintfn from rejectorPlugins.
				continue
			}

			hint, err := hintfn.QueueingHintFn(logger, pod, oldObj, newObj)
			if err != nil {
				// If the QueueingHintFn returned an error, we should treat the event as Queue so that we can prevent
				// the Pod from being stuck in the unschedulable pod pool.
				oldObjMeta, newObjMeta, asErr := util.As[klog.KMetadata](oldObj, newObj)
				if asErr != nil {
					logger.Error(err, "QueueingHintFn returns error", "event", event, "plugin", hintfn.PluginName, "pod", klog.KObj(pod))
				} else {
					logger.Error(err, "QueueingHintFn returns error", "event", event, "plugin", hintfn.PluginName, "pod", klog.KObj(pod), "oldObj", klog.KObj(oldObjMeta), "newObj", klog.KObj(newObjMeta))
				}
				hint = framework.Queue
			}
			if hint == framework.QueueSkip {
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

// runPreEnqueuePlugins iterates PreEnqueue function in each registered PreEnqueuePlugin.
// It returns true if all PreEnqueue function run successfully; otherwise returns false
// upon the first failure.
// Note: we need to associate the failed plugin to `pInfo`, so that the pod can be moved back
// to activeQ by related cluster event.
func (p *PriorityQueue) runPreEnqueuePlugins(ctx context.Context, pInfo *framework.QueuedPodInfo) bool {
	logger := klog.FromContext(ctx)
	var s *framework.Status
	pod := pInfo.Pod
	startTime := p.clock.Now()
	defer func() {
		metrics.FrameworkExtensionPointDuration.WithLabelValues(preEnqueue, s.Code().String(), pod.Spec.SchedulerName).Observe(metrics.SinceInSeconds(startTime))
	}()

	shouldRecordMetric := rand.Intn(100) < p.pluginMetricsSamplePercent
	for _, pl := range p.preEnqueuePluginMap[pod.Spec.SchedulerName] {
		s = p.runPreEnqueuePlugin(ctx, pl, pod, shouldRecordMetric)
		if s.IsSuccess() {
			continue
		}
		pInfo.UnschedulablePlugins.Insert(pl.Name())
		metrics.UnschedulableReason(pl.Name(), pod.Spec.SchedulerName).Inc()
		if s.Code() == framework.Error {
			logger.Error(s.AsError(), "Unexpected error running PreEnqueue plugin", "pod", klog.KObj(pod), "plugin", pl.Name())
		} else {
			logger.V(4).Info("Status after running PreEnqueue plugin", "pod", klog.KObj(pod), "plugin", pl.Name(), "status", s)
		}
		return false
	}
	return true
}

func (p *PriorityQueue) runPreEnqueuePlugin(ctx context.Context, pl framework.PreEnqueuePlugin, pod *v1.Pod, shouldRecordMetric bool) *framework.Status {
	if !shouldRecordMetric {
		return pl.PreEnqueue(ctx, pod)
	}
	startTime := p.clock.Now()
	s := pl.PreEnqueue(ctx, pod)
	p.metricsRecorder.ObservePluginDurationAsync(preEnqueue, pl.Name(), s.Code().String(), p.clock.Since(startTime).Seconds())
	return s
}

// addToActiveQ tries to add pod to active queue. It returns 2 parameters:
// 1. a boolean flag to indicate whether the pod is added successfully.
// 2. an error for the caller to act on.
func (p *PriorityQueue) addToActiveQ(logger klog.Logger, pInfo *framework.QueuedPodInfo) (bool, error) {
	pInfo.Gated = !p.runPreEnqueuePlugins(context.Background(), pInfo)
	if pInfo.Gated {
		// Add the Pod to unschedulablePods if it's not passing PreEnqueuePlugins.
		p.unschedulablePods.addOrUpdate(pInfo)
		return false, nil
	}
	if pInfo.InitialAttemptTimestamp == nil {
		now := p.clock.Now()
		pInfo.InitialAttemptTimestamp = &now
	}
	if err := p.activeQ.Add(pInfo); err != nil {
		logger.Error(err, "Error adding pod to the active queue", "pod", klog.KObj(pInfo.Pod))
		return false, err
	}
	return true, nil
}

// Add adds a pod to the active queue. It should be called only when a new pod
// is added so there is no chance the pod is already in active/unschedulable/backoff queues
func (p *PriorityQueue) Add(logger klog.Logger, pod *v1.Pod) error {
	p.lock.Lock()
	defer p.lock.Unlock()

	pInfo := p.newQueuedPodInfo(pod)
	gated := pInfo.Gated
	if added, err := p.addToActiveQ(logger, pInfo); !added {
		return err
	}
	if p.unschedulablePods.get(pod) != nil {
		logger.Error(nil, "Error: pod is already in the unschedulable queue", "pod", klog.KObj(pod))
		p.unschedulablePods.delete(pod, gated)
	}
	// Delete pod from backoffQ if it is backing off
	if err := p.podBackoffQ.Delete(pInfo); err == nil {
		logger.Error(nil, "Error: pod is already in the podBackoff queue", "pod", klog.KObj(pod))
	}
	logger.V(5).Info("Pod moved to an internal scheduling queue", "pod", klog.KObj(pod), "event", PodAdd, "queue", activeQ)
	metrics.SchedulerQueueIncomingPods.WithLabelValues("active", PodAdd).Inc()
	p.addNominatedPodUnlocked(logger, pInfo.PodInfo, nil)
	p.cond.Broadcast()

	return nil
}

// Activate moves the given pods to activeQ iff they're in unschedulablePods or backoffQ.
func (p *PriorityQueue) Activate(logger klog.Logger, pods map[string]*v1.Pod) {
	p.lock.Lock()
	defer p.lock.Unlock()

	activated := false
	for _, pod := range pods {
		if p.activate(logger, pod) {
			activated = true
		}
	}

	if activated {
		p.cond.Broadcast()
	}
}

func (p *PriorityQueue) activate(logger klog.Logger, pod *v1.Pod) bool {
	// Verify if the pod is present in activeQ.
	if _, exists, _ := p.activeQ.Get(newQueuedPodInfoForLookup(pod)); exists {
		// No need to activate if it's already present in activeQ.
		return false
	}
	var pInfo *framework.QueuedPodInfo
	// Verify if the pod is present in unschedulablePods or backoffQ.
	if pInfo = p.unschedulablePods.get(pod); pInfo == nil {
		// If the pod doesn't belong to unschedulablePods or backoffQ, don't activate it.
		if obj, exists, _ := p.podBackoffQ.Get(newQueuedPodInfoForLookup(pod)); !exists {
			logger.Error(nil, "To-activate pod does not exist in unschedulablePods or backoffQ", "pod", klog.KObj(pod))
			return false
		} else {
			pInfo = obj.(*framework.QueuedPodInfo)
		}
	}

	if pInfo == nil {
		// Redundant safe check. We shouldn't reach here.
		logger.Error(nil, "Internal error: cannot obtain pInfo")
		return false
	}

	gated := pInfo.Gated
	if added, _ := p.addToActiveQ(logger, pInfo); !added {
		return false
	}
	p.unschedulablePods.delete(pInfo.Pod, gated)
	p.podBackoffQ.Delete(pInfo)
	metrics.SchedulerQueueIncomingPods.WithLabelValues("active", ForceActivate).Inc()
	p.addNominatedPodUnlocked(logger, pInfo.PodInfo, nil)
	return true
}

// isPodBackingoff returns true if a pod is still waiting for its backoff timer.
// If this returns true, the pod should not be re-tried.
func (p *PriorityQueue) isPodBackingoff(podInfo *framework.QueuedPodInfo) bool {
	if podInfo.Gated {
		return false
	}
	boTime := p.getBackoffTime(podInfo)
	return boTime.After(p.clock.Now())
}

// SchedulingCycle returns current scheduling cycle.
func (p *PriorityQueue) SchedulingCycle() int64 {
	p.lock.RLock()
	defer p.lock.RUnlock()
	return p.schedulingCycle
}

// determineSchedulingHintForInFlightPod looks at the unschedulable plugins of the given Pod
// and determines the scheduling hint for this Pod while checking the events that happened during in-flight.
func (p *PriorityQueue) determineSchedulingHintForInFlightPod(logger klog.Logger, pInfo *framework.QueuedPodInfo) queueingStrategy {
	logger.V(5).Info("Checking events for in-flight pod", "pod", klog.KObj(pInfo.Pod), "unschedulablePlugins", pInfo.UnschedulablePlugins, "inFlightEventsSize", p.inFlightEvents.Len(), "inFlightPodsSize", len(p.inFlightPods))

	// AddUnschedulableIfNotPresent is called with the Pod at the end of scheduling or binding.
	// So, given pInfo should have been Pop()ed before,
	// we can assume pInfo must be recorded in inFlightPods and thus inFlightEvents.
	inFlightPod, ok := p.inFlightPods[pInfo.Pod.UID]
	if !ok {
		// This can happen while updating a pod. In that case pInfo.UnschedulablePlugins should
		// be empty. If it is not, we may have a problem.
		if len(pInfo.UnschedulablePlugins) != 0 {
			logger.Error(nil, "In flight Pod isn't found in the scheduling queue. If you see this error log, it's likely a bug in the scheduler.", "pod", klog.KObj(pInfo.Pod))
			return queueAfterBackoff
		}
		if p.inFlightEvents.Len() > len(p.inFlightPods) {
			return queueAfterBackoff
		}
		return queueSkip
	}

	rejectorPlugins := pInfo.UnschedulablePlugins.Union(pInfo.PendingPlugins)
	if len(rejectorPlugins) == 0 {
		// No failed plugins are associated with this Pod.
		// Meaning something unusual (a temporal failure on kube-apiserver, etc) happened and this Pod gets moved back to the queue.
		// In this case, we should retry scheduling it because this Pod may not be retried until the next flush.
		return queueAfterBackoff
	}

	// check if there is an event that makes this Pod schedulable based on pInfo.UnschedulablePlugins.
	queueingStrategy := queueSkip
	for event := inFlightPod.Next(); event != nil; event = event.Next() {
		e, ok := event.Value.(*clusterEvent)
		if !ok {
			// Must be another in-flight Pod (*v1.Pod). Can be ignored.
			continue
		}
		logger.V(5).Info("Checking event for in-flight pod", "pod", klog.KObj(pInfo.Pod), "event", e.event.Label)

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

// addUnschedulableIfNotPresentWithoutQueueingHint inserts a pod that cannot be scheduled into
// the queue, unless it is already in the queue. Normally, PriorityQueue puts
// unschedulable pods in `unschedulablePods`. But if there has been a recent move
// request, then the pod is put in `podBackoffQ`.
// TODO: This function is called only when p.isSchedulingQueueHintEnabled is false,
// and this will be removed after SchedulingQueueHint goes to stable and the feature gate is removed.
func (p *PriorityQueue) addUnschedulableWithoutQueueingHint(logger klog.Logger, pInfo *framework.QueuedPodInfo, podSchedulingCycle int64) error {
	pod := pInfo.Pod
	// Refresh the timestamp since the pod is re-added.
	pInfo.Timestamp = p.clock.Now()

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
		if err := p.podBackoffQ.Add(pInfo); err != nil {
			return fmt.Errorf("error adding pod %v to the backoff queue: %v", klog.KObj(pod), err)
		}
		logger.V(5).Info("Pod moved to an internal scheduling queue", "pod", klog.KObj(pod), "event", ScheduleAttemptFailure, "queue", backoffQ)
		metrics.SchedulerQueueIncomingPods.WithLabelValues("backoff", ScheduleAttemptFailure).Inc()
	} else {
		p.unschedulablePods.addOrUpdate(pInfo)
		logger.V(5).Info("Pod moved to an internal scheduling queue", "pod", klog.KObj(pod), "event", ScheduleAttemptFailure, "queue", unschedulablePods)
		metrics.SchedulerQueueIncomingPods.WithLabelValues("unschedulable", ScheduleAttemptFailure).Inc()
	}

	p.addNominatedPodUnlocked(logger, pInfo.PodInfo, nil)
	return nil
}

// AddUnschedulableIfNotPresent inserts a pod that cannot be scheduled into
// the queue, unless it is already in the queue. Normally, PriorityQueue puts
// unschedulable pods in `unschedulablePods`. But if there has been a recent move
// request, then the pod is put in `podBackoffQ`.
func (p *PriorityQueue) AddUnschedulableIfNotPresent(logger klog.Logger, pInfo *framework.QueuedPodInfo, podSchedulingCycle int64) error {
	p.lock.Lock()
	defer p.lock.Unlock()

	// In any case, this Pod will be moved back to the queue and we should call Done.
	defer p.done(pInfo.Pod.UID)

	pod := pInfo.Pod
	if p.unschedulablePods.get(pod) != nil {
		return fmt.Errorf("Pod %v is already present in unschedulable queue", klog.KObj(pod))
	}

	if _, exists, _ := p.activeQ.Get(pInfo); exists {
		return fmt.Errorf("Pod %v is already present in the active queue", klog.KObj(pod))
	}
	if _, exists, _ := p.podBackoffQ.Get(pInfo); exists {
		return fmt.Errorf("Pod %v is already present in the backoff queue", klog.KObj(pod))
	}

	if !p.isSchedulingQueueHintEnabled {
		// fall back to the old behavior which doesn't depend on the queueing hint.
		return p.addUnschedulableWithoutQueueingHint(logger, pInfo, podSchedulingCycle)
	}

	// Refresh the timestamp since the pod is re-added.
	pInfo.Timestamp = p.clock.Now()

	// If a move request has been received, move it to the BackoffQ, otherwise move
	// it to unschedulablePods.
	rejectorPlugins := pInfo.UnschedulablePlugins.Union(pInfo.PendingPlugins)
	for plugin := range rejectorPlugins {
		metrics.UnschedulableReason(plugin, pInfo.Pod.Spec.SchedulerName).Inc()
	}

	// We check whether this Pod may change its scheduling result by any of events that happened during scheduling.
	schedulingHint := p.determineSchedulingHintForInFlightPod(logger, pInfo)

	// In this case, we try to requeue this Pod to activeQ/backoffQ.
	queue := p.requeuePodViaQueueingHint(logger, pInfo, schedulingHint, ScheduleAttemptFailure)
	logger.V(3).Info("Pod moved to an internal scheduling queue", "pod", klog.KObj(pod), "event", ScheduleAttemptFailure, "queue", queue, "schedulingCycle", podSchedulingCycle, "hint", schedulingHint, "unschedulable plugins", rejectorPlugins)
	if queue == activeQ {
		// When the Pod is moved to activeQ, need to let p.cond know so that the Pod will be pop()ed out.
		p.cond.Broadcast()
	}

	p.addNominatedPodUnlocked(logger, pInfo.PodInfo, nil)
	return nil
}

// flushBackoffQCompleted Moves all pods from backoffQ which have completed backoff in to activeQ
func (p *PriorityQueue) flushBackoffQCompleted(logger klog.Logger) {
	p.lock.Lock()
	defer p.lock.Unlock()
	activated := false
	for {
		rawPodInfo := p.podBackoffQ.Peek()
		if rawPodInfo == nil {
			break
		}
		pInfo := rawPodInfo.(*framework.QueuedPodInfo)
		pod := pInfo.Pod
		if p.isPodBackingoff(pInfo) {
			break
		}
		_, err := p.podBackoffQ.Pop()
		if err != nil {
			logger.Error(err, "Unable to pop pod from backoff queue despite backoff completion", "pod", klog.KObj(pod))
			break
		}
		if added, _ := p.addToActiveQ(logger, pInfo); added {
			logger.V(5).Info("Pod moved to an internal scheduling queue", "pod", klog.KObj(pod), "event", BackoffComplete, "queue", activeQ)
			metrics.SchedulerQueueIncomingPods.WithLabelValues("active", BackoffComplete).Inc()
			activated = true
		}
	}

	if activated {
		p.cond.Broadcast()
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
		p.movePodsToActiveOrBackoffQueue(logger, podsToMove, UnschedulableTimeout, nil, nil)
	}
}

// Pop removes the head of the active queue and returns it. It blocks if the
// activeQ is empty and waits until a new item is added to the queue. It
// increments scheduling cycle when a pod is popped.
func (p *PriorityQueue) Pop(logger klog.Logger) (*framework.QueuedPodInfo, error) {
	p.lock.Lock()
	defer p.lock.Unlock()
	for p.activeQ.Len() == 0 {
		// When the queue is empty, invocation of Pop() is blocked until new item is enqueued.
		// When Close() is called, the p.closed is set and the condition is broadcast,
		// which causes this loop to continue and return from the Pop().
		if p.closed {
			logger.V(2).Info("Scheduling queue is closed")
			return nil, nil
		}
		p.cond.Wait()
	}
	obj, err := p.activeQ.Pop()
	if err != nil {
		return nil, err
	}
	pInfo := obj.(*framework.QueuedPodInfo)
	pInfo.Attempts++
	p.schedulingCycle++
	// In flight, no concurrent events yet.
	if p.isSchedulingQueueHintEnabled {
		p.inFlightPods[pInfo.Pod.UID] = p.inFlightEvents.PushBack(pInfo.Pod)
	}

	// Update metrics and reset the set of unschedulable plugins for the next attempt.
	for plugin := range pInfo.UnschedulablePlugins.Union(pInfo.PendingPlugins) {
		metrics.UnschedulableReason(plugin, pInfo.Pod.Spec.SchedulerName).Dec()
	}
	pInfo.UnschedulablePlugins.Clear()
	pInfo.PendingPlugins.Clear()

	return pInfo, nil
}

// Done must be called for pod returned by Pop. This allows the queue to
// keep track of which pods are currently being processed.
func (p *PriorityQueue) Done(pod types.UID) {
	p.lock.Lock()
	defer p.lock.Unlock()

	p.done(pod)
}

func (p *PriorityQueue) done(pod types.UID) {
	if !p.isSchedulingQueueHintEnabled {
		// do nothing if schedulingQueueHint is disabled.
		// In that case, we don't have inFlightPods and inFlightEvents.
		return
	}
	inFlightPod, ok := p.inFlightPods[pod]
	if !ok {
		// This Pod is already done()ed.
		return
	}
	delete(p.inFlightPods, pod)

	// Remove the pod from the list.
	p.inFlightEvents.Remove(inFlightPod)

	// Remove events which are only referred to by this Pod
	// so that the inFlightEvents list doesn't grow infinitely.
	// If the pod was at the head of the list, then all
	// events between it and the next pod are no longer needed
	// and can be removed.
	for {
		e := p.inFlightEvents.Front()
		if e == nil {
			// Empty list.
			break
		}
		if _, ok := e.Value.(*clusterEvent); !ok {
			// A pod, must stop pruning.
			break
		}
		p.inFlightEvents.Remove(e)
	}
}

// isPodUpdated checks if the pod is updated in a way that it may have become
// schedulable. It drops status of the pod and compares it with old version,
// except for pod.status.resourceClaimStatuses: changing that may have an
// effect on scheduling.
func isPodUpdated(oldPod, newPod *v1.Pod) bool {
	strip := func(pod *v1.Pod) *v1.Pod {
		p := pod.DeepCopy()
		p.ResourceVersion = ""
		p.Generation = 0
		p.Status = v1.PodStatus{
			ResourceClaimStatuses: pod.Status.ResourceClaimStatuses,
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
func (p *PriorityQueue) Update(logger klog.Logger, oldPod, newPod *v1.Pod) error {
	p.lock.Lock()
	defer p.lock.Unlock()

	if oldPod != nil {
		oldPodInfo := newQueuedPodInfoForLookup(oldPod)
		// If the pod is already in the active queue, just update it there.
		if oldPodInfo, exists, _ := p.activeQ.Get(oldPodInfo); exists {
			pInfo := updatePod(oldPodInfo, newPod)
			p.updateNominatedPodUnlocked(logger, oldPod, pInfo.PodInfo)
			return p.activeQ.Update(pInfo)
		}

		// If the pod is in the backoff queue, update it there.
		if oldPodInfo, exists, _ := p.podBackoffQ.Get(oldPodInfo); exists {
			pInfo := updatePod(oldPodInfo, newPod)
			p.updateNominatedPodUnlocked(logger, oldPod, pInfo.PodInfo)
			return p.podBackoffQ.Update(pInfo)
		}
	}

	// If the pod is in the unschedulable queue, updating it may make it schedulable.
	if usPodInfo := p.unschedulablePods.get(newPod); usPodInfo != nil {
		pInfo := updatePod(usPodInfo, newPod)
		p.updateNominatedPodUnlocked(logger, oldPod, pInfo.PodInfo)
		gated := usPodInfo.Gated
		if p.isSchedulingQueueHintEnabled {
			// When unscheduled Pods are updated, we check with QueueingHint
			// whether the update may make the pods schedulable.
			// Plugins have to implement a QueueingHint for Pod/Update event
			// if the rejection from them could be resolved by updating unscheduled Pods itself.
			hint := p.isPodWorthRequeuing(logger, pInfo, UnscheduledPodUpdate, oldPod, newPod)
			queue := p.requeuePodViaQueueingHint(logger, pInfo, hint, UnscheduledPodUpdate.Label)
			if queue != unschedulablePods {
				logger.V(5).Info("Pod moved to an internal scheduling queue because the Pod is updated", "pod", klog.KObj(newPod), "event", PodUpdate, "queue", queue)
				p.unschedulablePods.delete(usPodInfo.Pod, gated)
			}
			if queue == activeQ {
				p.cond.Broadcast()
			}
			return nil
		}
		if isPodUpdated(oldPod, newPod) {

			if p.isPodBackingoff(usPodInfo) {
				if err := p.podBackoffQ.Add(pInfo); err != nil {
					return err
				}
				p.unschedulablePods.delete(usPodInfo.Pod, gated)
				logger.V(5).Info("Pod moved to an internal scheduling queue", "pod", klog.KObj(pInfo.Pod), "event", PodUpdate, "queue", backoffQ)
				return nil
			}

			if added, err := p.addToActiveQ(logger, pInfo); !added {
				return err
			}
			p.unschedulablePods.delete(usPodInfo.Pod, gated)
			logger.V(5).Info("Pod moved to an internal scheduling queue", "pod", klog.KObj(pInfo.Pod), "event", BackoffComplete, "queue", activeQ)
			p.cond.Broadcast()
			return nil
		}

		// Pod update didn't make it schedulable, keep it in the unschedulable queue.
		p.unschedulablePods.addOrUpdate(pInfo)
		return nil
	}
	// If pod is not in any of the queues, we put it in the active queue.
	pInfo := p.newQueuedPodInfo(newPod)
	if added, err := p.addToActiveQ(logger, pInfo); !added {
		return err
	}
	p.addNominatedPodUnlocked(logger, pInfo.PodInfo, nil)
	logger.V(5).Info("Pod moved to an internal scheduling queue", "pod", klog.KObj(pInfo.Pod), "event", PodUpdate, "queue", activeQ)
	p.cond.Broadcast()
	return nil
}

// Delete deletes the item from either of the two queues. It assumes the pod is
// only in one queue.
func (p *PriorityQueue) Delete(pod *v1.Pod) error {
	p.lock.Lock()
	defer p.lock.Unlock()
	p.deleteNominatedPodIfExistsUnlocked(pod)
	pInfo := newQueuedPodInfoForLookup(pod)
	if err := p.activeQ.Delete(pInfo); err != nil {
		// The item was probably not found in the activeQ.
		p.podBackoffQ.Delete(pInfo)
		if pInfo = p.unschedulablePods.get(pod); pInfo != nil {
			p.unschedulablePods.delete(pod, pInfo.Gated)
		}
	}
	return nil
}

// AssignedPodAdded is called when a bound pod is added. Creation of this pod
// may make pending pods with matching affinity terms schedulable.
func (p *PriorityQueue) AssignedPodAdded(logger klog.Logger, pod *v1.Pod) {
	p.lock.Lock()
	p.movePodsToActiveOrBackoffQueue(logger, p.getUnschedulablePodsWithMatchingAffinityTerm(logger, pod), AssignedPodAdd, nil, pod)
	p.lock.Unlock()
}

// isPodResourcesResizedDown returns true if a pod CPU and/or memory resize request has been
// admitted by kubelet, is 'InProgress', and results in a net sizing down of updated resources.
// It returns false if either CPU or memory resource is net resized up, or if no resize is in progress.
func isPodResourcesResizedDown(pod *v1.Pod) bool {
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		// TODO(vinaykul,wangchen615,InPlacePodVerticalScaling): Fix this to determine when a
		// pod is truly resized down (might need oldPod if we cannot determine from Status alone)
		if pod.Status.Resize == v1.PodResizeStatusInProgress {
			return true
		}
	}
	return false
}

// AssignedPodUpdated is called when a bound pod is updated. Change of labels
// may make pending pods with matching affinity terms schedulable.
func (p *PriorityQueue) AssignedPodUpdated(logger klog.Logger, oldPod, newPod *v1.Pod) {
	p.lock.Lock()
	if isPodResourcesResizedDown(newPod) {
		p.moveAllToActiveOrBackoffQueue(logger, AssignedPodUpdate, oldPod, newPod, nil)
	} else {
		p.movePodsToActiveOrBackoffQueue(logger, p.getUnschedulablePodsWithMatchingAffinityTerm(logger, newPod), AssignedPodUpdate, oldPod, newPod)
	}
	p.lock.Unlock()
}

// NOTE: this function assumes a lock has been acquired in the caller.
// moveAllToActiveOrBackoffQueue moves all pods from unschedulablePods to activeQ or backoffQ.
// This function adds all pods and then signals the condition variable to ensure that
// if Pop() is waiting for an item, it receives the signal after all the pods are in the
// queue and the head is the highest priority pod.
func (p *PriorityQueue) moveAllToActiveOrBackoffQueue(logger klog.Logger, event framework.ClusterEvent, oldObj, newObj interface{}, preCheck PreEnqueueCheck) {
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
func (p *PriorityQueue) MoveAllToActiveOrBackoffQueue(logger klog.Logger, event framework.ClusterEvent, oldObj, newObj interface{}, preCheck PreEnqueueCheck) {
	p.lock.Lock()
	defer p.lock.Unlock()
	p.moveAllToActiveOrBackoffQueue(logger, event, oldObj, newObj, preCheck)
}

// requeuePodViaQueueingHint tries to requeue Pod to activeQ, backoffQ or unschedulable pod pool based on schedulingHint.
// It returns the queue name Pod goes.
//
// NOTE: this function assumes lock has been acquired in caller
func (p *PriorityQueue) requeuePodViaQueueingHint(logger klog.Logger, pInfo *framework.QueuedPodInfo, strategy queueingStrategy, event string) string {
	if strategy == queueSkip {
		p.unschedulablePods.addOrUpdate(pInfo)
		metrics.SchedulerQueueIncomingPods.WithLabelValues("unschedulable", event).Inc()
		return unschedulablePods
	}

	pod := pInfo.Pod
	if strategy == queueAfterBackoff && p.isPodBackingoff(pInfo) {
		if err := p.podBackoffQ.Add(pInfo); err != nil {
			logger.Error(err, "Error adding pod to the backoff queue, queue this Pod to unschedulable pod pool", "pod", klog.KObj(pod))
			p.unschedulablePods.addOrUpdate(pInfo)
			return unschedulablePods
		}

		metrics.SchedulerQueueIncomingPods.WithLabelValues("backoff", event).Inc()
		return backoffQ
	}

	// Reach here if schedulingHint is QueueImmediately, or schedulingHint is Queue but the pod is not backing off.

	added, err := p.addToActiveQ(logger, pInfo)
	if err != nil {
		logger.Error(err, "Error adding pod to the active queue, queue this Pod to unschedulable pod pool", "pod", klog.KObj(pod))
	}
	if added {
		metrics.SchedulerQueueIncomingPods.WithLabelValues("active", event).Inc()
		return activeQ
	}
	if pInfo.Gated {
		// In case the pod is gated, the Pod is pushed back to unschedulable Pods pool in addToActiveQ.
		return unschedulablePods
	}

	p.unschedulablePods.addOrUpdate(pInfo)
	metrics.SchedulerQueueIncomingPods.WithLabelValues("unschedulable", ScheduleAttemptFailure).Inc()
	return unschedulablePods
}

// NOTE: this function assumes lock has been acquired in caller
func (p *PriorityQueue) movePodsToActiveOrBackoffQueue(logger klog.Logger, podInfoList []*framework.QueuedPodInfo, event framework.ClusterEvent, oldObj, newObj interface{}) {
	if !p.isEventOfInterest(logger, event) {
		// No plugin is interested in this event.
		return
	}

	activated := false
	for _, pInfo := range podInfoList {
		// When handling events takes time, a scheduling throughput gets impacted negatively
		// because of a shared lock within PriorityQueue, which Pop() also requires.
		//
		// Scheduling-gated Pods never get schedulable with any events,
		// except the Pods themselves got updated, which isn't handled by movePodsToActiveOrBackoffQueue.
		// So, we can skip them early here so that they don't go through isPodWorthRequeuing,
		// which isn't fast enough when the number of scheduling-gated Pods in unschedulablePods is large.
		// This is a hotfix, which might be changed
		// once we have a better general solution for the shared lock issue.
		//
		// Note that we cannot skip all pInfo.Gated Pods here
		// because PreEnqueue plugins apart from the scheduling gate plugin may change the gating status
		// with these events.
		if pInfo.Gated && pInfo.UnschedulablePlugins.Has(names.SchedulingGates) {
			continue
		}

		schedulingHint := p.isPodWorthRequeuing(logger, pInfo, event, oldObj, newObj)
		if schedulingHint == queueSkip {
			// QueueingHintFn determined that this Pod isn't worth putting to activeQ or backoffQ by this event.
			logger.V(5).Info("Event is not making pod schedulable", "pod", klog.KObj(pInfo.Pod), "event", event.Label)
			continue
		}

		p.unschedulablePods.delete(pInfo.Pod, pInfo.Gated)
		queue := p.requeuePodViaQueueingHint(logger, pInfo, schedulingHint, event.Label)
		logger.V(4).Info("Pod moved to an internal scheduling queue", "pod", klog.KObj(pInfo.Pod), "event", event.Label, "queue", queue, "hint", schedulingHint)
		if queue == activeQ {
			activated = true
		}
	}

	p.moveRequestCycle = p.schedulingCycle

	if p.isSchedulingQueueHintEnabled && len(p.inFlightPods) != 0 {
		logger.V(5).Info("Event received while pods are in flight", "event", event.Label, "numPods", len(p.inFlightPods))
		// AddUnschedulableIfNotPresent might get called for in-flight Pods later, and in
		// AddUnschedulableIfNotPresent we need to know whether events were
		// observed while scheduling them.
		p.inFlightEvents.PushBack(&clusterEvent{
			event:  event,
			oldObj: oldObj,
			newObj: newObj,
		})
	}

	if activated {
		p.cond.Broadcast()
	}
}

// getUnschedulablePodsWithMatchingAffinityTerm returns unschedulable pods which have
// any affinity term that matches "pod".
// NOTE: this function assumes lock has been acquired in caller.
func (p *PriorityQueue) getUnschedulablePodsWithMatchingAffinityTerm(logger klog.Logger, pod *v1.Pod) []*framework.QueuedPodInfo {
	nsLabels := interpodaffinity.GetNamespaceLabelsSnapshot(logger, pod.Namespace, p.nsLister)

	var podsToMove []*framework.QueuedPodInfo
	for _, pInfo := range p.unschedulablePods.podInfoMap {
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
// This function is only used in tests.
func (p *PriorityQueue) PodsInActiveQ() []*v1.Pod {
	p.lock.RLock()
	defer p.lock.RUnlock()
	var result []*v1.Pod
	for _, pInfo := range p.activeQ.List() {
		result = append(result, pInfo.(*framework.QueuedPodInfo).Pod)
	}
	return result
}

var pendingPodsSummary = "activeQ:%v; backoffQ:%v; unschedulablePods:%v"

// PendingPods returns all the pending pods in the queue; accompanied by a debugging string
// recording showing the number of pods in each queue respectively.
// This function is used for debugging purposes in the scheduler cache dumper and comparer.
func (p *PriorityQueue) PendingPods() ([]*v1.Pod, string) {
	p.lock.RLock()
	defer p.lock.RUnlock()
	var result []*v1.Pod
	for _, pInfo := range p.activeQ.List() {
		result = append(result, pInfo.(*framework.QueuedPodInfo).Pod)
	}
	for _, pInfo := range p.podBackoffQ.List() {
		result = append(result, pInfo.(*framework.QueuedPodInfo).Pod)
	}
	for _, pInfo := range p.unschedulablePods.podInfoMap {
		result = append(result, pInfo.Pod)
	}
	return result, fmt.Sprintf(pendingPodsSummary, p.activeQ.Len(), p.podBackoffQ.Len(), len(p.unschedulablePods.podInfoMap))
}

// Close closes the priority queue.
func (p *PriorityQueue) Close() {
	p.lock.Lock()
	defer p.lock.Unlock()
	close(p.stop)
	p.closed = true
	p.cond.Broadcast()
}

// DeleteNominatedPodIfExists deletes <pod> from nominatedPods.
func (npm *nominator) DeleteNominatedPodIfExists(pod *v1.Pod) {
	npm.lock.Lock()
	npm.deleteNominatedPodIfExistsUnlocked(pod)
	npm.lock.Unlock()
}

func (npm *nominator) deleteNominatedPodIfExistsUnlocked(pod *v1.Pod) {
	npm.delete(pod)
}

// AddNominatedPod adds a pod to the nominated pods of the given node.
// This is called during the preemption process after a node is nominated to run
// the pod. We update the structure before sending a request to update the pod
// object to avoid races with the following scheduling cycles.
func (npm *nominator) AddNominatedPod(logger klog.Logger, pi *framework.PodInfo, nominatingInfo *framework.NominatingInfo) {
	npm.lock.Lock()
	npm.addNominatedPodUnlocked(logger, pi, nominatingInfo)
	npm.lock.Unlock()
}

// NominatedPodsForNode returns a copy of pods that are nominated to run on the given node,
// but they are waiting for other pods to be removed from the node.
func (npm *nominator) NominatedPodsForNode(nodeName string) []*framework.PodInfo {
	npm.lock.RLock()
	defer npm.lock.RUnlock()
	// Make a copy of the nominated Pods so the caller can mutate safely.
	pods := make([]*framework.PodInfo, len(npm.nominatedPods[nodeName]))
	for i := 0; i < len(pods); i++ {
		pods[i] = npm.nominatedPods[nodeName][i].DeepCopy()
	}
	return pods
}

func (p *PriorityQueue) podsCompareBackoffCompleted(podInfo1, podInfo2 interface{}) bool {
	pInfo1 := podInfo1.(*framework.QueuedPodInfo)
	pInfo2 := podInfo2.(*framework.QueuedPodInfo)
	bo1 := p.getBackoffTime(pInfo1)
	bo2 := p.getBackoffTime(pInfo2)
	return bo1.Before(bo2)
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

// getBackoffTime returns the time that podInfo completes backoff
func (p *PriorityQueue) getBackoffTime(podInfo *framework.QueuedPodInfo) time.Time {
	duration := p.calculateBackoffDuration(podInfo)
	backoffTime := podInfo.Timestamp.Add(duration)
	return backoffTime
}

// calculateBackoffDuration is a helper function for calculating the backoffDuration
// based on the number of attempts the pod has made.
func (p *PriorityQueue) calculateBackoffDuration(podInfo *framework.QueuedPodInfo) time.Duration {
	duration := p.podInitialBackoffDuration
	for i := 1; i < podInfo.Attempts; i++ {
		// Use subtraction instead of addition or multiplication to avoid overflow.
		if duration > p.podMaxBackoffDuration-duration {
			return p.podMaxBackoffDuration
		}
		duration += duration
	}
	return duration
}

func updatePod(oldPodInfo interface{}, newPod *v1.Pod) *framework.QueuedPodInfo {
	pInfo := oldPodInfo.(*framework.QueuedPodInfo)
	pInfo.Update(newPod)
	return pInfo
}

// UnschedulablePods holds pods that cannot be scheduled. This data structure
// is used to implement unschedulablePods.
type UnschedulablePods struct {
	// podInfoMap is a map key by a pod's full-name and the value is a pointer to the QueuedPodInfo.
	podInfoMap map[string]*framework.QueuedPodInfo
	keyFunc    func(*v1.Pod) string
	// unschedulableRecorder/gatedRecorder updates the counter when elements of an unschedulablePodsMap
	// get added or removed, and it does nothing if it's nil.
	unschedulableRecorder, gatedRecorder metrics.MetricRecorder
}

// addOrUpdate adds a pod to the unschedulable podInfoMap.
func (u *UnschedulablePods) addOrUpdate(pInfo *framework.QueuedPodInfo) {
	podID := u.keyFunc(pInfo.Pod)
	if _, exists := u.podInfoMap[podID]; !exists {
		if pInfo.Gated && u.gatedRecorder != nil {
			u.gatedRecorder.Inc()
		} else if !pInfo.Gated && u.unschedulableRecorder != nil {
			u.unschedulableRecorder.Inc()
		}
	}
	u.podInfoMap[podID] = pInfo
}

// delete deletes a pod from the unschedulable podInfoMap.
// The `gated` parameter is used to figure out which metric should be decreased.
func (u *UnschedulablePods) delete(pod *v1.Pod, gated bool) {
	podID := u.keyFunc(pod)
	if _, exists := u.podInfoMap[podID]; exists {
		if gated && u.gatedRecorder != nil {
			u.gatedRecorder.Dec()
		} else if !gated && u.unschedulableRecorder != nil {
			u.unschedulableRecorder.Dec()
		}
	}
	delete(u.podInfoMap, podID)
}

// get returns the QueuedPodInfo if a pod with the same key as the key of the given "pod"
// is found in the map. It returns nil otherwise.
func (u *UnschedulablePods) get(pod *v1.Pod) *framework.QueuedPodInfo {
	podKey := u.keyFunc(pod)
	if pInfo, exists := u.podInfoMap[podKey]; exists {
		return pInfo
	}
	return nil
}

// clear removes all the entries from the unschedulable podInfoMap.
func (u *UnschedulablePods) clear() {
	u.podInfoMap = make(map[string]*framework.QueuedPodInfo)
	if u.unschedulableRecorder != nil {
		u.unschedulableRecorder.Clear()
	}
	if u.gatedRecorder != nil {
		u.gatedRecorder.Clear()
	}
}

// newUnschedulablePods initializes a new object of UnschedulablePods.
func newUnschedulablePods(unschedulableRecorder, gatedRecorder metrics.MetricRecorder) *UnschedulablePods {
	return &UnschedulablePods{
		podInfoMap:            make(map[string]*framework.QueuedPodInfo),
		keyFunc:               util.GetPodFullName,
		unschedulableRecorder: unschedulableRecorder,
		gatedRecorder:         gatedRecorder,
	}
}

// nominator is a structure that stores pods nominated to run on nodes.
// It exists because nominatedNodeName of pod objects stored in the structure
// may be different than what scheduler has here. We should be able to find pods
// by their UID and update/delete them.
type nominator struct {
	// podLister is used to verify if the given pod is alive.
	podLister listersv1.PodLister
	// nominatedPods is a map keyed by a node name and the value is a list of
	// pods which are nominated to run on the node. These are pods which can be in
	// the activeQ or unschedulablePods.
	nominatedPods map[string][]*framework.PodInfo
	// nominatedPodToNode is map keyed by a Pod UID to the node name where it is
	// nominated.
	nominatedPodToNode map[types.UID]string

	lock sync.RWMutex
}

func (npm *nominator) addNominatedPodUnlocked(logger klog.Logger, pi *framework.PodInfo, nominatingInfo *framework.NominatingInfo) {
	// Always delete the pod if it already exists, to ensure we never store more than
	// one instance of the pod.
	npm.delete(pi.Pod)

	var nodeName string
	if nominatingInfo.Mode() == framework.ModeOverride {
		nodeName = nominatingInfo.NominatedNodeName
	} else if nominatingInfo.Mode() == framework.ModeNoop {
		if pi.Pod.Status.NominatedNodeName == "" {
			return
		}
		nodeName = pi.Pod.Status.NominatedNodeName
	}

	if npm.podLister != nil {
		// If the pod was removed or if it was already scheduled, don't nominate it.
		updatedPod, err := npm.podLister.Pods(pi.Pod.Namespace).Get(pi.Pod.Name)
		if err != nil {
			logger.V(4).Info("Pod doesn't exist in podLister, aborted adding it to the nominator", "pod", klog.KObj(pi.Pod))
			return
		}
		if updatedPod.Spec.NodeName != "" {
			logger.V(4).Info("Pod is already scheduled to a node, aborted adding it to the nominator", "pod", klog.KObj(pi.Pod), "node", updatedPod.Spec.NodeName)
			return
		}
	}

	npm.nominatedPodToNode[pi.Pod.UID] = nodeName
	for _, npi := range npm.nominatedPods[nodeName] {
		if npi.Pod.UID == pi.Pod.UID {
			logger.V(4).Info("Pod already exists in the nominator", "pod", klog.KObj(npi.Pod))
			return
		}
	}
	npm.nominatedPods[nodeName] = append(npm.nominatedPods[nodeName], pi)
}

func (npm *nominator) delete(p *v1.Pod) {
	nnn, ok := npm.nominatedPodToNode[p.UID]
	if !ok {
		return
	}
	for i, np := range npm.nominatedPods[nnn] {
		if np.Pod.UID == p.UID {
			npm.nominatedPods[nnn] = append(npm.nominatedPods[nnn][:i], npm.nominatedPods[nnn][i+1:]...)
			if len(npm.nominatedPods[nnn]) == 0 {
				delete(npm.nominatedPods, nnn)
			}
			break
		}
	}
	delete(npm.nominatedPodToNode, p.UID)
}

// UpdateNominatedPod updates the <oldPod> with <newPod>.
func (npm *nominator) UpdateNominatedPod(logger klog.Logger, oldPod *v1.Pod, newPodInfo *framework.PodInfo) {
	npm.lock.Lock()
	defer npm.lock.Unlock()
	npm.updateNominatedPodUnlocked(logger, oldPod, newPodInfo)
}

func (npm *nominator) updateNominatedPodUnlocked(logger klog.Logger, oldPod *v1.Pod, newPodInfo *framework.PodInfo) {
	// In some cases, an Update event with no "NominatedNode" present is received right
	// after a node("NominatedNode") is reserved for this pod in memory.
	// In this case, we need to keep reserving the NominatedNode when updating the pod pointer.
	var nominatingInfo *framework.NominatingInfo
	// We won't fall into below `if` block if the Update event represents:
	// (1) NominatedNode info is added
	// (2) NominatedNode info is updated
	// (3) NominatedNode info is removed
	if NominatedNodeName(oldPod) == "" && NominatedNodeName(newPodInfo.Pod) == "" {
		if nnn, ok := npm.nominatedPodToNode[oldPod.UID]; ok {
			// This is the only case we should continue reserving the NominatedNode
			nominatingInfo = &framework.NominatingInfo{
				NominatingMode:    framework.ModeOverride,
				NominatedNodeName: nnn,
			}
		}
	}
	// We update irrespective of the nominatedNodeName changed or not, to ensure
	// that pod pointer is updated.
	npm.delete(oldPod)
	npm.addNominatedPodUnlocked(logger, newPodInfo, nominatingInfo)
}

// NewPodNominator creates a nominator as a backing of framework.PodNominator.
// A podLister is passed in so as to check if the pod exists
// before adding its nominatedNode info.
func NewPodNominator(podLister listersv1.PodLister) framework.PodNominator {
	return newPodNominator(podLister)
}

func newPodNominator(podLister listersv1.PodLister) *nominator {
	return &nominator{
		podLister:          podLister,
		nominatedPods:      make(map[string][]*framework.PodInfo),
		nominatedPodToNode: make(map[types.UID]string),
	}
}

func podInfoKeyFunc(obj interface{}) (string, error) {
	return cache.MetaNamespaceKeyFunc(obj.(*framework.QueuedPodInfo).Pod)
}
