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

package preemption

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	apipod "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// victimLookupTimeout limits how long the executor waits for the API server
// to tell whether a victim whose preemption failed got deleted nonetheless.
const victimLookupTimeout = 10 * time.Second

type pendingVictim struct {
	namespace string
	name      string
	// gatedBeforeAssume is non-nil with DRAPreemption and gets set by
	// gateUntilAssumed.
	gatedBeforeAssume *atomic.Bool
}

// ExecutorPreemptor is an interface that represents a preemptor.
// It's mainly used to abstract away fields needed for logging
// purposes during preemption calls done by Executor.
type ExecutorPreemptor interface {
	klog.KMetadata
	// UID returns the UID of the object that is triggering preemption.
	UID() types.UID
	// SchedulerName returns the scheduler name assigned to the object that is triggering preemption.
	SchedulerName() string
	// Obj returns the object that is triggering preemption.
	Obj() runtime.Object
	// Pods returns a map of pod name to pods that form a preemptor.
	Pods() map[string]*v1.Pod
	// Priority returns the priority of the preemptor.
	Priority() int32
	// Type returns the type of the preemptor.
	Type() fwk.EntityKeyType
}

// Executor is responsible for actuating the preemption process based on the provided candidate.
// It supports both synchronous as well as asynchronous preemption.
type Executor struct {
	mu sync.RWMutex
	fh fwk.Handle

	podLister corelisters.PodLister

	fts feature.Features

	// preempting is a set that records the pods/podgroups that are currently triggering preemption asynchronously,
	// which is used to prevent the pods and pods from podgroups from entering the scheduling cycle meanwhile.
	preempting sets.Set[types.UID]
	// lastVictimsPendingPreemption is a map that records the victim pods that are currently being preempted for a given preemptor pod/podgroup,
	// with a condition that the preemptor is waiting for one last victim to be preempted. This is used together with `preempting`
	// to prevent the pods/podgroups from entering the scheduling cycle while waiting for preemption to complete.
	lastVictimsPendingPreemption map[types.UID]pendingVictim

	// PreemptPod is a function that actually preempts a specific Pod. It returns true
	// when the victim was preempted only in scheduler memory, without a delete call.
	// This is exposed to be replaced during tests.
	PreemptPod func(ctx context.Context, c fwk.PreemptionCandidate, preemptor ExecutorPreemptor, victim *v1.Pod, pluginName string) (bool, error)
}

var _ fwk.PreemptionExecutor = &Executor{}

// NewExecutor creates a new preemption executor.
func NewExecutor(fh fwk.Handle, fts feature.Features) *Executor {
	e := &Executor{
		fh:                           fh,
		podLister:                    fh.SharedInformerFactory().Core().V1().Pods().Lister(),
		preempting:                   sets.New[types.UID](),
		lastVictimsPendingPreemption: make(map[types.UID]pendingVictim),
		fts:                          fts,
	}

	e.PreemptPod = func(ctx context.Context, c fwk.PreemptionCandidate, preemptor ExecutorPreemptor, victim *v1.Pod, pluginName string) (bool, error) {
		logger := klog.FromContext(ctx)

		preemptedInMemory := false
		eventMessage := fmt.Sprintf("Preempted by %s %v on node %v", preemptor.Type(), preemptor.UID(), c.Name())
		// If the victim is a WaitingPod, try to preempt it without a delete call (victim will go back to backoff queue).
		// Otherwise we should delete the victim.
		if waitingPod := e.fh.GetWaitingPod(victim.UID); waitingPod != nil {
			if waitingPod.Preempt(pluginName, "preempted") {
				logger.V(2).Info("Preemptor preempted a waiting pod", "preemptorType", preemptor.Type(), "preemptor", klog.KObj(preemptor), "waitingPod", klog.KObj(victim), "node", c.Name())
				preemptedInMemory = true
			}
		} else if podInPreBind := e.fh.GetPodInPreBind(victim.UID); podInPreBind != nil {
			// If the victim is in the preBind cancel the binding process.
			if podInPreBind.CancelPod(fmt.Sprintf("preempted by %s", pluginName)) {
				logger.V(2).Info("Preemptor rejected a pod in preBind", "preemptorType", preemptor.Type(), "preemptor", klog.KObj(preemptor), "podInPreBind", klog.KObj(victim), "node", c.Name())
				preemptedInMemory = true
			} else {
				logger.V(5).Info("Failed to reject a pod in preBind, falling back to deletion via api call", "preemptor", klog.KObj(preemptor), "podInPreBind", klog.KObj(victim), "node", c.Name())
			}
		}
		if !preemptedInMemory {
			condition := &v1.PodCondition{
				Type:               v1.DisruptionTarget,
				ObservedGeneration: apipod.CalculatePodConditionObservedGeneration(&victim.Status, victim.Generation, v1.DisruptionTarget),
				Status:             v1.ConditionTrue,
				Reason:             v1.PodReasonPreemptionByScheduler,
				Message:            fmt.Sprintf("%s: preempting to accommodate a higher priority %s", preemptor.SchedulerName(), preemptor.Type()),
			}
			newStatus := victim.Status.DeepCopy()
			updated := apipod.UpdatePodCondition(newStatus, condition)
			if updated {
				if err := schedutil.PatchPodStatus(ctx, fh.ClientSet(), victim.Name, victim.Namespace, &victim.Status, newStatus); err != nil {
					if !apierrors.IsNotFound(err) {
						logger.Error(err, "Could not add DisruptionTarget condition due to preemption", "preemptor", klog.KObj(preemptor), "victim", klog.KObj(victim))
						return false, err
					}
					logger.V(2).Info("Victim Pod is already deleted", "preemptor", klog.KObj(preemptor), "victim", klog.KObj(victim), "node", c.Name())
					return false, nil
				}
			}
			if err := schedutil.DeletePod(ctx, fh.ClientSet(), victim); err != nil {
				if !apierrors.IsNotFound(err) {
					logger.Error(err, "Tried to preempted pod", "pod", klog.KObj(victim), "preemptor", klog.KObj(preemptor))
					return false, err
				}
				logger.V(2).Info("Victim Pod is already deleted", "preemptor", klog.KObj(preemptor), "victim", klog.KObj(victim), "node", c.Name())
				return false, nil
			}
			logger.V(2).Info("Preemptor Pod preempted victim Pod", "preemptor", klog.KObj(preemptor), "victim", klog.KObj(victim), "node", c.Name())
		} else {
			eventMessage += " (in kube-scheduler memory)."
		}

		fh.EventRecorder().WithLogger(logger).Eventf(victim, preemptor.Obj(), v1.EventTypeNormal, "Preempted", "Preempting", eventMessage)

		return preemptedInMemory, nil
	}

	return e
}

// assumeVictim lets plugins track what the victim releases asynchronously,
// once its preemption is under way.
func (e *Executor) assumeVictim(ctx context.Context, victim *v1.Pod) {
	if runner, ok := e.fh.(framework.VictimExtensionsRunner); ok {
		runner.RunVictimExtensionsAssumeVictim(ctx, victim)
	}
}

// preemptVictim preempts the victim and lets plugins track it once its
// preemption is under way. That includes a failed call after which the API
// server shows the victim deleted, for example because the response of a
// successful delete call got lost.
func (e *Executor) preemptVictim(ctx context.Context, c fwk.PreemptionCandidate, preemptor ExecutorPreemptor, victim *v1.Pod, pluginName string) (bool, error) {
	preemptedInMemory, err := e.PreemptPod(ctx, c, preemptor, victim, pluginName)
	if err != nil {
		if e.victimDeletionStarted(ctx, victim) {
			e.assumeVictim(ctx, victim)
		}
		return false, err
	}
	e.assumeVictim(ctx, victim)
	return preemptedInMemory, nil
}

// victimDeletionStarted asks the API server whether the victim is being
// deleted or gone after its preemption failed. Only plugins with
// VictimExtensions need to know, so it always returns false without
// DRAPreemption.
func (e *Executor) victimDeletionStarted(ctx context.Context, victim *v1.Pod) bool {
	if !e.fts.EnableDRAPreemption {
		return false
	}
	// ctx gets cancelled when the preemption of another victim fails.
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), victimLookupTimeout)
	defer cancel()
	pod, err := e.fh.ClientSet().CoreV1().Pods(victim.Namespace).Get(ctx, victim.Name, metav1.GetOptions{})
	switch {
	case apierrors.IsNotFound(err):
		return true
	case err != nil:
		klog.FromContext(ctx).V(2).Info("Cannot check whether the victim got deleted after its preemption failed", "victim", klog.KObj(victim), "err", err)
		return false
	default:
		return pod.UID != victim.UID || pod.DeletionTimestamp != nil
	}
}

// ActuatePodPreemption actuates the preemption given preemptorPod to be scheduled on targetNode and a list of victims to be evicted.
// victims to be evicted.
func (e *Executor) ActuatePodPreemption(ctx context.Context, candidate fwk.PreemptionCandidate, preemptorPod *v1.Pod, pluginName string) *fwk.Status {
	podPreemptor := &podExecutorPreemptor{Pod: preemptorPod}
	if e.fts.EnableAsyncPreemption {
		e.prepareCandidateAsync(candidate, podPreemptor, pluginName)
		return nil
	}
	return e.prepareCandidate(ctx, candidate, podPreemptor, pluginName)
}

// ActuatePodGroupPreemption actuates the preemption given preemptor pods, pod group and a list of victims to be evicted.
func (e *Executor) ActuatePodGroupPreemption(ctx context.Context, candidate fwk.PreemptionCandidate, pgInfo fwk.PodGroupInfo, pluginName string) *fwk.Status {
	podGroupPreemptor := &podGroupExecutorPreemptor{PodGroupInfo: pgInfo, pods: pgInfo.GetAllUnscheduledPods()}
	if e.fts.EnableAsyncPreemption {
		e.prepareCandidateAsync(candidate, podGroupPreemptor, pluginName)
		return nil
	}
	return e.prepareCandidate(ctx, candidate, podGroupPreemptor, pluginName)
}

// prepareCandidateAsync triggers a goroutine for some preparation work:
// - Evict the victim pods
// - Reject the victim pods if they are in waitingPod map
// - Clear the low-priority pods' nominatedNodeName status if needed
// The Pod won't be retried until the goroutine triggered here completes.
//
// See http://kep.k8s.io/4832 for how the async preemption works.
func (e *Executor) prepareCandidateAsync(c fwk.PreemptionCandidate, preemptor ExecutorPreemptor, pluginName string) {
	observeVictims(preemptor, c)
	// Intentionally create a new context, not using a ctx from the scheduling cycle, to create ctx,
	// because this process could continue even after this scheduling cycle finishes.
	ctx, cancel := context.WithCancel(context.Background())
	logger := klog.FromContext(ctx)

	victimPods := make([]*v1.Pod, 0, len(c.Victims().Pods))
	for _, victim := range c.Victims().Pods {
		if victim.DeletionTimestamp != nil {
			// Graceful pod deletion has already started. Sending another API call is unnecessary.
			logger.V(2).Info("Victim Pod is already being deleted, skipping the API call for it", "preemptor", klog.KObj(preemptor), "node", c.Name(), "victim", klog.KObj(victim))
			e.assumeVictim(ctx, victim)
			continue
		}
		victimPods = append(victimPods, victim)
	}
	if len(victimPods) == 0 {
		cancel()
		return
	}

	errCh := parallelize.NewResultChannel[error]()
	// PreEnqueue only watches the last victim for completion, so activate when that victim won't emit a deletion event.
	preemptedLastVictimInMemory := false
	// With DRAPreemption, PreEnqueue may have held back the preemptor until AssumeVictim ran for the last victim.
	lastVictimGatedBeforeAssume := false
	preemptPod := func(index int) {
		victim := victimPods[index]
		if _, err := e.preemptVictim(ctx, c, preemptor, victim, pluginName); err != nil {
			errCh.SendWithCancel(err, cancel)
		}
	}

	e.mu.Lock()
	e.preempting.Insert(preemptor.UID())
	e.mu.Unlock()

	go func() {
		logger := klog.FromContext(ctx)
		startTime := time.Now()
		result := metrics.GoroutineResultSuccess
		defer func() {
			metrics.PreemptionExecutionDuration.WithLabelValues(metrics.EntityTypeToLabel(preemptor.Type()), result).Observe(metrics.SinceInSeconds(startTime))
		}()
		defer func() {
			metrics.PreemptionGoroutinesDuration.WithLabelValues(result).Observe(metrics.SinceInSeconds(startTime))
		}()
		defer func() {
			metrics.PreemptionGoroutinesExecutionTotal.WithLabelValues(result).Inc()
		}()
		defer func() {
			if result == metrics.GoroutineResultError || preemptedLastVictimInMemory || lastVictimGatedBeforeAssume {
				// When API call isn't successful or no victim deletion event will be produced, the preemptor's
				// Pods may get stuck in the unschedulable pod pool in the worst case.
				e.fh.Activate(logger, preemptor.Pods())
			}
		}()
		defer cancel()
		logger.V(2).Info("Start the preemption asynchronously", "preemptor", klog.KObj(preemptor), "node", c.Name(), "numVictims", len(c.Victims().Pods), "numVictimsToDelete", len(victimPods))

		// Lower priority pods nominated to run the preemptor nominated nodes, may no
		// longer fit on them. So, we should remove their nomination. Removing their
		// nomination updates these pods and moves them to the active queue. It
		// lets scheduler find another place for them sooner than after waiting for preemption completion.
		if err := e.clearNominatedPods(ctx, preemptor.Priority(), c.Nodes()); err != nil {
			utilruntime.HandleErrorWithContext(ctx, err, "Cannot clear 'NominatedNodeName' field from lower priority pods on the same target node", "node", c.Name())
			result = metrics.GoroutineResultError
			// We do not return as this error is not critical.
		}

		preemptLastVictim := true
		if len(victimPods) > 1 {
			// In order to prevent requesting preemption of the same pod multiple times for the same preemptor,
			// preemptor is marked as "waiting for preemption of a victim" (by adding it to preempting map).
			// We can evict all victims in parallel, but the last one.
			// While deleting the last victim, we want the PreEnqueue check to be able to verify if the eviction
			// is in fact ongoing, or if it has already completed, but the function has not returned yet.
			// In the latter case, PreEnqueue (in `IsPodRunningPreemption`) reads the state of the last victim in
			// `lastVictimsPendingPreemption` and does not block the pod.
			// This helps us avoid the situation where pod removal might be notified to the scheduling queue before
			// the preemptor completes the deletion API calls and is removed from the `preempting` map - that way
			// the preemptor could end up stuck in the unschedulable pool, with all pod removal events being ignored.
			e.fh.Parallelizer().Until(ctx, len(victimPods)-1, preemptPod, pluginName)
			if err := errCh.Receive(); err != nil {
				utilruntime.HandleErrorWithContext(ctx, err, "Error occurred during async preemption")
				result = metrics.GoroutineResultError
				preemptLastVictim = false
			}
		}

		// If any of the previous victims failed to be preempted, then we can skip
		// the preemption attempt for the last victim Pod to expedite the preemptor's
		// re-entry to the scheduling cycle.
		if preemptLastVictim {
			lastVictim := victimPods[len(victimPods)-1]
			pending := pendingVictim{namespace: lastVictim.Namespace, name: lastVictim.Name}
			if e.fts.EnableDRAPreemption {
				pending.gatedBeforeAssume = &atomic.Bool{}
			}
			e.mu.Lock()
			e.lastVictimsPendingPreemption[preemptor.UID()] = pending
			e.mu.Unlock()

			preemptedInMemory, err := e.preemptVictim(ctx, c, preemptor, lastVictim, pluginName)
			if err != nil {
				utilruntime.HandleErrorWithContext(ctx, err, "Error occurred during async preemption of the last victim")
				result = metrics.GoroutineResultError
			} else {
				preemptedLastVictimInMemory = preemptedInMemory
			}
		}
		e.mu.Lock()
		e.preempting.Delete(preemptor.UID())
		// Read under the lock: once the preemptor is not in preempting anymore, isRunningPreemption doesn't gate it.
		if pending, ok := e.lastVictimsPendingPreemption[preemptor.UID()]; ok && pending.gatedBeforeAssume != nil {
			lastVictimGatedBeforeAssume = pending.gatedBeforeAssume.Load()
		}
		delete(e.lastVictimsPendingPreemption, preemptor.UID())
		e.mu.Unlock()

		logger.V(2).Info("Async Preemption finished completely", "preemptor", klog.KObj(preemptor), "node", c.Name(), "result", result)
	}()
}

// prepareCandidate does some preparation work before nominating the selected candidate:
// - Evict the victim pods
// - Reject the victim pods if they are in waitingPod map
// - Clear the low-priority pods' nominatedNodeName status if needed
func (e *Executor) prepareCandidate(ctx context.Context, c fwk.PreemptionCandidate, preemptor ExecutorPreemptor, pluginName string) *fwk.Status {
	observeVictims(preemptor, c)
	startTime := time.Now()
	metricsResult := metrics.GoroutineResultSuccess
	defer func() {
		metrics.PreemptionExecutionDuration.WithLabelValues(metrics.EntityTypeToLabel(preemptor.Type()), metricsResult).Observe(metrics.SinceInSeconds(startTime))
	}()

	fh := e.fh

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	logger := klog.FromContext(ctx)
	errCh := parallelize.NewResultChannel[error]()
	fh.Parallelizer().Until(ctx, len(c.Victims().Pods), func(index int) {
		victim := c.Victims().Pods[index]
		if victim.DeletionTimestamp != nil {
			// Graceful pod deletion has already started. Sending another API call is unnecessary.
			logger.V(2).Info("Victim Pod is already being deleted, skipping the API call for it", "preemptor", klog.KObj(preemptor), "node", c.Name(), "victim", klog.KObj(victim))
			e.assumeVictim(ctx, victim)
			return
		}
		if _, err := e.preemptVictim(ctx, c, preemptor, victim, pluginName); err != nil {
			errCh.SendWithCancel(err, cancel)
		}
	}, pluginName)
	if err := errCh.Receive(); err != nil {
		metricsResult = metrics.GoroutineResultError
		return fwk.AsStatus(err)
	}

	// Lower priority pods nominated to run the preemptor nominated nodes, may no
	// longer fit on them. So, we should remove their nomination. Removing their
	// nomination updates these pods and moves them to the active queue. It
	// lets scheduler find another place for them sooner than after waiting for preemption completion.
	if err := e.clearNominatedPods(ctx, preemptor.Priority(), c.Nodes()); err != nil {
		utilruntime.HandleErrorWithContext(ctx, err, "Cannot clear 'NominatedNodeName' field")
		// We do not return as this error is not critical.
	}

	return nil
}

func observeVictims(preemptor ExecutorPreemptor, candidate fwk.PreemptionCandidate) {
	numVictims := float64(len(candidate.Victims().Pods))
	if preemptor.Type() == fwk.PodKeyType {
		metrics.PreemptionVictims.Observe(numVictims)
	} else {
		preemptorLabel := metrics.EntityTypeToLabel(preemptor.Type())
		metrics.WorkloadPreemptionVictims.WithLabelValues(preemptorLabel).Observe(numVictims)
		for victimType, n := range candidate.NumDisruptionsByVictimType() {
			if n > 0 {
				metrics.PreemptionWorkloadDisruptions.WithLabelValues(preemptorLabel, metrics.EntityTypeToLabel(victimType)).Observe(float64(n))
			}
		}
	}

	numPDBViolations := float64(candidate.Victims().NumPDBViolations)
	if numPDBViolations > 0 {
		metrics.PreemptionPDBViolations.WithLabelValues(metrics.EntityTypeToLabel(preemptor.Type())).Add(numPDBViolations)
	}
}

// nominatedNodeName returns the nominated node of pod from fh if it implements
// NominatedNodeNameGetter, otherwise what the pod object shows.
func nominatedNodeName(fh fwk.Handle, pod *v1.Pod) string {
	if nominator, ok := fh.(framework.NominatedNodeNameGetter); ok {
		return nominator.NominatedNodeName(pod)
	}
	return pod.Status.NominatedNodeName
}

// IsPodGroupWaitingForVictims returns true if previously preempted victims on the
// pod group's nominated nodes are still terminating in the current snapshot.
func (e *Executor) IsPodGroupWaitingForVictims(pgInfo fwk.PodGroupInfo) bool {
	nominatedNodes := sets.New[string]()
	for _, pod := range pgInfo.GetAllUnscheduledPods() {
		if nnn := nominatedNodeName(e.fh, pod); len(nnn) > 0 {
			nominatedNodes.Insert(nnn)
		}
	}

	preemptorPriority := pgInfo.GetPriority()
	nodeLister := e.fh.SnapshotSharedLister().NodeInfos()
	for nomNodeName := range nominatedNodes {
		nodeInfo, err := nodeLister.Get(nomNodeName)
		if err != nil || nodeInfo == nil {
			continue
		}
		for _, p := range nodeInfo.GetPods() {
			if corev1helpers.PodPriority(p.GetPod()) < preemptorPriority && PodTerminatingByPreemption(p.GetPod()) {
				return true
			}
		}
	}

	return false
}

// IsPodRunningPreemption returns true if the pod is currently triggering preemption asynchronously.
func (e *Executor) IsPodRunningPreemption(podUID types.UID) bool {
	return e.isRunningPreemption(podUID)
}

// IsPodGroupRunningPreemption returns true if the pod group is currently triggering preemption asynchronously.
func (e *Executor) IsPodGroupRunningPreemption(podGroupUID types.UID) bool {
	return e.isRunningPreemption(podGroupUID)
}

func (e *Executor) isRunningPreemption(uid types.UID) bool {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if !e.preempting.Has(uid) {
		return false
	}

	victim, ok := e.lastVictimsPendingPreemption[uid]
	if !ok {
		// Since preemptor is in `preempting` but last victim is not registered yet, the async preemption is pending.
		return true
	}
	// Preemptor is waiting for preemption of one last victim. We can check if the victim has already been deleted.
	victimPod, err := e.podLister.Pods(victim.namespace).Get(victim.name)
	if err != nil {
		// Victim already deleted, preemption is done.
		return victim.gateUntilAssumed()
	}
	if victimPod.DeletionTimestamp != nil {
		// Victim deletion has started, preemption is done.
		return victim.gateUntilAssumed()
	}
	// Preemption of the last pod is still ongoing.
	return true
}

// gateUntilAssumed is called when the informer shows the victim deleted or
// terminating. That may happen before the delete call returned and
// AssumeVictim told plugins that resources of the victim are being released.
// Until then, PodEligibleToPreempt of VictimExtensions would let the
// preemptor preempt more pods. With DRAPreemption, it therefore returns true
// and records that the goroutine has to activate the preemptor when it is
// done. The event which would have let the preemptor pass may be gone by then.
func (v pendingVictim) gateUntilAssumed() bool {
	if v.gatedBeforeAssume == nil {
		return false
	}
	v.gatedBeforeAssume.Store(true)
	return true
}

type nominatedPodOnNode struct {
	pod      *v1.Pod
	nodeName string
}

// clearNominatedPods clears the nominations of pods with priority lower than
// the preemptor on the candidate nodes. With DRAPreemption, nominations get
// cleared in memory first so that devices held for displaced pods are released
// and the pods get requeued even when the informer did not show their
// nomination yet.
func (e *Executor) clearNominatedPods(ctx context.Context, priority int32, nodeNames []string) utilerrors.Aggregate {
	logger := klog.FromContext(ctx)
	if !e.fts.EnableDRAPreemption {
		nominatedPods := getLowerPriorityNominatedPods(logger, e.fh, priority, nodeNames)
		return clearNominatedNodeName(ctx, e.fh.ClientSet(), e.fh.APICacher(), nominatedPods...)
	}

	clearer, _ := e.fh.(framework.NominationClearer)
	var cleared []nominatedPodOnNode
	for _, nodeName := range nodeNames {
		for _, pi := range e.fh.NominatedPodsForNode(logger, nodeName) {
			if corev1helpers.PodPriority(pi.GetPod()) < priority {
				pod := pi.GetPod()
				if clearer != nil && !clearer.ClearNominatedPod(logger, pod, nodeName) {
					continue
				}
				cleared = append(cleared, nominatedPodOnNode{pod: pod, nodeName: nodeName})
			}
		}
	}

	var errs []error
	cs := e.fh.ClientSet()
	apiCacher := e.fh.APICacher()
	for _, item := range cleared {
		p := item.pod
		if clearer != nil && !clearer.IsNominatedTo(p, "") {
			// Another scheduling cycle nominated the pod after ClearNominatedPod woke it up.
			continue
		}
		if p.Status.NominatedNodeName != item.nodeName {
			// The pod object from the informer may not show the nomination yet
			// although the scheduler already set it in the API. Without the
			// old name, patching to "" would look like a no-op and leave the
			// nomination in the API.
			p = p.DeepCopy()
			p.Status.NominatedNodeName = item.nodeName
		}
		if err := clearNominatedNodeName(ctx, cs, apiCacher, p); err != nil {
			if apiCacher == nil && clearer != nil {
				clearer.NominationNotPersisted(p, "")
			}
			errs = append(errs, err.Errors()...)
		}
	}
	return utilerrors.NewAggregate(errs)
}

// clearNominatedNodeName internally submit a patch request to API server
// to set each pods[*].Status.NominatedNodeName> to "".
func clearNominatedNodeName(ctx context.Context, cs clientset.Interface, apiCacher fwk.APICacher, pods ...*v1.Pod) utilerrors.Aggregate {
	var errs []error
	for _, p := range pods {
		if apiCacher != nil {
			// When API cacher is available, use it to clear the NominatedNodeName.
			_, err := apiCacher.PatchPodStatus(p, nil, &fwk.NominatingInfo{NominatedNodeName: "", NominatingMode: fwk.ModeOverride})
			if err != nil {
				errs = append(errs, err)
			}
		} else {
			if len(p.Status.NominatedNodeName) == 0 {
				continue
			}
			podStatusCopy := p.Status.DeepCopy()
			podStatusCopy.NominatedNodeName = ""
			if err := schedutil.PatchPodStatus(ctx, cs, p.Name, p.Namespace, &p.Status, podStatusCopy); err != nil {
				errs = append(errs, err)
			}
		}
	}
	return utilerrors.NewAggregate(errs)
}

// getLowerPriorityNominatedPods returns pods whose priority is smaller than the
// priority of the given "pod" and are nominated to run on the given nodes.
// Note: We could possibly check if the nominated lower priority pods still fit
// and return those that no longer fit, but that would require lots of
// manipulation of NodeInfo and PreFilter state per nominated pod. It may not be
// worth the complexity, especially because we generally expect to have a very
// small number of nominated pods per node.
func getLowerPriorityNominatedPods(logger klog.Logger, pn fwk.PodNominator, priority int32, nodeNames []string) []*v1.Pod {
	var lowerPriorityPods []*v1.Pod
	for _, n := range nodeNames {
		for _, pi := range pn.NominatedPodsForNode(logger, n) {
			if corev1helpers.PodPriority(pi.GetPod()) < priority {
				lowerPriorityPods = append(lowerPriorityPods, pi.GetPod())
			}
		}
	}
	return lowerPriorityPods
}

// podExecutorPreemptor is a wrapper around single pod used by preemption execution.
type podExecutorPreemptor struct {
	*v1.Pod
}

func (p *podExecutorPreemptor) UID() types.UID {
	return p.Pod.UID
}

func (p *podExecutorPreemptor) SchedulerName() string {
	return p.Spec.SchedulerName
}

func (p *podExecutorPreemptor) Obj() runtime.Object {
	return p
}

func (p *podExecutorPreemptor) Pods() map[string]*v1.Pod {
	return map[string]*v1.Pod{p.Name: p.Pod}
}

func (p *podExecutorPreemptor) Priority() int32 {
	return corev1helpers.PodPriority(p.Pod)
}

func (p *podExecutorPreemptor) Type() fwk.EntityKeyType {
	return fwk.PodKeyType
}

// podGroupExecutorPreemptor is a wrapper around PodGroupInfo used by preemption execution.
type podGroupExecutorPreemptor struct {
	fwk.PodGroupInfo
	pods []*v1.Pod
}

func (p *podGroupExecutorPreemptor) UID() types.UID {
	return p.GetUID()
}

func (p *podGroupExecutorPreemptor) SchedulerName() string {
	// All pods in a pod group should use the same scheduler name.
	return p.pods[0].Spec.SchedulerName
}

func (p *podGroupExecutorPreemptor) Obj() runtime.Object {
	return p.GetObject()
}

func (p *podGroupExecutorPreemptor) Priority() int32 {
	return p.GetPriority()
}

func (p *podGroupExecutorPreemptor) Pods() map[string]*v1.Pod {
	m := make(map[string]*v1.Pod, len(p.pods))
	for _, pod := range p.pods {
		m[pod.Name] = pod
	}
	return m
}

func (p *podGroupExecutorPreemptor) Type() fwk.EntityKeyType {
	return p.GetType()
}
