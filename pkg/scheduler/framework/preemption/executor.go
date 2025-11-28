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
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
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
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

type pendingVictim struct {
	namespace string
	name      string
}

// Executor is responsible for actuating the preemption process based on the provided candidate.
// It supports both synchronous as well as asynchronous preemption.
type Executor struct {
	mu sync.RWMutex
	fh fwk.Handle

	podLister corelisters.PodLister

	// preempting is a set that records the pods that are currently triggering preemption asynchronously,
	// which is used to prevent the pods from entering the scheduling cycle meanwhile.
	preempting sets.Set[types.UID]
	// lastVictimsPendingPreemption is a map that records the victim pods that are currently being preempted for a given preemptor pod,
	// with a condition that the preemptor is waiting for one last victim to be preempted. This is used together with `preempting`
	// to prevent the pods from entering the scheduling cycle while waiting for preemption to complete.
	lastVictimsPendingPreemption map[types.UID]pendingVictim
	// PreemptPod is a function that actually makes API calls to preempt a specific Pod.
	// This is exposed to be replaced during tests.
	PreemptPod func(ctx context.Context, c Candidate, preemptor, victim *v1.Pod, pluginName string) error
}

// newExecutor creates a new preemption executor.
func newExecutor(fh fwk.Handle) *Executor {
	e := &Executor{
		fh:                           fh,
		podLister:                    fh.SharedInformerFactory().Core().V1().Pods().Lister(),
		preempting:                   sets.New[types.UID](),
		lastVictimsPendingPreemption: make(map[types.UID]pendingVictim),
	}

	e.PreemptPod = func(ctx context.Context, c Candidate, preemptor, victim *v1.Pod, pluginName string) error {
		logger := klog.FromContext(ctx)

		skipAPICall := false
		eventMessage := fmt.Sprintf("Preempted by pod %v on node %v", preemptor.UID, c.Name())
		// If the victim is a WaitingPod, try to preempt it without a delete call (victim will go back to backoff queue).
		// Otherwise we should delete the victim.
		if waitingPod := e.fh.GetWaitingPod(victim.UID); waitingPod != nil {
			if waitingPod.Preempt(pluginName, "preempted") {
				logger.V(2).Info("Preemptor pod preempted a waiting pod", "preemptor", klog.KObj(preemptor), "waitingPod", klog.KObj(victim), "node", c.Name())
				skipAPICall = true
			}
		} else if podInPreBind := e.fh.GetPodInPreBind(victim.UID); podInPreBind != nil {
			// If the victim is in the preBind cancel the binding process.
			if podInPreBind.CancelPod(fmt.Sprintf("preempted by %s", pluginName)) {
				logger.V(2).Info("Preemptor pod rejected a pod in preBind", "preemptor", klog.KObj(preemptor), "podInPreBind", klog.KObj(victim), "node", c.Name())
				skipAPICall = true
			} else {
				logger.V(5).Info("Failed to reject a pod in preBind, falling back to deletion via api call", "preemptor", klog.KObj(preemptor), "podInPreBind", klog.KObj(victim), "node", c.Name())
			}
		}
		if !skipAPICall {
			condition := &v1.PodCondition{
				Type:               v1.DisruptionTarget,
				ObservedGeneration: apipod.CalculatePodConditionObservedGeneration(&victim.Status, victim.Generation, v1.DisruptionTarget),
				Status:             v1.ConditionTrue,
				Reason:             v1.PodReasonPreemptionByScheduler,
				Message:            fmt.Sprintf("%s: preempting to accommodate a higher priority pod", preemptor.Spec.SchedulerName),
			}
			newStatus := victim.Status.DeepCopy()
			updated := apipod.UpdatePodCondition(newStatus, condition)
			if updated {
				if err := util.PatchPodStatus(ctx, fh.ClientSet(), victim.Name, victim.Namespace, &victim.Status, newStatus); err != nil {
					if !apierrors.IsNotFound(err) {
						logger.Error(err, "Could not add DisruptionTarget condition due to preemption", "preemptor", klog.KObj(preemptor), "victim", klog.KObj(victim))
						return err
					}
					logger.V(2).Info("Victim Pod is already deleted", "preemptor", klog.KObj(preemptor), "victim", klog.KObj(victim), "node", c.Name())
					return nil
				}
			}
			if err := util.DeletePod(ctx, fh.ClientSet(), victim); err != nil {
				if !apierrors.IsNotFound(err) {
					logger.Error(err, "Tried to preempted pod", "pod", klog.KObj(victim), "preemptor", klog.KObj(preemptor))
					return err
				}
				logger.V(2).Info("Victim Pod is already deleted", "preemptor", klog.KObj(preemptor), "victim", klog.KObj(victim), "node", c.Name())
				return nil
			}
			logger.V(2).Info("Preemptor Pod preempted victim Pod", "preemptor", klog.KObj(preemptor), "victim", klog.KObj(victim), "node", c.Name())
		} else {
			eventMessage += " (in kube-scheduler memory)."
		}

		fh.EventRecorder().Eventf(victim, preemptor, v1.EventTypeNormal, "Preempted", "Preempting", eventMessage)

		return nil
	}

	return e
}

// prepareCandidateAsync triggers a goroutine for some preparation work:
// - Evict the victim pods
// - Reject the victim pods if they are in waitingPod map
// - Clear the low-priority pods' nominatedNodeName status if needed
// The Pod won't be retried until the goroutine triggered here completes.
//
// See http://kep.k8s.io/4832 for how the async preemption works.
func (e *Executor) prepareCandidateAsync(c Candidate, pod *v1.Pod, pluginName string) {
	// Intentionally create a new context, not using a ctx from the scheduling cycle, to create ctx,
	// because this process could continue even after this scheduling cycle finishes.
	ctx, cancel := context.WithCancel(context.Background())
	logger := klog.FromContext(ctx)

	victimPods := make([]*v1.Pod, 0, len(c.Victims().Pods))
	for _, victim := range c.Victims().Pods {
		if victim.DeletionTimestamp != nil {
			// Graceful pod deletion has already started. Sending another API call is unnecessary.
			logger.V(2).Info("Victim Pod is already being deleted, skipping the API call for it", "preemptor", klog.KObj(pod), "node", c.Name(), "victim", klog.KObj(victim))
			continue
		}
		victimPods = append(victimPods, victim)
	}
	if len(victimPods) == 0 {
		cancel()
		return
	}

	metrics.PreemptionVictims.Observe(float64(len(c.Victims().Pods)))

	errCh := parallelize.NewResultChannel[error]()
	preemptPod := func(index int) {
		victim := victimPods[index]
		if err := e.PreemptPod(ctx, c, pod, victim, pluginName); err != nil {
			errCh.SendWithCancel(err, cancel)
		}
	}

	e.mu.Lock()
	e.preempting.Insert(pod.UID)
	e.mu.Unlock()

	go func() {
		logger := klog.FromContext(ctx)
		startTime := time.Now()
		result := metrics.GoroutineResultSuccess
		defer metrics.PreemptionGoroutinesDuration.WithLabelValues(result).Observe(metrics.SinceInSeconds(startTime))
		defer metrics.PreemptionGoroutinesExecutionTotal.WithLabelValues(result).Inc()
		defer func() {
			if result == metrics.GoroutineResultError {
				// When API call isn't successful, the Pod may get stuck in the unschedulable pod pool in the worst case.
				// So, we should move the Pod to the activeQ.
				e.fh.Activate(logger, map[string]*v1.Pod{pod.Name: pod})
			}
		}()
		defer cancel()
		logger.V(2).Info("Start the preemption asynchronously", "preemptor", klog.KObj(pod), "node", c.Name(), "numVictims", len(c.Victims().Pods), "numVictimsToDelete", len(victimPods))

		// Lower priority pods nominated to run on this node, may no longer fit on
		// this node. So, we should remove their nomination. Removing their
		// nomination updates these pods and moves them to the active queue. It
		// lets scheduler find another place for them sooner than after waiting for preemption completion.
		nominatedPods := getLowerPriorityNominatedPods(e.fh, pod, c.Name())
		if err := clearNominatedNodeName(ctx, e.fh.ClientSet(), e.fh.APICacher(), nominatedPods...); err != nil {
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
			e.mu.Lock()
			e.lastVictimsPendingPreemption[pod.UID] = pendingVictim{namespace: lastVictim.Namespace, name: lastVictim.Name}
			e.mu.Unlock()

			if err := e.PreemptPod(ctx, c, pod, lastVictim, pluginName); err != nil {
				utilruntime.HandleErrorWithContext(ctx, err, "Error occurred during async preemption of the last victim")
				result = metrics.GoroutineResultError
			}
		}
		e.mu.Lock()
		e.preempting.Delete(pod.UID)
		delete(e.lastVictimsPendingPreemption, pod.UID)
		e.mu.Unlock()

		logger.V(2).Info("Async Preemption finished completely", "preemptor", klog.KObj(pod), "node", c.Name(), "result", result)
	}()
}

// prepareCandidate does some preparation work before nominating the selected candidate:
// - Evict the victim pods
// - Reject the victim pods if they are in waitingPod map
// - Clear the low-priority pods' nominatedNodeName status if needed
func (e *Executor) prepareCandidate(ctx context.Context, c Candidate, pod *v1.Pod, pluginName string) *fwk.Status {
	metrics.PreemptionVictims.Observe(float64(len(c.Victims().Pods)))

	fh := e.fh
	cs := e.fh.ClientSet()

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	logger := klog.FromContext(ctx)
	errCh := parallelize.NewResultChannel[error]()
	fh.Parallelizer().Until(ctx, len(c.Victims().Pods), func(index int) {
		victim := c.Victims().Pods[index]
		if victim.DeletionTimestamp != nil {
			// Graceful pod deletion has already started. Sending another API call is unnecessary.
			logger.V(2).Info("Victim Pod is already being deleted, skipping the API call for it", "preemptor", klog.KObj(pod), "node", c.Name(), "victim", klog.KObj(victim))
			return
		}
		if err := e.PreemptPod(ctx, c, pod, victim, pluginName); err != nil {
			errCh.SendWithCancel(err, cancel)
		}
	}, pluginName)
	if err := errCh.Receive(); err != nil {
		return fwk.AsStatus(err)
	}

	// Lower priority pods nominated to run on this node, may no longer fit on
	// this node. So, we should remove their nomination. Removing their
	// nomination updates these pods and moves them to the active queue. It
	// lets scheduler find another place for them sooner than after waiting for preemption completion.
	nominatedPods := getLowerPriorityNominatedPods(fh, pod, c.Name())
	if err := clearNominatedNodeName(ctx, cs, fh.APICacher(), nominatedPods...); err != nil {
		utilruntime.HandleErrorWithContext(ctx, err, "Cannot clear 'NominatedNodeName' field")
		// We do not return as this error is not critical.
	}

	return nil
}

// IsPodRunningPreemption returns true if the pod is currently triggering preemption asynchronously.
func (e *Executor) IsPodRunningPreemption(podUID types.UID) bool {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if !e.preempting.Has(podUID) {
		return false
	}

	victim, ok := e.lastVictimsPendingPreemption[podUID]
	if !ok {
		// Since pod is in `preempting` but last victim is not registered yet, the async preemption is pending.
		return true
	}
	// Pod is waiting for preemption of one last victim. We can check if the victim has already been deleted.
	victimPod, err := e.podLister.Pods(victim.namespace).Get(victim.name)
	if err != nil {
		// Victim already deleted, preemption is done.
		return false
	}
	if victimPod.DeletionTimestamp != nil {
		// Victim deletion has started, preemption is done.
		return false
	}
	// Preemption of the last pod is still ongoing.
	return true
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
			if err := util.PatchPodStatus(ctx, cs, p.Name, p.Namespace, &p.Status, podStatusCopy); err != nil {
				errs = append(errs, err)
			}
		}
	}
	return utilerrors.NewAggregate(errs)
}

// getLowerPriorityNominatedPods returns pods whose priority is smaller than the
// priority of the given "pod" and are nominated to run on the given node.
// Note: We could possibly check if the nominated lower priority pods still fit
// and return those that no longer fit, but that would require lots of
// manipulation of NodeInfo and PreFilter state per nominated pod. It may not be
// worth the complexity, especially because we generally expect to have a very
// small number of nominated pods per node.
func getLowerPriorityNominatedPods(pn fwk.PodNominator, pod *v1.Pod, nodeName string) []*v1.Pod {
	podInfos := pn.NominatedPodsForNode(nodeName)

	if len(podInfos) == 0 {
		return nil
	}

	var lowerPriorityPods []*v1.Pod
	podPriority := corev1helpers.PodPriority(pod)
	for _, pi := range podInfos {
		if corev1helpers.PodPriority(pi.GetPod()) < podPriority {
			lowerPriorityPods = append(lowerPriorityPods, pi.GetPod())
		}
	}
	return lowerPriorityPods
}
