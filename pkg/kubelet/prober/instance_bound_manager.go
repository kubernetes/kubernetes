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

package prober

import (
	"context"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	kubeutil "k8s.io/kubernetes/pkg/kubelet/util"
)

// allProbeTypes is the iteration order for whole-container operations.
var allProbeTypes = [...]probeType{liveness, readiness, startup}

// instanceBoundManager drives probe workers from the pod sync loop instead of
// letting them discover containers for themselves.
//
// Each map entry is a "slot" -- one probe type of one container of one pod --
// holding a worker bound to a single container ID. Two rules keep that binding
// honest:
//
//   - Workers are only ever created on the pod worker goroutine, which is
//     serialized per pod, so lifecycle events for one pod are totally ordered.
//     Callers from other goroutines (RemovePod, CleanupPods) only remove.
//   - Every stop names a container ID and only affects a worker bound to that
//     ID. Late, duplicate or reordered events become no-ops instead of races.
//
// Together these make the "which container am I probing" question unanswerable
// in the wrong way, rather than answerable but easy to get wrong.
type instanceBoundManager struct {
	// mu guards workers. There is no polling read path left to justify an
	// RWMutex.
	mu      sync.Mutex
	workers map[probeKey]*instanceBoundWorker

	// ctx is the parent of every worker context. It deliberately outlives any
	// pod sync context: the pod worker cancels its sync context when a pod
	// starts terminating, but probes must keep running until the container
	// stops, so a failing readiness probe can take the pod out of service
	// during graceful termination.
	ctx context.Context

	// prober executes the probe actions.
	prober *prober

	// readinessManager manages the results of readiness probes
	readinessManager results.Manager

	// livenessManager manages the results of liveness probes
	livenessManager results.Manager

	// startupManager manages the results of startup probes
	startupManager results.Manager
}

// NewInstanceBoundManager creates a probe Manager whose workers are started and
// stopped with the containers they probe. ctx must outlive the manager; every
// probe worker's context descends from it.
func NewInstanceBoundManager(
	ctx context.Context,
	livenessManager results.Manager,
	readinessManager results.Manager,
	startupManager results.Manager,
	runner kubecontainer.CommandRunner,
	recorder record.EventRecorderLogger) Manager {

	return newInstanceBoundManager(ctx, livenessManager, readinessManager, startupManager, runner, recorder)
}

func newInstanceBoundManager(
	ctx context.Context,
	livenessManager results.Manager,
	readinessManager results.Manager,
	startupManager results.Manager,
	runner kubecontainer.CommandRunner,
	recorder record.EventRecorderLogger) *instanceBoundManager {

	return &instanceBoundManager{
		ctx:              ctx,
		prober:           newProber(runner, recorder),
		workers:          make(map[probeKey]*instanceBoundWorker),
		readinessManager: readinessManager,
		livenessManager:  livenessManager,
		startupManager:   startupManager,
	}
}

// EnsureProbes reconciles probe workers against the runtime's observed container
// status (podStatus).
//
// Probe management uses two complementary mechanisms:
//   - Event hooks (StartProbes / StopProbes): invoked immediately when the kubelet
//     starts or stops a container, minimizing latency.
//   - State reconciliation (EnsureProbes): authoritative periodic reconciliation
//     covering cases where event hooks do not fire (e.g., containers running across
//     a kubelet restart, containers that crashed or exited on their own, or missed events).
//
// EnsureProbes runs on the serialized pod worker goroutine before the runtime sync
// executes. Because it reads the runtime snapshot before any lifecycle operations or
// hooks for this sync iteration fire, state reconciliation and event hooks cannot race.
//
// For each probe-bearing container in the pod:
//  1. Running and already tracked: keep probing (and start any probe workers the
//     container is newly eligible for).
//  2. Running and untracked: adopt it by reconstructing prior probe state from the
//     pod API status and start probing.
//  3. Not running: stop its probe workers and clear cached results.
func (m *instanceBoundManager) EnsureProbes(ctx context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus) {
	if podStatus == nil {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for _, container := range probedContainers(pod) {
		status := runningContainerStatus(podStatus, container.Name)
		if status == nil {
			// Rule 3. The container is gone, was never started, or exited on
			// its own; either way nothing should still be probing it.
			m.stopContainerWorkersLocked(ctx, pod.UID, container.Name)
			continue
		}

		target := probeTarget{
			pod:         pod,
			container:   container,
			containerID: status.ID,
			podIPs:      podStatus.IPs,
			startedAt:   status.StartedAt,
		}

		if m.tracksContainerLocked(pod.UID, container.Name, status.ID) {
			// Rule 1. Idempotent: this also fills in any worker the container
			// has since become eligible for.
			m.startProbesLocked(ctx, target, false /* adopted */)
			continue
		}

		// Rule 2.
		m.adoptLocked(ctx, target)
	}
}

// adoptLocked starts probing a container that is already running but that this
// manager knows nothing about -- overwhelmingly, one that survived a kubelet
// restart.
//
// There is no persisted probe state, so what the container's probes "were" has
// to be reconstructed. The one rule is: believe the last thing the kubelet told
// the API server, and let the first real probe correct it within a period. That
// is deliberately a rule rather than a heuristic -- it replaces guessing from
// container start timestamps whether a container predates the kubelet.
func (m *instanceBoundManager) adoptLocked(ctx context.Context, target probeTarget) {
	// Trust the reported state only if it is about this same container
	// instance. An API status with no container ID recorded yet is matched by
	// name; one naming a different instance tells us nothing about this one.
	reported := apiContainerStatus(target.pod, target.container.Name)
	if reported != nil && reported.ContainerID != "" && reported.ContainerID != target.containerID.String() {
		reported = nil
	}

	if target.container.StartupProbe != nil {
		// A container the API says has started stays started, so a kubelet
		// restart does not re-run a startup probe that already passed and does
		// not deadlock a sidecar waiting for a result that will never arrive.
		result := results.Unknown
		if reported != nil && reported.Started != nil && *reported.Started {
			result = results.Success
		}
		m.startupManager.Seed(target.containerID, result)
	}
	if target.container.ReadinessProbe != nil {
		// A kubelet restart flips Ready in neither direction.
		result := results.Failure
		if reported != nil && reported.Ready {
			result = results.Success
		}
		m.readinessManager.Seed(target.containerID, result)
	}
	if target.container.LivenessProbe != nil {
		// Never kill a container over anything that happened before the
		// restart.
		m.livenessManager.Seed(target.containerID, results.Success)
	}

	klog.FromContext(ctx).V(4).Info("Adopting probes for a running container", "pod", klog.KObj(target.pod),
		"podUID", target.pod.UID, "containerName", target.container.Name, "containerID", target.containerID.String())
	m.startProbesLocked(ctx, target, true /* adopted */)
}

// tracksContainerLocked reports whether this manager already knows about a
// container instance: either it has a worker bound to it, or it has results
// cached for it. The second case covers a container whose only probe was a
// startup probe that has already passed, which correctly has no worker left.
func (m *instanceBoundManager) tracksContainerLocked(podUID types.UID, containerName string, containerID kubecontainer.ContainerID) bool {
	key := probeKey{podUID: podUID, containerName: containerName}
	for _, probeType := range allProbeTypes {
		key.probeType = probeType
		if w, ok := m.workers[key]; ok && w.containerID == containerID {
			return true
		}
		if _, ok := m.resultsManager(probeType).Get(containerID); ok {
			return true
		}
	}
	return false
}

// stopContainerWorkersLocked stops every worker for one container of a pod,
// whichever instance they are bound to, and forgets their results.
func (m *instanceBoundManager) stopContainerWorkersLocked(ctx context.Context, podUID types.UID, containerName string) {
	stale := sets.New[kubecontainer.ContainerID]()
	key := probeKey{podUID: podUID, containerName: containerName}
	for _, probeType := range allProbeTypes {
		key.probeType = probeType
		w, ok := m.workers[key]
		if !ok {
			continue
		}
		klog.FromContext(ctx).V(4).Info("Stopping probe worker for a container that is no longer running",
			"probeType", probeType, "containerName", containerName, "containerID", w.containerID.String())
		m.stopWorkerLocked(key, w)
		stale.Insert(w.containerID)
	}
	for _, id := range stale.UnsortedList() {
		m.removeResults(id)
	}
}

// StartProbes starts probing a container instance that has just started. It is
// called from the container runtime once CRI StartContainer has succeeded, and
// is idempotent for a given container ID.
func (m *instanceBoundManager) StartProbes(ctx context.Context, pod *v1.Pod, container *v1.Container, containerID kubecontainer.ContainerID, podIPs []string, startedAt time.Time) {
	target := probeTarget{
		pod:         pod,
		container:   *container,
		containerID: containerID,
		podIPs:      podIPs,
		startedAt:   startedAt,
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.startProbesLocked(ctx, target, false /* adopted */)
}

// startProbesLocked installs the workers a container instance should have right
// now, replacing any worker left over from an earlier instance of the same
// container.
func (m *instanceBoundManager) startProbesLocked(ctx context.Context, target probeTarget, adopted bool) {
	m.evictStaleWorkersLocked(ctx, target.pod.UID, target.container.Name, target.containerID)

	if target.container.StartupProbe != nil && !m.startupSucceeded(target.containerID) {
		// Readiness and liveness are gated on the startup probe. They are
		// created by onStartupSucceeded, so that a worker existing always means
		// a worker that is actively probing -- there is no idle "waiting for
		// startup" state for anyone else to have to reason about.
		m.ensureWorkerLocked(ctx, startup, target, adopted)
		return
	}

	m.ensureWorkerLocked(ctx, readiness, target, adopted)
	m.ensureWorkerLocked(ctx, liveness, target, adopted)
}

// ensureWorkerLocked starts a worker for one probe type of target, unless the
// container has no such probe or a worker is already probing this very
// container instance.
func (m *instanceBoundManager) ensureWorkerLocked(ctx context.Context, probeType probeType, target probeTarget, adopted bool) {
	spec := probeSpec(probeType, &target.container)
	if spec == nil {
		return
	}

	key := probeKey{target.pod.UID, target.container.Name, probeType}
	if w, ok := m.workers[key]; ok {
		if w.containerID == target.containerID {
			// Already probing this instance. Both the container-start hook and
			// reconciliation can reach here for the same container.
			return
		}
		// Stale workers are evicted before we get here, but a slot bound to a
		// different container must never be silently overwritten.
		m.stopWorkerLocked(key, w)
	}

	resultsManager := m.resultsManager(probeType)
	cached, known := resultsManager.Get(target.containerID)

	if known && isTerminalResult(probeType, cached) {
		// This probe has already reached a verdict this container instance
		// cannot come back from: the sync loop is about to kill it. Probing
		// again could only overwrite the verdict that is about to be acted on,
		// and the restart brings new workers with it anyway.
		return
	}

	// Seed the cache so consumers always find a result for a container that has
	// a worker: a readiness-probed container is not Ready until it passes, a
	// liveness-probed one is not killed for a probe that has not run yet, and a
	// startup-probed one is neither started nor failed. Seeding is skipped if
	// something already knows better -- adoption fills the cache from the last
	// state reported to the API before starting workers.
	if !known {
		resultsManager.Set(target.containerID, initialValue(probeType), target.pod)
	}

	opts := instanceBoundWorkerOptions{
		probeType:      probeType,
		target:         target,
		prober:         m.prober,
		resultsManager: resultsManager,
		adopted:        adopted,
		onExit:         m.removeWorker,
	}
	if probeType == startup {
		opts.onStartupSucceeded = m.onStartupSucceeded
	}

	w := newInstanceBoundWorker(m.ctx, opts)
	m.workers[key] = w
	klog.FromContext(ctx).V(4).Info("Starting probe worker", "probeType", probeType, "pod", klog.KObj(target.pod),
		"podUID", target.pod.UID, "containerName", target.container.Name, "containerID", target.containerID.String())
	go w.run()
}

// evictStaleWorkersLocked stops every worker for this container name that is
// bound to some other container instance, and forgets the results those
// instances produced.
func (m *instanceBoundManager) evictStaleWorkersLocked(ctx context.Context, podUID types.UID, containerName string, current kubecontainer.ContainerID) {
	stale := sets.New[kubecontainer.ContainerID]()
	key := probeKey{podUID: podUID, containerName: containerName}
	for _, probeType := range allProbeTypes {
		key.probeType = probeType
		w, ok := m.workers[key]
		if !ok || w.containerID == current {
			continue
		}
		klog.FromContext(ctx).V(4).Info("Stopping probe worker for a replaced container", "probeType", probeType,
			"containerName", containerName, "staleContainerID", w.containerID.String(), "containerID", current.String())
		m.stopWorkerLocked(key, w)
		stale.Insert(w.containerID)
	}
	for _, id := range stale.UnsortedList() {
		m.removeResults(id)
	}
}

// onStartupSucceeded brings up the readiness and liveness workers for a
// container whose startup probe has just passed. It runs on the startup
// worker's goroutine, just before that worker exits.
func (m *instanceBoundManager) onStartupSucceeded(w *instanceBoundWorker) {
	m.mu.Lock()
	defer m.mu.Unlock()

	key := probeKey{w.pod.UID, w.container.Name, startup}
	if current, ok := m.workers[key]; !ok || current != w {
		// The container was killed, or replaced by a new instance, while this
		// probe was in flight. Bringing up probes for it now would be probing a
		// container that no longer exists.
		return
	}

	m.ensureWorkerLocked(m.ctx, readiness, w.probeTarget, w.adopted)
	m.ensureWorkerLocked(m.ctx, liveness, w.probeTarget, w.adopted)
}

// StopLivenessAndStartupProbes stops the liveness and startup workers bound to
// this container instance, aborting any probe they are executing. It is called
// before the PreStop hook, so that a container which is on its way out is not
// killed again for failing a probe, and so that no exec probe is running inside
// it while it is torn down.
//
// Readiness keeps running: a container that is shutting down should be taken
// out of service, which is exactly what its readiness probe failing does.
func (m *instanceBoundManager) StopLivenessAndStartupProbes(containerID kubecontainer.ContainerID) {
	m.stopWorkersForContainer(containerID, liveness, startup)
}

// StopProbes stops every worker bound to this container instance and forgets
// its results. It is called once the container has actually stopped.
func (m *instanceBoundManager) StopProbes(containerID kubecontainer.ContainerID) {
	m.stopWorkersForContainer(containerID, allProbeTypes[:]...)
	m.removeResults(containerID)
}

// stopWorkersForContainer stops the given probe types for one container
// instance. A worker bound to any other instance is left alone, so a stop that
// arrives late -- after the container has already been replaced -- cannot take
// down its successor's workers.
func (m *instanceBoundManager) stopWorkersForContainer(containerID kubecontainer.ContainerID, probeTypes ...probeType) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, probeType := range probeTypes {
		for key, w := range m.workers {
			if key.probeType != probeType || w.containerID != containerID {
				continue
			}
			m.stopWorkerLocked(key, w)
		}
	}
}

// stopWorkerLocked stops a worker and vacates its slot immediately, rather than
// waiting for the worker's goroutine to notice, so that the slot reads as free
// to anything that looks at it next.
func (m *instanceBoundManager) stopWorkerLocked(key probeKey, w *instanceBoundWorker) {
	w.stop()
	delete(m.workers, key)
}

// removeWorker is called by a worker as it exits. It is a compare-and-delete:
// by the time a worker gets here its slot may already hold a worker for a newer
// container instance, which must not be dropped.
func (m *instanceBoundManager) removeWorker(w *instanceBoundWorker) {
	m.mu.Lock()
	defer m.mu.Unlock()

	key := probeKey{w.pod.UID, w.container.Name, w.probeType}
	if m.workers[key] == w {
		delete(m.workers, key)
	}
}

// StopLivenessAndStartup stops the pod's liveness and startup workers. Container
// kill hooks (StopLivenessAndStartupProbes) already stop individual probes before
// PreStop hooks execute; this method is kept on the Manager interface for backwards
// compatibility with the legacy probe manager and will be removed in the future.
func (m *instanceBoundManager) StopLivenessAndStartup(pod *v1.Pod) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for key, w := range m.workers {
		if key.podUID != pod.UID || key.probeType == readiness {
			continue
		}
		m.stopWorkerLocked(key, w)
	}
}

// RemovePod stops all of the pod's probes and forgets their results.
func (m *instanceBoundManager) RemovePod(pod *v1.Pod) {
	m.removePodWorkers(func(podUID types.UID) bool { return podUID == pod.UID })
}

// CleanupPods stops the probes of every pod that should no longer be running.
// It is the safety net for pods that never reach the sync loop at all, such as
// a runtime pod left behind by a kubelet restart after its pod was deleted.
func (m *instanceBoundManager) CleanupPods(desiredPods map[types.UID]sets.Empty) {
	m.removePodWorkers(func(podUID types.UID) bool {
		_, desired := desiredPods[podUID]
		return !desired
	})
}

func (m *instanceBoundManager) removePodWorkers(match func(types.UID) bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	stale := sets.New[kubecontainer.ContainerID]()
	for key, w := range m.workers {
		if !match(key.podUID) {
			continue
		}
		m.stopWorkerLocked(key, w)
		// Metric series are scoped by pod and container name rather than container instance.
		// Delete them only when the pod is removed to preserve counters across container restarts.
		w.deleteMetrics()
		stale.Insert(w.containerID)
	}
	for _, id := range stale.UnsortedList() {
		m.removeResults(id)
	}
}

// removeResults drops every cached probe result for a container instance.
func (m *instanceBoundManager) removeResults(containerID kubecontainer.ContainerID) {
	m.readinessManager.Remove(containerID)
	m.livenessManager.Remove(containerID)
	m.startupManager.Remove(containerID)
}

func (m *instanceBoundManager) resultsManager(probeType probeType) results.Manager {
	switch probeType {
	case readiness:
		return m.readinessManager
	case liveness:
		return m.livenessManager
	case startup:
		return m.startupManager
	}
	return nil
}

// startupSucceeded reports whether the container is known to have passed its
// startup probe.
func (m *instanceBoundManager) startupSucceeded(containerID kubecontainer.ContainerID) bool {
	result, ok := m.startupManager.Get(containerID)
	return ok && result == results.Success
}

// UpdatePodStatus fills in Started and Ready for each container from the probe
// result caches.
//
// It is a pure read of those caches. There is no need to consult the worker map
// to find out whether a probe "has run yet", and no need to poke a worker into
// probing early, because a worker is only created once it will actually probe,
// and its cache entry is seeded at creation. Whatever this reads, some worker
// deliberately put there.
func (m *instanceBoundManager) UpdatePodStatus(ctx context.Context, pod *v1.Pod, podStatus *v1.PodStatus) {
	logger := klog.FromContext(ctx)

	for i := range podStatus.ContainerStatuses {
		containerStatus := &podStatus.ContainerStatuses[i]
		containerSpec := findContainerSpec(pod.Spec.Containers, containerStatus.Name)

		started := m.isContainerStarted(logger, containerStatus, containerSpec)
		containerStatus.Started = &started
		if !started {
			continue
		}
		containerStatus.Ready = m.isContainerReady(logger, containerStatus, containerSpec)
	}

	for i := range podStatus.InitContainerStatuses {
		containerStatus := &podStatus.InitContainerStatuses[i]

		initContainer, ok := kubeutil.GetContainerByIndex(pod.Spec.InitContainers, podStatus.InitContainerStatuses, i)
		if !ok {
			logger.V(4).Info("Mismatch between pod spec and status, likely programmer error", "pod", klog.KObj(pod), "containerName", containerStatus.Name)
			continue
		}

		started := m.isContainerStarted(logger, containerStatus, &initContainer)
		containerStatus.Started = &started

		if !podutil.IsRestartableInitContainer(&initContainer) {
			// A plain init container is "ready" once it has done its job.
			if containerStatus.State.Terminated != nil && containerStatus.State.Terminated.ExitCode == 0 {
				containerStatus.Ready = true
			}
			continue
		}

		if !started {
			continue
		}
		containerStatus.Ready = m.isContainerReady(logger, containerStatus, &initContainer)
	}
}

// isContainerStarted reports whether the container has passed its startup
// probe, or has no startup probe to pass.
func (m *instanceBoundManager) isContainerStarted(logger klog.Logger, containerStatus *v1.ContainerStatus, containerSpec *v1.Container) bool {
	if containerStatus.State.Running == nil {
		return false
	}
	if containerSpec == nil || containerSpec.StartupProbe == nil {
		return true
	}
	result, ok := m.startupManager.Get(kubecontainer.ParseContainerID(logger, containerStatus.ContainerID))
	return ok && result == results.Success
}

// isContainerReady reports whether the container is passing its readiness
// probe, or has no readiness probe to pass. Only call it for a started
// container.
func (m *instanceBoundManager) isContainerReady(logger klog.Logger, containerStatus *v1.ContainerStatus, containerSpec *v1.Container) bool {
	if containerStatus.State.Running == nil {
		return false
	}
	if containerSpec == nil || containerSpec.ReadinessProbe == nil {
		return true
	}
	result, ok := m.readinessManager.Get(kubecontainer.ParseContainerID(logger, containerStatus.ContainerID))
	return ok && result == results.Success
}

// workerCount returns the total number of probe workers. For testing.
func (m *instanceBoundManager) workerCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.workers)
}

// probedContainers returns the pod's containers that have at least one probe.
// Restartable init containers are included: they are long-running containers
// that can be probed like any other, and they are exactly where losing probe
// state across a kubelet restart is most damaging.
func probedContainers(pod *v1.Pod) []v1.Container {
	var probed []v1.Container
	for _, c := range pod.Spec.Containers {
		if hasProbe(&c) {
			probed = append(probed, c)
		}
	}
	for _, c := range pod.Spec.InitContainers {
		if podutil.IsRestartableInitContainer(&c) && hasProbe(&c) {
			probed = append(probed, c)
		}
	}
	return probed
}

func hasProbe(container *v1.Container) bool {
	return container.ReadinessProbe != nil || container.LivenessProbe != nil || container.StartupProbe != nil
}

// runningContainerStatus returns the runtime status of the container's running
// instance, or nil if it has none. A container name can appear more than once
// in a pod status -- previous, exited instances are reported too -- and only the
// running one is a thing to probe.
func runningContainerStatus(podStatus *kubecontainer.PodStatus, containerName string) *kubecontainer.Status {
	for _, status := range podStatus.ContainerStatuses {
		if status.Name == containerName && status.State == kubecontainer.ContainerStateRunning {
			return status
		}
	}
	return nil
}

// apiContainerStatus returns the container's status as last reported to the API
// server, which is where adoption reconstructs probe state from.
func apiContainerStatus(pod *v1.Pod, containerName string) *v1.ContainerStatus {
	for i, status := range pod.Status.ContainerStatuses {
		if status.Name == containerName {
			return &pod.Status.ContainerStatuses[i]
		}
	}
	for i, status := range pod.Status.InitContainerStatuses {
		if status.Name == containerName {
			return &pod.Status.InitContainerStatuses[i]
		}
	}
	return nil
}

func findContainerSpec(containers []v1.Container, containerName string) *v1.Container {
	for i := range containers {
		if containers[i].Name == containerName {
			return &containers[i]
		}
	}
	return nil
}

func probeSpec(probeType probeType, container *v1.Container) *v1.Probe {
	switch probeType {
	case readiness:
		return container.ReadinessProbe
	case liveness:
		return container.LivenessProbe
	case startup:
		return container.StartupProbe
	}
	return nil
}

// isTerminalResult reports whether a result ends probing of a container
// instance for good. A failed liveness or startup probe gets the container
// killed, and the replacement is a different instance with its own workers.
func isTerminalResult(probeType probeType, result results.Result) bool {
	return (probeType == liveness || probeType == startup) && result == results.Failure
}

// initialValue is what a probe's result is taken to be before the probe has
// actually run.
func initialValue(probeType probeType) results.Result {
	switch probeType {
	case readiness:
		// Not ready until proven ready.
		return results.Failure
	case liveness:
		// Never kill a container over a probe that has not run.
		return results.Success
	default:
		return results.Unknown
	}
}
