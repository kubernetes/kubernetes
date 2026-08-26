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

// containerBoundManager manages probe workers bound to specific container instances.
//
// Each slot (pod, container, probe type) holds a worker tied to a single container
// ID. Workers are driven by explicit lifecycle events, and stop operations match
// against the target container ID so late or out-of-order events are safely no-ops.
type containerBoundManager struct {
	// mu guards workers.
	mu      sync.Mutex
	workers map[probeKey]*containerBoundWorker

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

func NewContainerBoundManager(
	ctx context.Context,
	livenessManager results.Manager,
	readinessManager results.Manager,
	startupManager results.Manager,
	runner kubecontainer.CommandRunner,
	recorder record.EventRecorderLogger) Manager {

	return &containerBoundManager{
		ctx:              ctx,
		prober:           newProber(runner, recorder),
		workers:          make(map[probeKey]*containerBoundWorker),
		readinessManager: readinessManager,
		livenessManager:  livenessManager,
		startupManager:   startupManager,
	}
}

// EnsureProbes performs authoritative periodic reconciliation of probe workers
// against observed runtime status. While event hooks (StartProbes/StopProbes)
// handle expected container lifecycles, EnsureProbes is the source of truth for
// unhooked events (e.g. kubelet restarts, container crashes, or missed events).
func (m *containerBoundManager) EnsureProbes(ctx context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus) {
	if podStatus == nil {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for _, container := range probedContainers(pod) {
		status := runningContainerStatus(podStatus, container.Name)
		if status == nil {
			// Stop probing containers that exited or were never started.
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
			// Already tracked: ensure any newly eligible probe workers are running.
			m.startProbesLocked(ctx, target, false /* adopted */)
			continue
		}

		m.adoptLocked(ctx, target)
	}
}

// adoptLocked starts probing an untracked running container (e.g., after a kubelet
// restart). Because probe state is not persisted, initial probe results are seeded
// from the pod's last reported API status until live probes take over.
func (m *containerBoundManager) adoptLocked(ctx context.Context, target probeTarget) {
	// Only trust reported API status if it matches this specific container instance.
	reported := apiContainerStatus(target.pod, target.container.Name)
	if reported != nil && reported.ContainerID != "" && reported.ContainerID != target.containerID.String() {
		reported = nil
	}

	if target.container.StartupProbe != nil {
		// Preserve started state to avoid re-running passed startup probes or blocking sidecars.
		result := results.Unknown
		if reported != nil && reported.Started != nil && *reported.Started {
			result = results.Success
		}
		m.startupManager.Seed(target.containerID, result)
	}
	if target.container.ReadinessProbe != nil {
		// Preserve readiness status across restarts until live probes execute.
		result := results.Failure
		if reported != nil && reported.Ready {
			result = results.Success
		}
		m.readinessManager.Seed(target.containerID, result)
	}
	if target.container.LivenessProbe != nil {
		// Seed liveness as Success to avoid killing surviving containers before live probes run.
		m.livenessManager.Seed(target.containerID, results.Success)
	}

	klog.FromContext(ctx).V(4).Info("Adopting probes for a running container", "pod", klog.KObj(target.pod),
		"podUID", target.pod.UID, "containerName", target.container.Name, "containerID", target.containerID.String())
	m.startProbesLocked(ctx, target, true /* adopted */)
}

// tracksContainerLocked reports whether a container instance is already tracked
// via an active worker or cached probe results (e.g., passed startup probes).
func (m *containerBoundManager) tracksContainerLocked(podUID types.UID, containerName string, containerID kubecontainer.ContainerID) bool {
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
func (m *containerBoundManager) stopContainerWorkersLocked(ctx context.Context, podUID types.UID, containerName string) {
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
func (m *containerBoundManager) StartProbes(ctx context.Context, pod *v1.Pod, container *v1.Container, containerID kubecontainer.ContainerID, podIPs []string, startedAt time.Time) {
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
func (m *containerBoundManager) startProbesLocked(ctx context.Context, target probeTarget, adopted bool) {
	m.evictStaleWorkersLocked(ctx, target.pod.UID, target.container.Name, target.containerID)

	if target.container.StartupProbe != nil && !m.startupSucceeded(target.containerID) {
		// Gate readiness and liveness workers on startup probe completion.
		m.ensureWorkerLocked(ctx, startup, target, adopted)
		return
	}

	m.ensureWorkerLocked(ctx, readiness, target, adopted)
	m.ensureWorkerLocked(ctx, liveness, target, adopted)
}

// ensureWorkerLocked starts a worker for one probe type of target, unless the
// container has no such probe or a worker is already probing this very
// container instance.
func (m *containerBoundManager) ensureWorkerLocked(ctx context.Context, probeType probeType, target probeTarget, adopted bool) {
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
		// Do not re-probe instances already failed by a terminal probe verdict.
		return
	}

	// Seed initial cache results for new workers if not already populated.
	if !known {
		resultsManager.Set(target.containerID, initialValue(probeType), target.pod)
	}

	opts := containerBoundWorkerOptions{
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

	w := newContainerBoundWorker(m.ctx, opts)
	m.workers[key] = w
	klog.FromContext(ctx).V(4).Info("Starting probe worker", "probeType", probeType, "pod", klog.KObj(target.pod),
		"podUID", target.pod.UID, "containerName", target.container.Name, "containerID", target.containerID.String())
	go w.run()
}

// evictStaleWorkersLocked stops every worker for this container name that is
// bound to some other container instance, and forgets the results those
// instances produced.
func (m *containerBoundManager) evictStaleWorkersLocked(ctx context.Context, podUID types.UID, containerName string, current kubecontainer.ContainerID) {
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
func (m *containerBoundManager) onStartupSucceeded(w *containerBoundWorker) {
	m.mu.Lock()
	defer m.mu.Unlock()

	key := probeKey{w.pod.UID, w.container.Name, startup}
	if current, ok := m.workers[key]; !ok || current != w {
		// Skip if container was stopped or replaced while the startup probe was in flight.
		return
	}

	m.ensureWorkerLocked(m.ctx, readiness, w.probeTarget, w.adopted)
	m.ensureWorkerLocked(m.ctx, liveness, w.probeTarget, w.adopted)
}

// StopProbes stops the specified probe types bound to this container instance.
// When all probes are stopped, its cached results are also removed.
func (m *containerBoundManager) StopProbes(containerID kubecontainer.ContainerID, probeTypes kubecontainer.ProbeType) {
	m.stopWorkersForContainer(containerID, probeTypes)
	if probeTypes&allProbes == allProbes {
		m.removeResults(containerID)
	}
}

// stopWorkersForContainer stops the specified probe types for a container instance,
// ignoring workers bound to newer instances to avoid racing with replacements.
func (m *containerBoundManager) stopWorkersForContainer(containerID kubecontainer.ContainerID, probeTypes probeType) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for key, w := range m.workers {
		if key.probeType&probeTypes == 0 || w.containerID != containerID {
			continue
		}
		m.stopWorkerLocked(key, w)
	}
}

// stopWorkerLocked stops a worker and immediately removes it from the map.
func (m *containerBoundManager) stopWorkerLocked(key probeKey, w *containerBoundWorker) {
	w.stop()
	delete(m.workers, key)
}

// removeWorker removes an exiting worker only if its slot has not been reassigned
// to a newer container instance.
func (m *containerBoundManager) removeWorker(w *containerBoundWorker) {
	m.mu.Lock()
	defer m.mu.Unlock()

	key := probeKey{w.pod.UID, w.container.Name, w.probeType}
	if m.workers[key] == w {
		delete(m.workers, key)
	}
}

// StopLivenessAndStartup stops the pod's liveness and startup workers. Kept on
// Manager for backwards compatibility with the legacy probe manager.
func (m *containerBoundManager) StopLivenessAndStartup(pod *v1.Pod) {
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
func (m *containerBoundManager) RemovePod(pod *v1.Pod) {
	m.removePodWorkers(func(podUID types.UID) bool { return podUID == pod.UID })
}

// CleanupPods stops probes for pods that should no longer run (e.g. deleted while
// kubelet was down).
func (m *containerBoundManager) CleanupPods(desiredPods map[types.UID]sets.Empty) {
	m.removePodWorkers(func(podUID types.UID) bool {
		_, desired := desiredPods[podUID]
		return !desired
	})
}

func (m *containerBoundManager) removePodWorkers(match func(types.UID) bool) {
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
func (m *containerBoundManager) removeResults(containerID kubecontainer.ContainerID) {
	m.readinessManager.Remove(containerID)
	m.livenessManager.Remove(containerID)
	m.startupManager.Remove(containerID)
}

func (m *containerBoundManager) resultsManager(probeType probeType) results.Manager {
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
func (m *containerBoundManager) startupSucceeded(containerID kubecontainer.ContainerID) bool {
	result, ok := m.startupManager.Get(containerID)
	return ok && result == results.Success
}

// UpdatePodStatus updates Started and Ready container conditions directly from
// cached probe results.
func (m *containerBoundManager) UpdatePodStatus(ctx context.Context, pod *v1.Pod, podStatus *v1.PodStatus) {
	logger := klog.FromContext(ctx)

	for i := range podStatus.ContainerStatuses {
		containerStatus := &podStatus.ContainerStatuses[i]
		containerSpec := findContainerSpec(pod.Spec.Containers, containerStatus.Name)

		started := m.isContainerStarted(logger, containerStatus, containerSpec)
		containerStatus.Started = &started
		if !started {
			containerStatus.Ready = false
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
			} else {
				containerStatus.Ready = false
			}
			continue
		}

		if !started {
			containerStatus.Ready = false
			continue
		}
		containerStatus.Ready = m.isContainerReady(logger, containerStatus, &initContainer)
	}
}

// isContainerStarted reports whether the container has passed its startup
// probe, or has no startup probe to pass.
func (m *containerBoundManager) isContainerStarted(logger klog.Logger, containerStatus *v1.ContainerStatus, containerSpec *v1.Container) bool {
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
func (m *containerBoundManager) isContainerReady(logger klog.Logger, containerStatus *v1.ContainerStatus, containerSpec *v1.Container) bool {
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
func (m *containerBoundManager) workerCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.workers)
}

// probedContainers returns all regular and restartable init containers with defined probes.
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

// runningContainerStatus returns the status of the container's running instance,
// ignoring terminated instances with the same container name.
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
	if status, ok := podutil.GetContainerStatus(pod.Status.ContainerStatuses, containerName); ok {
		return &status
	}
	if status, ok := podutil.GetContainerStatus(pod.Status.InitContainerStatuses, containerName); ok {
		return &status
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

// isTerminalResult reports whether a result ends probing of a container instance
// due to a failed liveness or startup probe.
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
