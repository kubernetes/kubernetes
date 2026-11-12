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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	"k8s.io/component-base/metrics"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	kubeutil "k8s.io/kubernetes/pkg/kubelet/util"
)

// containerKey uniquely identifies a container slot within a pod.
type containerKey struct {
	podUID        types.UID
	containerName string
}

// probeMetricLabels contains metadata required to delete Prometheus metric series for a probe.
type probeMetricLabels struct {
	podUID        types.UID
	podName       string
	podNamespace  string
	containerName string
	probeType     kubecontainer.ProbeType
}

// containerBoundManager binds each worker to one container ID so stale lifecycle
// events cannot affect a replacement container.
type containerBoundManager struct {
	mutable bool
	// Track even containers without probes so adding startup cannot undo their started state.
	targets     map[containerKey]probeTarget
	terminating sets.Set[types.UID]

	// mu guards workers, containers, targets, terminating, and probedMetrics.
	mu sync.Mutex

	// Workers can exit before the container, after a terminal probe result.
	workers map[probeKey]*containerBoundWorker

	// Keep container IDs after workers exit to evict results on restart or removal.
	containers map[containerKey]kubecontainer.ContainerID

	// Keep labels after workers exit to delete metric series on pod removal.
	probedMetrics map[probeKey]probeMetricLabels

	// Outlive pod sync cancellation so readiness probes continue during termination.
	ctx context.Context

	prober *prober

	readinessManager results.Manager
	livenessManager  results.Manager
	startupManager   results.Manager
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
		mutable:          utilfeature.DefaultFeatureGate.Enabled(features.MutableContainerProbes),
		targets:          make(map[containerKey]probeTarget),
		terminating:      sets.New[types.UID](),
		prober:           newProber(runner, recorder),
		workers:          make(map[probeKey]*containerBoundWorker),
		containers:       make(map[containerKey]kubecontainer.ContainerID),
		probedMetrics:    make(map[probeKey]probeMetricLabels),
		readinessManager: readinessManager,
		livenessManager:  livenessManager,
		startupManager:   startupManager,
	}
}

// EnsureProbes reconciles workers after kubelet restarts, container crashes,
// or missed lifecycle events.
func (m *containerBoundManager) EnsureProbes(ctx context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus) {
	if podStatus == nil {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	containers := probedContainers(pod)
	if m.mutable {
		if m.terminating.Has(pod.UID) || pod.DeletionTimestamp != nil || pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
			return
		}
		pod = pod.DeepCopy()
		containers = append([]v1.Container{}, pod.Spec.Containers...)
		for _, c := range pod.Spec.InitContainers {
			if podutil.IsRestartableInitContainer(&c) {
				containers = append(containers, c)
			}
		}
	}
	for _, container := range containers {
		status := runningContainerStatus(podStatus, container.Name)
		if status == nil {
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
			if m.mutable && !kubetypes.IsStaticPod(pod) && !kubetypes.IsMirrorPod(pod) {
				m.reconcileProbesLocked(target)
			}
			m.startProbesLocked(ctx, target, false /* adopted */)
			continue
		}

		m.adoptLocked(ctx, target)
	}
}

// adoptLocked restores probe state from API status until live probes take over.
func (m *containerBoundManager) adoptLocked(ctx context.Context, target probeTarget) {
	// Only trust reported API status if it matches this specific container instance.
	reported := apiContainerStatus(target.pod, target.container.Name)
	if reported != nil && reported.ContainerID != target.containerID.String() {
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
		"containerName", target.container.Name, "containerID", target.containerID.String())
	m.startProbesLocked(ctx, target, true /* adopted */)
}

func (m *containerBoundManager) tracksContainerLocked(podUID types.UID, containerName string, containerID kubecontainer.ContainerID) bool {
	ck := containerKey{podUID: podUID, containerName: containerName}
	id, ok := m.containers[ck]
	return ok && id == containerID
}

// stopContainerWorkersLocked stops every worker for one container of a pod,
// whichever instance they are bound to, and forgets their results.
func (m *containerBoundManager) stopContainerWorkersLocked(ctx context.Context, podUID types.UID, containerName string) {
	ck := containerKey{podUID: podUID, containerName: containerName}

	key := probeKey{podUID: podUID, containerName: containerName}
	for probeType := kubecontainer.LivenessProbe; probeType <= kubecontainer.StartupProbe; probeType <<= 1 {
		key.probeType = probeType
		w, ok := m.workers[key]
		if !ok {
			continue
		}
		klog.FromContext(ctx).V(4).Info("Stopping probe worker for a container that is no longer running",
			"probeType", key.probeType, "podUID", podUID, "containerName", containerName, "containerID", w.containerID.String())
		m.stopWorkerLocked(key, w)
	}
	if id, ok := m.containers[ck]; ok {
		m.removeResults(id)
		delete(m.containers, ck)
		delete(m.targets, ck)
	}
}

// StartProbes starts probing a container instance that has just started. It is
// called from the container runtime once CRI StartContainer has succeeded, and
// is idempotent for a given container ID.
func (m *containerBoundManager) StartProbes(ctx context.Context, pod *v1.Pod, container *v1.Container, containerID kubecontainer.ContainerID, podIPs []string, startedAt time.Time) {
	if m.mutable {
		pod = pod.DeepCopy()
		container = container.DeepCopy()
		podIPs = append([]string(nil), podIPs...)
	}
	target := probeTarget{
		pod:         pod,
		container:   *container,
		containerID: containerID,
		podIPs:      podIPs,
		startedAt:   startedAt,
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.mutable && m.tracksContainerLocked(pod.UID, container.Name, containerID) {
		return
	}
	m.startProbesLocked(ctx, target, false /* adopted */)
}

func (m *containerBoundManager) startProbesLocked(ctx context.Context, target probeTarget, adopted bool) {
	if m.mutable && (m.terminating.Has(target.pod.UID) || target.pod.DeletionTimestamp != nil) {
		return
	}
	ck := containerKey{podUID: target.pod.UID, containerName: target.container.Name}
	m.evictStaleWorkersLocked(ctx, target.pod.UID, target.container.Name, target.containerID)
	if oldID, ok := m.containers[ck]; ok && oldID != target.containerID {
		m.removeResults(oldID)
	}

	m.containers[ck] = target.containerID
	if m.mutable {
		m.targets[ck] = target
	}
	if target.container.StartupProbe != nil && !m.startupSucceeded(target.containerID) {
		// Gate readiness and liveness workers on startup probe completion.
		m.ensureWorkerLocked(ctx, kubecontainer.StartupProbe, target, adopted)
		return
	}

	m.ensureWorkerLocked(ctx, kubecontainer.ReadinessProbe, target, adopted)
	m.ensureWorkerLocked(ctx, kubecontainer.LivenessProbe, target, adopted)
}

func (m *containerBoundManager) ensureWorkerLocked(ctx context.Context, probeType kubecontainer.ProbeType, target probeTarget, adopted bool) {
	spec := probeSpec(probeType, &target.container)
	if spec == nil {
		return
	}

	key := probeKey{target.pod.UID, target.container.Name, probeType}
	if w, ok := m.workers[key]; ok {
		if w.containerID == target.containerID {
			// Both the start hook and reconciliation can reach this instance.
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

	// Initialization must not block on the updates channel while holding m.mu.
	if !known {
		resultsManager.Seed(target.containerID, initialValue(probeType))
	}

	opts := containerBoundWorkerOptions{
		probeType:      probeType,
		target:         target,
		prober:         m.prober,
		resultsManager: resultsManager,
		adopted:        adopted,
		onExit:         m.removeWorker,
		mutable:        m.mutable,
	}
	if probeType == kubecontainer.StartupProbe {
		opts.onStartupSucceeded = m.onStartupSucceeded
	}

	m.probedMetrics[key] = probeMetricLabels{
		podUID:        target.pod.UID,
		podName:       target.pod.Name,
		podNamespace:  target.pod.Namespace,
		containerName: target.container.Name,
		probeType:     probeType,
	}

	w := newContainerBoundWorker(m.ctx, opts)
	m.workers[key] = w
	klog.FromContext(ctx).V(4).Info("Starting probe worker", "probeType", probeType, "pod", klog.KObj(target.pod),
		"podUID", target.pod.UID, "containerName", target.container.Name, "containerID", target.containerID.String())
	go w.run()
}

// evictStaleWorkersLocked stops every worker for this container name that is
// bound to some other container instance.
func (m *containerBoundManager) evictStaleWorkersLocked(ctx context.Context, podUID types.UID, containerName string, current kubecontainer.ContainerID) {
	key := probeKey{podUID: podUID, containerName: containerName}
	for probeType := kubecontainer.LivenessProbe; probeType <= kubecontainer.StartupProbe; probeType <<= 1 {
		key.probeType = probeType
		w, ok := m.workers[key]
		if !ok || w.containerID == current {
			continue
		}
		klog.FromContext(ctx).V(4).Info("Stopping probe worker for a replaced container", "probeType", key.probeType,
			"podUID", podUID, "containerName", containerName, "staleContainerID", w.containerID.String(), "containerID", current.String())
		m.stopWorkerLocked(key, w)
	}
}

// onStartupSucceeded runs on the startup worker's goroutine before it exits.
func (m *containerBoundManager) onStartupSucceeded(w *containerBoundWorker) {
	m.mu.Lock()
	defer m.mu.Unlock()

	key := probeKey{w.pod.UID, w.container.Name, kubecontainer.StartupProbe}
	if current, ok := m.workers[key]; !ok || current != w {
		// Skip if container was stopped or replaced while the startup probe was in flight.
		return
	}

	target := w.probeTarget
	if m.mutable {
		if !m.startupSucceeded(w.containerID) || m.terminating.Has(w.pod.UID) {
			return
		}
		target = m.targets[containerKey{w.pod.UID, w.container.Name}]
	}
	m.ensureWorkerLocked(m.ctx, kubecontainer.ReadinessProbe, target, w.adopted)
	m.ensureWorkerLocked(m.ctx, kubecontainer.LivenessProbe, target, w.adopted)
}

// StopProbes stops the specified probe types bound to this container instance.
// When all probes are stopped, its cached results are also removed.
func (m *containerBoundManager) StopProbes(containerID kubecontainer.ContainerID, probeTypes kubecontainer.ProbeType) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for key, w := range m.workers {
		if key.probeType&probeTypes == 0 || w.containerID != containerID {
			continue
		}
		m.stopWorkerLocked(key, w)
	}

	if probeTypes&kubecontainer.AllProbes == kubecontainer.AllProbes {
		m.removeResults(containerID)
		for ck, id := range m.containers {
			if id == containerID {
				delete(m.containers, ck)
				delete(m.targets, ck)
			}
		}
	}
}

func (m *containerBoundManager) stopWorkerLocked(key probeKey, w *containerBoundWorker) {
	w.stopMutable()
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
	if m.mutable {
		m.terminating.Insert(pod.UID)
	}

	for key, w := range m.workers {
		if key.podUID != pod.UID || key.probeType == kubecontainer.ReadinessProbe {
			continue
		}
		m.stopWorkerLocked(key, w)
	}
}

// RemovePod stops all of the pod's probes and forgets their results.
func (m *containerBoundManager) RemovePod(pod *v1.Pod) {
	m.mu.Lock()
	if m.mutable {
		m.terminating.Insert(pod.UID)
	}
	m.mu.Unlock()
	m.removePodWorkers(func(podUID types.UID) bool { return podUID == pod.UID })
}

// CleanupPods stops probes for pods that should no longer run (e.g. deleted while
// kubelet was down).
func (m *containerBoundManager) CleanupPods(desiredPods map[types.UID]sets.Empty) {
	m.mu.Lock()
	for uid := range m.terminating {
		if _, ok := desiredPods[uid]; !ok {
			m.terminating.Delete(uid)
		}
	}
	m.mu.Unlock()
	m.removePodWorkers(func(podUID types.UID) bool {
		_, desired := desiredPods[podUID]
		return !desired
	})
}

func (m *containerBoundManager) removePodWorkers(match func(types.UID) bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for key, w := range m.workers {
		if !match(key.podUID) {
			continue
		}
		m.stopWorkerLocked(key, w)
	}

	for ck, id := range m.containers {
		if !match(ck.podUID) {
			continue
		}
		m.removeResults(id)
		delete(m.containers, ck)
		delete(m.targets, ck)
	}

	// Metric series are scoped by pod and container name rather than container instance.
	// Delete them only when the pod is removed to preserve counters across container restarts.
	for key, labels := range m.probedMetrics {
		if !match(key.podUID) {
			continue
		}
		deleteMetricSeries(labels)
		delete(m.probedMetrics, key)
	}
}

func deleteMetricSeries(l probeMetricLabels) {
	basicLabels := metrics.Labels{
		"probe_type": l.probeType.String(),
		"container":  l.containerName,
		"pod":        l.podName,
		"namespace":  l.podNamespace,
		"pod_uid":    string(l.podUID),
	}

	proberDurationLabels := metrics.Labels{
		"probe_type": l.probeType.String(),
		"container":  l.containerName,
		"pod":        l.podName,
		"namespace":  l.podNamespace,
	}

	for _, result := range []string{probeResultSuccessful, probeResultFailed, probeResultUnknown} {
		lbls := deepCopyPrometheusLabels(basicLabels)
		lbls["result"] = result
		ProberResults.Delete(lbls)
	}
	ProberDuration.Delete(proberDurationLabels)
}

func (m *containerBoundManager) removeResults(containerID kubecontainer.ContainerID) {
	m.readinessManager.Remove(containerID)
	m.livenessManager.Remove(containerID)
	m.startupManager.Remove(containerID)
}

func (m *containerBoundManager) resultsManager(probeType kubecontainer.ProbeType) results.Manager {
	switch probeType {
	case kubecontainer.ReadinessProbe:
		return m.readinessManager
	case kubecontainer.LivenessProbe:
		return m.livenessManager
	case kubecontainer.StartupProbe:
		return m.startupManager
	}
	return nil
}

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

// Only call isContainerReady for a started container.
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

// runningContainerStatus only considers the newest instance of the container.
func runningContainerStatus(podStatus *kubecontainer.PodStatus, containerName string) *kubecontainer.Status {
	status := podStatus.FindContainerStatusByName(containerName)
	if status != nil && status.State == kubecontainer.ContainerStateRunning {
		return status
	}
	return nil
}

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

func probeSpec(probeType kubecontainer.ProbeType, container *v1.Container) *v1.Probe {
	switch probeType {
	case kubecontainer.ReadinessProbe:
		return container.ReadinessProbe
	case kubecontainer.LivenessProbe:
		return container.LivenessProbe
	case kubecontainer.StartupProbe:
		return container.StartupProbe
	}
	return nil
}

// isTerminalResult reports whether a result ends probing of a container instance
// due to a failed liveness or startup probe.
func isTerminalResult(probeType kubecontainer.ProbeType, result results.Result) bool {
	return (probeType == kubecontainer.LivenessProbe || probeType == kubecontainer.StartupProbe) && result == results.Failure
}

func initialValue(probeType kubecontainer.ProbeType) results.Result {
	switch probeType {
	case kubecontainer.ReadinessProbe:
		// Not ready until proven ready.
		return results.Failure
	case kubecontainer.LivenessProbe:
		// Never kill a container over a probe that has not run.
		return results.Success
	default:
		return results.Unknown
	}
}

// Reconcile configuration in the existing worker so cancelled calls cannot overlap replacements.
func (m *containerBoundManager) reconcileProbesLocked(target probeTarget) {
	ck := containerKey{target.pod.UID, target.container.Name}
	old := m.targets[ck]
	for _, typ := range []kubecontainer.ProbeType{kubecontainer.StartupProbe, kubecontainer.ReadinessProbe, kubecontainer.LivenessProbe} {
		spec := probeSpec(typ, &target.container)
		previous := probeSpec(typ, &old.container)
		complete := typ == kubecontainer.StartupProbe && (previous == nil || m.startupSucceeded(target.containerID))
		if w := m.workers[probeKey{target.pod.UID, target.container.Name, typ}]; w != nil {
			w.updateMutable(target, complete)
		} else if previous != nil && spec == nil {
			m.resultsManager(typ).Remove(target.containerID)
		}
		if typ == kubecontainer.StartupProbe && spec != nil && previous == nil && m.workers[probeKey{target.pod.UID, target.container.Name, typ}] == nil {
			m.startupManager.Seed(target.containerID, results.Success)
		}
	}
	m.targets[ck] = target
}
