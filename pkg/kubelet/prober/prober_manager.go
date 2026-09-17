/*
Copyright 2015 The Kubernetes Authors.

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
	"sync/atomic"
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
	"k8s.io/kubernetes/pkg/kubelet/status"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	kubeutil "k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/utils/clock"
)

// ProberResults stores the cumulative number of a probe by result as prometheus metrics.
var ProberResults = metrics.NewCounterVec(
	&metrics.CounterOpts{
		Subsystem:      "prober",
		Name:           "probe_total",
		Help:           "Cumulative number of a liveness, readiness or startup probe for a container by result.",
		StabilityLevel: metrics.BETA,
	},
	[]string{"probe_type",
		"result",
		"container",
		"pod",
		"namespace",
		"pod_uid"},
)

// ProberDuration stores the duration of a successful probe lifecycle by result as prometheus metrics.
var ProberDuration = metrics.NewHistogramVec(
	&metrics.HistogramOpts{
		Subsystem:      "prober",
		Name:           "probe_duration_seconds",
		Help:           "Duration in seconds for a probe response.",
		StabilityLevel: metrics.ALPHA,
	},
	[]string{"probe_type",
		"container",
		"pod",
		"namespace"},
)

// Manager manages pod probing. It reconciles a probe "worker" for every container that specifies a
// probe. The worker periodically probes its assigned container and caches the results. The manager
// uses the cached probe results to set the appropriate Ready state in the PodStatus when requested.
type Manager interface {
	// AddPod creates workers using the original add-only lifecycle.
	AddPod(ctx context.Context, pod *v1.Pod)

	// ReconcilePod makes the probe workers for a pod match its current spec. statusChanged is true
	// when enabling or disabling a probe can change the derived Started or Ready state.
	ReconcilePod(ctx context.Context, pod *v1.Pod) (statusChanged bool, err error)

	// IsResultCurrent rejects updates superseded by a publication, probe removal, or container replacement.
	IsResultCurrent(update results.Update, probeType ProbeType) bool

	// StopLivenessAndStartup handles stopping liveness and startup probes during termination.
	StopLivenessAndStartup(pod *v1.Pod)

	// RemovePod handles cleaning up the removed pod state, including terminating probe workers and
	// deleting cached results.
	RemovePod(pod *v1.Pod)

	// CleanupPods handles cleaning up pods which should no longer be running.
	// It takes a map of "desired pods" which should not be cleaned up.
	CleanupPods(desiredPods map[types.UID]sets.Empty)

	// UpdatePodStatus modifies the given PodStatus with the appropriate Ready state for each
	// container based on container running status, cached probe results and worker states.
	UpdatePodStatus(context.Context, *v1.Pod, *v1.PodStatus)
}

type manager struct {
	// Feature gates are fixed for the lifetime of this manager.
	mutable bool

	// Workers include disabled probes so re-enabling cannot overlap an old in-flight call.
	workers map[probeKey]*worker
	// Lock for accessing & mutating workers
	workerLock      sync.RWMutex
	nextProbeID     atomic.Uint64
	reconciledPods  sets.Set[types.UID]
	terminatingPods sets.Set[types.UID]

	// The statusManager cache provides pod IP and container IDs for probing.
	statusManager status.Manager

	// readinessManager manages the results of readiness probes
	readinessManager results.Manager

	// livenessManager manages the results of liveness probes
	livenessManager results.Manager

	// startupManager manages the results of startup probes
	startupManager results.Manager

	// prober executes the probe actions.
	prober *prober

	start time.Time
}

// NewManager creates a Manager for pod probing.
func NewManager(
	statusManager status.Manager,
	livenessManager results.Manager,
	readinessManager results.Manager,
	startupManager results.Manager,
	runner kubecontainer.CommandRunner,
	recorder record.EventRecorderLogger) Manager {

	prober := newProber(runner, recorder)
	m := &manager{
		mutable:          utilfeature.DefaultFeatureGate.Enabled(features.MutableContainerProbes),
		statusManager:    statusManager,
		prober:           prober,
		readinessManager: readinessManager,
		livenessManager:  livenessManager,
		startupManager:   startupManager,
		workers:          make(map[probeKey]*worker),
		start:            clock.RealClock{}.Now(),
	}

	if m.mutable {
		m.reconciledPods = sets.New[types.UID]()
		m.terminatingPods = sets.New[types.UID]()
	}

	return m
}

// Key uniquely identifying container probes
type probeKey struct {
	podUID        types.UID
	containerName string
	probeType     probeType
}

// ProbeType identifies a liveness, readiness, or startup probe.
type ProbeType int

type probeType = ProbeType

const (
	// ProbeTypeLiveness identifies a liveness probe result.
	ProbeTypeLiveness ProbeType = iota
	// ProbeTypeReadiness identifies a readiness probe result.
	ProbeTypeReadiness
	// ProbeTypeStartup identifies a startup probe result.
	ProbeTypeStartup

	liveness  = ProbeTypeLiveness
	readiness = ProbeTypeReadiness
	startup   = ProbeTypeStartup

	probeResultSuccessful string = "successful"
	probeResultFailed     string = "failed"
	probeResultUnknown    string = "unknown"
)

// For debugging.
func (t ProbeType) String() string {
	switch t {
	case readiness:
		return "Readiness"
	case liveness:
		return "Liveness"
	case startup:
		return "Startup"
	default:
		return "UNKNOWN"
	}
}

func getRestartableInitContainers(pod *v1.Pod) []v1.Container {
	var restartableInitContainers []v1.Container
	for _, c := range pod.Spec.InitContainers {
		if podutil.IsRestartableInitContainer(&c) {
			restartableInitContainers = append(restartableInitContainers, c)
		}
	}
	return restartableInitContainers
}

func (m *manager) AddPod(ctx context.Context, pod *v1.Pod) {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()

	logger := klog.FromContext(ctx)
	// Detach the workers' context from the caller's: the pod worker cancels the
	// sync context when the pod begins terminating, but probe workers must keep
	// probing until the container stops so a failing readiness probe can mark
	// the pod NotReady during graceful termination. Workers are stopped
	// explicitly via their stop channel (RemovePod/CleanupPods).
	//
	// TODO(#140977): This also means nothing cancels an in-flight probe. worker.stop()
	// only signals stopCh, which is checked between probes, so an exec probe that is
	// already running keeps executing in a container that is being killed until its
	// own TimeoutSeconds elapses. The fix is a per-worker cancellable context
	// cancelled by stop(), not the pod sync context, which cancels too early.
	ctx = context.WithoutCancel(ctx)
	key := probeKey{podUID: pod.UID}
	for _, c := range append(pod.Spec.Containers, getRestartableInitContainers(pod)...) {
		key.containerName = c.Name

		if c.StartupProbe != nil {
			key.probeType = startup
			if _, ok := m.workers[key]; ok {
				logger.V(8).Info("Startup probe already exists for container",
					"pod", klog.KObj(pod), "containerName", c.Name)
				continue
			}
			w := newWorker(m, startup, pod, c)
			m.workers[key] = w
			go w.run(ctx)
		}

		if c.ReadinessProbe != nil {
			key.probeType = readiness
			if _, ok := m.workers[key]; ok {
				logger.V(8).Info("Readiness probe already exists for container",
					"pod", klog.KObj(pod), "containerName", c.Name)
				continue
			}
			w := newWorker(m, readiness, pod, c)
			m.workers[key] = w
			go w.run(ctx)
		}

		if c.LivenessProbe != nil {
			key.probeType = liveness
			if _, ok := m.workers[key]; ok {
				logger.V(8).Info("Liveness probe already exists for container",
					"pod", klog.KObj(pod), "containerName", c.Name)
				continue
			}
			w := newWorker(m, liveness, pod, c)
			m.workers[key] = w
			go w.run(ctx)
		}
	}
}

type desiredProbe struct {
	pod       *v1.Pod
	container v1.Container
	probeType probeType
}

func desiredProbeWorkers(pod *v1.Pod) map[probeKey]desiredProbe {
	desired := make(map[probeKey]desiredProbe)
	for _, c := range append(pod.Spec.Containers, getRestartableInitContainers(pod)...) {
		for _, probeType := range [...]probeType{startup, readiness, liveness} {
			if probeForType(c, probeType) == nil {
				continue
			}

			key := probeKey{podUID: pod.UID, containerName: c.Name, probeType: probeType}
			desired[key] = desiredProbe{pod: pod, container: c, probeType: probeType}
		}
	}

	return desired
}

// ReconcilePod applies mutable probe configuration only while MutableContainerProbes is enabled.
func (m *manager) ReconcilePod(ctx context.Context, pod *v1.Pod) (bool, error) {
	if !m.mutable {
		m.AddPod(ctx, pod)
		return false, nil
	}

	// Readiness must outlive the pod sync context during graceful termination. Explicit probe
	// shutdown, rather than cancellation of this caller, owns the lifetime of each worker.
	ctx = context.WithoutCancel(ctx)
	if pod.DeletionTimestamp != nil || pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
		return false, nil
	}

	status, found := m.statusManager.GetPodStatus(pod.UID)
	if found && (status.Phase == v1.PodSucceeded || status.Phase == v1.PodFailed) {
		return false, nil
	}

	// All workers may share this snapshot because neither reconciliation nor execution mutates it.
	pod = pod.DeepCopy()
	desired := desiredProbeWorkers(pod)
	mutable := utilfeature.DefaultFeatureGate.Enabled(features.MutableContainerProbes) && !kubetypes.IsStaticPod(pod) && !kubetypes.IsMirrorPod(pod)
	var toStart, toWake []*worker
	var cancellations []context.CancelFunc
	statusChanged := false

	m.workerLock.Lock()
	if m.terminatingPods.Has(pod.UID) {
		m.workerLock.Unlock()
		return false, nil
	}

	previouslyReconciled := m.reconciledPods.Has(pod.UID)
	m.reconciledPods.Insert(pod.UID)

	for key, w := range m.workers {
		if key.podUID != pod.UID {
			continue
		}

		probe, wanted := desired[key]
		delete(desired, key)
		if !mutable {
			continue
		}

		container := probe.container
		if !wanted {
			// Only the probe's absence matters here; keep the rest of its immutable snapshot.
			cfg := w.configSnapshot()
			container = *cfg.container.DeepCopy()
			switch key.probeType {
			case readiness:
				container.ReadinessProbe = nil
			case liveness:
				container.LivenessProbe = nil
			case startup:
				container.StartupProbe = nil
			}
		}

		changed, cancel := w.applyConfig(ctx, pod, container)
		if cancel != nil {
			cancellations = append(cancellations, cancel)
		}

		if changed && key.probeType != liveness {
			statusChanged = true
		}

		toWake = append(toWake, w)
	}

	for key, probe := range desired {
		w := newWorker(m, probe.probeType, probe.pod, probe.container)
		if mutable {
			w.initializeProbe(ctx, previouslyReconciled)
			if key.probeType != liveness {
				statusChanged = true
			}
		}

		m.workers[key] = w
		toStart = append(toStart, w)
	}
	m.workerLock.Unlock()

	for _, cancel := range cancellations {
		cancel()
	}

	for _, w := range toWake {
		w.wake()
	}

	for _, w := range toStart {
		go w.run(ctx)
	}

	return statusChanged, nil
}

func (m *manager) IsResultCurrent(update results.Update, probeType ProbeType) bool {
	if !m.mutable {
		return true
	}

	// Preserve callers that inject updates without going through the probe result cache.
	if update.ProbeID == 0 {
		return true
	}

	var cache results.Manager
	switch probeType {
	case liveness:
		cache = m.livenessManager
	case readiness:
		cache = m.readinessManager
	case startup:
		cache = m.startupManager
	default:
		return false
	}

	status, ok := m.statusManager.GetPodStatus(update.PodUID)
	if !ok {
		return false
	}

	c, ok := containerStatusByName(status, update.ContainerName)
	return ok && c.ContainerID == update.ContainerID.String() && cache.IsCurrent(update)
}

func (m *manager) StopLivenessAndStartup(pod *v1.Pod) {
	if m.mutable {
		m.stopPodWorkers(pod.UID, false)
		return
	}

	m.workerLock.RLock()
	defer m.workerLock.RUnlock()

	key := probeKey{podUID: pod.UID}
	for _, c := range append(pod.Spec.Containers, getRestartableInitContainers(pod)...) {
		key.containerName = c.Name
		for _, probeType := range [...]probeType{liveness, startup} {
			key.probeType = probeType
			if worker, ok := m.workers[key]; ok {
				worker.stop()
			}
		}
	}
}

func (m *manager) RemovePod(pod *v1.Pod) {
	if m.mutable {
		m.stopPodWorkers(pod.UID, true)
		return
	}

	m.workerLock.RLock()
	defer m.workerLock.RUnlock()

	key := probeKey{podUID: pod.UID}
	for _, c := range append(pod.Spec.Containers, getRestartableInitContainers(pod)...) {
		key.containerName = c.Name
		for _, probeType := range [...]probeType{readiness, liveness, startup} {
			key.probeType = probeType
			if worker, ok := m.workers[key]; ok {
				worker.stop()
			}
		}
	}
}

func (m *manager) stopPodWorkers(uid types.UID, includeReadiness bool) {
	var workers []*worker
	m.workerLock.Lock()
	m.terminatingPods.Insert(uid)
	for key, w := range m.workers {
		if key.podUID == uid && (includeReadiness || key.probeType != readiness) {
			workers = append(workers, w)
		}
	}
	m.workerLock.Unlock()

	for _, w := range workers {
		w.stop()
	}
}

func (m *manager) CleanupPods(desiredPods map[types.UID]sets.Empty) {
	if m.mutable {
		m.cleanupMutablePods(desiredPods)
		return
	}

	m.workerLock.RLock()
	defer m.workerLock.RUnlock()

	for key, worker := range m.workers {
		if _, ok := desiredPods[key.podUID]; !ok {
			worker.stop()
		}
	}
}

func (m *manager) cleanupMutablePods(desiredPods map[types.UID]sets.Empty) {
	var workers []*worker
	m.workerLock.Lock()
	activePods := sets.New[types.UID]()
	for key, w := range m.workers {
		activePods.Insert(key.podUID)
		if _, desired := desiredPods[key.podUID]; !desired {
			m.terminatingPods.Insert(key.podUID)
			workers = append(workers, w)
		}
	}

	for uid := range m.terminatingPods {
		if _, desired := desiredPods[uid]; !desired && !activePods.Has(uid) {
			m.terminatingPods.Delete(uid)
		}
	}

	for uid := range m.reconciledPods {
		if _, desired := desiredPods[uid]; !desired && !activePods.Has(uid) {
			m.reconciledPods.Delete(uid)
		}
	}
	m.workerLock.Unlock()

	for _, w := range workers {
		w.stop()
	}
}

func (m *manager) isContainerStarted(logger klog.Logger, pod *v1.Pod, containerStatus *v1.ContainerStatus) bool {
	if containerStatus.State.Running == nil {
		return false
	}

	if result, ok := m.startupManager.Get(kubecontainer.ParseContainerID(logger, containerStatus.ContainerID)); ok {
		return result == results.Success
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.ChangeContainerStatusOnKubeletRestart) && containerStatus.Started != nil && *containerStatus.Started {
		return true
	}

	// if there is a startup probe which hasn't run yet, the container is not
	// started.
	if _, exists := m.getWorker(pod.UID, containerStatus.Name, startup); exists {
		return false
	}

	// there is no startup probe, so the container is started.
	return true
}

// setReadyStateOnKubeletRestart sets the ready state of a container to false if it was started
// before kubelet restarted and has a readiness probe, but the pod is not ready yet.
// This is to avoid flapping ready status of containers that were ready before kubelet restarted.
func (m *manager) setReadyStateOnKubeletRestart(logger klog.Logger, ready *bool, pod *v1.Pod, containerStatus *v1.ContainerStatus, containerSpec *v1.Container) {
	var containerStartTime time.Time
	if containerStatus.State.Running != nil {
		containerStartTime = containerStatus.State.Running.StartedAt.Time
	}

	if !containerStartTime.IsZero() && containerStartTime.Before(kubeletRestartGracePeriod(m.start)) {
		// At this point, the Pod may be in one of the following two states:
		// - It has not yet been added to the readinessManager. In this case, we directly set the container status to Ready.
		// - It has been added to the readinessManager, but the probe has not yet started execution.
		// Therefore, in this case, we also need to set the container status to Ready.
		if !*ready {
			if _, ok := m.readinessManager.Get(kubecontainer.ParseContainerID(logger, containerStatus.ContainerID)); !ok {
				*ready = true
			}
		}
		if containerSpec.ReadinessProbe != nil {
			podIsReady := false
			for _, c := range pod.Status.Conditions {
				if c.Type == v1.PodReady && c.Status == v1.ConditionTrue {
					podIsReady = true
					break
				}
			}
			if !podIsReady {
				*ready = false
			}
		}
	}
}

func (m *manager) UpdatePodStatus(ctx context.Context, pod *v1.Pod, podStatus *v1.PodStatus) {
	logger := klog.FromContext(ctx)
	for i, c := range podStatus.ContainerStatuses {
		started := m.isContainerStarted(logger, pod, &podStatus.ContainerStatuses[i])
		podStatus.ContainerStatuses[i].Started = &started

		if !started {
			continue
		}

		var ready bool
		if c.State.Running == nil {
			ready = false
		} else if result, ok := m.readinessManager.Get(kubecontainer.ParseContainerID(logger, c.ContainerID)); ok && result == results.Success {
			ready = true
		} else {
			// The check whether there is a probe which hasn't run yet.
			w, exists := m.getWorker(pod.UID, c.Name, readiness)
			ready = !exists // no readinessProbe -> always ready
			if exists {
				// Trigger an immediate run of the readinessProbe to update ready state
				select {
				case w.manualTriggerCh <- struct{}{}:
				default: // Non-blocking.
					logger.Info("Failed to trigger a manual run", "probe", w.probeType.String())
				}
			}

			if !utilfeature.DefaultFeatureGate.Enabled(features.ChangeContainerStatusOnKubeletRestart) {
				// Find the container spec for the container status.
				var containerSpec *v1.Container
				for j := range pod.Spec.Containers {
					if pod.Spec.Containers[j].Name == c.Name {
						containerSpec = &pod.Spec.Containers[j]
						break
					}
				}
				if containerSpec != nil {
					m.setReadyStateOnKubeletRestart(logger, &ready, pod, &podStatus.ContainerStatuses[i], containerSpec)
				}
			}
		}
		podStatus.ContainerStatuses[i].Ready = ready
	}

	for i, c := range podStatus.InitContainerStatuses {
		started := m.isContainerStarted(logger, pod, &podStatus.InitContainerStatuses[i])
		podStatus.InitContainerStatuses[i].Started = &started

		initContainer, ok := kubeutil.GetContainerByIndex(pod.Spec.InitContainers, podStatus.InitContainerStatuses, i)
		if !ok {
			logger.V(4).Info("Mismatch between pod spec and status, likely programmer error", "pod", klog.KObj(pod), "containerName", c.Name)
			continue
		}
		if !podutil.IsRestartableInitContainer(&initContainer) {
			if c.State.Terminated != nil && c.State.Terminated.ExitCode == 0 {
				podStatus.InitContainerStatuses[i].Ready = true
			}
			continue
		}

		if !started {
			continue
		}

		var ready bool
		if c.State.Running == nil {
			ready = false
		} else if result, ok := m.readinessManager.Get(kubecontainer.ParseContainerID(logger, c.ContainerID)); ok && result == results.Success {
			ready = true
		} else {
			// The check whether there is a probe which hasn't run yet.
			w, exists := m.getWorker(pod.UID, c.Name, readiness)
			ready = !exists // no readinessProbe -> always ready
			if exists {
				// Trigger an immediate run of the readinessProbe to update ready state
				select {
				case w.manualTriggerCh <- struct{}{}:
				default: // Non-blocking.
					logger.Info("Failed to trigger a manual run", "probe", w.probeType.String())
				}
			}
			if !utilfeature.DefaultFeatureGate.Enabled(features.ChangeContainerStatusOnKubeletRestart) {
				m.setReadyStateOnKubeletRestart(logger, &ready, pod, &podStatus.InitContainerStatuses[i], &initContainer)
			}
		}
		podStatus.InitContainerStatuses[i].Ready = ready
	}
}

func (m *manager) getWorker(podUID types.UID, containerName string, probeType probeType) (*worker, bool) {
	m.workerLock.RLock()
	defer m.workerLock.RUnlock()
	worker, ok := m.workers[probeKey{podUID, containerName, probeType}]
	if !m.mutable {
		return worker, ok
	}

	return worker, ok && worker.isEnabled()
}

// Called by the worker after exiting. A worker replaced under the same key must not remove its
// replacement or the replacement's cached result.
func (m *manager) removeWorker(key probeKey, worker *worker) bool {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()

	if current, ok := m.workers[key]; !ok || current != worker {
		return false
	}

	delete(m.workers, key)

	return true
}

// workerCount returns the total number of probe workers. For testing.
func (m *manager) workerCount() int {
	m.workerLock.RLock()
	defer m.workerLock.RUnlock()
	return len(m.workers)
}

// kubeletRestartGracePeriod returns a time point that is 10 seconds before the kubelet start time.
// This grace period is used to determine if a container was already running before kubelet restarted.
// If a container's start time is before this grace period, it indicates the container was running
// prior to kubelet restart and should not be immediately marked as failed to avoid unnecessary
// status changes for containers that were previously ready.
func kubeletRestartGracePeriod(start time.Time) time.Time {
	return start.Add(-time.Second * 10)
}

func (m *manager) removeLegacyWorker(podUID types.UID, containerName string, probeType probeType) {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()

	delete(m.workers, probeKey{podUID, containerName, probeType})
}
