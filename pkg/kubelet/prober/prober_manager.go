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

// Manager manages pod probing. It creates a probe "worker" for every container that specifies a
// probe (AddPod). The worker periodically probes its assigned container and caches the results. The
// manager use the cached probe results to set the appropriate Ready state in the PodStatus when
// requested (UpdatePodStatus). Updating probe parameters is not currently supported.
type Manager interface {
	// AddPod creates new probe workers for every container probe. This should be called for every
	// pod created.
	AddPod(ctx context.Context, pod *v1.Pod)

	// StartContainerProbes starts probe workers for a started container. It is idempotent
	// and safe to call multiple times for the same container.
	StartContainerProbes(ctx context.Context, pod *v1.Pod, container *v1.Container, containerID kubecontainer.ContainerID, podIP string, containerStartTime time.Time)

	// StopContainerProbes handles stopping probe workers for a terminated container.
	StopContainerProbes(podUID types.UID, containerName string)

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
	// Map of active workers for probes
	workers map[probeKey]*worker
	// Lock for accessing & mutating workers
	workerLock sync.RWMutex

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
	return &manager{
		statusManager:    statusManager,
		prober:           prober,
		readinessManager: readinessManager,
		livenessManager:  livenessManager,
		startupManager:   startupManager,
		workers:          make(map[probeKey]*worker),
		start:            clock.RealClock{}.Now(),
	}
}

// Key uniquely identifying container probes
type probeKey struct {
	podUID        types.UID
	containerName string
	probeType     probeType
}

// Type of probe (liveness, readiness or startup)
type probeType int

const (
	liveness probeType = iota
	readiness
	startup

	probeResultSuccessful string = "successful"
	probeResultFailed     string = "failed"
	probeResultUnknown    string = "unknown"
)

// For debugging.
func (t probeType) String() string {
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
			if !utilfeature.DefaultFeatureGate.Enabled(features.ContainerLifecycleProber) {
				w.running.Store(true)
				go w.run(ctx)
			}
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
			if !utilfeature.DefaultFeatureGate.Enabled(features.ContainerLifecycleProber) {
				w.running.Store(true)
				go w.run(ctx)
			}
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
			if !utilfeature.DefaultFeatureGate.Enabled(features.ContainerLifecycleProber) {
				w.running.Store(true)
				go w.run(ctx)
			}
		}
	}
}

// StartContainerProbes starts probe workers for a started container. It is idempotent for a
// given container instance and safe to call multiple times (e.g. on every SyncPod). If the
// containerID differs from the one currently being probed (a container restart that did not
// go through StopContainerProbes, such as a crash-restart), the stale workers are torn down
// and recreated for the new instance.
func (m *manager) StartContainerProbes(ctx context.Context, pod *v1.Pod, container *v1.Container, containerID kubecontainer.ContainerID, podIP string, containerStartTime time.Time) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ContainerLifecycleProber) {
		return
	}
	m.workerLock.Lock()
	defer m.workerLock.Unlock()

	m.resetStaleWorkersLocked(pod, container, containerID)
	m.ensureWorkersLocked(pod, *container)

	if container.StartupProbe != nil {
		// Only the startup probe runs initially; readiness and liveness are activated once
		// the startup probe succeeds (see worker.startDependents).
		m.startWorkerLocked(ctx, pod, container, startup, containerID, podIP, containerStartTime)
	} else {
		m.startLivenessAndReadinessLocked(ctx, pod, container, containerID, podIP, containerStartTime)
	}
}

// resetStaleWorkersLocked stops any workers for the container that are bound to a
// previous container instance and removes them from m.workers. This handles container restarts
// that do not go through killContainer/StopContainerProbes. Caller must hold workerLock.
func (m *manager) resetStaleWorkersLocked(pod *v1.Pod, container *v1.Container, containerID kubecontainer.ContainerID) {
	key := probeKey{podUID: pod.UID, containerName: container.Name}
	for _, probeType := range [...]probeType{readiness, liveness, startup} {
		key.probeType = probeType
		if w, ok := m.workers[key]; ok && !w.containerID.IsEmpty() && w.containerID != containerID {
			w.stop()
			delete(m.workers, key)
		}
	}
}

// startWorkerLocked activates the probe worker of the given type for a container instance,
// launching its probe loop. It is a no-op if no probe of that type is configured or the worker
// is already probing the given container instance.
//
// 1-to-1 Worker-to-Goroutine Lifecycle:
// To avoid race conditions between exiting probe goroutines and newly launched container instances,
// worker structs are not reused across container runs. Each startWorkerLocked call creates a fresh
// *worker instance for the specific containerID. When a previous worker's goroutine exits, its cleanup
// defer (removeWorkerIfCurrent) only removes m.workers entries that still point to that specific worker,
// preventing stale goroutines from deleting active workers.
// Caller must hold workerLock.
func (m *manager) startWorkerLocked(ctx context.Context, pod *v1.Pod, container *v1.Container, probeType probeType, containerID kubecontainer.ContainerID, podIP string, containerStartTime time.Time) {
	key := probeKey{podUID: pod.UID, containerName: container.Name, probeType: probeType}
	w, ok := m.workers[key]
	if !ok {
		// No probe of this type is configured for the container.
		return
	}
	if w.running.Load() && w.containerID == containerID {
		// Already probing this container instance.
		return
	}
	if w.running.Load() {
		// Probing a different container instance; signal it to stop.
		w.stop()
	}

	w = newWorker(m, probeType, pod, *container)
	w.containerID = containerID
	w.podIP = podIP
	w.containerStartTime = containerStartTime
	w.running.Store(true)
	m.workers[key] = w
	go w.run(ctx)
}

// startLivenessAndReadinessLocked activates readiness and liveness probe workers for a container.
// Caller must hold workerLock.
func (m *manager) startLivenessAndReadinessLocked(ctx context.Context, pod *v1.Pod, container *v1.Container, containerID kubecontainer.ContainerID, podIP string, containerStartTime time.Time) {
	m.startWorkerLocked(ctx, pod, container, readiness, containerID, podIP, containerStartTime)
	m.startWorkerLocked(ctx, pod, container, liveness, containerID, podIP, containerStartTime)
}

// startLivenessAndReadiness is called upon startup probe success to immediately transition to
// running readiness and liveness probes without sync loop delay.
func (m *manager) startLivenessAndReadiness(ctx context.Context, pod *v1.Pod, container *v1.Container, containerID kubecontainer.ContainerID, podIP string, containerStartTime time.Time) {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()
	m.startLivenessAndReadinessLocked(ctx, pod, container, containerID, podIP, containerStartTime)
}

func (m *manager) StopContainerProbes(podUID types.UID, containerName string) {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()

	key := probeKey{podUID: podUID, containerName: containerName}
	for _, probeType := range [...]probeType{readiness, liveness, startup} {
		key.probeType = probeType
		if worker, ok := m.workers[key]; ok {
			worker.stop()
			if !worker.running.Load() {
				delete(m.workers, key)
			}
		}
	}
}

// ensureWorkersLocked guarantees that probe worker entries are allocated in
// m.workers for any configured probes. Caller must hold workerLock.
func (m *manager) ensureWorkersLocked(pod *v1.Pod, c v1.Container) {
	key := probeKey{podUID: pod.UID, containerName: c.Name}
	if c.StartupProbe != nil {
		key.probeType = startup
		if _, ok := m.workers[key]; !ok {
			m.workers[key] = newWorker(m, startup, pod, c)
		}
	}
	if c.ReadinessProbe != nil {
		key.probeType = readiness
		if _, ok := m.workers[key]; !ok {
			m.workers[key] = newWorker(m, readiness, pod, c)
		}
	}
	if c.LivenessProbe != nil {
		key.probeType = liveness
		if _, ok := m.workers[key]; !ok {
			m.workers[key] = newWorker(m, liveness, pod, c)
		}
	}
}

func (m *manager) StopLivenessAndStartup(pod *v1.Pod) {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()

	key := probeKey{podUID: pod.UID}
	for _, c := range pod.Spec.Containers {
		key.containerName = c.Name
		for _, probeType := range [...]probeType{liveness, startup} {
			key.probeType = probeType
			if worker, ok := m.workers[key]; ok {
				worker.stop()
				if !worker.running.Load() {
					delete(m.workers, key)
				}
			}
		}
	}
}

func (m *manager) RemovePod(pod *v1.Pod) {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()

	key := probeKey{podUID: pod.UID}
	for _, c := range append(pod.Spec.Containers, getRestartableInitContainers(pod)...) {
		key.containerName = c.Name
		for _, probeType := range [...]probeType{readiness, liveness, startup} {
			key.probeType = probeType
			if worker, ok := m.workers[key]; ok {
				worker.stop()
				if !worker.running.Load() {
					delete(m.workers, key)
				}
			}
		}
	}
}

func (m *manager) CleanupPods(desiredPods map[types.UID]sets.Empty) {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()

	for key, worker := range m.workers {
		if _, ok := desiredPods[key.podUID]; !ok {
			worker.stop()
			if !worker.running.Load() {
				delete(m.workers, key)
			}
		}
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
			if exists && !utilfeature.DefaultFeatureGate.Enabled(features.ContainerLifecycleProber) {
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
			if exists && !utilfeature.DefaultFeatureGate.Enabled(features.ContainerLifecycleProber) {
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
	return worker, ok
}

// removeWorkerIfCurrent is called by a worker after exiting. It deletes the worker from the
// map only if it is still the registered worker for its key, so that a replacement worker
// installed for a new container instance is not removed by the exiting goroutine of a prior one.
func (m *manager) removeWorkerIfCurrent(w *worker) {
	m.workerLock.Lock()
	defer m.workerLock.Unlock()
	key := probeKey{w.pod.UID, w.container.Name, w.probeType}
	if m.workers[key] == w {
		delete(m.workers, key)
	}
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
