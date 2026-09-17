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
	"math/rand"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/metrics"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
)

// worker handles the periodic probing of its assigned container. Each worker has a go-routine
// associated with it which runs the probe loop until the container permanently terminates, or the
// stop channel is closed. The worker uses the probe Manager's statusManager to get up-to-date
// container IDs.
type worker struct {
	probeState
	key     probeKey
	mutable bool

	// Channel for stopping the probe.
	stopCh chan struct{}

	// Channel for triggering the probe manually.
	manualTriggerCh chan struct{}

	// Channel for waking the scheduler after a configuration change.
	updateCh chan struct{}

	// The type of the worker.
	probeType probeType

	// The probe value during the initial delay.
	initialValue results.Result

	// Where to store this workers results.
	resultsManager results.Manager
	probeManager   *manager

	// proberResultsMetricLabels holds the labels attached to this worker
	// for the ProberResults metric by result.
	proberResultsSuccessfulMetricLabels metrics.Labels
	proberResultsFailedMetricLabels     metrics.Labels
	proberResultsUnknownMetricLabels    metrics.Labels
	// proberDurationMetricLabels holds the labels attached to this worker
	// for the ProberDuration metric by result.
	proberDurationSuccessfulMetricLabels metrics.Labels
	proberDurationUnknownMetricLabels    metrics.Labels
}

// probeState survives disabling a probe so re-enabling it cannot overlap a call that has not
// returned after cancellation. Its mutex serializes configuration changes and result commits;
// probe I/O and cancellation never run under this lock.
type probeState struct {
	mu        sync.Mutex
	pod       *v1.Pod
	container v1.Container
	spec      *v1.Probe
	enabled   bool
	stopped   bool
	// Probe identity survives config updates; executionEpoch only invalidates attempts.
	probeID             uint64
	executionEpoch      uint64
	attempt             *probeAttempt
	containerID         kubecontainer.ContainerID
	containerStartedAt  time.Time
	firstProbeStarted   bool
	startupComplete     bool
	needsInitialization bool
	lastResult          results.Result
	resultRun           int
	onHold              bool
	nextProbeTime       time.Time
	lastProbeCompletion time.Time
}

type probeAttempt struct {
	cancel      context.CancelFunc
	epoch       uint64
	containerID kubecontainer.ContainerID
}

type workerConfig struct {
	pod            *v1.Pod
	container      v1.Container
	spec           *v1.Probe
	executionEpoch uint64
}

func probeForType(container v1.Container, probeType probeType) *v1.Probe {
	switch probeType {
	case readiness:
		return container.ReadinessProbe
	case liveness:
		return container.LivenessProbe
	case startup:
		return container.StartupProbe
	default:
		return nil
	}
}

func containerStatusByName(status v1.PodStatus, name string) (v1.ContainerStatus, bool) {
	if c, ok := podutil.GetContainerStatus(status.ContainerStatuses, name); ok {
		return c, true
	}

	return podutil.GetContainerStatus(status.InitContainerStatuses, name)
}

func isInitContainer(pod *v1.Pod, containerName string) bool {
	for _, initContainer := range pod.Spec.InitContainers {
		if initContainer.Name == containerName {
			return true
		}
	}
	return false
}

// Creates and starts a new probe worker.
func newWorker(
	m *manager,
	probeType probeType,
	pod *v1.Pod,
	container v1.Container) *worker {
	if !m.mutable {
		return newLegacyWorker(m, probeType, pod, container)
	}

	w := &worker{
		mutable: true,
		probeState: probeState{
			pod: pod.DeepCopy(), container: *container.DeepCopy(),
			enabled: probeForType(container, probeType) != nil,
			probeID: m.nextProbeID.Add(1), executionEpoch: 1,
			lastResult: results.Unknown, needsInitialization: true,
			nextProbeTime: time.Now(),
		},
		key:             probeKey{pod.UID, container.Name, probeType},
		stopCh:          make(chan struct{}),
		manualTriggerCh: make(chan struct{}, 1),
		updateCh:        make(chan struct{}, 1),
		probeType:       probeType,
		probeManager:    m,
	}

	switch probeType {
	case readiness:
		w.spec = w.container.ReadinessProbe
		w.resultsManager = m.readinessManager
		w.initialValue = results.Failure
	case liveness:
		w.spec = w.container.LivenessProbe
		w.resultsManager = m.livenessManager
		w.initialValue = results.Success
	case startup:
		w.spec = w.container.StartupProbe
		w.resultsManager = m.startupManager
		w.initialValue = results.Unknown
	}

	if w.enabled {
		period := probePeriod(w.spec)
		if period > time.Since(m.start) {
			w.nextProbeTime = w.nextProbeTime.Add(time.Duration(rand.Float64() * float64(period)))
		}
	}

	basicMetricLabels := metrics.Labels{
		"probe_type": w.probeType.String(),
		"container":  w.container.Name,
		"pod":        w.pod.Name,
		"namespace":  w.pod.Namespace,
		"pod_uid":    string(w.pod.UID),
	}

	proberDurationLabels := metrics.Labels{
		"probe_type": w.probeType.String(),
		"container":  w.container.Name,
		"pod":        w.pod.Name,
		"namespace":  w.pod.Namespace,
	}

	w.proberResultsSuccessfulMetricLabels = deepCopyPrometheusLabels(basicMetricLabels)
	w.proberResultsSuccessfulMetricLabels["result"] = probeResultSuccessful

	w.proberResultsFailedMetricLabels = deepCopyPrometheusLabels(basicMetricLabels)
	w.proberResultsFailedMetricLabels["result"] = probeResultFailed

	w.proberResultsUnknownMetricLabels = deepCopyPrometheusLabels(basicMetricLabels)
	w.proberResultsUnknownMetricLabels["result"] = probeResultUnknown

	w.proberDurationSuccessfulMetricLabels = deepCopyPrometheusLabels(proberDurationLabels)
	w.proberDurationUnknownMetricLabels = deepCopyPrometheusLabels(proberDurationLabels)

	return w
}

func (w *worker) configSnapshot() workerConfig {
	w.mu.Lock()
	defer w.mu.Unlock()

	return w.configLocked()
}

func (w *worker) configLocked() workerConfig {
	return workerConfig{pod: w.pod, container: w.container, spec: w.spec, executionEpoch: w.executionEpoch}
}

func (w *worker) updateConfig(pod *v1.Pod, container v1.Container) {
	_, cancel := w.applyConfig(context.Background(), pod.DeepCopy(), *container.DeepCopy())
	if cancel != nil {
		cancel()
	}

	w.wake()
}

// applyConfig consumes immutable inputs. The caller invokes cancellation and wakes the worker
// after releasing the registry lock, so runtime callbacks cannot block another pod's reconcile.
func (w *worker) applyConfig(ctx context.Context, pod *v1.Pod, container v1.Container) (bool, context.CancelFunc) {
	if !w.mutable {
		return false, nil
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	if w.stopped {
		return false, nil
	}

	observed := w.observeStatusLocked(ctx)
	oldSpec, newSpec := w.spec, probeForType(container, w.probeType)
	wasEnabled := w.enabled

	w.pod, w.container, w.spec = pod, container, newSpec
	w.enabled = newSpec != nil

	if !w.enabled {
		if !wasEnabled {
			return false, nil
		}

		// Removing startup ungates this running container even if its status update is still queued.
		if w.probeType == startup && observed != nil && observed.status.State.Running != nil {
			w.observeContainerLocked(observed)
			w.startupComplete = true
		}

		w.executionEpoch++
		w.resetResultsLocked()
		w.onHold = false
		w.removeResultLocked()
		w.cleanupMetricsLocked()

		return true, w.attemptCancelLocked()
	}

	if !wasEnabled {
		w.probeID = w.probeManager.nextProbeID.Add(1)
		w.executionEpoch++
		w.resetResultsLocked()
		w.onHold = false
		w.needsInitialization = true
		w.lastProbeCompletion = time.Time{}
		w.nextProbeTime = time.Now()
		w.initializeProbeLocked(observed, true)

		return true, w.attemptCancelLocked()
	}

	var cancel context.CancelFunc
	if observed != nil && observed.id != w.containerID {
		w.observeContainerLocked(observed)
		w.initializeResultLocked(observed.status)
		cancel = w.attemptCancelLocked()
	}

	executionChanged := oldSpec.TimeoutSeconds != newSpec.TimeoutSeconds || !apiequality.Semantic.DeepEqual(oldSpec.ProbeHandler, newSpec.ProbeHandler)
	if executionChanged || oldSpec.SuccessThreshold != newSpec.SuccessThreshold || oldSpec.FailureThreshold != newSpec.FailureThreshold {
		w.resetResultsLocked()
	}

	if executionChanged {
		w.executionEpoch++

		// Subsequent schedule edits must not defer the first attempt with the new handler.
		w.lastProbeCompletion = time.Time{}
		w.nextProbeTime = time.Now()
		return false, w.attemptCancelLocked()
	}

	if oldSpec.PeriodSeconds != newSpec.PeriodSeconds || (!w.firstProbeStarted && oldSpec.InitialDelaySeconds != newSpec.InitialDelaySeconds) {
		w.nextProbeTime = w.nextRunLocked(time.Now())
	}

	return false, cancel
}

func (w *worker) resetResultsLocked() {
	w.lastResult = results.Unknown
	w.resultRun = 0
}

func (w *worker) attemptCancelLocked() context.CancelFunc {
	if w.attempt != nil {
		return w.attempt.cancel
	}

	return nil
}

func (w *worker) initializeProbeLocked(observed *containerObservation, added bool) {
	if observed == nil {
		return
	}

	w.observeContainerLocked(observed)
	if observed.status.State.Running != nil && w.probeType == startup &&
		(w.startupComplete || (observed.status.Started != nil && *observed.status.Started)) {
		w.startupComplete = true
		w.onHold = true
		w.needsInitialization = false
		w.publishLocked(results.Success)

		return
	}

	if added {
		// A newly added probe must initialize even for a container predating kubelet startup.
		w.publishLocked(w.initialValue)
		w.needsInitialization = false
	} else {
		w.initializeResultLocked(observed.status)
	}
}

// The first reconciliation restores probes after kubelet restart; later additions initialize them.
func (w *worker) initializeProbe(ctx context.Context, added bool) {
	w.mu.Lock()
	defer w.mu.Unlock()

	initialRun := w.nextProbeTime
	w.initializeProbeLocked(w.observeStatusLocked(ctx), added)

	// Discovering a container must preserve the staggering chosen for kubelet startup.
	w.nextProbeTime = initialRun
}

func (w *worker) holdForContainer(containerID kubecontainer.ContainerID) {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.containerID = containerID
	w.onHold = true
	w.startupComplete = w.probeType == startup
	w.needsInitialization = false
}

func (w *worker) isEnabled() bool {
	if !w.mutable {
		return true
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	return w.enabled && !w.stopped
}

func (w *worker) wake() {
	select {
	case w.updateCh <- struct{}{}:
	default:
	}
}

func probePeriod(spec *v1.Probe) time.Duration {
	period := time.Duration(spec.PeriodSeconds) * time.Second
	if period <= 0 {
		return time.Second
	}

	return period
}

func (w *worker) nextRunLocked(now time.Time) time.Time {
	if !w.firstProbeStarted && !w.containerStartedAt.IsZero() {
		due := w.containerStartedAt.Add(time.Duration(w.spec.InitialDelaySeconds) * time.Second)
		if now.Before(due) {
			return due
		}
	}

	if !w.lastProbeCompletion.IsZero() {
		return w.lastProbeCompletion.Add(probePeriod(w.spec))
	}

	return now
}

func (w *worker) run(ctx context.Context) {
	if !w.mutable {
		w.runLegacy(ctx)
		return
	}

	defer func() {
		w.stop()
		w.probeManager.removeWorker(w.key, w)
	}()

	for w.waitForNextProbe() {
		if !w.probe(ctx, true) {
			return
		}
	}
}

func (w *worker) waitForNextProbe() bool {
	for {
		w.mu.Lock()
		stopped, enabled, due := w.stopped, w.enabled, w.nextProbeTime
		w.mu.Unlock()

		if stopped {
			return false
		}

		var timer *time.Timer
		var tick <-chan time.Time
		if enabled {
			timer = time.NewTimer(max(time.Until(due), 0))
			tick = timer.C
		}

		select {
		case <-w.stopCh:
			if timer != nil {
				stopTimer(timer)
			}

			return false
		case <-tick:
			// Configuration may have postponed this deadline after the timer was armed.
			return true
		case <-w.manualTriggerCh:
			if timer != nil {
				stopTimer(timer)
			}

			w.mu.Lock()
			if w.enabled && !w.stopped {
				w.nextProbeTime = time.Now()
			}
			w.mu.Unlock()
		case <-w.updateCh:
			if timer != nil {
				stopTimer(timer)
			}
		}
	}
}

func stopTimer(timer *time.Timer) {
	if !timer.Stop() {
		select {
		case <-timer.C:
		default:
		}
	}
}

func (w *worker) stop() {
	if !w.mutable {
		w.stopLegacy()
		return
	}

	w.mu.Lock()
	if w.stopped {
		w.mu.Unlock()
		return
	}

	w.stopped = true
	w.enabled = false
	w.executionEpoch++
	cancel := w.attemptCancelLocked()
	w.removeResultLocked()
	w.cleanupMetricsLocked()
	close(w.stopCh)
	w.mu.Unlock()

	if cancel != nil {
		cancel()
	}
}

func (w *worker) cleanupMetricsLocked() {
	ProberResults.Delete(w.proberResultsSuccessfulMetricLabels)
	ProberResults.Delete(w.proberResultsFailedMetricLabels)
	ProberResults.Delete(w.proberResultsUnknownMetricLabels)
	ProberDuration.Delete(w.proberDurationSuccessfulMetricLabels)
	ProberDuration.Delete(w.proberDurationUnknownMetricLabels)
}

func (w *worker) removeResultLocked() {
	if !w.containerID.IsEmpty() {
		w.resultsManager.Remove(w.containerID)
	}
}

func (w *worker) publishLocked(result results.Result) {
	if !w.containerID.IsEmpty() {
		w.resultsManager.SetWithMetadata(w.containerID, result, w.pod, results.Metadata{
			ContainerName: w.key.containerName, ProbeID: w.probeID,
		})
	}
}

type containerObservation struct {
	id     kubecontainer.ContainerID
	status v1.ContainerStatus
}

func (w *worker) observeStatusLocked(ctx context.Context) *containerObservation {
	// This is a local cache read. Serialize it with state transitions so reconciliation cannot
	// restore an older container ID after the worker has already observed its replacement.
	status, found := w.probeManager.statusManager.GetPodStatus(w.key.podUID)
	c, ok := containerStatusByName(status, w.key.containerName)
	if !found || !ok || c.ContainerID == "" {
		return nil
	}

	return &containerObservation{id: kubecontainer.ParseContainerID(klog.FromContext(ctx), c.ContainerID), status: c}
}

func (w *worker) observeContainerLocked(observed *containerObservation) {
	if w.containerID != observed.id {
		w.removeResultLocked()
		w.containerID = observed.id
		w.executionEpoch++
		w.resetResultsLocked()
		w.onHold = false
		w.startupComplete = false
		w.firstProbeStarted = false
		w.needsInitialization = true
		w.lastProbeCompletion = time.Time{}
		w.nextProbeTime = time.Now()
	}

	w.containerStartedAt = time.Time{}
	if observed.status.State.Running != nil {
		w.containerStartedAt = observed.status.State.Running.StartedAt.Time
	}
}

func (w *worker) initializeResultLocked(c v1.ContainerStatus) {
	w.needsInitialization = false
	if w.probeType == startup && w.startupComplete {
		w.publishLocked(results.Success)
		w.onHold = true
		return
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.ChangeContainerStatusOnKubeletRestart) {
		w.publishLocked(w.initialValue)
		return
	}

	if isInitContainer(w.pod, w.container.Name) && podutil.IsRestartableInitContainer(&w.container) &&
		w.probeType == startup && c.Started != nil && *c.Started {
		w.publishLocked(results.Success)
	}

	if c.State.Running == nil || c.State.Running.StartedAt.IsZero() || !c.State.Running.StartedAt.Time.Before(kubeletRestartGracePeriod(w.probeManager.start)) {
		w.publishLocked(w.initialValue)
	}
}

func (w *worker) prepareProbe(ctx context.Context, cfg workerConfig, status v1.PodStatus, found, scheduled bool) (context.Context, *probeAttempt, workerConfig, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.stopped {
		return nil, nil, workerConfig{}, false
	}

	if !w.enabled || w.attempt != nil || w.executionEpoch != cfg.executionEpoch {
		return nil, nil, workerConfig{}, true
	}

	now := time.Now()
	if scheduled && now.Before(w.nextProbeTime) {
		return nil, nil, workerConfig{}, true
	}

	w.nextProbeTime = now.Add(probePeriod(w.spec))
	if !found {
		return nil, nil, workerConfig{}, true
	}

	if status.Phase == v1.PodFailed || status.Phase == v1.PodSucceeded {
		return nil, nil, workerConfig{}, false
	}

	c, ok := containerStatusByName(status, w.key.containerName)
	if !ok || c.ContainerID == "" {
		return nil, nil, workerConfig{}, true
	}

	w.observeContainerLocked(&containerObservation{id: kubecontainer.ParseContainerID(klog.FromContext(ctx), c.ContainerID), status: c})
	if w.needsInitialization {
		w.initializeResultLocked(c)
	}

	// A newly observed container may have reset nextProbeTime; skipped probes still poll periodically.
	w.nextProbeTime = now.Add(probePeriod(w.spec))
	if w.onHold {
		return nil, nil, workerConfig{}, true
	}

	isRestartableInitContainer := isInitContainer(w.pod, w.container.Name) && podutil.IsRestartableInitContainer(&w.container)
	if c.State.Running == nil {
		w.publishLocked(results.Failure)
		keepGoing := c.State.Terminated == nil || w.pod.Spec.RestartPolicy != v1.RestartPolicyNever || isRestartableInitContainer
		if utilfeature.DefaultFeatureGate.Enabled(features.ContainerRestartRules) {
			keepGoing = c.State.Terminated == nil
			if !keepGoing {
				keepGoing = podutil.ContainerShouldRestart(w.container, w.pod.Spec, c.State.Terminated.ExitCode) ||
					(utilfeature.DefaultFeatureGate.Enabled(features.RestartAllContainersOnContainerExits) && kubecontainer.ShouldAllContainersRestart(w.pod, nil, &status))
			}
		}

		return nil, nil, workerConfig{}, keepGoing
	}

	if w.pod.DeletionTimestamp != nil && w.probeType != readiness {
		w.publishLocked(results.Success)
		return nil, nil, workerConfig{}, false
	}

	if !w.firstProbeStarted {
		initialDelayEnd := c.State.Running.StartedAt.Add(time.Duration(w.spec.InitialDelaySeconds) * time.Second)
		if now.Before(initialDelayEnd) {
			w.nextProbeTime = initialDelayEnd
			return nil, nil, workerConfig{}, true
		}
	}

	if c.Started != nil && *c.Started {
		if w.probeType == startup {
			w.startupComplete = true
			return nil, nil, workerConfig{}, true
		}
	} else if w.probeType != startup {
		return nil, nil, workerConfig{}, true
	}

	attemptCtx, cancel := context.WithCancel(ctx)
	attempt := &probeAttempt{cancel: cancel, epoch: w.executionEpoch, containerID: w.containerID}
	w.attempt = attempt
	w.firstProbeStarted = true
	return attemptCtx, attempt, w.configLocked(), true
}

func (w *worker) finishAttempt(attempt *probeAttempt) {
	attempt.cancel()

	w.mu.Lock()
	defer w.mu.Unlock()

	if w.attempt == attempt {
		w.attempt = nil
	}
}

// doProbe is also used by unit tests to execute one iteration without waiting for the timer.
func (w *worker) doProbe(ctx context.Context) bool {
	if !w.mutable {
		return w.doProbeLegacy(ctx)
	}

	return w.probe(ctx, false)
}

func (w *worker) probe(ctx context.Context, scheduled bool) (keepGoing bool) {
	defer func() { recover() }()
	defer runtime.HandleCrashWithContext(ctx, func(context.Context, interface{}) { keepGoing = true })

	cfg := w.configSnapshot()
	status, found := w.probeManager.statusManager.GetPodStatus(w.key.podUID)
	attemptCtx, attempt, cfg, keepGoing := w.prepareProbe(ctx, cfg, status, found, scheduled)
	if attempt == nil {
		return keepGoing
	}

	defer w.finishAttempt(attempt)

	start := time.Now()
	result, err := w.probeManager.prober.probe(attemptCtx, w.probeType, cfg.pod, status, cfg.container, attempt.containerID)

	latest, found := w.probeManager.statusManager.GetPodStatus(w.key.podUID)
	c, hasContainer := containerStatusByName(latest, w.key.containerName)

	w.mu.Lock()
	defer w.mu.Unlock()

	if w.stopped {
		return false
	}

	if !w.enabled {
		return true
	}

	if attempt.epoch != w.executionEpoch || attempt.containerID != w.containerID || !found || !hasContainer ||
		c.ContainerID != attempt.containerID.String() || latest.Phase == v1.PodFailed || latest.Phase == v1.PodSucceeded {
		w.nextProbeTime = time.Now()
		return true
	}

	w.lastProbeCompletion = time.Now()
	w.nextProbeTime = w.lastProbeCompletion.Add(probePeriod(w.spec))
	if err != nil {
		return true
	}
	switch result {
	case results.Success:
		ProberResults.With(w.proberResultsSuccessfulMetricLabels).Inc()
		ProberDuration.With(w.proberDurationSuccessfulMetricLabels).Observe(time.Since(start).Seconds())
	case results.Failure:
		ProberResults.With(w.proberResultsFailedMetricLabels).Inc()
	default:
		ProberResults.With(w.proberResultsUnknownMetricLabels).Inc()
		ProberDuration.With(w.proberDurationUnknownMetricLabels).Observe(time.Since(start).Seconds())
	}

	if w.lastResult == result {
		w.resultRun++
	} else {
		w.lastResult, w.resultRun = result, 1
	}

	if (result == results.Failure && w.resultRun < int(w.spec.FailureThreshold)) ||
		(result == results.Success && w.resultRun < int(w.spec.SuccessThreshold)) {
		return true
	}

	w.publishLocked(result)
	if (w.probeType == liveness && result == results.Failure) || w.probeType == startup {
		// Preserve a completed startup or a pending restart until the container changes.
		w.onHold = true
		w.resultRun = 0
		if w.probeType == startup && result == results.Success {
			w.startupComplete = true
		}
	}

	return true
}

func deepCopyPrometheusLabels(m metrics.Labels) metrics.Labels {
	ret := make(metrics.Labels, len(m))
	for k, v := range m {
		ret[k] = v
	}
	return ret
}
