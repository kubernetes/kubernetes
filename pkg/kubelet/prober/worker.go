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
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/component-base/metrics"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
)

// worker handles the periodic probing of its assigned container. Each worker has a go-routine
// associated with it which runs the probe loop until the container permanently terminates, or the
// stop channel is closed. The worker uses the probe Manager's statusManager to get up-to-date
// container IDs.
type worker struct {
	// Channel for stopping the probe.
	stopCh chan struct{}

	// Channel for triggering the probe manually.
	manualTriggerCh chan struct{}

	// The pod containing this probe (read-only)
	pod *v1.Pod

	// The container to probe (read-only)
	container v1.Container

	// Describes the probe configuration (read-only)
	spec *v1.Probe

	// The type of the worker.
	probeType probeType

	// The probe value during the initial delay.
	initialValue results.Result

	// Where to store this workers results.
	resultsManager results.Manager
	probeManager   *manager

	// The last known container ID for this worker.
	containerID kubecontainer.ContainerID
	// The last probe result for this worker.
	lastResult results.Result
	// How many times in a row the probe has returned the same result.
	resultRun int

	// If set, skip probing.
	onHold bool

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

// Creates and starts a new probe worker.
func newWorker(
	m *manager,
	probeType probeType,
	pod *v1.Pod,
	container v1.Container) *worker {

	w := &worker{
		stopCh:          make(chan struct{}, 1), // Buffer so stop() can be non-blocking.
		manualTriggerCh: make(chan struct{}, 1), // Buffer so prober_manager can do non-blocking calls to doProbe.
		pod:             pod,
		container:       container,
		probeType:       probeType,
		probeManager:    m,
	}

	switch probeType {
	case readiness:
		w.spec = container.ReadinessProbe
		w.resultsManager = m.readinessManager
		w.initialValue = results.Failure
	case liveness:
		w.spec = container.LivenessProbe
		w.resultsManager = m.livenessManager
		w.initialValue = results.Success
	case startup:
		w.spec = container.StartupProbe
		w.resultsManager = m.startupManager
		w.initialValue = results.Unknown
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

// run periodically probes the container.
func (w *worker) run() {
	ctx := context.Background()
	probeTickerPeriod := time.Duration(w.spec.PeriodSeconds) * time.Second

	// If kubelet restarted the probes could be started in rapid succession.
	// Let the worker wait for a random portion of tickerPeriod before probing.
	// Do it only if the kubelet has started recently.
	if probeTickerPeriod > time.Since(w.probeManager.start) {
		time.Sleep(time.Duration(rand.Float64() * float64(probeTickerPeriod)))
	}

	probeTicker := time.NewTicker(probeTickerPeriod)

	defer func() {
		// Clean up.
		probeTicker.Stop()
		if !w.containerID.IsEmpty() {
			w.resultsManager.Remove(w.containerID)
		}

		w.probeManager.removeWorker(w.pod.UID, w.container.Name, w.probeType)
		ProberResults.Delete(w.proberResultsSuccessfulMetricLabels)
		ProberResults.Delete(w.proberResultsFailedMetricLabels)
		ProberResults.Delete(w.proberResultsUnknownMetricLabels)
		ProberDuration.Delete(w.proberDurationSuccessfulMetricLabels)
		ProberDuration.Delete(w.proberDurationUnknownMetricLabels)
	}()

probeLoop:
	for w.doProbe(ctx) {
		// Wait for next probe tick.
		select {
		case <-w.stopCh:
			break probeLoop
		case <-probeTicker.C:
			// continue
		case <-w.manualTriggerCh:
			// Updating the periodic timer to run the probe again at intervals of probeTickerPeriod
			// starting from the moment a manual run occurs.
			probeTicker.Reset(probeTickerPeriod)
			klog.V(4).Infof("Triggered Probe by manual run, probeType=%s, pod=%s/%s, podUID=%s, containerName=%s",
				w.probeType, w.pod.Namespace, w.pod.Name, w.pod.UID, w.container.Name)
			// continue
		}
	}
}

// stop stops the probe worker. The worker handles cleanup and removes itself from its manager.
// It is safe to call stop multiple times.
func (w *worker) stop() {
	select {
	case w.stopCh <- struct{}{}:
	default: // Non-blocking.
	}
}

// doProbe probes the container once and records the result.
// Returns whether the worker should continue.
func (w *worker) doProbe(ctx context.Context) (keepGoing bool) {
	defer func() { recover() }() // Actually eat panics (HandleCrash takes care of logging)
	defer runtime.HandleCrash(func(_ interface{}) { keepGoing = true })

	startTime := time.Now()
	status, ok := w.probeManager.statusManager.GetPodStatus(w.pod.UID)
	if !ok {
		klog.V(3).Infof("No status for pod %s/%s", w.pod.Namespace, w.pod.Name)
		return true
	}

	// Worker should terminate if pod is terminated.
	if status.Phase == v1.PodFailed || status.Phase == v1.PodSucceeded {
		klog.V(3).Infof("Pod is terminated, exiting probe worker, pod=%s/%s, phase=%s",
			w.pod.Namespace, w.pod.Name, status.Phase)
		if w.probeType == readiness && !w.containerID.IsEmpty() {
			w.resultsManager.Set(w.containerID, results.Failure, w.pod)
		}
		return false
	}

	// Check container status
	isRestartableInitContainer := podutil.IsRestartableInitContainer(&w.container)
	var c v1.ContainerStatus
	if c, ok = podutil.GetContainerStatus(status.InitContainerStatuses, w.container.Name); !ok || len(c.ContainerID) == 0 {
		if c, ok = podutil.GetContainerStatus(status.ContainerStatuses, w.container.Name); !ok || len(c.ContainerID) == 0 {
			klog.V(3).Infof("Probe target container not found, pod=%s/%s, containerName=%s",
				w.pod.Namespace, w.pod.Name, w.container.Name)
			return true // Wait for more information.
		}
	}

	if w.containerID.String() != c.ContainerID {
		if !w.containerID.IsEmpty() {
			w.resultsManager.Remove(w.containerID)
		}
		w.containerID = kubecontainer.ParseContainerID(c.ContainerID)
		w.resultsManager.Set(w.containerID, w.initialValue, w.pod)
		// We've got a new container; resume probing.
		w.onHold = false
	}

	if w.onHold {
		klog.V(4).Infof("Probe skipped, worker on hold, probeType=%s, pod=%s/%s, containerName=%s",
			w.probeType, w.pod.Namespace, w.pod.Name, w.container.Name)
		return true
	}

	// Handle pod deletion
	if w.pod.ObjectMeta.DeletionTimestamp != nil && (w.probeType == liveness || w.probeType == startup) {
		klog.V(3).Infof("Pod deletion requested, setting probe result to success, probeType=%s, pod=%s/%s, containerName=%s",
			w.probeType, w.pod.Namespace, w.pod.Name, w.container.Name)
		if w.probeType == startup {
			klog.Infof("Pod deletion requested before container has fully started, pod=%s/%s, containerName=%s",
				w.pod.Namespace, w.pod.Name, w.container.Name)
		}
		w.resultsManager.Set(w.containerID, results.Success, w.pod)
		return false
	}

	// Handle non-running or terminated containers
	if c.State.Running == nil {
		klog.V(3).Infof("Non-running container probed, pod=%s/%s, containerName=%s",
			w.pod.Namespace, w.pod.Name, w.container.Name)
		if c.State.Terminated != nil {
			if w.probeType == liveness || w.probeType == readiness {
				w.resultsManager.Set(w.containerID, results.Failure, w.pod)
				return false // Stop for liveness/readiness if terminated
			} else if w.probeType == startup {
				if isRestartableInitContainer {
					w.resultsManager.Set(w.containerID, results.Failure, w.pod)
					return true // Continue for sidecar containers with restartPolicy: Always
				}
				w.resultsManager.Set(w.containerID, results.Failure, w.pod)
				return c.State.Terminated == nil || w.pod.Spec.RestartPolicy != v1.RestartPolicyNever
			}
		} else {
			if w.probeType == liveness || w.probeType == readiness {
				w.resultsManager.Set(w.containerID, results.Failure, w.pod)
				return false // Stop for non-running, non-terminated liveness/readiness
			} else if w.probeType == startup {
				w.resultsManager.Set(w.containerID, results.Unknown, w.pod)
				return true // Continue for startup probes
			}
		}
	}

	// Probe disabled for InitialDelaySeconds.
	if c.State.Running != nil && int32(time.Since(c.State.Running.StartedAt.Time).Seconds()) < w.spec.InitialDelaySeconds {
		klog.V(4).Infof("Probe skipped, before initial delay, probeType=%s, pod=%s/%s, containerName=%s",
			w.probeType, w.pod.Namespace, w.pod.Name, w.container.Name)
		w.resultsManager.Set(w.containerID, w.initialValue, w.pod)
		return true
	}

	// Handle Started field for startup and other probes
	if c.Started != nil && *c.Started && w.probeType == startup {
		klog.V(4).Infof("Startup probe stopped, container started, pod=%s/%s, containerName=%s",
			w.pod.Namespace, w.pod.Name, w.container.Name)
		w.resultsManager.Set(w.containerID, results.Success, w.pod)
		w.onHold = true
		return false
	} else if (w.probeType == liveness || w.probeType == readiness) && (c.Started == nil || !*c.Started) {
		klog.V(4).Infof("Probe skipped, container not started, probeType=%s, pod=%s/%s, containerName=%s",
			w.probeType, w.pod.Namespace, w.pod.Name, w.container.Name)
		w.resultsManager.Set(w.containerID, w.initialValue, w.pod)
		return true
	}

	// Execute the probe
	result, err := w.probeManager.prober.probe(ctx, w.probeType, w.pod, status, w.container, w.containerID)
	klog.V(4).Infof("Debug probe, pod=%s/%s, containerName=%s, probeType=%s, result=%v, err=%v",
		w.pod.Namespace, w.pod.Name, w.container.Name, w.probeType, result, err)
	if err != nil {
		klog.Errorf("Probe execution failed, probeType=%s, pod=%s/%s, containerName=%s, err=%v",
			w.probeType, w.pod.Namespace, w.pod.Name, w.container.Name, err)
		return true
	}

	switch result {
	case results.Success:
		ProberResults.With(w.proberResultsSuccessfulMetricLabels).Inc()
		ProberDuration.With(w.proberDurationSuccessfulMetricLabels).Observe(time.Since(startTime).Seconds())
	case results.Failure:
		ProberResults.With(w.proberResultsFailedMetricLabels).Inc()
	default:
		ProberResults.With(w.proberResultsUnknownMetricLabels).Inc()
		ProberDuration.With(w.proberDurationUnknownMetricLabels).Observe(time.Since(startTime).Seconds())
	}

	if w.lastResult == result {
		w.resultRun++
	} else {
		w.lastResult = result
		w.resultRun = 1
	}

	if (result == results.Failure && w.resultRun < int(w.spec.FailureThreshold)) ||
		(result == results.Success && w.resultRun < int(w.spec.SuccessThreshold)) {
		return true
	}

	w.resultsManager.Set(w.containerID, result, w.pod)

	if w.probeType == liveness && result == results.Failure {
		w.onHold = true
		w.resultRun = 0
	} else if w.probeType == startup && (result == results.Success || result == results.Failure) {
		w.onHold = true
		w.resultRun = 0
		return false
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
