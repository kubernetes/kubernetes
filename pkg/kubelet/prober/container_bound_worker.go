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
	"math/rand"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/component-base/metrics"
	"k8s.io/klog/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
)

// probeTarget contains the immutable snapshot of a container instance needed for probing.
type probeTarget struct {
	pod         *v1.Pod
	container   v1.Container
	containerID kubecontainer.ContainerID
	podIPs      []string
	startedAt   time.Time
}

// podIP returns the address the probe handlers dial.
func (t probeTarget) podIP() string {
	if len(t.podIPs) == 0 {
		return ""
	}
	return t.podIPs[0]
}

type containerBoundWorkerOptions struct {
	probeType      probeType
	target         probeTarget
	prober         *prober
	resultsManager results.Manager

	// adopted indicates the container was already running (e.g. after kubelet restart).
	// Initial probe timing is jittered to prevent a thundering herd.
	adopted bool

	onExit             func(*containerBoundWorker)
	onStartupSucceeded func(*containerBoundWorker)
}

// containerBoundWorker periodically probes a single container instance for its lifetime.
// A restarted container is assigned a new worker instance.
type containerBoundWorker struct {
	probeTarget
	probeType probeType

	spec *v1.Probe

	prober         *prober
	resultsManager results.Manager

	adopted            bool
	onExit             func(*containerBoundWorker)
	onStartupSucceeded func(*containerBoundWorker)

	ctx    context.Context
	cancel context.CancelFunc

	lastResult results.Result
	resultRun  int

	proberResultsSuccessfulMetricLabels metrics.Labels
	proberResultsFailedMetricLabels     metrics.Labels
	proberResultsUnknownMetricLabels    metrics.Labels
	proberDurationMetricLabels          metrics.Labels
}

// newContainerBoundWorker creates a probe worker bound to target. ctx should outlive
// the pod sync context so probes continue running during graceful termination.
func newContainerBoundWorker(ctx context.Context, opts containerBoundWorkerOptions) *containerBoundWorker {
	w := &containerBoundWorker{
		probeTarget:        opts.target,
		probeType:          opts.probeType,
		prober:             opts.prober,
		resultsManager:     opts.resultsManager,
		adopted:            opts.adopted,
		onExit:             opts.onExit,
		onStartupSucceeded: opts.onStartupSucceeded,
	}
	w.ctx, w.cancel = context.WithCancel(ctx)

	switch opts.probeType {
	case readiness:
		w.spec = w.container.ReadinessProbe
	case liveness:
		w.spec = w.container.LivenessProbe
	case startup:
		w.spec = w.container.StartupProbe
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

	w.proberDurationMetricLabels = deepCopyPrometheusLabels(proberDurationLabels)

	return w
}

// run probes the container until the worker is stopped or a probe result
// terminates it. It is expected to be called in its own goroutine.
func (w *containerBoundWorker) run() {
	defer func() {
		w.cancel()
		if w.onExit != nil {
			w.onExit(w)
		}
		// Cached results are preserved upon worker exit until container teardown.
	}()

	if !w.wait(w.firstProbeDelay()) {
		return
	}

	probeTicker := time.NewTicker(time.Duration(w.spec.PeriodSeconds) * time.Second)
	defer probeTicker.Stop()

	for w.doProbe() {
		select {
		case <-w.ctx.Done():
			return
		case <-probeTicker.C:
		}
	}
}

// stop ends the worker and aborts any probe it is currently executing.
// It is safe to call stop multiple times.
func (w *containerBoundWorker) stop() {
	w.cancel()
}

// firstProbeDelay calculates the delay before the first probe run, accounting for
// container start time and adding jitter for adopted containers.
func (w *containerBoundWorker) firstProbeDelay() time.Duration {
	period := time.Duration(w.spec.PeriodSeconds) * time.Second
	delay := max(time.Until(w.startedAt.Add(time.Duration(w.spec.InitialDelaySeconds)*time.Second)), 0)
	if w.adopted {
		delay += time.Duration(rand.Float64() * float64(period))
	}
	return delay
}

// wait sleeps for d, reporting whether the worker should keep going.
func (w *containerBoundWorker) wait(d time.Duration) bool {
	if d <= 0 {
		return w.ctx.Err() == nil
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-w.ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

// doProbe probes the container once and records the result.
// Returns whether the worker should continue.
func (w *containerBoundWorker) doProbe() (keepGoing bool) {
	ctx := w.ctx
	defer func() { _ = recover() }() // Actually eat panics (HandleCrash takes care of logging)
	defer runtime.HandleCrashWithContext(ctx, func(ctx context.Context, _ interface{}) { keepGoing = true })

	startTime := time.Now()
	result, err := w.prober.probe(ctx, w.probeType, w.pod, w.podIP(), w.container, w.containerID)
	if err != nil {
		// Prober error, or a probe cancelled by stop(): throw away the result.
		return true
	}

	switch result {
	case results.Success:
		ProberResults.With(w.proberResultsSuccessfulMetricLabels).Inc()
		ProberDuration.With(w.proberDurationMetricLabels).Observe(time.Since(startTime).Seconds())
	case results.Failure:
		ProberResults.With(w.proberResultsFailedMetricLabels).Inc()
	default:
		ProberResults.With(w.proberResultsUnknownMetricLabels).Inc()
		ProberDuration.With(w.proberDurationMetricLabels).Observe(time.Since(startTime).Seconds())
	}

	if w.lastResult == result {
		w.resultRun++
	} else {
		w.lastResult = result
		w.resultRun = 1
	}

	if (result == results.Failure && w.resultRun < int(w.spec.FailureThreshold)) ||
		(result == results.Success && w.resultRun < int(w.spec.SuccessThreshold)) {
		// Success or failure is below threshold - leave the probe state unchanged.
		return true
	}

	w.resultsManager.Set(w.containerID, result, w.pod)

	switch {
	case w.probeType == startup:
		// Startup worker exits once completed; on success it activates readiness/liveness workers.
		if result == results.Success && w.onStartupSucceeded != nil {
			w.onStartupSucceeded(w)
		}
		return false

	case w.probeType == liveness && result == results.Failure:
		// Stop worker immediately on terminal failure to avoid redundant probing before container kill.
		klog.FromContext(ctx).V(3).Info("Liveness probe failed past its threshold, stopping probe worker",
			"pod", klog.KObj(w.pod), "podUID", w.pod.UID, "containerName", w.container.Name, "containerID", w.containerID.String())
		return false
	}

	return true
}
