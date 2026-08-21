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

// probeTarget is an immutable snapshot of the container instance a worker
// probes, taken when the container starts (or when a running container is
// adopted). Everything a worker needs to probe is in here, so a worker never
// has to ask anyone what it is probing, and can never end up probing a
// different container than the one it was created for.
type probeTarget struct {
	pod         *v1.Pod
	container   v1.Container
	containerID kubecontainer.ContainerID
	podIPs      []string
	// startedAt is when the container itself started, which is what
	// InitialDelaySeconds is measured from.
	startedAt time.Time
}

// podIP returns the address the probe handlers dial.
func (t probeTarget) podIP() string {
	if len(t.podIPs) == 0 {
		return ""
	}
	return t.podIPs[0]
}

type instanceBoundWorkerOptions struct {
	probeType      probeType
	target         probeTarget
	prober         *prober
	resultsManager results.Manager

	// adopted marks a worker created for a container that was already running,
	// rather than by the container-start hook. Its first probe is spread over
	// one probe period so that reconstructing probes for a whole node's worth
	// of containers after a kubelet restart does not produce a thundering herd.
	adopted bool

	// onExit is called once, from the worker's own goroutine, as it returns.
	onExit func(*instanceBoundWorker)
	// onStartupSucceeded is called by a startup worker, before it exits, when
	// its probe has succeeded. It is nil for other probe types.
	onStartupSucceeded func(*instanceBoundWorker)
}

// instanceBoundWorker periodically probes one container instance. Unlike
// worker, it is bound to a single container ID for its whole life: a container
// restart is not something a worker handles, it is a new worker. That is what
// lets it be this small -- there is no container ID to re-resolve, no state to
// hold while waiting for a replacement container, and no restart policy to
// reason about. Deciding whether a container comes back is the sync loop's job;
// the worker's only job is to probe, and to stop when told.
type instanceBoundWorker struct {
	probeTarget
	probeType probeType

	// spec is the probe configuration for probeType, snapshotted with the rest
	// of the target.
	spec *v1.Probe

	prober *prober
	// resultsManager is the cache for this worker's probe type.
	resultsManager results.Manager

	adopted            bool
	onExit             func(*instanceBoundWorker)
	onStartupSucceeded func(*instanceBoundWorker)

	// ctx is cancelled by stop(), which aborts an in-flight probe as well as
	// ending the loop.
	ctx    context.Context
	cancel context.CancelFunc

	// The following are owned by the worker's goroutine.
	//
	// lastResult is the previous probe result.
	lastResult results.Result
	// resultRun is how many times in a row the probe has returned lastResult.
	resultRun int

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

// newInstanceBoundWorker creates a worker bound to opts.target. ctx should be
// the probe manager's own context, not a pod sync context: a worker must keep
// probing while its pod terminates so that a failing readiness probe can mark
// the pod NotReady, and it is stopped explicitly instead.
func newInstanceBoundWorker(ctx context.Context, opts instanceBoundWorkerOptions) *instanceBoundWorker {
	w := &instanceBoundWorker{
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

	w.proberDurationSuccessfulMetricLabels = deepCopyPrometheusLabels(proberDurationLabels)
	w.proberDurationUnknownMetricLabels = deepCopyPrometheusLabels(proberDurationLabels)

	return w
}

// run probes the container until the worker is stopped or a probe result
// terminates it. It is expected to be called in its own goroutine.
func (w *instanceBoundWorker) run() {
	defer func() {
		w.cancel()
		if w.onExit != nil {
			w.onExit(w)
		}
	}()

	// The results this worker cached are deliberately left in place. They are
	// still what the container's status says, and are removed by whoever tears
	// the container's probes down.

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
func (w *instanceBoundWorker) stop() {
	w.cancel()
}

// firstProbeDelay is how long to wait before the first probe: the remainder of
// InitialDelaySeconds measured from when the container actually started, rather
// than from when the worker happened to be created. For an adopted worker, whose
// container generally started long ago, a random spread over one probe period is
// added instead of firing immediately alongside every other adopted worker.
func (w *instanceBoundWorker) firstProbeDelay() time.Duration {
	period := time.Duration(w.spec.PeriodSeconds) * time.Second
	delay := max(time.Until(w.startedAt.Add(time.Duration(w.spec.InitialDelaySeconds)*time.Second)), 0)
	if w.adopted {
		delay += time.Duration(rand.Float64() * float64(period))
	}
	return delay
}

// wait sleeps for d, reporting whether the worker should keep going.
func (w *instanceBoundWorker) wait(d time.Duration) bool {
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
func (w *instanceBoundWorker) doProbe() (keepGoing bool) {
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
		// Success or failure is below threshold - leave the probe state unchanged.
		return true
	}

	w.resultsManager.Set(w.containerID, result, w.pod)

	switch {
	case w.probeType == startup:
		// The container has either started or is doomed; either way this
		// worker has answered the only question it was created to answer. On
		// success the manager brings up the readiness and liveness workers for
		// this same container instance.
		if result == results.Success && w.onStartupSucceeded != nil {
			w.onStartupSucceeded(w)
		}
		return false

	case w.probeType == liveness && result == results.Failure:
		// The sync loop will kill the container. Stopping the worker immediately
		// avoids running exec or network probes against a terminating container,
		// which wastes resources and logs spurious failures. If restarted, the
		// container will receive new probe workers.
		klog.FromContext(ctx).V(3).Info("Liveness probe failed past its threshold, stopping probe worker",
			"pod", klog.KObj(w.pod), "podUID", w.pod.UID, "containerName", w.container.Name, "containerID", w.containerID.String())
		return false
	}

	return true
}

// deleteMetrics drops this worker's metric label sets. The labels identify a
// pod and container name, not a container instance, so they outlive any one
// worker: dropping them when a container restarts would reset the counters
// under a series that is about to be used again. Only pod-scoped teardown
// calls this.
func (w *instanceBoundWorker) deleteMetrics() {
	ProberResults.Delete(w.proberResultsSuccessfulMetricLabels)
	ProberResults.Delete(w.proberResultsFailedMetricLabels)
	ProberResults.Delete(w.proberResultsUnknownMetricLabels)
	ProberDuration.Delete(w.proberDurationSuccessfulMetricLabels)
	ProberDuration.Delete(w.proberDurationUnknownMetricLabels)
}
