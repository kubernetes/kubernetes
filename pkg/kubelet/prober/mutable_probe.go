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
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
)

// Mutable state belongs to one container instance. Disabled workers remain alive
// until teardown so even a handler that ignores cancellation cannot overlap itself.
type mutableProbeState struct {
	mu                   sync.Mutex
	target               probeTarget
	spec                 *v1.Probe
	epoch                uint64
	cancel               context.CancelFunc
	wake                 chan struct{}
	stopped, first, hold bool
	due, completed       time.Time
	last                 results.Result
	consecutive          int
}

func (w *containerBoundWorker) initMutable() {
	w.mutable = &mutableProbeState{
		target: w.probeTarget, spec: w.spec, wake: make(chan struct{}, 1),
		due: time.Now().Add(w.firstProbeDelay()), last: results.Unknown,
	}
}

func mutableProbePeriod(spec *v1.Probe) time.Duration {
	return time.Duration(max(spec.PeriodSeconds, 1)) * time.Second
}

func (w *containerBoundWorker) updateMutable(target probeTarget, startupComplete bool) {
	s := w.mutable
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped {
		return
	}
	old, next := s.spec, probeSpec(w.probeType, &target.container)
	s.target, s.spec = target, next
	if apiequality.Semantic.DeepEqual(old, next) {
		return
	}
	if old == nil || next == nil {
		s.epoch++
		if s.cancel != nil {
			s.cancel()
		}
		s.last, s.consecutive = results.Unknown, 0
		s.hold = startupComplete
		s.completed = time.Time{}
		s.due = time.Now()
		if next == nil {
			w.resultsManager.Remove(w.containerID)
		} else {
			value := initialValue(w.probeType)
			if startupComplete {
				value = results.Success
			}
			w.resultsManager.SetWithMetadata(w.containerID, value, target.pod, results.Metadata{ContainerName: target.container.Name})
		}
	} else {
		executionChanged := old.TimeoutSeconds != next.TimeoutSeconds || !apiequality.Semantic.DeepEqual(old.ProbeHandler, next.ProbeHandler)
		if executionChanged || old.SuccessThreshold != next.SuccessThreshold || old.FailureThreshold != next.FailureThreshold {
			s.last, s.consecutive = results.Unknown, 0
		}
		// Invalidate the attempt, preserving any published verdict until a new one commits.
		if executionChanged {
			s.epoch++
			if s.cancel != nil {
				s.cancel()
			}
			s.completed = time.Time{}
			s.due = time.Now()
		} else if old.PeriodSeconds != next.PeriodSeconds || (!s.first && old.InitialDelaySeconds != next.InitialDelaySeconds) {
			s.due = time.Now()
			if !s.completed.IsZero() {
				s.due = s.completed.Add(mutableProbePeriod(next))
			}
		}
	}
	if next != nil && !s.first {
		s.due = maxTime(s.due, w.startedAt.Add(time.Duration(next.InitialDelaySeconds)*time.Second))
	}
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

func maxTime(a, b time.Time) time.Time {
	if a.Before(b) {
		return b
	}
	return a
}

func (w *containerBoundWorker) stopMutable() {
	if s := w.mutable; s != nil {
		s.mu.Lock()
		s.stopped = true
		s.epoch++
		if s.cancel != nil {
			s.cancel()
		}
		s.mu.Unlock()
	}
	w.cancel()
}

func (w *containerBoundWorker) runMutable() {
	defer func() {
		w.stopMutable()
		if w.onExit != nil {
			w.onExit(w)
		}
	}()
	s := w.mutable
	for {
		s.mu.Lock()
		active, due := s.spec != nil && !s.hold && !s.stopped, s.due
		s.mu.Unlock()
		var timer *time.Timer
		var tick <-chan time.Time
		if active {
			timer = time.NewTimer(max(time.Until(due), 0))
			tick = timer.C
		}
		select {
		case <-w.ctx.Done():
			if timer != nil {
				timer.Stop()
			}
			return
		case <-s.wake:
			if timer != nil {
				timer.Stop()
			}
		case <-tick:
			w.probeMutable()
		}
	}
}

func (w *containerBoundWorker) probeMutable() {
	defer func() { _ = recover() }()
	defer runtime.HandleCrashWithContext(w.ctx)
	s := w.mutable
	s.mu.Lock()
	if s.stopped || s.hold || s.spec == nil || time.Now().Before(s.due) {
		s.mu.Unlock()
		return
	}
	target, epoch := s.target, s.epoch
	ctx, cancel := context.WithCancel(w.ctx)
	s.cancel, s.first = cancel, true
	s.due = time.Now().Add(mutableProbePeriod(s.spec))
	s.mu.Unlock()
	defer cancel()
	start := time.Now()
	result, err := w.prober.probe(ctx, w.probeType, target.pod, target.podIP(), target.container, w.containerID)
	startupSucceeded := w.commitMutable(ctx, epoch, target, result, err, start)
	// The manager acquires its registry lock here; never call it under the worker lock.
	if startupSucceeded && w.onStartupSucceeded != nil {
		w.onStartupSucceeded(w)
	}
}

// Serialize publication with configuration updates and container teardown.
func (w *containerBoundWorker) commitMutable(ctx context.Context, epoch uint64, target probeTarget, result results.Result, err error, start time.Time) bool {
	s := w.mutable
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cancel = nil
	if s.stopped || epoch != s.epoch || ctx.Err() != nil {
		return false
	}
	s.completed = time.Now()
	s.due = s.completed.Add(mutableProbePeriod(s.spec))
	if err != nil {
		return false
	}
	switch result {
	case results.Success:
		ProberResults.With(w.proberResultsSuccessfulMetricLabels).Inc()
		ProberDuration.With(w.proberDurationMetricLabels).Observe(time.Since(start).Seconds())
	case results.Failure:
		ProberResults.With(w.proberResultsFailedMetricLabels).Inc()
	default:
		ProberResults.With(w.proberResultsUnknownMetricLabels).Inc()
		ProberDuration.With(w.proberDurationMetricLabels).Observe(time.Since(start).Seconds())
	}
	if s.last == result {
		s.consecutive++
	} else {
		s.last, s.consecutive = result, 1
	}
	if (result == results.Failure && s.consecutive < int(s.spec.FailureThreshold)) || (result == results.Success && s.consecutive < int(s.spec.SuccessThreshold)) {
		return false
	}
	w.resultsManager.SetWithMetadata(w.containerID, result, target.pod, results.Metadata{ContainerName: target.container.Name})
	startupSucceeded := w.probeType == kubecontainer.StartupProbe && result == results.Success
	if w.probeType == kubecontainer.StartupProbe || isTerminalResult(w.probeType, result) {
		s.hold = true
	}
	return startupSucceeded
}
