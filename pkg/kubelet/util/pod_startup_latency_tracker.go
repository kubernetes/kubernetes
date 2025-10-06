/*
Copyright 2022 The Kubernetes Authors.

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

package util

import (
	"context"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/utils/clock"
)

// PodStartupLatencyTracker records key moments for startup latency calculation,
// e.g. image pulling or pod observed running on watch.
type PodStartupLatencyTracker interface {
	ObservedPodOnWatch(pod *v1.Pod, when time.Time)
	RecordImageStartedPulling(podUID types.UID)
	RecordImageFinishedPulling(podUID types.UID)
	RecordInitContainerStarted(podUID types.UID, startedAt time.Time)
	RecordInitContainerFinished(podUID types.UID, finishedAt time.Time)
	RecordStatusUpdated(pod *v1.Pod)
	DeletePodStartupState(podUID types.UID)
}

type basicPodStartupLatencyTracker struct {
	// protect against concurrent read and write on pods map
	lock sync.Mutex
	pods map[types.UID]*perPodState
	// Track pods that were excluded from SLI due to unschedulability
	// These pods should never be re-added even if they later become schedulable
	excludedPods map[types.UID]bool
	// metrics for the first network pod only
	firstNetworkPodSeen bool
	// For testability
	clock clock.Clock
}

type imagePullSession struct {
	start time.Time
	end   time.Time
}
type perPodState struct {
	// Session-based image pulling tracking for accurate overlap handling
	imagePullSessions       []imagePullSession
	imagePullSessionsStarts []time.Time // Track multiple concurrent pull starts
	// Init container tracking
	totalInitContainerRuntime time.Duration
	currentInitContainerStart time.Time
	// first time, when pod status changed into Running
	observedRunningTime time.Time
	// log, if pod latency was already Observed
	metricRecorded bool
}

// NewPodStartupLatencyTracker creates an instance of PodStartupLatencyTracker
func NewPodStartupLatencyTracker() PodStartupLatencyTracker {
	return &basicPodStartupLatencyTracker{
		pods:         map[types.UID]*perPodState{},
		excludedPods: map[types.UID]bool{},
		clock:        clock.RealClock{},
	}
}

func (p *basicPodStartupLatencyTracker) ObservedPodOnWatch(pod *v1.Pod, when time.Time) {
	p.lock.Lock()
	defer p.lock.Unlock()

	// if the pod is terminal, we do not have to track it anymore for startup
	if pod.Status.Phase == v1.PodFailed || pod.Status.Phase == v1.PodSucceeded {
		delete(p.pods, pod.UID)
		return
	}

	state := p.pods[pod.UID]
	if state == nil {
		// if pod was previously unschedulable, don't track it again
		if p.excludedPods[pod.UID] {
			return
		}

		// create a new record for pod
		if pod.Status.StartTime.IsZero() {
			if isPodUnschedulable(pod) {
				p.excludedPods[pod.UID] = true
				return
			}

			// if pod is schedulable then track it
			state = &perPodState{}
			p.pods[pod.UID] = state
		}
		return
	}

	// remove existing pods from tracking (this handles cases where scheduling state becomes known later)
	if isPodUnschedulable(pod) {
		delete(p.pods, pod.UID)
		p.excludedPods[pod.UID] = true
		return
	}

	if state.observedRunningTime.IsZero() {
		// skip, pod didn't start yet
		return
	}

	if state.metricRecorded {
		// skip, pod's latency already recorded
		return
	}

	if hasPodStartedSLO(pod) {
		// TODO: it needs to be replaced by a proper context in the future
		ctx := context.TODO()
		logger := klog.FromContext(ctx)
		podStartingDuration := when.Sub(pod.CreationTimestamp.Time)
		podStartSLOduration := podStartingDuration

		totalImagesPullingTime := calculateImagePullingTime(state.imagePullSessions)
		if totalImagesPullingTime > 0 {
			podStartSLOduration -= totalImagesPullingTime
		}

		if state.totalInitContainerRuntime > 0 {
			podStartSLOduration -= state.totalInitContainerRuntime
		}

		podIsStateful := isStatefulPod(pod)

		logger.Info("Observed pod startup duration",
			"pod", klog.KObj(pod),
			"podStartSLOduration", podStartSLOduration.Seconds(),
			"podStartE2EDuration", podStartingDuration,
			"totalImagesPullingTime", totalImagesPullingTime,
			"totalInitContainerRuntime", state.totalInitContainerRuntime,
			"isStatefulPod", podIsStateful,
			"podCreationTimestamp", pod.CreationTimestamp.Time,
			"imagePullSessionsCount", len(state.imagePullSessions),
			"imagePullSessionsStartsCount", len(state.imagePullSessionsStarts),
			"observedRunningTime", state.observedRunningTime,
			"watchObservedRunningTime", when)

		metrics.PodStartTotalDuration.WithLabelValues().Observe(podStartingDuration.Seconds())
		if !podIsStateful {
			metrics.PodStartSLIDuration.WithLabelValues().Observe(podStartSLOduration.Seconds())
			// if is the first Pod with network track the start values
			// these metrics will help to identify problems with the CNI plugin
			if !pod.Spec.HostNetwork && !p.firstNetworkPodSeen {
				metrics.FirstNetworkPodStartSLIDuration.Set(podStartSLOduration.Seconds())
				p.firstNetworkPodSeen = true
			}
		}
		state.metricRecorded = true
	}
}

// calculateImagePullingTime computes the total time spent pulling images,
// accounting for overlapping pull sessions properly
func calculateImagePullingTime(sessions []imagePullSession) time.Duration {
	if len(sessions) == 0 {
		return 0
	}

	var totalTime time.Duration
	var currentEnd time.Time

	for i, session := range sessions {
		if session.end.IsZero() {
			continue
		}

		if i == 0 || session.start.After(currentEnd) {
			// First session or no overlap with previous session
			totalTime += session.end.Sub(session.start)
			currentEnd = session.end
		} else if session.end.After(currentEnd) {
			// Partial overlap - add only the non-overlapping part
			totalTime += session.end.Sub(currentEnd)
			currentEnd = session.end
		}
		// If session.end <= currentEnd, it's completely overlapped
	}

	return totalTime
}

func (p *basicPodStartupLatencyTracker) RecordImageStartedPulling(podUID types.UID) {
	p.lock.Lock()
	defer p.lock.Unlock()

	state := p.pods[podUID]
	if state == nil {
		return
	}

	now := p.clock.Now()
	state.imagePullSessionsStarts = append(state.imagePullSessionsStarts, now)
}

func (p *basicPodStartupLatencyTracker) RecordImageFinishedPulling(podUID types.UID) {
	p.lock.Lock()
	defer p.lock.Unlock()

	state := p.pods[podUID]
	if state == nil {
		return
	}

	now := p.clock.Now()

	// Complete the oldest pull session if we have active starts
	if len(state.imagePullSessionsStarts) > 0 {
		// Take the first (oldest) start and create a session
		startTime := state.imagePullSessionsStarts[0]
		session := imagePullSession{
			start: startTime,
			end:   now,
		}
		state.imagePullSessions = append(state.imagePullSessions, session)
		state.imagePullSessionsStarts = state.imagePullSessionsStarts[1:]
	}
}

func (p *basicPodStartupLatencyTracker) RecordInitContainerStarted(podUID types.UID, startedAt time.Time) {
	p.lock.Lock()
	defer p.lock.Unlock()

	state := p.pods[podUID]
	if state == nil {
		return
	}

	state.currentInitContainerStart = startedAt
}

func (p *basicPodStartupLatencyTracker) RecordInitContainerFinished(podUID types.UID, finishedAt time.Time) {
	p.lock.Lock()
	defer p.lock.Unlock()

	state := p.pods[podUID]
	if state == nil {
		return
	}

	if !state.currentInitContainerStart.IsZero() {
		initDuration := finishedAt.Sub(state.currentInitContainerStart)
		if initDuration > 0 {
			state.totalInitContainerRuntime += initDuration
		}
		state.currentInitContainerStart = time.Time{}
	}
}

func (p *basicPodStartupLatencyTracker) RecordStatusUpdated(pod *v1.Pod) {
	p.lock.Lock()
	defer p.lock.Unlock()

	state := p.pods[pod.UID]
	if state == nil {
		return
	}

	if state.metricRecorded {
		// skip, pod latency already recorded
		return
	}

	if !state.observedRunningTime.IsZero() {
		// skip, pod already started
		return
	}

	// TODO: it needs to be replaced by a proper context in the future
	ctx := context.TODO()
	logger := klog.FromContext(ctx)
	if hasPodStartedSLO(pod) {
		logger.V(3).Info("Mark when the pod was running for the first time", "pod", klog.KObj(pod), "rv", pod.ResourceVersion)
		state.observedRunningTime = p.clock.Now()
	}
}

// hasPodStartedSLO, check if for given pod, each container has been started at least once
//
// This should reflect "Pod startup latency SLI" definition
// ref: https://github.com/kubernetes/community/blob/master/sig-scalability/slos/pod_startup_latency.md
func hasPodStartedSLO(pod *v1.Pod) bool {
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.State.Running == nil || cs.State.Running.StartedAt.IsZero() {
			return false
		}
	}

	return true
}

// isStatefulPod determines if a pod is stateful according to the SLI documentation:
// "A stateful pod is defined as a pod that mounts at least one volume with sources
// other than secrets, config maps, downward API and empty dir."
// This should reflect the "stateful pod" definition
// ref: https://github.com/kubernetes/community/blob/master/sig-scalability/slos/pod_startup_latency.md
func isStatefulPod(pod *v1.Pod) bool {
	for _, volume := range pod.Spec.Volumes {
		// Check if this volume is NOT a stateless/ephemeral type
		if volume.Secret == nil &&
			volume.ConfigMap == nil &&
			volume.DownwardAPI == nil &&
			volume.EmptyDir == nil {
			return true
		}
	}
	return false
}

// isPodUnschedulable determines if a pod should be excluded from SLI tracking
// according to the SLI definition: "By schedulable pod we mean a pod that has to be
// immediately (without actions from any other components) schedulable in the cluster
// without causing any preemption."
// Any pod with PodScheduled=False is not immediately schedulable and should be excluded.
func isPodUnschedulable(pod *v1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == v1.PodScheduled && condition.Status == v1.ConditionFalse {
			return true
		}
	}
	return false
}

func (p *basicPodStartupLatencyTracker) DeletePodStartupState(podUID types.UID) {
	p.lock.Lock()
	defer p.lock.Unlock()

	delete(p.pods, podUID)
	delete(p.excludedPods, podUID)
}
