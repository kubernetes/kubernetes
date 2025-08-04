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
	// metrics for the first network pod only
	firstNetworkPodSeen bool
	// For testability
	clock clock.Clock
}

type perPodState struct {
	firstStartedPulling     time.Time
	lastFinishedPulling     time.Time
	firstInitContainerStart time.Time
	lastInitContainerFinish time.Time
	// first time, when pod status changed into Running
	observedRunningTime time.Time
	// log, if pod latency was already Observed
	metricRecorded bool
}

// NewPodStartupLatencyTracker creates an instance of PodStartupLatencyTracker
func NewPodStartupLatencyTracker() PodStartupLatencyTracker {
	return &basicPodStartupLatencyTracker{
		pods:  map[types.UID]*perPodState{},
		clock: clock.RealClock{},
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
		// create a new record for pod, only if it was not yet acknowledged by the Kubelet
		// this is required, as we want to log metric only for those pods, that where scheduled
		// after Kubelet started
		if pod.Status.StartTime.IsZero() {
			p.pods[pod.UID] = &perPodState{}
		}

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
		var podStartSLOduration time.Duration
		var excludedTimeStart time.Time
		var excludedTimeEnd time.Time

		// Add time from pod creation to first excluded activity starts (either image pulling or init containers starting)
		if !state.firstStartedPulling.IsZero() && !state.firstInitContainerStart.IsZero() {
			if state.firstStartedPulling.Before(state.firstInitContainerStart) {
				excludedTimeStart = state.firstStartedPulling
			} else {
				excludedTimeStart = state.firstInitContainerStart
			}
		} else if !state.firstStartedPulling.IsZero() {
			excludedTimeStart = state.firstStartedPulling
		} else if !state.firstInitContainerStart.IsZero() {
			excludedTimeStart = state.firstInitContainerStart
		}

		if !excludedTimeStart.IsZero() {
			preExcludedDuration := excludedTimeStart.Sub(pod.CreationTimestamp.Time)
			if preExcludedDuration > 0 {
				podStartSLOduration += preExcludedDuration
			}
		}

		// Add gap between image pulling end and init container start if there is any
		if !state.lastFinishedPulling.IsZero() && !state.firstInitContainerStart.IsZero() {
			// Only count gap if init container starts after image pulling ends (no overlap)
			if state.firstInitContainerStart.After(state.lastFinishedPulling) {
				gapDuration := state.firstInitContainerStart.Sub(state.lastFinishedPulling)
				if gapDuration > 0 {
					podStartSLOduration += gapDuration
				}
			}
		}

		// Add time from last dependency completion to containers running
		if state.lastFinishedPulling.After(state.lastInitContainerFinish) {
			excludedTimeEnd = state.lastFinishedPulling
		} else if !state.lastInitContainerFinish.IsZero() {
			excludedTimeEnd = state.lastInitContainerFinish
		} else if !state.lastFinishedPulling.IsZero() {
			excludedTimeEnd = state.lastFinishedPulling
		}

		if !excludedTimeEnd.IsZero() {
			postExcludedDuration := when.Sub(excludedTimeEnd)
			if postExcludedDuration > 0 {
				podStartSLOduration += postExcludedDuration
			}
		} else if excludedTimeStart.IsZero() {
			// No dependencies at all, count entire duration
			podStartSLOduration = podStartingDuration
		}

		isStatefulPod := isStatefulPod(pod)

		logger.Info("Observed pod startup duration",
			"pod", klog.KObj(pod),
			"podStartSLOduration", podStartSLOduration.Seconds(),
			"podStartE2EDuration", podStartingDuration,
			"isStatefulPod", isStatefulPod,
			"podCreationTimestamp", pod.CreationTimestamp.Time,
			"firstStartedPulling", state.firstStartedPulling,
			"lastFinishedPulling", state.lastFinishedPulling,
			"firstInitContainerStart", state.firstInitContainerStart,
			"lastInitContainerFinish", state.lastInitContainerFinish,
			"observedRunningTime", state.observedRunningTime,
			"watchObservedRunningTime", when)

		metrics.PodStartTotalDuration.WithLabelValues().Observe(podStartingDuration.Seconds())
		if !isStatefulPod {
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

func (p *basicPodStartupLatencyTracker) RecordImageStartedPulling(podUID types.UID) {
	p.lock.Lock()
	defer p.lock.Unlock()

	state := p.pods[podUID]
	if state == nil {
		return
	}

	if state.firstStartedPulling.IsZero() {
		state.firstStartedPulling = p.clock.Now()
	}
}

func (p *basicPodStartupLatencyTracker) RecordImageFinishedPulling(podUID types.UID) {
	p.lock.Lock()
	defer p.lock.Unlock()

	state := p.pods[podUID]
	if state == nil {
		return
	}

	if !state.firstStartedPulling.IsZero() {
		state.lastFinishedPulling = p.clock.Now() // Now is always grater than values from the past.
	}
}

func (p *basicPodStartupLatencyTracker) RecordInitContainerStarted(podUID types.UID, startedAt time.Time) {
	p.lock.Lock()
	defer p.lock.Unlock()

	state := p.pods[podUID]
	if state == nil {
		return
	}

	if state.firstInitContainerStart.IsZero() || startedAt.Before(state.firstInitContainerStart) {
		state.firstInitContainerStart = startedAt
	}
}

func (p *basicPodStartupLatencyTracker) RecordInitContainerFinished(podUID types.UID, finishedAt time.Time) {
	p.lock.Lock()
	defer p.lock.Unlock()

	state := p.pods[podUID]
	if state == nil {
		return
	}

	if finishedAt.After(state.lastInitContainerFinish) {
		state.lastInitContainerFinish = finishedAt
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
			volume.EmptyDir == nil &&
			volume.Projected == nil &&
			volume.GitRepo == nil &&
			volume.Image == nil &&
			volume.Ephemeral == nil &&
			(volume.CSI == nil || volume.CSI.VolumeAttributes["csi.storage.k8s.io/ephemeral"] != "true") {
			return true
		}
	}
	return false
}

func (p *basicPodStartupLatencyTracker) DeletePodStartupState(podUID types.UID) {
	p.lock.Lock()
	defer p.lock.Unlock()

	delete(p.pods, podUID)
}
