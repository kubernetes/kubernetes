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
	RecordStatusUpdated(pod *v1.Pod)
	DeletePodStartupState(podUID types.UID)
}

type basicPodStartupLatencyTracker struct {
	// protect against concurrent read and write on pods map
	lock sync.Mutex
	pods map[types.UID]*perPodState
	// For testability
	clock clock.Clock
}

type perPodState struct {
	firstStartedPulling time.Time
	lastFinishedPulling time.Time
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
		podStartingDuration := when.Sub(pod.CreationTimestamp.Time)
		imagePullingDuration := state.lastFinishedPulling.Sub(state.firstStartedPulling)
		podStartSLOduration := (podStartingDuration - imagePullingDuration).Seconds()

		klog.InfoS("Observed pod startup duration",
			"pod", klog.KObj(pod),
			"podStartSLOduration", podStartSLOduration,
			"podStartE2EDuration", podStartingDuration,
			"podCreationTimestamp", pod.CreationTimestamp.Time,
			"firstStartedPulling", state.firstStartedPulling,
			"lastFinishedPulling", state.lastFinishedPulling,
			"observedRunningTime", state.observedRunningTime,
			"watchObservedRunningTime", when)

		metrics.PodStartSLIDuration.WithLabelValues().Observe(podStartSLOduration)
		metrics.PodStartTotalDuration.WithLabelValues().Observe(podStartingDuration.Seconds())
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

	state.lastFinishedPulling = p.clock.Now() // Now is always grater than values from the past.
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

	if hasPodStartedSLO(pod) {
		klog.V(3).InfoS("Mark when the pod was running for the first time", "pod", klog.KObj(pod), "rv", pod.ResourceVersion)
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

func (p *basicPodStartupLatencyTracker) DeletePodStartupState(podUID types.UID) {
	p.lock.Lock()
	defer p.lock.Unlock()

	delete(p.pods, podUID)
}
