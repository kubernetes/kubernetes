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
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

type PodStartupLatencyTracker struct {
	// protect against concurrent read and write on pods map
	lock sync.Mutex
	pods map[types.UID]*perPodState
}

type perPodState struct {
	firstStartedPulling time.Time
	lastFinishedPulling time.Time

	firstResourceVersionWhenPodWasReportedAsRunning string
	recorded                                        bool
}

func NewPodStartupLatencyTracker() *PodStartupLatencyTracker {
	return &PodStartupLatencyTracker{
		pods: make(map[types.UID]*perPodState),
	}
}

// ObservedPodOnWatch to be called from somewhere where we look for pods.
func (p *PodStartupLatencyTracker) ObservedPodOnWatch(pod *v1.Pod, when time.Time) {
	p.lock.Lock()
	defer p.lock.Unlock()

	state := p.pods[pod.UID]
	if state == nil {
		p.pods[pod.UID] = &perPodState{recorded: false}
		return
	}
	if state.recorded {
		// Already recorded latency for this pod.
		return
	}
	if state.firstResourceVersionWhenPodWasReportedAsRunning == pod.ResourceVersion { // Should we cast to int64 and compare? Probably...
		imagePullingDuration := state.lastFinishedPulling.Sub(state.firstStartedPulling)
		podStartingDuration := when.Sub(pod.CreationTimestamp.Time)
		duration := (imagePullingDuration - podStartingDuration).Seconds()

		metrics.PodStartSLODuration.Observe(duration)
		state.recorded = true
	}
}

func (p *PodStartupLatencyTracker) RecordImageStartedPulling(podUID types.UID) {
	p.lock.Lock()
	defer p.lock.Unlock()

	state := p.pods[podUID]
	if state == nil {
		// warning: ObservedPodOnWatch should happened before and 'state per pod' created
		return
	}
	if state.firstStartedPulling.IsZero() {
		state.firstStartedPulling = time.Now()
	}
}

func (p *PodStartupLatencyTracker) RecordImageFinishedPulling(podUID types.UID) {
	p.lock.Lock()
	defer p.lock.Unlock()

	state := p.pods[podUID]
	if state == nil {
		// warning: ObservedPodOnWatch should happened before and 'state per pod' created
		return
	}
	state.lastFinishedPulling = time.Now() // Now is always grater than values from the past.
}

func (p *PodStartupLatencyTracker) RecordStatusUpdated(pod *v1.Pod) {
	p.lock.Lock()
	defer p.lock.Unlock()

	podUID := pod.UID
	state := p.pods[podUID]
	if state == nil {
		return
	}
	if state.firstResourceVersionWhenPodWasReportedAsRunning != "" {
		// Already started.
		return
	}
	if hasPodStartedSLO(pod) {
		// It's first time we see pod startup running. Let's record pod.ResourceVersion
		state.firstResourceVersionWhenPodWasReportedAsRunning = pod.ResourceVersion
	}
}

// hasPodStartedSLO would reflect 1:1 Pod startup latency SLI/SLO definition
// ref: https://github.com/kubernetes/community/blob/master/sig-scalability/slos/pod_startup_latency.md
func hasPodStartedSLO(pod *v1.Pod) bool {
	// if any container haven't started nor the , the pod has not started
	for _, status := range pod.Status.ContainerStatuses {
		if status.Started == nil || (!*status.Started && status.RestartCount == 0) {
			return false
		}
	}

	return true
}

func (p *PodStartupLatencyTracker) DeletePodStartupState(podUID types.UID) {
	p.lock.Lock()
	defer p.lock.Unlock()

	delete(p.pods, podUID)
}
