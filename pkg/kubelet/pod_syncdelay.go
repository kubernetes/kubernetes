/*
Copyright 2014 The Kubernetes Authors.

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

package kubelet

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/workqueue"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	"k8s.io/kubernetes/pkg/kubelet/podworks"
	"k8s.io/kubernetes/pkg/kubelet/status"
	"k8s.io/kubernetes/pkg/util/observer"
	"k8s.io/utils/clock"
)

const (
	channelCapacity = 100
)

type podStatusRecord struct {
	startTime             metav1.Time
	activeDeadlineSeconds int64
}

type podSyncDelay struct {
	podManager             kubepod.Manager
	podStatusProvider      status.PodStatusProvider
	queue                  workqueue.RateLimitingInterface
	podsStatus             map[types.UID]podStatusRecord
	fixActiveDeadlineEvent observer.SubjectEvent
	eventChannel           chan types.UID
	clock                  clock.WithTicker
}

type podUID types.UID

func (u podUID) ForceFix() bool {
	return true
}

func newPodSyncDelay(podManager kubepod.Manager, podStatusProvider status.PodStatusProvider, clock clock.WithTicker) *podSyncDelay {
	syncDelay := &podSyncDelay{
		podManager:        podManager,
		podStatusProvider: podStatusProvider,
		clock:             clock,
		queue:             workqueue.NewRateLimitingQueueWithCustomClock(workqueue.DefaultControllerRateLimiter(), clock),
		podsStatus:        make(map[types.UID]podStatusRecord),
		eventChannel:      make(chan types.UID, channelCapacity),
	}

	syncDelay.fixActiveDeadlineEvent = observer.NewSubjectEvent(syncDelay.handleActiveDeadline, nil)
	kubepod.Observer.Attach(kubepod.POD_MANAGER_CHANGED, syncDelay.podManagerChangedNotify, syncDelay.handlePodManagerChangedNotify)
	podworks.Observer.Attach(podworks.POD_SYNC_DELAY, syncDelay.podSyncDelayNotify, syncDelay.handlePodSyncDelay)

	go syncDelay.run()
	return syncDelay
}

func (p *podSyncDelay) podManagerChangedNotify(se observer.SubjectEvent) {
	p.queue.Add(se)
}

func (p *podSyncDelay) podSyncDelayNotify(se observer.SubjectEvent) {
	e := se.Event.(podSyncDelayEvent)
	duration := e.duration
	e.duration = 0
	se.Event = e

	p.queue.AddAfter(se, duration)
}

func (p *podSyncDelay) handlePodSyncDelay(e interface{}) error {
	event := e.(podSyncDelayEvent)
	p.eventChannel <- event.poduid
	return nil
}

func (p *podSyncDelay) handlePodManagerChangedNotify(e interface{}) error {
	poduid := e.(types.UID)
	p.activeDeadline(poduid)
	return nil
}

func (p *podSyncDelay) activeDeadline(poduid types.UID) {
	pod, exists := p.podManager.GetPodByUID(poduid)
	if !exists {
		delete(p.podsStatus, poduid)
		return
	}

	if pod.Spec.ActiveDeadlineSeconds == nil {
		return
	}

	// get the latest status to determine if it was started
	podStatus, ok := p.podStatusProvider.GetPodStatus(pod.UID)
	if !ok {
		podStatus = pod.Status
	}
	// we have no start time so just return
	if podStatus.StartTime.IsZero() {
		return
	}

	status, okStatus := p.podsStatus[poduid]
	if okStatus {
		if !status.startTime.Equal(podStatus.StartTime) ||
			status.activeDeadlineSeconds != *pod.Spec.ActiveDeadlineSeconds {
			p.podsStatus[poduid] = podStatusRecord{
				startTime:             *podStatus.StartTime,
				activeDeadlineSeconds: *pod.Spec.ActiveDeadlineSeconds,
			}
		}
	} else {
		p.podsStatus[poduid] = podStatusRecord{
			startTime:             *podStatus.StartTime,
			activeDeadlineSeconds: *pod.Spec.ActiveDeadlineSeconds,
		}
	}

	start := podStatus.StartTime.Time
	duration := p.clock.Since(start)
	allowedDuration := time.Duration(*pod.Spec.ActiveDeadlineSeconds) * time.Second
	if duration >= allowedDuration {
		p.eventChannel <- poduid
		return
	} else {
		allowedDuration = allowedDuration - duration
	}
	p.fixActiveDeadlineEvent.Event = podUID(poduid)
	p.queue.AddAfter(p.fixActiveDeadlineEvent, allowedDuration)
}

func (p *podSyncDelay) handleActiveDeadline(e interface{}) error {
	poduid := types.UID(e.(podUID))
	p.activeDeadline(poduid)
	return nil
}

func (p *podSyncDelay) run() {
	for p.execute() {
	}
}

func (p *podSyncDelay) execute() bool {
	item, quit := p.queue.Get()
	if quit {
		return false
	}

	defer p.queue.Done(item)

	if se, ok := item.(observer.SubjectEvent); ok {
		se.Handle()
	}

	return true
}

func (p *podSyncDelay) watch() chan types.UID {
	return p.eventChannel
}
