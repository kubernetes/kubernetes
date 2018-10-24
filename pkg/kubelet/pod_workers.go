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
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/kubelet/util/queue"
)

// OnCompleteFunc is a function that is invoked when an operation completes.
// If err is non-nil, the operation did not complete successfully.
type OnCompleteFunc func(err error)

// PodStatusFunc is a function that is invoked to generate a pod status.
type PodStatusFunc func(pod *v1.Pod, podStatus *kubecontainer.PodStatus) v1.PodStatus

// KillPodOptions are options when performing a pod update whose update type is kill.
type KillPodOptions struct {
	// PodStatusFunc is the function to invoke to set pod status in response to a kill request.
	PodStatusFunc PodStatusFunc
	// PodTerminationGracePeriodSecondsOverride is optional override to use if a pod is being killed as part of kill operation.
	PodTerminationGracePeriodSecondsOverride *int64
}

// UpdatePodOptions is an options struct to pass to a UpdatePod operation.
type UpdatePodOptions struct {
	// pod to update
	Pod *v1.Pod
	// the mirror pod for the pod to update, if it is a static pod
	MirrorPod *v1.Pod
	// the type of update (create, update, sync, kill)
	UpdateType kubetypes.SyncPodType
	// optional callback function when operation completes
	// this callback is not guaranteed to be completed since a pod worker may
	// drop update requests if it was fulfilling a previous request.  this is
	// only guaranteed to be invoked in response to a kill pod request which is
	// always delivered.
	OnCompleteFunc OnCompleteFunc
	// if update type is kill, use the specified options to kill the pod.
	KillPodOptions *KillPodOptions
}

// PodWorkers is an abstract interface for testability.
type PodWorkers interface {
	UpdatePod(options *UpdatePodOptions)
	ForgetNonExistingPodWorkers(desiredPods map[types.UID]empty)
	ForgetWorker(uid types.UID)
}

// syncPodOptions provides the arguments to a SyncPod operation.
type syncPodOptions struct {
	// the mirror pod for the pod to sync, if it is a static pod
	mirrorPod *v1.Pod
	// pod to sync
	pod *v1.Pod
	// the type of update (create, update, sync)
	updateType kubetypes.SyncPodType
	// the current status
	podStatus *kubecontainer.PodStatus
	// if update type is kill, use the specified options to kill the pod.
	killPodOptions *KillPodOptions
}

// the function to invoke to perform a sync.
type syncPodFnType func(options syncPodOptions) error

const (
	// jitter factor for resyncInterval
	workerResyncIntervalJitterFactor = 0.5

	// jitter factor for backOffPeriod and backOffOnTransientErrorPeriod
	workerBackOffPeriodJitterFactor = 0.5

	// backoff period when transient error occurred.
	backOffOnTransientErrorPeriod = time.Second
)

type podWorkers struct {
	// Protects all per worker fields.
	podLock sync.Mutex

	// Tracks all running per-pod goroutines - per-pod goroutine will be
	// processing updates received through its corresponding channel.
	podUpdates map[types.UID]chan UpdatePodOptions
	// Track the current state of per-pod goroutines.
	// Currently all update request for a given pod coming when another
	// update of this pod is being processed are ignored.
	isWorking map[types.UID]bool
	// Tracks the last undelivered work item for this pod - a work item is
	// undelivered if it comes in while the worker is working.
	lastUndeliveredWorkUpdate map[types.UID]UpdatePodOptions

	workQueue queue.WorkQueue

	// This function is run to sync the desired stated of pod.
	// NOTE: This function has to be thread-safe - it can be called for
	// different pods at the same time.
	syncPodFn syncPodFnType

	// The EventRecorder to use
	recorder record.EventRecorder

	// backOffPeriod is the duration to back off when there is a sync error.
	backOffPeriod time.Duration

	// resyncInterval is the duration to wait until the next sync.
	resyncInterval time.Duration

	// podCache stores kubecontainer.PodStatus for all pods.
	podCache kubecontainer.Cache
}

func newPodWorkers(syncPodFn syncPodFnType, recorder record.EventRecorder, workQueue queue.WorkQueue,
	resyncInterval, backOffPeriod time.Duration, podCache kubecontainer.Cache) *podWorkers {
	return &podWorkers{
		podUpdates:                map[types.UID]chan UpdatePodOptions{},
		isWorking:                 map[types.UID]bool{},
		lastUndeliveredWorkUpdate: map[types.UID]UpdatePodOptions{},
		syncPodFn:                 syncPodFn,
		recorder:                  recorder,
		workQueue:                 workQueue,
		resyncInterval:            resyncInterval,
		backOffPeriod:             backOffPeriod,
		podCache:                  podCache,
	}
}

func (p *podWorkers) managePodLoop(podUpdates <-chan UpdatePodOptions) {
	var lastSyncTime time.Time
	for update := range podUpdates {
		err := func() error {
			podUID := update.Pod.UID
			// This is a blocking call that would return only if the cache
			// has an entry for the pod that is newer than minRuntimeCache
			// Time. This ensures the worker doesn't start syncing until
			// after the cache is at least newer than the finished time of
			// the previous sync.
			status, err := p.podCache.GetNewerThan(podUID, lastSyncTime)
			if err != nil {
				// This is the legacy event thrown by manage pod loop
				// all other events are now dispatched from syncPodFn
				p.recorder.Eventf(update.Pod, v1.EventTypeWarning, events.FailedSync, "error determining status: %v", err)
				return err
			}
			err = p.syncPodFn(syncPodOptions{
				mirrorPod:      update.MirrorPod,
				pod:            update.Pod,
				podStatus:      status,
				killPodOptions: update.KillPodOptions,
				updateType:     update.UpdateType,
			})
			lastSyncTime = time.Now()
			return err
		}()
		// notify the call-back function if the operation succeeded or not
		if update.OnCompleteFunc != nil {
			update.OnCompleteFunc(err)
		}
		if err != nil {
			// IMPORTANT: we do not log errors here, the syncPodFn is responsible for logging errors
			glog.Errorf("Error syncing pod %s (%q), skipping: %v", update.Pod.UID, format.Pod(update.Pod), err)
		}
		p.wrapUp(update.Pod.UID, err)
	}
}

// Apply the new setting to the specified pod.
// If the options provide an OnCompleteFunc, the function is invoked if the update is accepted.
// Update requests are ignored if a kill pod request is pending.
func (p *podWorkers) UpdatePod(options *UpdatePodOptions) {
	pod := options.Pod
	uid := pod.UID
	var podUpdates chan UpdatePodOptions
	var exists bool

	p.podLock.Lock()
	defer p.podLock.Unlock()
	if podUpdates, exists = p.podUpdates[uid]; !exists {
		// We need to have a buffer here, because checkForUpdates() method that
		// puts an update into channel is called from the same goroutine where
		// the channel is consumed. However, it is guaranteed that in such case
		// the channel is empty, so buffer of size 1 is enough.
		podUpdates = make(chan UpdatePodOptions, 1)
		p.podUpdates[uid] = podUpdates

		// Creating a new pod worker either means this is a new pod, or that the
		// kubelet just restarted. In either case the kubelet is willing to believe
		// the status of the pod for the first pod worker sync. See corresponding
		// comment in syncPod.
		go func() {
			defer runtime.HandleCrash()
			p.managePodLoop(podUpdates)
		}()
	}
	if !p.isWorking[pod.UID] {
		p.isWorking[pod.UID] = true
		podUpdates <- *options
	} else {
		// if a request to kill a pod is pending, we do not let anything overwrite that request.
		update, found := p.lastUndeliveredWorkUpdate[pod.UID]
		if !found || update.UpdateType != kubetypes.SyncPodKill {
			p.lastUndeliveredWorkUpdate[pod.UID] = *options
		}
	}
}

func (p *podWorkers) removeWorker(uid types.UID) {
	if ch, ok := p.podUpdates[uid]; ok {
		close(ch)
		delete(p.podUpdates, uid)
		// If there is an undelivered work update for this pod we need to remove it
		// since per-pod goroutine won't be able to put it to the already closed
		// channel when it finishes processing the current work update.
		if _, cached := p.lastUndeliveredWorkUpdate[uid]; cached {
			delete(p.lastUndeliveredWorkUpdate, uid)
		}
	}
}
func (p *podWorkers) ForgetWorker(uid types.UID) {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	p.removeWorker(uid)
}

func (p *podWorkers) ForgetNonExistingPodWorkers(desiredPods map[types.UID]empty) {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	for key := range p.podUpdates {
		if _, exists := desiredPods[key]; !exists {
			p.removeWorker(key)
		}
	}
}

func (p *podWorkers) wrapUp(uid types.UID, syncErr error) {
	// Requeue the last update if the last sync returned error.
	switch {
	case syncErr == nil:
		// No error; requeue at the regular resync interval.
		p.workQueue.Enqueue(uid, wait.Jitter(p.resyncInterval, workerResyncIntervalJitterFactor))
	case strings.Contains(syncErr.Error(), NetworkNotReadyErrorMsg):
		// Network is not ready; back off for short period of time and retry as network might be ready soon.
		p.workQueue.Enqueue(uid, wait.Jitter(backOffOnTransientErrorPeriod, workerBackOffPeriodJitterFactor))
	default:
		// Error occurred during the sync; back off and then retry.
		p.workQueue.Enqueue(uid, wait.Jitter(p.backOffPeriod, workerBackOffPeriodJitterFactor))
	}
	p.checkForUpdates(uid)
}

func (p *podWorkers) checkForUpdates(uid types.UID) {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	if workUpdate, exists := p.lastUndeliveredWorkUpdate[uid]; exists {
		p.podUpdates[uid] <- workUpdate
		delete(p.lastUndeliveredWorkUpdate, uid)
	} else {
		p.isWorking[uid] = false
	}
}

// killPodNow returns a KillPodFunc that can be used to kill a pod.
// It is intended to be injected into other modules that need to kill a pod.
func killPodNow(podWorkers PodWorkers, recorder record.EventRecorder) eviction.KillPodFunc {
	return func(pod *v1.Pod, status v1.PodStatus, gracePeriodOverride *int64) error {
		// determine the grace period to use when killing the pod
		gracePeriod := int64(0)
		if gracePeriodOverride != nil {
			gracePeriod = *gracePeriodOverride
		} else if pod.Spec.TerminationGracePeriodSeconds != nil {
			gracePeriod = *pod.Spec.TerminationGracePeriodSeconds
		}

		// we timeout and return an error if we don't get a callback within a reasonable time.
		// the default timeout is relative to the grace period (we settle on 10s to wait for kubelet->runtime traffic to complete in sigkill)
		timeout := int64(gracePeriod + (gracePeriod / 2))
		minTimeout := int64(10)
		if timeout < minTimeout {
			timeout = minTimeout
		}
		timeoutDuration := time.Duration(timeout) * time.Second

		// open a channel we block against until we get a result
		type response struct {
			err error
		}
		ch := make(chan response, 1)
		podWorkers.UpdatePod(&UpdatePodOptions{
			Pod:        pod,
			UpdateType: kubetypes.SyncPodKill,
			OnCompleteFunc: func(err error) {
				ch <- response{err: err}
			},
			KillPodOptions: &KillPodOptions{
				PodStatusFunc: func(p *v1.Pod, podStatus *kubecontainer.PodStatus) v1.PodStatus {
					return status
				},
				PodTerminationGracePeriodSecondsOverride: gracePeriodOverride,
			},
		})

		// wait for either a response, or a timeout
		select {
		case r := <-ch:
			return r.err
		case <-time.After(timeoutDuration):
			recorder.Eventf(pod, v1.EventTypeWarning, events.ExceededGracePeriod, "Container runtime did not kill the pod within specified grace period.")
			return fmt.Errorf("timeout waiting to kill pod")
		}
	}
}
