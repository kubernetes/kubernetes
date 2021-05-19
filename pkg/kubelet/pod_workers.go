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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/queue"
)

// OnCompleteFunc is a function that is invoked when an operation completes.
// If err is non-nil, the operation did not complete successfully.
type OnCompleteFunc func(err error)

// PodStatusFunc is a function that is invoked to override the pod status when a pod is killed.
type PodStatusFunc func(podStatus *v1.PodStatus)

// KillPodOptions are options when performing a pod update whose update type is kill.
type KillPodOptions struct {
	// Evict is true if this is a pod triggered eviction - once a pod is evicted some resources are
	// more aggressively reaped than during normal pod operation (stopped containers).
	Evict bool
	// PodStatusFunc is invoked (if set) and overrides the status of the pod at the time the pod is killed.
	// The provided status is populated from the latest state.
	PodStatusFunc PodStatusFunc
	// PodTerminationGracePeriodSecondsOverride is optional override to use if a pod is being killed as part of kill operation.
	PodTerminationGracePeriodSecondsOverride *int64
}

// UpdatePodOptions is an options struct to pass to a UpdatePod operation.
type UpdatePodOptions struct {
	// Pod to update. Required.
	Pod *v1.Pod
	// MirrorPod is the mirror pod if Pod is a static pod. Optional when UpdateType
	// is kill or terminated.
	MirrorPod *v1.Pod
	// RunningPod is a runtime pod that is no longer present in config. Required
	// if Pod is nil, ignored if Pod is set.
	RunningPod *kubecontainer.Pod
	// The type of update (create, update, sync, kill).
	UpdateType kubetypes.SyncPodType
	// OnCompleteFunc is an optional callback function for when operation
	// completes. This method is always called, but another operation may
	// have occurred in the meantime so callers must check the latest status.
	OnCompleteFunc OnCompleteFunc
	// KillPodOptions is used to override the default termination behavior of the
	// pod or to update the pod status after an operation is completed. Since a
	// pod can be killed for multiple reasons, PodStatusFunc is invoked in order
	// and later kills have an opportunity to override the status (i.e. a preemption
	// may be later turned into an eviction).
	KillPodOptions *KillPodOptions
}

// PodWorkers is an abstract interface for testability.
type PodWorkers interface {
	// UpdatePod notifies the pod worker of a change to a pod, which will then
	// be processed in FIFO order by a goroutine per pod UID. The state of the
	// pod will be passed to the syncPod method until either the pod is marked
	// as deleted, it reaches a terminal phase (Succeeded/Failed), or the pod
	// is evicted by the kubelet. Once that occurs the syncTerminatingPod method
	// will be called until it exits successfully, and after that all further
	// UpdatePod() calls will be ignored for that pod until it has been forgotten
	// due to significant time passing. A pod that is terminated will never be
	// restarted.
	UpdatePod(options UpdatePodOptions)
	// SyncKnownPods removes workers for pods that are not in the desiredPods set
	// and have been terminated for a significant period of time. Once this method
	// has been called once, the workers are assumed to be fully initialized and
	// subsequent calls to ShouldPodContentBeRemoved on unknown pods will return
	// true.
	SyncKnownPods(desiredPods []*v1.Pod) map[types.UID]kubelettypes.SyncPodType
	// CouldHaveRunningContainers returns true before the pod workers have synced,
	// once the pod workers see the pod (syncPod could be called), and returns false
	// after the pod has been terminated (running containers guaranteed stopped).
	CouldHaveRunningContainers(uid types.UID) bool
	// IsPodTerminationRequested returns true when pod termination has been requested
	// until the termination completes and the pod is removed from config. This should
	// not be used in cleanup loops because it will return false if the pod has already
	// been cleaned up - use ShouldPodContainersBeTerminating instead. Also, this method
	// may return true while containers are still being initialized by the pod worker.
	IsPodTerminationRequested(uid types.UID) bool
	// ShouldPodContainersBeTerminating returns true before the pod workers have synced,
	// or once a pod has started terminating. This check is similar to
	// ShouldPodRuntimeBeRemoved but is also true after pod termination is requested.
	ShouldPodContainersBeTerminating(uid types.UID) bool
	// ShouldPodRuntimeBeRemoved returns true if runtime managers within the Kubelet
	// should aggressively cleanup pod resources that are not containers or on disk
	// content, like attached volumes. This is true when a pod is not yet observed
	// by a worker after the first sync (meaning it can't be running yet) or after
	// all running containers are stopped.
	// TODO: Once pod logs are separated from running containers, this method should
	// be used to gate whether containers are kept.
	ShouldPodRuntimeBeRemoved(uid types.UID) bool
	// ShouldPodContentBeRemoved returns true if resource managers within the Kubelet
	// should aggressively cleanup all content related to the pod. This is true
	// during pod eviction (when we wish to remove that content to free resources)
	// as well as after the request to delete a pod has resulted in containers being
	// stopped (which is a more graceful action). Note that a deleting pod can still
	// be evicted.
	ShouldPodContentBeRemoved(uid types.UID) bool
	// IsPodForMirrorPodTerminatingByFullName returns true if a static pod with the
	// provided pod name is currently terminating and has yet to complete. It is
	// intended to be used only during orphan mirror pod cleanup to prevent us from
	// deleting a terminating static pod from the apiserver before the pod is shut
	// down.
	IsPodForMirrorPodTerminatingByFullName(podFullname string) bool
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
}

// the function to invoke to perform a sync (reconcile the kubelet state to the desired shape of the pod)
type syncPodFnType func(options syncPodOptions) error

// the function to invoke to terminate a pod (ensure no running processes are present)
type syncTerminatingPodFnType func(pod *v1.Pod, podStatus *kubecontainer.PodStatus, runningPod *kubecontainer.Pod, gracePeriod *int64, podStatusFn func(*v1.PodStatus)) error

// the function to invoke to cleanup a pod that is terminated
type syncTerminatedPodFnType func(pod *v1.Pod, podStatus *kubecontainer.PodStatus) error

const (
	// jitter factor for resyncInterval
	workerResyncIntervalJitterFactor = 0.5

	// jitter factor for backOffPeriod and backOffOnTransientErrorPeriod
	workerBackOffPeriodJitterFactor = 0.5

	// backoff period when transient error occurred.
	backOffOnTransientErrorPeriod = time.Second
)

type podSyncStatus struct {
	syncedAt      time.Time
	terminatingAt time.Time
	deleted       bool
	terminating   bool
	gracePeriod   int64
	evicted       bool

	terminatedAt time.Time
	finished     bool
}

func (s podSyncStatus) IsTerminating() bool        { return !s.terminatingAt.IsZero() }
func (s podSyncStatus) IsStartedTerminating() bool { return s.terminating }
func (s podSyncStatus) IsTerminated() bool         { return !s.terminatedAt.IsZero() }
func (s podSyncStatus) IsFinished() bool           { return s.finished }
func (s podSyncStatus) IsEvicted() bool            { return s.evicted }
func (s podSyncStatus) IsDeleted() bool            { return s.deleted }

// podWorkers keeps track of operations on pods and ensures each pod is
// reconciled with the container runtime and other subsystems. The worker
// also tracks which pods are in flight for starting, which pods are
// shutting down but still have running containers, and which pods have
// terminated recently and are guaranteed to have no running containers.
//
// A pod passed to a pod worker is either being synced (expected to be
// running), terminating (has running containers but no new containers are
// expected to start), terminated (has no running containers but may still
// have resources being consumed), or cleaned up (no resources remaining).
// Once a pod is set to be "torn down" it cannot be started again for that
// UID (corresponding to a delete or eviction) until the kubelet is
// restarted or 15m have passed (if a UID is reused).
//
// Pod workers provide a consistent source of information to other kubelet
// loops about the status of the pod and whether containers can be
// running. The ShouldPodContentBeRemoved() method tracks whether a pod's
// contents should still exist, which includes non-existent pods after
// SyncKnownPods() has been called once (as per the contract, all existing
// pods should be provided via UpdatePod before SyncKnownPods is invoked).
//
// Other components in the Kubelet can request a termination of the pod
// via the UpdatePod method or the killPodNow wrapper - this will ensure
// the components of the pod are stopped until the kubelet is restarted
// or permanently (if the phase of the pod is set to a terminal phase
// in the pod status change).
type podWorkers struct {
	// Protects all per worker fields.
	podLock sync.Mutex
	// podsSynced is true once the pod worker has been synced at least once,
	// which means that all working pods have been started via UpdatePod().
	podsSynced bool
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
	// Tracks by UID the termination status of a pod - syncing, terminating,
	// terminated, and evicted.
	podSyncStatus map[types.UID]*podSyncStatus
	// Tracks when a static pod is being killed and is removed when the
	// static pod transitions to the killed state.
	terminatingStaticPodFullnames map[string]struct{}

	workQueue queue.WorkQueue

	// This function is run to sync the desired state of pod.
	// NOTE: This function has to be thread-safe - it can be called for
	// different pods at the same time.

	syncPodFn            syncPodFnType
	syncTerminatingPodFn syncTerminatingPodFnType
	syncTerminatedPodFn  syncTerminatedPodFnType

	// The EventRecorder to use
	recorder record.EventRecorder

	// backOffPeriod is the duration to back off when there is a sync error.
	backOffPeriod time.Duration

	// resyncInterval is the duration to wait until the next sync.
	resyncInterval time.Duration

	// podCache stores kubecontainer.PodStatus for all pods.
	podCache kubecontainer.Cache
}

func newPodWorkers(
	syncPodFn syncPodFnType,
	syncTerminatingPodFn syncTerminatingPodFnType,
	syncTerminatedPodFn syncTerminatedPodFnType,
	recorder record.EventRecorder,
	workQueue queue.WorkQueue,
	resyncInterval, backOffPeriod time.Duration,
	podCache kubecontainer.Cache,
) PodWorkers {
	return &podWorkers{
		podSyncStatus:                 map[types.UID]*podSyncStatus{},
		podUpdates:                    map[types.UID]chan UpdatePodOptions{},
		isWorking:                     map[types.UID]bool{},
		lastUndeliveredWorkUpdate:     map[types.UID]UpdatePodOptions{},
		terminatingStaticPodFullnames: map[string]struct{}{},
		syncPodFn:                     syncPodFn,
		syncTerminatingPodFn:          syncTerminatingPodFn,
		syncTerminatedPodFn:           syncTerminatedPodFn,
		recorder:                      recorder,
		workQueue:                     workQueue,
		resyncInterval:                resyncInterval,
		backOffPeriod:                 backOffPeriod,
		podCache:                      podCache,
	}
}

func (p *podWorkers) CouldHaveRunningContainers(uid types.UID) bool {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	if status, ok := p.podSyncStatus[uid]; ok {
		return !status.IsTerminated()
	}
	// once all pods are synced, any pod without sync status is known to not be running.
	return !p.podsSynced
}

func (p *podWorkers) IsPodTerminationRequested(uid types.UID) bool {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	if status, ok := p.podSyncStatus[uid]; ok {
		// the pod may still be setting up at this point.
		return status.IsTerminating()
	}
	// an unknown pod is considered not to be terminating (use ShouldPodContainersBeTerminating in
	// cleanup loops to avoid failing to cleanup pods that have already been removed from config)
	return false
}

func (p *podWorkers) ShouldPodContainersBeTerminating(uid types.UID) bool {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	if status, ok := p.podSyncStatus[uid]; ok {
		// we wait until the pod worker goroutine observes the termination, which means syncPod will not
		// be executed again, which means no new containers can be started
		return status.IsStartedTerminating()
	}
	// once we've synced, if the pod isn't known to the workers we should be tearing them
	// down
	return p.podsSynced
}

func (p *podWorkers) ShouldPodRuntimeBeRemoved(uid types.UID) bool {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	if status, ok := p.podSyncStatus[uid]; ok {
		return status.IsTerminated()
	}
	// a pod that hasn't been sent to the pod worker yet should have no runtime components once we have
	// synced all content.
	return p.podsSynced
}

func (p *podWorkers) ShouldPodContentBeRemoved(uid types.UID) bool {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	if status, ok := p.podSyncStatus[uid]; ok {
		return status.IsEvicted() || (status.IsDeleted() && status.IsTerminated())
	}
	// a pod that hasn't been sent to the pod worker yet should have no content on disk once we have
	// synced all content.
	return p.podsSynced
}

func (p *podWorkers) IsPodForMirrorPodTerminatingByFullName(podFullName string) bool {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	_, ok := p.terminatingStaticPodFullnames[podFullName]
	return ok
}

// UpdatePod carries a configuration change or termination state to a pod. A pod is either runnable,
// terminating, or terminated, and will transition to terminating if deleted on the apiserver, it is
// discovered to have a terminal phase (Succeeded or Failed), or if it is evicted by the kubelet.
func (p *podWorkers) UpdatePod(options UpdatePodOptions) {
	// only the pod worker can use the SyncPodTerminated update type
	if options.UpdateType == kubetypes.SyncPodTerminated {
		options.UpdateType = kubetypes.SyncPodKill
	}

	// handle when the pod is an orphan (no config) and we only have runtime status by running only
	// the terminating part of the lifecycle
	pod := options.Pod
	var isRuntimePod bool
	if options.RunningPod != nil {
		if options.Pod == nil {
			pod = options.RunningPod.ToAPIPod()
			if options.UpdateType != kubetypes.SyncPodKill {
				klog.InfoS("Pod update is ignored, runtime pods can only be killed", "pod", klog.KObj(pod), "podUID", pod.UID)
				return
			}
			options.Pod = pod
			isRuntimePod = true
		} else {
			options.RunningPod = nil
			klog.InfoS("Pod update included RunningPod which is only valid when Pod is not specified", "pod", klog.KObj(options.Pod), "podUID", options.Pod.UID)
		}
	}
	uid := pod.UID

	p.podLock.Lock()
	defer p.podLock.Unlock()

	var podUpdates chan UpdatePodOptions
	var exists bool

	// decide what to do with this pod - we are either setting it up, tearing it down, or ignoring it
	now := time.Now()
	status, ok := p.podSyncStatus[uid]
	if !ok {
		klog.V(4).InfoS("Pod is being synced for the first time", "pod", klog.KObj(pod), "podUID", pod.UID)
		status = &podSyncStatus{
			syncedAt: now,
		}
		p.podSyncStatus[uid] = status
	}

	// once a pod is terminated by UID, it cannot reenter the pod worker (until the UID is purged by housekeeping)
	if status.IsFinished() {
		klog.V(4).InfoS("Pod is finished processing, no further updates", "pod", klog.KObj(pod), "podUID", pod.UID)
		return
	}

	// check for a transition to terminating
	var becameTerminating bool
	if !status.IsTerminating() {
		switch {
		case isRuntimePod:
			klog.V(4).InfoS("Pod is orphaned and must be torn down", "pod", klog.KObj(pod), "podUID", pod.UID)
			status.terminatingAt = now
			becameTerminating = true
		case pod.Status.Phase == v1.PodFailed, pod.Status.Phase == v1.PodSucceeded:
			klog.V(4).InfoS("Pod is in a terminal phase (success/failed), begin teardown", "pod", klog.KObj(pod), "podUID", pod.UID)
			status.terminatingAt = now
			becameTerminating = true
		case pod.DeletionTimestamp != nil:
			klog.V(4).InfoS("Pod is marked for graceful deletion, begin teardown", "pod", klog.KObj(pod), "podUID", pod.UID)
			status.terminatingAt = now
			becameTerminating = true
		case options.UpdateType == kubetypes.SyncPodKill:
			if options.KillPodOptions != nil && options.KillPodOptions.Evict {
				klog.V(4).InfoS("Pod is being evicted by the kubelet, begin teardown", "pod", klog.KObj(pod), "podUID", pod.UID)
				status.evicted = true
			} else {
				klog.V(4).InfoS("Pod is being removed by the kubelet, begin teardown", "pod", klog.KObj(pod), "podUID", pod.UID)
			}
			status.terminatingAt = now
			becameTerminating = true
		}
	}
	if pod.DeletionTimestamp != nil || isRuntimePod {
		status.deleted = true
	}

	// once a pod is terminating, all updates are kills and the grace period can only decrease
	var wasGracePeriodShortened bool
	switch {
	case status.IsTerminated():
		options.UpdateType = kubetypes.SyncPodTerminated
		options.KillPodOptions = nil

	case status.IsTerminating():
		options.UpdateType = kubetypes.SyncPodKill
		if options.KillPodOptions == nil {
			options.KillPodOptions = &KillPodOptions{}
		}

		gracePeriod, gracePeriodShortened := calculateEffectiveGracePeriod(status, pod, options.KillPodOptions)

		wasGracePeriodShortened = gracePeriodShortened
		status.gracePeriod = gracePeriod
		// always set the grace period for syncTerminatingPod so we don't have to recalculate,
		// will never be zero.
		options.KillPodOptions.PodTerminationGracePeriodSecondsOverride = &gracePeriod

		// if a static pod comes through, start tracking it explicitly (cleared by the pod worker loop)
		if kubelettypes.IsStaticPod(pod) {
			p.terminatingStaticPodFullnames[kubecontainer.GetPodFullName(pod)] = struct{}{}
		}
	}

	// start the pod worker goroutine if it doesn't exist
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

	// dispatch a request to the pod worker if none are running
	if !p.isWorking[pod.UID] {
		p.isWorking[pod.UID] = true
		podUpdates <- options
		return
	}

	// never lose an OnCompleteFunc or a PodStatusFunc
	if undelivered, ok := p.lastUndeliveredWorkUpdate[pod.UID]; ok {
		if undelivered.OnCompleteFunc != nil {
			if latest := options.OnCompleteFunc; latest != nil {
				original := undelivered.OnCompleteFunc
				options.OnCompleteFunc = func(err error) {
					original(err)
					latest(err)
				}
			} else {
				options.OnCompleteFunc = undelivered.OnCompleteFunc
			}
		}
		if undelivered.KillPodOptions != nil && undelivered.KillPodOptions.PodStatusFunc != nil {
			// once a pod is killed, all subsequent UpdatePod methods must be "kill" and so we should
			// be able to carry the PodStatusFunc over
			if options.KillPodOptions != nil {
				if latest := options.KillPodOptions.PodStatusFunc; latest != nil {
					original := undelivered.KillPodOptions.PodStatusFunc
					options.KillPodOptions.PodStatusFunc = func(podStatus *v1.PodStatus) {
						original(podStatus)
						latest(podStatus)
					}
				} else {
					options.OnCompleteFunc = undelivered.OnCompleteFunc
				}
			}
		}
	}

	// decide whether to cancel the current running sync
	if becameTerminating || wasGracePeriodShortened {
		p.lastUndeliveredWorkUpdate[pod.UID] = options
		// TODO: queue the next update, then cancel the inflight job by invoking context cancellation
		return
	}

	// queue a request if we have decided not to cancel the existing request
	p.lastUndeliveredWorkUpdate[pod.UID] = options
}

// calculateEffectiveGracePeriod sets the initial grace period for a newly terminating pod or allows a
// shorter grace period to be provided, returning the desired value.
func calculateEffectiveGracePeriod(status *podSyncStatus, pod *v1.Pod, options *KillPodOptions) (int64, bool) {
	// enforce the restriction that a grace period can only decrease and track whatever our value is,
	// then ensure a calculated value is passed down to lower levels
	gracePeriod := status.gracePeriod
	// this value is bedrock truth - the apiserver owns telling us this value calculated by apiserver
	if override := pod.DeletionGracePeriodSeconds; override != nil {
		if gracePeriod == 0 || *override < gracePeriod {
			gracePeriod = *override
		}
	}
	// we allow other parts of the kubelet (namely eviction) to request this pod be terminated faster
	if options != nil {
		if override := options.PodTerminationGracePeriodSecondsOverride; override != nil {
			if gracePeriod == 0 || *override < gracePeriod {
				gracePeriod = *override
			}
		}
	}
	// make a best effort to default this value to the pod's desired intent, in the event
	// the kubelet provided no requested value (graceful termination?)
	if gracePeriod == 0 && pod.Spec.TerminationGracePeriodSeconds != nil {
		gracePeriod = *pod.Spec.TerminationGracePeriodSeconds
	}
	// no matter what, we always supply a grace period of 1
	if gracePeriod < 1 {
		gracePeriod = 1
	}
	return gracePeriod, status.gracePeriod != 0 && status.gracePeriod != gracePeriod
}

func (p *podWorkers) managePodLoop(podUpdates <-chan UpdatePodOptions) {
	var lastSyncTime time.Time
	var lastKillTime time.Time
	for update := range podUpdates {
		klog.V(4).InfoS("DEBUG: Processing pod event", "pod", klog.KObj(update.Pod), "podUID", update.Pod.UID, "updateType", update.UpdateType)
		err := func() error {
			// The worker is responsible for ensuring the sync method sees the appropriate
			// status updates on resyncs (the result of the last sync), transitions to
			// terminating (no wait), or on terminated (whatever the most recent state is).
			// Only syncing and terminating can generate pod status changes, while terminated
			// pods ensure the most recent status makes it to the api server.
			var status *kubecontainer.PodStatus
			var err error
			switch {
			case update.RunningPod != nil:
				// when we receive a running pod, we don't need status at all to terminate a pod
			case update.UpdateType == kubetypes.SyncPodKill:
				if lastKillTime.IsZero() {
					// if this is the first time we are killing the pod, we don't want to wait on the pod
					// cache at all as it may be out of date and we want to begin termination as quickly
					// as possible
					lastKillTime = time.Now()
					status = nil
					break
				}
				if lastSyncTime.IsZero() {
					// in the event we have never synced this pod before, we may have no container status
					status = nil
					break
				}
				// if this is a retry of termination or a delete shorten, we expect cache to eventually get
				// filled and can thus simply use the last syncTime
				status, err = p.podCache.GetNewerThan(update.Pod.UID, lastSyncTime)
			case update.UpdateType == kubetypes.SyncPodTerminated:
				// when a pod is terminated we don't expect to get updates, so simply use whatever status is
				// most recent
				status, err = p.podCache.Get(update.Pod.UID)
			default:
				// wait until we have observed status changes since the last
				// syncPod method completed, so we see the result of our updates
				status, err = p.podCache.GetNewerThan(update.Pod.UID, lastSyncTime)
			}
			if err != nil {
				// This is the legacy event thrown by manage pod loop all other events are now dispatched
				// from syncPodFn
				p.recorder.Eventf(update.Pod, v1.EventTypeWarning, events.FailedSync, "error determining status: %v", err)
				return err
			}

			// Take the appropriate action (illegal phases are prevented by UpdatePod)
			switch {
			case update.UpdateType == kubetypes.SyncPodTerminated:
				err = p.syncTerminatedPodFn(update.Pod, status)

			case update.UpdateType == kubetypes.SyncPodKill:
				p.acknowledgeTerminating(update.Pod)
				var gracePeriod *int64
				var podStatusFn func(*v1.PodStatus)
				if update.KillPodOptions != nil {
					gracePeriod = update.KillPodOptions.PodTerminationGracePeriodSecondsOverride
					podStatusFn = update.KillPodOptions.PodStatusFunc
				}
				err = p.syncTerminatingPodFn(update.Pod, status, update.RunningPod, gracePeriod, podStatusFn)

			default:
				err = p.syncPodFn(syncPodOptions{
					mirrorPod:  update.MirrorPod,
					pod:        update.Pod,
					podStatus:  status,
					updateType: update.UpdateType,
				})
			}

			lastSyncTime = time.Now()
			return err
		}()
		// notify the call-back function if the operation succeeded or not
		if update.OnCompleteFunc != nil {
			update.OnCompleteFunc(err)
		}
		if err != nil {
			// IMPORTANT: we do not log errors here, the syncPodFn is responsible for logging errors
			klog.ErrorS(err, "Error syncing pod, skipping", "pod", klog.KObj(update.Pod), "podUID", update.Pod.UID)
		}

		// if we've successfully terminated the pod, we can move to the next phase
		if err == nil {
			switch {
			case update.UpdateType == kubetypes.SyncPodTerminated:
				p.completeTerminated(update.Pod)
				klog.V(4).InfoS("DEBUG: Processing pod event done", "pod", klog.KObj(update.Pod), "podUID", update.Pod.UID, "updateType", update.UpdateType)
				return
			case update.UpdateType == kubetypes.SyncPodKill:
				if update.RunningPod != nil {
					p.completeTerminatingRuntimePod(update.Pod)
					klog.V(4).InfoS("DEBUG: Processing pod event done", "pod", klog.KObj(update.Pod), "podUID", update.Pod.UID, "updateType", update.UpdateType)
					return
				}
				p.completeTerminating(update.Pod)
			}
		}
		p.completeWork(update.Pod, err)
		klog.V(4).InfoS("DEBUG: Processing pod event done", "pod", klog.KObj(update.Pod), "podUID", update.Pod.UID, "updateType", update.UpdateType)
	}
}

// acknowledgeTerminating sets the terminating flag on the pod status once the pod worker sees
// the termination state so that other components know no new containers will be started in this
// pod.
func (p *podWorkers) acknowledgeTerminating(pod *v1.Pod) {
	p.podLock.Lock()
	defer p.podLock.Unlock()

	if status, ok := p.podSyncStatus[pod.UID]; ok && !status.terminatingAt.IsZero() && !status.terminating {
		klog.V(4).InfoS("Pod worker has observed request to terminate", "pod", klog.KObj(pod), "podUID", pod.UID)
		status.terminating = true
	}
}

// completeTerminating is invoked when syncTerminatingPod completes successfully, which means
// no container is running, no container will be started in the future, and we are ready for
// cleanup.  This updates the termination state which prevents future syncs and will ensure
// other kubelet loops know this pod is not running any containers.
func (p *podWorkers) completeTerminating(pod *v1.Pod) {
	p.podLock.Lock()
	defer p.podLock.Unlock()

	klog.V(4).InfoS("Pod terminated all containers successfully", "pod", klog.KObj(pod), "podUID", pod.UID)

	// if a static pod is being tracked, forget it
	delete(p.terminatingStaticPodFullnames, kubecontainer.GetPodFullName(pod))

	if status, ok := p.podSyncStatus[pod.UID]; ok {
		if status.terminatingAt.IsZero() {
			klog.V(4).InfoS("Pod worker was terminated but did not have terminatingAt set, likely programmer error", "pod", klog.KObj(pod), "podUID", pod.UID)
		}
		status.terminatedAt = time.Now()
	}

	p.lastUndeliveredWorkUpdate[pod.UID] = UpdatePodOptions{UpdateType: kubetypes.SyncPodTerminated, Pod: pod}
}

// completeTerminatingRuntimePod is invoked when syncTerminatingPod completes successfully,
// which means an orphaned pod (no config) is terminated and we can exit. Since orphaned
// pods have no API representation, we want to exit the loop at this point
// cleanup.  This updates the termination state which prevents future syncs and will ensure
// other kubelet loops know this pod is not running any containers.
func (p *podWorkers) completeTerminatingRuntimePod(pod *v1.Pod) {
	p.podLock.Lock()
	defer p.podLock.Unlock()

	klog.V(4).InfoS("Pod terminated all orphaned containers successfully and worker can now stop", "pod", klog.KObj(pod), "podUID", pod.UID)

	// if a static pod is being tracked, forget it
	delete(p.terminatingStaticPodFullnames, kubecontainer.GetPodFullName(pod))

	if status, ok := p.podSyncStatus[pod.UID]; ok {
		if status.terminatingAt.IsZero() {
			klog.V(4).InfoS("Pod worker was terminated but did not have terminatingAt set, likely programmer error", "pod", klog.KObj(pod), "podUID", pod.UID)
		}
		status.terminatedAt = time.Now()
		status.finished = true
	}

	ch, ok := p.podUpdates[pod.UID]
	if ok {
		close(ch)
	}
	delete(p.podUpdates, pod.UID)
	delete(p.lastUndeliveredWorkUpdate, pod.UID)
	delete(p.isWorking, pod.UID)
	delete(p.terminatingStaticPodFullnames, kubecontainer.GetPodFullName(pod))
}

// completeTerminated is invoked after syncTerminatedPod completes successfully and means we
// can stop the pod worker. The pod is finalized at this point.
func (p *podWorkers) completeTerminated(pod *v1.Pod) {
	p.podLock.Lock()
	defer p.podLock.Unlock()

	klog.V(4).InfoS("Pod is complete and the worker can now stop", "pod", klog.KObj(pod), "podUID", pod.UID)

	ch, ok := p.podUpdates[pod.UID]
	if ok {
		close(ch)
	}
	delete(p.podUpdates, pod.UID)
	delete(p.lastUndeliveredWorkUpdate, pod.UID)
	delete(p.isWorking, pod.UID)
	delete(p.terminatingStaticPodFullnames, kubecontainer.GetPodFullName(pod))

	if status, ok := p.podSyncStatus[pod.UID]; ok {
		if status.terminatingAt.IsZero() {
			klog.V(4).InfoS("Pod worker is complete but did not have terminatingAt set, likely programmer error", "pod", klog.KObj(pod), "podUID", pod.UID)
		}
		if status.terminatedAt.IsZero() {
			klog.V(4).InfoS("Pod worker is complete but did not have terminatedAt set, likely programmer error", "pod", klog.KObj(pod), "podUID", pod.UID)
		}
		status.finished = true
	}
}

// completeWork requeues on error or the next sync interval and then immediately executes any pending
// work.
func (p *podWorkers) completeWork(pod *v1.Pod, syncErr error) {
	// Requeue the last update if the last sync returned error.
	switch {
	case syncErr == nil:
		// No error; requeue at the regular resync interval.
		p.workQueue.Enqueue(pod.UID, wait.Jitter(p.resyncInterval, workerResyncIntervalJitterFactor))
	case strings.Contains(syncErr.Error(), NetworkNotReadyErrorMsg):
		// Network is not ready; back off for short period of time and retry as network might be ready soon.
		p.workQueue.Enqueue(pod.UID, wait.Jitter(backOffOnTransientErrorPeriod, workerBackOffPeriodJitterFactor))
	default:
		// Error occurred during the sync; back off and then retry.
		p.workQueue.Enqueue(pod.UID, wait.Jitter(p.backOffPeriod, workerBackOffPeriodJitterFactor))
	}
	p.completeWorkQueueNext(pod.UID)
}

// completeWorkQueueNext holds the lock and either queues the next work item for the worker or
// clears the working status.
func (p *podWorkers) completeWorkQueueNext(uid types.UID) {
	p.podLock.Lock()
	defer p.podLock.Unlock()
	if workUpdate, exists := p.lastUndeliveredWorkUpdate[uid]; exists {
		p.podUpdates[uid] <- workUpdate
		delete(p.lastUndeliveredWorkUpdate, uid)
	} else {
		p.isWorking[uid] = false
	}
}

// SyncKnownPods will purge any fully terminated pods that are not in the desiredPods
// list, which means SyncKnownPods must be called in a threadsafe manner from calls
// to UpdatePods for new pods. It returns a map of known workers that are not finished
// with a value of SyncPodTerminated, SyncPodKill, or SyncPodSync depending on whether
// the pod is terminated, terminating, or syncing.
func (p *podWorkers) SyncKnownPods(desiredPods []*v1.Pod) map[types.UID]kubelettypes.SyncPodType {
	workers := make(map[types.UID]kubelettypes.SyncPodType)
	known := make(map[types.UID]struct{})
	for _, pod := range desiredPods {
		known[pod.UID] = struct{}{}
	}

	p.podLock.Lock()
	defer p.podLock.Unlock()

	p.podsSynced = true
	now := time.Now()
	for uid := range p.podUpdates {
		if _, exists := known[uid]; !exists {
			p.removeWorker(uid, now)
		}

		if status, ok := p.podSyncStatus[uid]; ok {
			switch {
			case !status.terminatedAt.IsZero():
				workers[uid] = kubelettypes.SyncPodTerminated
			case !status.terminatingAt.IsZero():
				workers[uid] = kubelettypes.SyncPodKill
			default:
				workers[uid] = kubelettypes.SyncPodSync
			}
		} else {
			workers[uid] = kubelettypes.SyncPodSync
		}
	}
	return workers
}

func (p *podWorkers) removeWorker(uid types.UID, now time.Time) {
	status, ok := p.podSyncStatus[uid]
	if !ok {
		// already forgotten, or forgotten too early
		klog.V(4).InfoS("Pod worker has been requested for removal but is not a known pod", "podUID", uid)
		return
	}

	if !status.finished {
		klog.V(4).InfoS("Pod worker has been requested for removal but is still not fully terminated", "podUID", uid)
		return
	}

	klog.V(4).InfoS("Pod has been terminated and is no longer known to the kubelet, remove all history", "podUID", uid)
	delete(p.podSyncStatus, uid)
	delete(p.podUpdates, uid)
	delete(p.lastUndeliveredWorkUpdate, uid)
}

// killPodNow returns a KillPodFunc that can be used to kill a pod.
// It is intended to be injected into other modules that need to kill a pod.
func killPodNow(podWorkers PodWorkers, recorder record.EventRecorder) eviction.KillPodFunc {
	return func(pod *v1.Pod, isEvicted bool, gracePeriodOverride *int64, statusFn func(*v1.PodStatus)) error {
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
		podWorkers.UpdatePod(UpdatePodOptions{
			Pod:        pod,
			UpdateType: kubetypes.SyncPodKill,
			OnCompleteFunc: func(err error) {
				ch <- response{err: err}
			},
			KillPodOptions: &KillPodOptions{
				Evict:                                    isEvicted,
				PodStatusFunc:                            statusFn,
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
