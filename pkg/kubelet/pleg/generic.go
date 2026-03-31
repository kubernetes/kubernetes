/*
Copyright 2015 The Kubernetes Authors.

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

package pleg

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/utils/clock"
)

// GenericPLEG is an extremely simple generic PLEG that relies solely on
// periodic listing to discover container changes. It should be used
// as temporary replacement for container runtimes do not support a proper
// event generator yet.
//
// Note that GenericPLEG assumes that a container would not be created,
// terminated, and garbage collected within one relist period. If such an
// incident happens, GenericPLEG would miss all events regarding this
// container. In the case of relisting failure, the window may become longer.
// Note that this assumption is not unique -- many kubelet internal components
// rely on terminated containers as tombstones for bookkeeping purposes. The
// garbage collector is implemented to work with such situations. However, to
// guarantee that kubelet can handle missing container events, it is
// recommended to set the relist period short and have an auxiliary, longer
// periodic sync in kubelet as the safety net.
type GenericPLEG struct {
	// The container runtime.
	runtime kubecontainer.Runtime
	// The channel from which the subscriber listens events.
	eventChannel chan *PodLifecycleEvent
	// The internal cache for pod/container information.
	// Guarded by relistLock.
	podRecords podRecords
	// Time of the last relisting.
	relistTime atomic.Value
	// Cache for storing the runtime states required for syncing pods.
	cache kubecontainer.Cache
	// For testability.
	clock clock.Clock
	// Pods that failed to have their status retrieved during a relist. These pods will be
	// retried during the next relisting.
	podsToReinspect sync.Map // map: podUID -> empty
	// Stop the Generic PLEG by closing the channel.
	stopCh chan struct{}
	// Locks the relisting of the Generic PLEG
	relistLock sync.Mutex
	// Indicates if the Generic PLEG is running or not
	isRunning bool
	// Locks the start/stop operation of Generic PLEG
	runningMu sync.Mutex
	// Indicates relisting related parameters
	relistDuration *RelistDuration
	// logger is used for contextual logging
	logger klog.Logger
	// relistRequests is the queue of on-demand pod relist requests.
	relistRequests chan relistRequest
	// globalRelistTimer controls the periodic global relist.
	globalRelistTimer clock.Timer
}

// Empty placeholder value for podsToReinspect (shared pointer reduces allocations).
var empty = &struct{}{}

// plegContainerState has a one-to-one mapping to the
// kubecontainer.State except for the non-existent state. This state
// is introduced here to complete the state transition scenarios.
type plegContainerState string

const (
	plegContainerRunning     plegContainerState = "running"
	plegContainerExited      plegContainerState = "exited"
	plegContainerUnknown     plegContainerState = "unknown"
	plegContainerNonExistent plegContainerState = "non-existent"
)

func convertState(state kubecontainer.State) plegContainerState {
	switch state {
	case kubecontainer.ContainerStateCreated:
		// kubelet doesn't use the "created" state yet, hence convert it to "unknown".
		return plegContainerUnknown
	case kubecontainer.ContainerStateRunning:
		return plegContainerRunning
	case kubecontainer.ContainerStateExited:
		return plegContainerExited
	case kubecontainer.ContainerStateUnknown:
		return plegContainerUnknown
	default:
		panic(fmt.Sprintf("unrecognized container state: %v", state))
	}
}

// relistRequest is used to track on-demand single-pod relist requests.
type relistRequest struct {
	// podUID is the UID of the pod to relist.
	podUID types.UID
	// timestamp is the time the request was made. If the pod has already been relisted by the time
	// this request is executed, the request will be skipped.
	timestamp time.Time
}

type podRecord struct {
	old     *kubecontainer.Pod
	current *kubecontainer.Pod
}

type podRecords map[types.UID]*podRecord

// NewGenericPLEG instantiates a new GenericPLEG object and return it.
func NewGenericPLEG(logger klog.Logger, runtime kubecontainer.Runtime, eventChannel chan *PodLifecycleEvent,
	relistDuration *RelistDuration, cache kubecontainer.Cache,
	clock clock.Clock) PodLifecycleEventGenerator {
	if cache == nil {
		panic("cache cannot be nil")
	}
	return &GenericPLEG{
		logger:         logger,
		relistDuration: relistDuration,
		runtime:        runtime,
		eventChannel:   eventChannel,
		podRecords:     make(podRecords),
		cache:          cache,
		clock:          clock,
		relistRequests: make(chan relistRequest, 200),
	}
}

// Watch returns a channel from which the subscriber can receive PodLifecycleEvent
// events.
// TODO: support multiple subscribers.
func (g *GenericPLEG) Watch() chan *PodLifecycleEvent {
	return g.eventChannel
}

// Start spawns a goroutine to relist periodically.
func (g *GenericPLEG) Start() {
	g.runningMu.Lock()
	defer g.runningMu.Unlock()
	if g.isRunning {
		return
	}

	g.isRunning = true
	g.stopCh = make(chan struct{})
	// Set 0 to trigger the first relist immediately
	// The RelistPeriod is set after the first relist.
	g.globalRelistTimer = g.clock.NewTimer(0)

	go func() {
		for g.workerLoopIteration() {
		}
	}()
}

func (g *GenericPLEG) Stop() {
	g.runningMu.Lock()
	defer g.runningMu.Unlock()
	if g.isRunning {
		close(g.stopCh)
		g.isRunning = false
		g.globalRelistTimer.Stop()
	}
}

// workerLoopIteration is the primary work dispatcher for the PLEG. It uses non-blocking channel
// reads for a prioritized work selection:
// 1. stop: exit the workerLoop if stopCh has been closed.
// 2. global relist: the global relist timer takes priority over single-pod relists.
// 3. single pod relist: execute a request from the single-pod relist queue.
func (g *GenericPLEG) workerLoopIteration() bool {
	// Use non-blocking selects for prioritized selection.

	// First priority: stopCh
	select {
	case <-g.stopCh:
		return false
	default:
	}

	// Second priority: global Relist
	select {
	case <-g.globalRelistTimer.C():
		g.Relist()
		g.globalRelistTimer.Reset(g.relistDuration.RelistPeriod)
		return true
	default:
	}

	// Fallback: blocking select
	select {
	case <-g.stopCh:
		return false
	case <-g.globalRelistTimer.C():
		g.Relist()
		g.globalRelistTimer.Reset(g.relistDuration.RelistPeriod)
	case req := <-g.relistRequests:
		if req.timestamp.After(g.getRelistTime()) {
			// The request hasn't already been served by the global relist.
			g.relistPod(req.podUID)
		}
	}

	return true
}

// This must not be called while the GenericPLEG is running.
func (g *GenericPLEG) Update(relistDuration *RelistDuration) {
	g.relistDuration = relistDuration
}

// Healthy check if PLEG work properly.
// relistThreshold is the maximum interval between two relist.
func (g *GenericPLEG) Healthy() (bool, error) {
	relistTime := g.getRelistTime()
	if relistTime.IsZero() {
		return false, fmt.Errorf("pleg has yet to be successful")
	}
	// Expose as metric so you can alert on `time()-pleg_last_seen_seconds > nn`
	metrics.PLEGLastSeen.Set(float64(relistTime.Unix()))
	elapsed := g.clock.Since(relistTime)
	if elapsed > g.relistDuration.RelistThreshold {
		return false, fmt.Errorf("pleg was last seen active %v ago; threshold is %v", elapsed, g.relistDuration.RelistThreshold)
	}
	return true, nil
}

func generateEvents(logger klog.Logger, podID types.UID, cid string, oldState, newState plegContainerState) []*PodLifecycleEvent {
	if newState == oldState {
		return nil
	}

	logger.V(4).Info("GenericPLEG", "podUID", podID, "containerID", cid, "oldState", oldState, "newState", newState)
	switch newState {
	case plegContainerRunning:
		return []*PodLifecycleEvent{{ID: podID, Type: ContainerStarted, Data: cid}}
	case plegContainerExited:
		return []*PodLifecycleEvent{{ID: podID, Type: ContainerDied, Data: cid}}
	case plegContainerUnknown:
		return []*PodLifecycleEvent{{ID: podID, Type: ContainerChanged, Data: cid}}
	case plegContainerNonExistent:
		switch oldState {
		case plegContainerExited:
			// We already reported that the container died before.
			return []*PodLifecycleEvent{{ID: podID, Type: ContainerRemoved, Data: cid}}
		default:
			return []*PodLifecycleEvent{{ID: podID, Type: ContainerDied, Data: cid}, {ID: podID, Type: ContainerRemoved, Data: cid}}
		}
	default:
		panic(fmt.Sprintf("unrecognized container state: %v", newState))
	}
}

func (g *GenericPLEG) getRelistTime() time.Time {
	val := g.relistTime.Load()
	if val == nil {
		return time.Time{}
	}
	return val.(time.Time)
}

func (g *GenericPLEG) updateRelistTime(timestamp time.Time) {
	g.relistTime.Store(timestamp)
}

// Relist queries the container runtime for list of pods/containers, compare
// with the internal pods/containers, and generates events accordingly.
func (g *GenericPLEG) Relist() {
	g.relistLock.Lock()
	defer g.relistLock.Unlock()

	ctx := context.Background()

	g.logger.V(5).Info("GenericPLEG: Relisting")

	if lastRelistTime := g.getRelistTime(); !lastRelistTime.IsZero() {
		metrics.PLEGRelistInterval.Observe(metrics.SinceInSeconds(lastRelistTime))
	}

	timestamp := g.clock.Now()
	defer func() {
		metrics.PLEGRelistDuration.Observe(metrics.SinceInSeconds(timestamp))
	}()

	// Get all the pods.
	podList, err := g.runtime.GetPods(ctx, true)
	if err != nil {
		g.logger.Error(err, "GenericPLEG: Unable to retrieve pods")
		return
	}

	g.updateRelistTime(timestamp)

	pods := kubecontainer.Pods(podList)
	// update running pod and container count
	updateRunningPodAndContainerMetrics(pods)
	g.podRecords.setCurrent(pods)

	for pid := range g.podRecords {
		g.reconcilePodRecord(ctx, pid)
	}

	// Update the cache timestamp.  This needs to happen *after*
	// all pods have been properly updated in the cache.
	g.cache.UpdateTime(timestamp)
}

// reconcilePodRecord compares the old & current pod record and checks for event-worthy state
// changes. If any events are generated, or the pod needs reinspection, it fetches the updated
// status, updates the pod cache, and delivers the events. This function should only be called after
// the current pod record has been updated.
func (g *GenericPLEG) reconcilePodRecord(ctx context.Context, pid types.UID) {
	// Compare the old and the current pods, and generate events.
	oldPod := g.podRecords.getOld(pid)
	pod := g.podRecords.getCurrent(pid)
	// Get all containers in the old and the new pod.
	allContainers := getContainersFromPods(oldPod, pod)
	var events []*PodLifecycleEvent
	for _, container := range allContainers {
		containerEvents := computeEvents(g.logger, oldPod, pod, &container.ID)
		events = append(events, containerEvents...)
	}

	_, reinspect := g.podsToReinspect.LoadAndDelete(pid)

	if len(events) == 0 && !reinspect {
		// Nothing else needed for this pod.
		return
	}

	// updateCache() will inspect the pod and update the cache. If an
	// error occurs during the inspection, we want PLEG to retry again
	// in the next relist. To achieve this, we do not update the
	// associated podRecord of the pod, so that the change will be
	// detect again in the next relist.
	// TODO: If many pods changed during the same relist period,
	// inspecting the pod and getting the PodStatus to update the cache
	// serially may take a while. We should be aware of this and
	// parallelize if needed.
	status, updated, err := g.updateCache(ctx, pod, pid)
	if err != nil {
		// Rely on updateCache calling GetPodStatus to log the actual error.
		g.logger.V(4).Info("PLEG: Ignoring events for pod", "pod", klog.KRef(pod.Namespace, pod.Name), "err", err)

		// make sure we try to reinspect the pod during the next relisting
		g.podsToReinspect.Store(pid, empty)

		return
	} else if utilfeature.DefaultFeatureGate.Enabled(features.EventedPLEG) {
		if !updated {
			return
		}
	}

	if len(events) == 0 {
		// Make sure we always trigger a PodSync after a full reinspection.
		events = append(events, &PodLifecycleEvent{ID: pid, Type: PodSync})
	}

	// Update the internal storage and send out the events.
	g.podRecords.update(pid)

	// Map from containerId to exit code; used as a temporary cache for lookup
	containerExitCode := make(map[string]int)

	for i := range events {
		// Filter out events that are not reliable and no other components use yet.
		if events[i].Type == ContainerChanged {
			continue
		}
		select {
		case g.eventChannel <- events[i]:
		default:
			metrics.PLEGDiscardEvents.Inc()
			g.logger.Error(nil, "Event channel is full, discard this relist() cycle event")
		}
		// Log exit code of containers when they finished in a particular event
		if events[i].Type == ContainerDied {
			// Fill up containerExitCode map for ContainerDied event when first time appeared
			if len(containerExitCode) == 0 && pod != nil {
				if err == nil {
					for _, containerStatus := range status.ContainerStatuses {
						containerExitCode[containerStatus.ID.ID] = containerStatus.ExitCode
					}
				}
			}
			if containerID, ok := events[i].Data.(string); ok {
				if exitCode, ok := containerExitCode[containerID]; ok && pod != nil {
					g.logger.V(2).Info("Generic (PLEG): container finished", "podID", pod.ID, "containerID", containerID, "exitCode", exitCode)
				}
			}
		}
	}
}

func (g *GenericPLEG) relistPod(podUID types.UID) {
	g.relistLock.Lock()
	defer g.relistLock.Unlock()

	ctx := context.Background()

	g.logger.V(5).Info("GenericPLEG: Relisting Pod", "podUID", podUID)

	startTime := g.clock.Now()
	defer func() {
		metrics.PLEGPodRelistDuration.Observe(metrics.SinceInSeconds(startTime))
	}()
	pod, err := g.runtime.GetPod(ctx, podUID)
	if err != nil {
		// We don't need to do anything other than log the error here, since it will be retried by
		// the next global relist.
		g.logger.Error(err, "GenericPLEG: Unable to retrieve pod", "podUID", podUID)
		return
	}

	// Update current pod record
	if pr, ok := g.podRecords[pod.ID]; ok {
		pr.current = pod
	} else {
		g.podRecords[pod.ID] = &podRecord{current: pod}
	}

	g.reconcilePodRecord(ctx, podUID)

	// Since we're not setting the global timestamp, we need to mark this pod as observed in case it
	// wasn't modified.
	g.cache.SetObservedTime(podUID, pod.Timestamp)
}

func getContainersFromPods(pods ...*kubecontainer.Pod) []*kubecontainer.Container {
	cidSet := sets.New[string]()
	var containers []*kubecontainer.Container
	fillCidSet := func(cs []*kubecontainer.Container) {
		for _, c := range cs {
			cid := c.ID.ID
			if cidSet.Has(cid) {
				continue
			}
			cidSet.Insert(cid)
			containers = append(containers, c)
		}
	}

	for _, p := range pods {
		if p == nil {
			continue
		}
		fillCidSet(p.Containers)
		// Update sandboxes as containers
		// TODO: keep track of sandboxes explicitly.
		fillCidSet(p.Sandboxes)
	}
	return containers
}

func computeEvents(logger klog.Logger, oldPod, newPod *kubecontainer.Pod, cid *kubecontainer.ContainerID) []*PodLifecycleEvent {
	var pid types.UID
	if oldPod != nil {
		pid = oldPod.ID
	} else if newPod != nil {
		pid = newPod.ID
	}
	oldState := getContainerState(oldPod, cid)
	newState := getContainerState(newPod, cid)
	return generateEvents(logger, pid, cid.ID, oldState, newState)
}

// getPodIP preserves an older cached status' pod IP if the new status has no pod IPs
// and its sandboxes have exited
func (g *GenericPLEG) getPodIPs(pid types.UID, status *kubecontainer.PodStatus) []string {
	if len(status.IPs) != 0 {
		return status.IPs
	}

	oldStatus, err := g.cache.Get(pid)
	if err != nil || len(oldStatus.IPs) == 0 {
		return nil
	}

	for _, sandboxStatus := range status.SandboxStatuses {
		// If at least one sandbox is ready, then use this status update's pod IP
		if sandboxStatus.State == runtimeapi.PodSandboxState_SANDBOX_READY {
			return status.IPs
		}
	}

	// For pods with no ready containers or sandboxes (like exited pods)
	// use the old status' pod IP
	return oldStatus.IPs
}

// updateCache tries to update the pod status in the kubelet cache and returns true if the
// pod status was actually updated in the cache. It will return false if the pod status
// was ignored by the cache.
func (g *GenericPLEG) updateCache(ctx context.Context, pod *kubecontainer.Pod, pid types.UID) (*kubecontainer.PodStatus, bool, error) {
	if pod == nil {
		// The pod is missing in the current relist. This means that
		// the pod has no visible (active or inactive) containers.
		g.logger.V(4).Info("PLEG: Delete status for pod", "podUID", string(pid))
		g.cache.Delete(pid)
		return nil, true, nil
	}

	timestamp := pod.Timestamp
	status, err := g.runtime.GetPodStatus(ctx, pod)
	if err != nil {
		// nolint:logcheck // Not using the result of klog.V inside the
		// if branch is okay, we just use it to determine whether the
		// additional "podStatus" key and its value should be added.
		if klog.V(6).Enabled() {
			g.logger.Error(err, "PLEG: Write status", "pod", klog.KRef(pod.Namespace, pod.Name), "podStatus", status)
		} else {
			g.logger.Error(err, "PLEG: Write status", "pod", klog.KRef(pod.Namespace, pod.Name))
		}
	} else {
		if klogV := g.logger.V(6); klogV.Enabled() {
			g.logger.Info("PLEG: Write status", "pod", klog.KRef(pod.Namespace, pod.Name), "podStatus", status)
		} else {
			g.logger.V(4).Info("PLEG: Write status", "pod", klog.KRef(pod.Namespace, pod.Name))
		}
		// Preserve the pod IP across cache updates if the new IP is empty.
		// When a pod is torn down, kubelet may race with PLEG and retrieve
		// a pod status after network teardown, but the kubernetes API expects
		// the completed pod's IP to be available after the pod is dead.
		status.IPs = g.getPodIPs(pid, status)
	}

	// When we use Generic PLEG only, the PodStatus is saved in the cache without
	// any validation of the existing status against the current timestamp.
	// This works well when there is only Generic PLEG setting the PodStatus in the cache however,
	// if we have multiple entities, such as Evented PLEG, while trying to set the PodStatus in the
	// cache we may run into the racy timestamps given each of them were to calculate the timestamps
	// in their respective execution flow. While Generic PLEG calculates this timestamp and gets
	// the PodStatus, we can only calculate the corresponding timestamp in
	// Evented PLEG after the event has been received by the Kubelet.
	// For more details refer to:
	// https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/3386-kubelet-evented-pleg#timestamp-of-the-pod-status
	if utilfeature.DefaultFeatureGate.Enabled(features.EventedPLEG) && isEventedPLEGInUse() && status != nil {
		timestamp = status.TimeStamp
	}

	return status, g.cache.Set(pod.ID, status, err, timestamp), err
}

func (g *GenericPLEG) RequestReinspect(podUID types.UID) {
	g.podsToReinspect.Store(podUID, empty)
}

func (g *GenericPLEG) RequestRelist(podUID types.UID) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.PLEGOnDemandRelist) {
		return
	}

	select {
	case g.relistRequests <- relistRequest{podUID, time.Now()}:
	default:
		g.logger.Error(nil, "Relist request channel full; dropping relist request", "podUID", podUID)
	}
}

func getContainerState(pod *kubecontainer.Pod, cid *kubecontainer.ContainerID) plegContainerState {
	// Default to the non-existent state.
	state := plegContainerNonExistent
	if pod == nil {
		return state
	}
	c := pod.FindContainerByID(*cid)
	if c != nil {
		return convertState(c.State)
	}
	// Search through sandboxes too.
	c = pod.FindSandboxByID(*cid)
	if c != nil {
		return convertState(c.State)
	}

	return state
}

func updateRunningPodAndContainerMetrics(pods []*kubecontainer.Pod) {
	runningSandboxNum := 0
	// intermediate map to store the count of each "container_state"
	containerStateCount := make(map[string]int)

	for _, pod := range pods {
		containers := pod.Containers
		for _, container := range containers {
			// update the corresponding "container_state" in map to set value for the gaugeVec metrics
			containerStateCount[string(container.State)]++
		}

		sandboxes := pod.Sandboxes

		for _, sandbox := range sandboxes {
			if sandbox.State == kubecontainer.ContainerStateRunning {
				runningSandboxNum++
				// every pod should only have one running sandbox
				break
			}
		}
	}
	for key, value := range containerStateCount {
		metrics.RunningContainerCount.WithLabelValues(key).Set(float64(value))
	}

	// Set the number of running pods in the parameter
	metrics.RunningPodCount.Set(float64(runningSandboxNum))
}

func (pr podRecords) getOld(id types.UID) *kubecontainer.Pod {
	r, ok := pr[id]
	if !ok {
		return nil
	}
	return r.old
}

func (pr podRecords) getCurrent(id types.UID) *kubecontainer.Pod {
	r, ok := pr[id]
	if !ok {
		return nil
	}
	return r.current
}

func (pr podRecords) setCurrent(pods []*kubecontainer.Pod) {
	for i := range pr {
		pr[i].current = nil
	}
	for _, pod := range pods {
		if r, ok := pr[pod.ID]; ok {
			r.current = pod
		} else {
			pr[pod.ID] = &podRecord{current: pod}
		}
	}
}

func (pr podRecords) update(id types.UID) {
	r, ok := pr[id]
	if !ok {
		return
	}
	pr.updateInternal(id, r)
}

func (pr podRecords) updateInternal(id types.UID, r *podRecord) {
	if r.current == nil {
		// Pod no longer exists; delete the entry.
		delete(pr, id)
		return
	}
	r.old = r.current
	r.current = nil
}
