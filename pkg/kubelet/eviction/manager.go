/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package eviction

import (
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	qosutil "k8s.io/kubernetes/pkg/kubelet/qos/util"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	// the reason reported back in status.
	reason = "Evicted"
	// the message associated with the reason.
	message = "The node was low on compute resources."
)

// resourceToRankFunc maps a resource to ranking function for that resource.
var resourceToRankFunc = map[api.ResourceName]rankFunc{
	api.ResourceMemory: rankMemoryPressure,
}

// signalToNodeCondition maps a signal to the node condition to report if threshold is met.
var signalToNodeCondition = map[Signal]api.NodeConditionType{
	SignalMemoryAvailable: api.NodeMemoryPressure,
}

// managerImpl implements NodeStabilityManager
type managerImpl struct {
	//  used to track time
	clock util.Clock
	// the function to invoke to kill a pod
	killPodFunc KillPodFunc
	// lock protects access to node conditions
	lock sync.RWMutex
	// node conditions are the set of conditions present
	nodeConditions sets.String
	// nodeRef is a reference to the node
	nodeRef *api.ObjectReference
	// used to record events about the node
	recorder record.EventRecorder
	// used to measure usage stats on system
	summaryProvider stats.SummaryProvider
	// thresholds are the configured thresholds
	thresholds []Threshold
	// thresholdObserved captures when a threshold was observed
	thresholdObserved map[Threshold]time.Time
}

// ensure it implements the required interface
var _ Manager = &managerImpl{}

// NewManager returns a configured Manager
func NewManager(
	podLifecycleTarget lifecycle.PodLifecycleTarget,
	summaryProvider stats.SummaryProvider,
	thresholds []Threshold,
	killPodFunc KillPodFunc,
	recorder record.EventRecorder,
	nodeRef *api.ObjectReference,
	clock util.Clock) (Manager, error) {
	manager := &managerImpl{
		clock:             clock,
		killPodFunc:       killPodFunc,
		thresholds:        thresholds,
		recorder:          recorder,
		summaryProvider:   summaryProvider,
		nodeRef:           nodeRef,
		thresholdObserved: map[Threshold]time.Time{},
	}
	podLifecycleTarget.AddPodAdmitHandler(manager)
	return manager, nil
}

// Admit rejects a pod if its not safe to admit for node stability.
func (m *managerImpl) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	m.lock.RLock()
	defer m.lock.RUnlock()
	if len(m.nodeConditions) == 0 {
		return lifecycle.PodAdmitResult{Admit: true}
	}
	notBestEffort := qosutil.BestEffort != qosutil.GetPodQos(attrs.Pod)
	if notBestEffort {
		return lifecycle.PodAdmitResult{Admit: true}
	}
	glog.Warningf("Failed to admit pod %v - %s", format.Pod(attrs.Pod), "node has conditions: %v", m.nodeConditions)
	// we reject all best effort pods until we are stable.
	return lifecycle.PodAdmitResult{
		Admit:   false,
		Reason:  reason,
		Message: message,
	}
}

// Start starts the control loop to observe and response to low compute resources.
func (m *managerImpl) Start(podFunc ActivePodsFunc) {
	// TODO: expose the housekeeping interval
	housekeepingInterval := time.Second * 10
	go wait.Until(func() { m.synchronize(podFunc) }, housekeepingInterval, wait.NeverStop)
}

// IsUnderMemoryPressure returns true if the node is under memory pressure.
func (m *managerImpl) IsUnderMemoryPressure() bool {
	m.lock.RLock()
	defer m.lock.RUnlock()
	return m.nodeConditions.Has(string(api.NodeMemoryPressure))
}

// synchronize is the main control loop that enforces eviction thresholds.
func (m *managerImpl) synchronize(podFunc ActivePodsFunc) {
	// if we have nothing to do, just return
	if len(m.thresholds) == 0 {
		return
	}

	// find current usage so we can see if we violate an eviction threshold.
	summary, err := m.summaryProvider.Get()
	if err != nil {
		glog.Errorf("eviction manager: unexpected err: %v", err)
		return
	}

	// build an evaluation context for current eviction signals
	observations := map[Signal]resource.Quantity{}
	observations[SignalMemoryAvailable] = *resource.NewQuantity(int64(*summary.Node.Memory.AvailableBytes), resource.BinarySI)

	// track the amount of resource required to be reclaimed.
	var thresholdToHandle *Threshold
	toReclaim := api.ResourceList{}
	for i := range m.thresholds {
		threshold := m.thresholds[i]
		observed, found := observations[threshold.Signal]
		if !found {
			glog.Warningf("eviction manager: no observation found for eviction signal %v", threshold.Signal)
			continue
		}

		// determine if we have met the specified threshold
		thresholdMet := false
		thresholdResult := threshold.Value.Cmp(observed)
		switch threshold.Operator {
		case OpLessThan:
			thresholdMet = thresholdResult > 0
		}

		// this threshold was not met, delete any previous observations for this threshold
		if !thresholdMet {
			delete(m.thresholdObserved, threshold)
			glog.V(2).Infof("eviction manager: eviction criteria not met for %v, observed: %v", formatThreshold(threshold), observed.String())
			continue
		}

		// record when this threshold was met to know how to handle grace period
		observedAt, found := m.thresholdObserved[threshold]
		if !found {
			observedAt = m.clock.Now()
			m.thresholdObserved[threshold] = observedAt
		}
		if threshold.GracePeriod > 0 {
			duration := m.clock.Since(observedAt)
			if duration < threshold.GracePeriod {
				glog.V(2).Infof("eviction manager: eviction criteria not met for %v, observed: %v, duration: %v", formatThreshold(threshold), observed.String(), duration)
				continue
			}
		}

		// determine the amount of resource to reclaim based on the operator.
		var x, y resource.Quantity
		switch threshold.Operator {
		case OpLessThan:
			x = *(threshold.Value.Copy())
			y = observed
		}
		if err := x.Sub(y); err != nil {
			glog.Errorf("eviction manager: unexpected error determining amount of resource to reclaim: %v", err)
			continue
		}

		glog.V(2).Infof("eviction manager: eviction criteria met for %v, observed: %v", formatThreshold(threshold), observed.String())

		computeResource := signalToResource[threshold.Signal]
		toReclaim[computeResource] = x

		// we need to act on this threshold now
		thresholdToHandle = &threshold
		break
	}

	// any threshold met independent of grace period will trigger the node condition
	observedNodeConditions := sets.NewString()
	for threshold := range m.thresholdObserved {
		nodeCondition := signalToNodeCondition[threshold.Signal]
		observedNodeConditions.Insert(string(nodeCondition))
	}

	// update internal state
	m.lock.Lock()
	m.nodeConditions = observedNodeConditions
	m.lock.Unlock()

	// if there is noting to handle, we assume all is good.
	if thresholdToHandle == nil {
		glog.Infof("eviction manager: no pod evictions are required")
		return
	}

	// find out the resource under pressure...
	resourceToReclaim, ok := signalToResource[thresholdToHandle.Signal]
	if !ok {
		glog.Errorf("eviction manager: no resource to reclaim associated with signal %s", thresholdToHandle.Signal)
		return
	}

	// record an event about the resources we are now attempting to reclaim via eviction
	m.recorder.Eventf(m.nodeRef, api.EventTypeWarning, "EvictionThresholdMet", "Attempting to reclaim %s of %s", format.ResourceList(toReclaim), resourceToReclaim)

	// rank the pods for eviction
	rank, ok := resourceToRankFunc[resourceToReclaim]
	if !ok {
		glog.Errorf("eviction manager: no ranking function for resource %s", resourceToReclaim)
		return
	}

	// the only candidates viable for eviction are those pods that had anything running.
	activePods := podFunc()
	if len(activePods) == 0 {
		glog.Errorf("eviction manager: eviction thresholds have been met, but no pods are active to evict")
		return
	}

	// rank the running pods for eviction for the specified resource
	rank(activePods, cachedStatsFunc(summary.Pods))

	glog.Infof("eviction manager: pods ranked for eviction: %s", format.Pods(activePods))

	// we kill at most a single pod during each eviction interval
	for i := range activePods {
		pod := activePods[i]
		status := api.PodStatus{
			Phase:   api.PodFailed,
			Message: message,
			Reason:  reason,
		}
		// record that we are evicting the pod
		m.recorder.Eventf(pod, api.EventTypeWarning, reason, message)
		// TODO this needs to be based on soft or hard eviction threshold being met, soft eviction will allow a configured value.
		gracePeriodOverride := int64(0)
		// this is a blocking call and should only return when the pod and its containers are killed.
		err := m.killPodFunc(pod, status, &gracePeriodOverride)
		if err != nil {
			glog.Infof("eviction manager: pod %s failed to evict %v", format.Pod(pod), err)
			continue
		}
		// success, so we return until the next housekeeping interval
		glog.Infof("eviction manager: pod %s evicted successfully", format.Pod(pod))
		return
	}
	glog.Infof("eviction manager: unable to evict any pods from the node")
}
