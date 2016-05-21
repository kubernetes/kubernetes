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
	"sort"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	qosutil "k8s.io/kubernetes/pkg/kubelet/qos/util"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"
)

// managerImpl implements NodeStabilityManager
type managerImpl struct {
	//  used to track time
	clock util.Clock
	// config is how the manager is configured
	config Config
	// the function to invoke to kill a pod
	killPodFunc KillPodFunc
	// protects access to internal state
	sync.RWMutex
	// node conditions are the set of conditions present
	nodeConditions []api.NodeConditionType
	// captures when a node condition was last observed based on a threshold being met
	nodeConditionsLastObservedAt nodeConditionsObservedAt
	// nodeRef is a reference to the node
	nodeRef *api.ObjectReference
	// used to record events about the node
	recorder record.EventRecorder
	// used to measure usage stats on system
	summaryProvider stats.SummaryProvider
	// records when a threshold was first observed
	thresholdsFirstObservedAt thresholdsObservedAt
}

// ensure it implements the required interface
var _ Manager = &managerImpl{}

// NewManager returns a configured Manager and an associated admission handler to enforce eviction configuration.
func NewManager(
	summaryProvider stats.SummaryProvider,
	config Config,
	killPodFunc KillPodFunc,
	recorder record.EventRecorder,
	nodeRef *api.ObjectReference,
	clock util.Clock) (Manager, lifecycle.PodAdmitHandler, error) {
	manager := &managerImpl{
		clock:           clock,
		killPodFunc:     killPodFunc,
		config:          config,
		recorder:        recorder,
		summaryProvider: summaryProvider,
		nodeRef:         nodeRef,
		nodeConditionsLastObservedAt: nodeConditionsObservedAt{},
		thresholdsFirstObservedAt:    thresholdsObservedAt{},
	}
	return manager, manager, nil
}

// Admit rejects a pod if its not safe to admit for node stability.
func (m *managerImpl) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	m.RLock()
	defer m.RUnlock()
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
func (m *managerImpl) Start(podFunc ActivePodsFunc, monitoringInterval time.Duration) {
	go wait.Until(func() { m.synchronize(podFunc) }, monitoringInterval, wait.NeverStop)
}

// IsUnderMemoryPressure returns true if the node is under memory pressure.
func (m *managerImpl) IsUnderMemoryPressure() bool {
	m.RLock()
	defer m.RUnlock()
	return hasNodeCondition(m.nodeConditions, api.NodeMemoryPressure)
}

// synchronize is the main control loop that enforces eviction thresholds.
func (m *managerImpl) synchronize(podFunc ActivePodsFunc) {
	// if we have nothing to do, just return
	thresholds := m.config.Thresholds
	if len(thresholds) == 0 {
		return
	}

	// make observations and get a function to derive pod usage stats relative to those observations.
	observations, statsFunc, err := makeSignalObservations(m.summaryProvider)
	if err != nil {
		glog.Errorf("eviction manager: unexpected err: %v", err)
		return
	}

	// find the list of thresholds that are met independent of grace period
	now := m.clock.Now()

	// determine the set of thresholds met independent of grace period
	thresholds = thresholdsMet(thresholds, observations)

	// track when a threshold was first observed
	thresholdsFirstObservedAt := thresholdsFirstObservedAt(thresholds, m.thresholdsFirstObservedAt, now)

	// the set of node conditions that are triggered by currently observed thresholds
	nodeConditions := nodeConditions(thresholds)

	// track when a node condition was last observed
	nodeConditionsLastObservedAt := nodeConditionsLastObservedAt(nodeConditions, m.nodeConditionsLastObservedAt, now)

	// node conditions report true if it has been observed within the transition period window
	nodeConditions = nodeConditionsObservedSince(nodeConditionsLastObservedAt, m.config.PressureTransitionPeriod, now)

	// determine the set of thresholds we need to drive eviction behavior (i.e. all grace periods are met)
	thresholds = thresholdsMetGracePeriod(thresholdsFirstObservedAt, now)

	// update internal state
	m.Lock()
	m.nodeConditions = nodeConditions
	m.thresholdsFirstObservedAt = thresholdsFirstObservedAt
	m.nodeConditionsLastObservedAt = nodeConditionsLastObservedAt
	m.Unlock()

	// determine the set of resources under starvation
	starvedResources := reclaimResources(thresholds)
	if len(starvedResources) == 0 {
		glog.V(3).Infof("eviction manager: no resources are starved")
		return
	}

	// rank the resources to reclaim by eviction priority
	sort.Sort(byEvictionPriority(starvedResources))
	resourceToReclaim := starvedResources[0]
	glog.Warningf("eviction manager: attempting to reclaim %v", resourceToReclaim)

	// determine if this is a soft or hard eviction associated with the resource
	softEviction := isSoftEviction(thresholds, resourceToReclaim)

	// record an event about the resources we are now attempting to reclaim via eviction
	m.recorder.Eventf(m.nodeRef, api.EventTypeWarning, "EvictionThresholdMet", "Attempting to reclaim %s", resourceToReclaim)

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
	rank(activePods, statsFunc)

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
		gracePeriodOverride := int64(0)
		if softEviction {
			gracePeriodOverride = m.config.MaxPodGracePeriodSeconds
		}
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
