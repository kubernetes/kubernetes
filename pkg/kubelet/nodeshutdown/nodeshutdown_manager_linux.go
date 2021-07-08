//go:build linux
// +build linux

/*
Copyright 2020 The Kubernetes Authors.

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

// Package nodeshutdown can watch for node level shutdown events and trigger graceful termination of pods running on the node prior to a system shutdown.
package nodeshutdown

import (
	"fmt"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	kubeletevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/nodeshutdown/systemd"
	"k8s.io/kubernetes/pkg/kubelet/prober"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/utils/clock"
)

const (
	nodeShutdownReason             = "Terminated"
	nodeShutdownMessage            = "Pod was terminated in response to imminent node shutdown."
	nodeShutdownNotAdmittedReason  = "NodeShutdown"
	nodeShutdownNotAdmittedMessage = "Pod was rejected as the node is shutting down."
	dbusReconnectPeriod            = 1 * time.Second
)

var systemDbus = func() (dbusInhibiter, error) {
	return systemd.NewDBusCon()
}

type dbusInhibiter interface {
	CurrentInhibitDelay() (time.Duration, error)
	InhibitShutdown() (systemd.InhibitLock, error)
	ReleaseInhibitLock(lock systemd.InhibitLock) error
	ReloadLogindConf() error
	MonitorShutdown() (<-chan bool, error)
	OverrideInhibitDelay(inhibitDelayMax time.Duration) error
}

// managerImpl has functions that can be used to interact with the Node Shutdown Manager.
type managerImpl struct {
	recorder     record.EventRecorder
	nodeRef      *v1.ObjectReference
	probeManager prober.Manager

	shutdownGracePeriodRequested    time.Duration
	shutdownGracePeriodCriticalPods time.Duration

	getPods        eviction.ActivePodsFunc
	killPodFunc    eviction.KillPodFunc
	syncNodeStatus func()

	dbusCon     dbusInhibiter
	inhibitLock systemd.InhibitLock

	nodeShuttingDownMutex sync.Mutex
	nodeShuttingDownNow   bool

	clock clock.Clock
}

// NewManager returns a new node shutdown manager.
func NewManager(conf *Config) (Manager, lifecycle.PodAdmitHandler) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.GracefulNodeShutdown) ||
		(conf.ShutdownGracePeriodRequested == 0 && conf.ShutdownGracePeriodCriticalPods == 0) {
		m := managerStub{}
		return m, m
	}
	if conf.Clock == nil {
		conf.Clock = clock.RealClock{}
	}
	manager := &managerImpl{
		probeManager:                    conf.ProbeManager,
		recorder:                        conf.Recorder,
		nodeRef:                         conf.NodeRef,
		getPods:                         conf.GetPodsFunc,
		killPodFunc:                     conf.KillPodFunc,
		syncNodeStatus:                  conf.SyncNodeStatusFunc,
		shutdownGracePeriodRequested:    conf.ShutdownGracePeriodRequested,
		shutdownGracePeriodCriticalPods: conf.ShutdownGracePeriodCriticalPods,
		clock:                           conf.Clock,
	}
	klog.InfoS("Creating node shutdown manager",
		"shutdownGracePeriodRequested", conf.ShutdownGracePeriodRequested,
		"shutdownGracePeriodCriticalPods", conf.ShutdownGracePeriodCriticalPods,
	)
	return manager, manager
}

// Admit rejects all pods if node is shutting
func (m *managerImpl) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	nodeShuttingDown := m.ShutdownStatus() != nil

	if nodeShuttingDown {
		return lifecycle.PodAdmitResult{
			Admit:   false,
			Reason:  nodeShutdownNotAdmittedReason,
			Message: nodeShutdownNotAdmittedMessage,
		}
	}
	return lifecycle.PodAdmitResult{Admit: true}
}

// Start starts the node shutdown manager and will start watching the node for shutdown events.
func (m *managerImpl) Start() error {
	stop, err := m.start()
	if err != nil {
		return err
	}
	go func() {
		for {
			if stop != nil {
				<-stop
			}

			time.Sleep(dbusReconnectPeriod)
			klog.V(1).InfoS("Restarting watch for node shutdown events")
			stop, err = m.start()
			if err != nil {
				klog.ErrorS(err, "Unable to watch the node for shutdown events")
			}
		}
	}()
	return nil
}

func (m *managerImpl) start() (chan struct{}, error) {
	systemBus, err := systemDbus()
	if err != nil {
		return nil, err
	}
	m.dbusCon = systemBus

	currentInhibitDelay, err := m.dbusCon.CurrentInhibitDelay()
	if err != nil {
		return nil, err
	}

	// If the logind's InhibitDelayMaxUSec as configured in (logind.conf) is less than shutdownGracePeriodRequested, attempt to update the value to shutdownGracePeriodRequested.
	if m.shutdownGracePeriodRequested > currentInhibitDelay {
		err := m.dbusCon.OverrideInhibitDelay(m.shutdownGracePeriodRequested)
		if err != nil {
			return nil, fmt.Errorf("unable to override inhibit delay by shutdown manager: %v", err)
		}

		err = m.dbusCon.ReloadLogindConf()
		if err != nil {
			return nil, err
		}

		// Read the current inhibitDelay again, if the override was successful, currentInhibitDelay will be equal to shutdownGracePeriodRequested.
		updatedInhibitDelay, err := m.dbusCon.CurrentInhibitDelay()
		if err != nil {
			return nil, err
		}

		if m.shutdownGracePeriodRequested > updatedInhibitDelay {
			return nil, fmt.Errorf("node shutdown manager was unable to update logind InhibitDelayMaxSec to %v (ShutdownGracePeriod), current value of InhibitDelayMaxSec (%v) is less than requested ShutdownGracePeriod", m.shutdownGracePeriodRequested, updatedInhibitDelay)
		}
	}

	err = m.aquireInhibitLock()
	if err != nil {
		return nil, err
	}

	events, err := m.dbusCon.MonitorShutdown()
	if err != nil {
		releaseErr := m.dbusCon.ReleaseInhibitLock(m.inhibitLock)
		if releaseErr != nil {
			return nil, fmt.Errorf("failed releasing inhibitLock: %v and failed monitoring shutdown: %v", releaseErr, err)
		}
		return nil, fmt.Errorf("failed to monitor shutdown: %v", err)
	}

	stop := make(chan struct{})
	go func() {
		// Monitor for shutdown events. This follows the logind Inhibit Delay pattern described on https://www.freedesktop.org/wiki/Software/systemd/inhibit/
		// 1. When shutdown manager starts, an inhibit lock is taken.
		// 2. When shutdown(true) event is received, process the shutdown and release the inhibit lock.
		// 3. When shutdown(false) event is received, this indicates a previous shutdown was cancelled. In this case, acquire the inhibit lock again.
		for {
			select {
			case isShuttingDown, ok := <-events:
				if !ok {
					klog.ErrorS(err, "Ended to watching the node for shutdown events")
					close(stop)
					return
				}
				klog.V(1).InfoS("Shutdown manager detected new shutdown event, isNodeShuttingDownNow", "event", isShuttingDown)

				var shutdownType string
				if isShuttingDown {
					shutdownType = "shutdown"
				} else {
					shutdownType = "cancelled"
				}
				klog.V(1).InfoS("Shutdown manager detected new shutdown event", "event", shutdownType)
				if isShuttingDown {
					m.recorder.Event(m.nodeRef, v1.EventTypeNormal, kubeletevents.NodeShutdown, "Shutdown manager detected shutdown event")
				} else {
					m.recorder.Event(m.nodeRef, v1.EventTypeNormal, kubeletevents.NodeShutdown, "Shutdown manager detected shutdown cancellation")
				}

				m.nodeShuttingDownMutex.Lock()
				m.nodeShuttingDownNow = isShuttingDown
				m.nodeShuttingDownMutex.Unlock()

				if isShuttingDown {
					// Update node status and ready condition
					go m.syncNodeStatus()

					m.processShutdownEvent()
				} else {
					m.aquireInhibitLock()
				}
			}
		}
	}()
	return stop, nil
}

func (m *managerImpl) aquireInhibitLock() error {
	lock, err := m.dbusCon.InhibitShutdown()
	if err != nil {
		return err
	}
	if m.inhibitLock != 0 {
		m.dbusCon.ReleaseInhibitLock(m.inhibitLock)
	}
	m.inhibitLock = lock
	return nil
}

// ShutdownStatus will return an error if the node is currently shutting down.
func (m *managerImpl) ShutdownStatus() error {
	m.nodeShuttingDownMutex.Lock()
	defer m.nodeShuttingDownMutex.Unlock()

	if m.nodeShuttingDownNow {
		return fmt.Errorf("node is shutting down")
	}
	return nil
}

func (m *managerImpl) processShutdownEvent() error {
	klog.V(1).InfoS("Shutdown manager processing shutdown event")
	activePods := m.getPods()

	nonCriticalPodGracePeriod := m.shutdownGracePeriodRequested - m.shutdownGracePeriodCriticalPods

	var wg sync.WaitGroup
	wg.Add(len(activePods))
	for _, pod := range activePods {
		go func(pod *v1.Pod) {
			defer wg.Done()

			var gracePeriodOverride int64
			if kubelettypes.IsCriticalPod(pod) {
				gracePeriodOverride = int64(m.shutdownGracePeriodCriticalPods.Seconds())
				m.clock.Sleep(nonCriticalPodGracePeriod)
			} else {
				gracePeriodOverride = int64(nonCriticalPodGracePeriod.Seconds())
			}

			// Stop probes for the pod
			m.probeManager.RemovePod(pod)

			// If the pod's spec specifies a termination gracePeriod which is less than the gracePeriodOverride calculated, use the pod spec termination gracePeriod.
			if pod.Spec.TerminationGracePeriodSeconds != nil && *pod.Spec.TerminationGracePeriodSeconds <= gracePeriodOverride {
				gracePeriodOverride = *pod.Spec.TerminationGracePeriodSeconds
			}

			klog.V(1).InfoS("Shutdown manager killing pod with gracePeriod", "pod", klog.KObj(pod), "gracePeriod", gracePeriodOverride)
			if err := m.killPodFunc(pod, false, &gracePeriodOverride, func(status *v1.PodStatus) {
				status.Message = nodeShutdownMessage
				status.Reason = nodeShutdownReason
			}); err != nil {
				klog.V(1).InfoS("Shutdown manager failed killing pod", "pod", klog.KObj(pod), "err", err)
			} else {
				klog.V(1).InfoS("Shutdown manager finished killing pod", "pod", klog.KObj(pod))
			}
		}(pod)
	}

	c := make(chan struct{})
	go func() {
		defer close(c)
		wg.Wait()
	}()

	// We want to ensure that inhibitLock is released, so only wait up to the shutdownGracePeriodRequested timeout.
	select {
	case <-c:
		break
	case <-time.After(m.shutdownGracePeriodRequested):
		klog.V(1).InfoS("Shutdown manager pod killing time out", "gracePeriod", m.shutdownGracePeriodRequested)
	}

	m.dbusCon.ReleaseInhibitLock(m.inhibitLock)
	klog.V(1).InfoS("Shutdown manager completed processing shutdown event, node will shutdown shortly")

	return nil
}
