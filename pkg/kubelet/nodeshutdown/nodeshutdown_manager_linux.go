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

	"github.com/godbus/dbus/v5"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	"k8s.io/kubernetes/pkg/kubelet/nodeshutdown/systemd"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

const (
	nodeShutdownReason  = "Shutdown"
	nodeShutdownMessage = "Node is shutting, evicting pods"
)

var systemDbus = func() (dbusInhibiter, error) {
	bus, err := dbus.SystemBus()
	if err != nil {
		return nil, err
	}
	return &systemd.DBusCon{SystemBus: bus}, nil
}

type dbusInhibiter interface {
	CurrentInhibitDelay() (time.Duration, error)
	InhibitShutdown() (systemd.InhibitLock, error)
	ReleaseInhibitLock(lock systemd.InhibitLock) error
	ReloadLogindConf() error
	MonitorShutdown() (<-chan bool, error)
	OverrideInhibitDelay(inhibitDelayMax time.Duration) error
}

// Manager has functions that can be used to interact with the Node Shutdown Manager.
type Manager struct {
	shutdownGracePeriodRequested    time.Duration
	shutdownGracePeriodCriticalPods time.Duration

	getPods eviction.ActivePodsFunc
	killPod eviction.KillPodFunc

	dbusCon     dbusInhibiter
	inhibitLock systemd.InhibitLock

	nodeShuttingDownMutex sync.Mutex
	nodeShuttingDownNow   bool

	clock clock.Clock
}

// NewManager returns a new node shutdown manager.
func NewManager(getPodsFunc eviction.ActivePodsFunc, killPodFunc eviction.KillPodFunc, shutdownGracePeriodRequested, shutdownGracePeriodCriticalPods time.Duration) *Manager {
	return &Manager{
		getPods:                         getPodsFunc,
		killPod:                         killPodFunc,
		shutdownGracePeriodRequested:    shutdownGracePeriodRequested,
		shutdownGracePeriodCriticalPods: shutdownGracePeriodCriticalPods,
		clock:                           clock.RealClock{},
	}
}

// Start starts the node shutdown manager and will start watching the node for shutdown events.
func (m *Manager) Start() error {
	if !utilfeature.DefaultFeatureGate.Enabled(features.GracefulNodeShutdown) {
		return nil
	}
	if m.shutdownGracePeriodRequested == 0 {
		return nil
	}

	systemBus, err := systemDbus()
	if err != nil {
		return err
	}
	m.dbusCon = systemBus

	currentInhibitDelay, err := m.dbusCon.CurrentInhibitDelay()
	if err != nil {
		return err
	}

	// If the logind's InhibitDelayMaxUSec as configured in (logind.conf) is less than shutdownGracePeriodRequested, attempt to update the value to shutdownGracePeriodRequested.
	if m.shutdownGracePeriodRequested > currentInhibitDelay {
		err := m.dbusCon.OverrideInhibitDelay(m.shutdownGracePeriodRequested)
		if err != nil {
			return fmt.Errorf("unable to override inhibit delay by shutdown manager: %v", err)
		}

		err = m.dbusCon.ReloadLogindConf()
		if err != nil {
			return err
		}

		// Read the current inhibitDelay again, if the override was successful, currentInhibitDelay will be equal to shutdownGracePeriodRequested.
		updatedInhibitDelay, err := m.dbusCon.CurrentInhibitDelay()
		if err != nil {
			return err
		}

		if updatedInhibitDelay != m.shutdownGracePeriodRequested {
			return fmt.Errorf("node shutdown manager was unable to update logind InhibitDelayMaxSec to %v (ShutdownGracePeriod), current value of InhibitDelayMaxSec (%v) is less than requested ShutdownGracePeriod", m.shutdownGracePeriodRequested, updatedInhibitDelay)
		}
	}

	err = m.aquireInhibitLock()
	if err != nil {
		return err
	}

	events, err := m.dbusCon.MonitorShutdown()
	if err != nil {
		releaseErr := m.dbusCon.ReleaseInhibitLock(m.inhibitLock)
		if releaseErr != nil {
			return fmt.Errorf("failed releasing inhibitLock: %v and failed monitoring shutdown: %v", releaseErr, err)
		}
		return fmt.Errorf("failed to monitor shutdown: %v", err)
	}

	go func() {
		// Monitor for shutdown events. This follows the logind Inhibit Delay pattern described on https://www.freedesktop.org/wiki/Software/systemd/inhibit/
		// 1. When shutdown manager starts, an inhibit lock is taken.
		// 2. When shutdown(true) event is received, process the shutdown and release the inhibit lock.
		// 3. When shutdown(false) event is received, this indicates a previous shutdown was cancelled. In this case, acquire the inhibit lock again.
		for {
			select {
			case isShuttingDown := <-events:
				klog.V(1).Infof("Shutdown manager detected new shutdown event, isNodeShuttingDownNow: %t", isShuttingDown)

				m.nodeShuttingDownMutex.Lock()
				m.nodeShuttingDownNow = isShuttingDown
				m.nodeShuttingDownMutex.Unlock()

				if isShuttingDown {
					m.processShutdownEvent()
				} else {
					m.aquireInhibitLock()
				}
			}
		}
	}()
	return nil
}

func (m *Manager) aquireInhibitLock() error {
	lock, err := m.dbusCon.InhibitShutdown()
	if err != nil {
		return err
	}
	m.inhibitLock = lock
	return nil
}

// ShutdownStatus will return an error if the node is currently shutting down.
func (m *Manager) ShutdownStatus() error {
	if !utilfeature.DefaultFeatureGate.Enabled(features.GracefulNodeShutdown) {
		return nil
	}

	m.nodeShuttingDownMutex.Lock()
	defer m.nodeShuttingDownMutex.Unlock()

	if m.nodeShuttingDownNow {
		return fmt.Errorf("node is shutting down")
	}
	return nil
}

func (m *Manager) processShutdownEvent() error {
	klog.V(1).Infof("Shutdown manager processing shutdown event")
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

			// If the pod's spec specifies a termination gracePeriod which is less than the gracePeriodOverride calculated, use the pod spec termination gracePeriod.
			if pod.Spec.TerminationGracePeriodSeconds != nil && *pod.Spec.TerminationGracePeriodSeconds <= gracePeriodOverride {
				gracePeriodOverride = *pod.Spec.TerminationGracePeriodSeconds
			}

			klog.V(1).Infof("Shutdown manager killing pod %q with gracePeriod: %v seconds", format.Pod(pod), gracePeriodOverride)

			status := v1.PodStatus{
				Phase:   v1.PodFailed,
				Message: nodeShutdownMessage,
				Reason:  nodeShutdownReason,
			}

			err := m.killPod(pod, status, &gracePeriodOverride)
			if err != nil {
				klog.V(1).Infof("Shutdown manager failed killing pod %q: %v", format.Pod(pod), err)
			} else {
				klog.V(1).Infof("Shutdown manager finished killing pod %q", format.Pod(pod))
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
		klog.V(1).Infof("Shutdown manager pod killing did not complete in %v", m.shutdownGracePeriodRequested)
	}

	m.dbusCon.ReleaseInhibitLock(m.inhibitLock)
	klog.V(1).Infof("Shutdown manager completed processing shutdown event, node will shutdown shortly")

	return nil
}
