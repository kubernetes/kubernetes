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
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"sync"
	"syscall"
	"time"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/nodeshutdown/systemd"
	"k8s.io/kubernetes/pkg/kubelet/prober"
	"k8s.io/utils/clock"
)

const (
	nodeShutdownReason             = "Terminated"
	nodeShutdownMessage            = "Pod was terminated in response to imminent node shutdown."
	nodeShutdownNotAdmittedReason  = "NodeShutdown"
	nodeShutdownNotAdmittedMessage = "Pod was rejected as the node is shutting down."
	dbusReconnectPeriod            = 1 * time.Second
	localStorageStateFile          = "graceful_node_shutdown_state"
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
	logger       klog.Logger
	recorder     record.EventRecorder
	nodeRef      *v1.ObjectReference
	probeManager prober.Manager

	shutdownGracePeriodByPodPriority []kubeletconfig.ShutdownGracePeriodByPodPriority

	getPods        eviction.ActivePodsFunc
	killPodFunc    eviction.KillPodFunc
	syncNodeStatus func()

	dbusCon     dbusInhibiter
	inhibitLock systemd.InhibitLock

	nodeShuttingDownMutex sync.Mutex
	nodeShuttingDownNow   bool

	clock clock.Clock

	enableMetrics bool
	storage       storage
}

// NewManager returns a new node shutdown manager.
func NewManager(conf *Config) (Manager, lifecycle.PodAdmitHandler) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.GracefulNodeShutdown) {
		m := managerStub{}
		conf.Logger.Info("Node graceful shutdown feature is disabled, node shutdown will be immediate")
		return m, m
	}

	shutdownGracePeriodByPodPriority := conf.ShutdownGracePeriodByPodPriority
	// Migration from the original configuration
	if !utilfeature.DefaultFeatureGate.Enabled(features.GracefulNodeShutdownBasedOnPodPriority) ||
		len(shutdownGracePeriodByPodPriority) == 0 {
		shutdownGracePeriodByPodPriority = migrateConfig(conf.ShutdownGracePeriodRequested, conf.ShutdownGracePeriodCriticalPods)
	}

	// Disable if the configuration is empty
	if len(shutdownGracePeriodByPodPriority) == 0 {
		m := managerStub{}
		conf.Logger.Info("No graceful configuration specified, node shutdown will be immediate")
		return m, m
	}

	// Sort by priority from low to high
	sort.Slice(shutdownGracePeriodByPodPriority, func(i, j int) bool {
		return shutdownGracePeriodByPodPriority[i].Priority < shutdownGracePeriodByPodPriority[j].Priority
	})

	if conf.Clock == nil {
		conf.Clock = clock.RealClock{}
	}
	manager := &managerImpl{
		logger:                           conf.Logger,
		probeManager:                     conf.ProbeManager,
		recorder:                         conf.Recorder,
		nodeRef:                          conf.NodeRef,
		getPods:                          conf.GetPodsFunc,
		killPodFunc:                      conf.KillPodFunc,
		syncNodeStatus:                   conf.SyncNodeStatusFunc,
		shutdownGracePeriodByPodPriority: shutdownGracePeriodByPodPriority,
		clock:                            conf.Clock,
		enableMetrics:                    utilfeature.DefaultFeatureGate.Enabled(features.GracefulNodeShutdownBasedOnPodPriority),
		storage: localStorage{
			Path: filepath.Join(conf.StateDirectory, localStorageStateFile),
		},
	}
	manager.logger.Info("Creating node shutdown manager",
		"shutdownGracePeriodRequested", conf.ShutdownGracePeriodRequested,
		"shutdownGracePeriodCriticalPods", conf.ShutdownGracePeriodCriticalPods,
		"shutdownGracePeriodByPodPriority", shutdownGracePeriodByPodPriority,
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

// setMetrics sets the metrics for the node shutdown manager.
func (m *managerImpl) setMetrics() {
	if m.enableMetrics && m.storage != nil {
		sta := state{}
		err := m.storage.Load(&sta)
		if err != nil {
			m.logger.Error(err, "Failed to load graceful shutdown state")
		} else {
			if !sta.StartTime.IsZero() {
				metrics.GracefulShutdownStartTime.Set(timestamp(sta.StartTime))
			}
			if !sta.EndTime.IsZero() {
				metrics.GracefulShutdownEndTime.Set(timestamp(sta.EndTime))
			}
		}
	}
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
			m.logger.V(1).Info("Restarting watch for node shutdown events")
			stop, err = m.start()
			if err != nil {
				m.logger.Error(err, "Unable to watch the node for shutdown events")
			}
		}
	}()

	m.setMetrics()
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

	// If the logind's InhibitDelayMaxUSec as configured in (logind.conf) is less than periodRequested, attempt to update the value to periodRequested.
	if periodRequested := m.periodRequested(); periodRequested > currentInhibitDelay {
		err := m.dbusCon.OverrideInhibitDelay(periodRequested)
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

		if periodRequested > updatedInhibitDelay {
			return nil, fmt.Errorf("node shutdown manager was unable to update logind InhibitDelayMaxSec to %v (ShutdownGracePeriod), current value of InhibitDelayMaxSec (%v) is less than requested ShutdownGracePeriod", periodRequested, updatedInhibitDelay)
		}
	}

	err = m.acquireInhibitLock()
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

	unifiedCh := make(chan bool, 1)
	sigUSR1 := make(chan os.Signal, 1)
	signal.Notify(sigUSR1, syscall.SIGUSR1)

	stop := make(chan struct{})
	go func() {
		// Monitor for shutdown events. This follows the logind Inhibit Delay pattern described on https://www.freedesktop.org/wiki/Software/systemd/inhibit/
		// 1. When shutdown manager starts, an inhibit lock is taken.
		// 2. When shutdown(true) event is received, process the shutdown and release the inhibit lock.
		// 3. When shutdown(false) event is received, this indicates a previous shutdown was cancelled. In this case, acquire the inhibit lock again.
		for {
			select {

			case <-sigUSR1:
				m.logger.V(1).Info("Received shutdown signal from USR1")
				unifiedCh <- true
			case isShuttingDown, ok := <-events:
				if !ok {
					m.logger.Error(err, "Ended to watching the node for shutdown events")
					close(stop)
					return
				}
				unifiedCh <- isShuttingDown

			case isShuttingDown := <-unifiedCh:
				m.logger.V(1).Info("Shutdown manager detected new shutdown event, isNodeShuttingDownNow", "event", isShuttingDown)

				var shutdownType string
				if isShuttingDown {
					shutdownType = "shutdown"
				} else {
					shutdownType = "cancelled"
				}
				m.logger.V(1).Info("Shutdown manager detected new shutdown event", "event", shutdownType)
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
					_ = m.acquireInhibitLock()
				}
			}
		}
	}()
	return stop, nil
}

func (m *managerImpl) acquireInhibitLock() error {
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
	m.logger.V(1).Info("Shutdown manager processing shutdown event")
	activePods := m.getPods()

	defer func() {
		m.dbusCon.ReleaseInhibitLock(m.inhibitLock)
		m.logger.V(1).Info("Shutdown manager completed processing shutdown event, node will shutdown shortly")
	}()

	if m.enableMetrics && m.storage != nil {
		startTime := time.Now()
		err := m.storage.Store(state{
			StartTime: startTime,
		})
		if err != nil {
			m.logger.Error(err, "Failed to store graceful shutdown state")
		}
		metrics.GracefulShutdownStartTime.Set(timestamp(startTime))
		metrics.GracefulShutdownEndTime.Set(0)

		defer func() {
			endTime := time.Now()
			err := m.storage.Store(state{
				StartTime: startTime,
				EndTime:   endTime,
			})
			if err != nil {
				m.logger.Error(err, "Failed to store graceful shutdown state")
			}
			metrics.GracefulShutdownStartTime.Set(timestamp(endTime))
		}()
	}

	groups := groupByPriority(m.shutdownGracePeriodByPodPriority, activePods)
	for _, group := range groups {
		// If there are no pods in a particular range,
		// then do not wait for pods in that priority range.
		if len(group.Pods) == 0 {
			continue
		}

		var wg sync.WaitGroup
		wg.Add(len(group.Pods))
		for _, pod := range group.Pods {
			go func(pod *v1.Pod, group podShutdownGroup) {
				defer wg.Done()

				gracePeriodOverride := group.ShutdownGracePeriodSeconds

				// If the pod's spec specifies a termination gracePeriod which is less than the gracePeriodOverride calculated, use the pod spec termination gracePeriod.
				if pod.Spec.TerminationGracePeriodSeconds != nil && *pod.Spec.TerminationGracePeriodSeconds <= gracePeriodOverride {
					gracePeriodOverride = *pod.Spec.TerminationGracePeriodSeconds
				}

				m.logger.V(1).Info("Shutdown manager killing pod with gracePeriod", "pod", klog.KObj(pod), "gracePeriod", gracePeriodOverride)

				if err := m.killPodFunc(pod, false, &gracePeriodOverride, func(status *v1.PodStatus) {
					// set the pod status to failed (unless it was already in a successful terminal phase)
					if status.Phase != v1.PodSucceeded {
						status.Phase = v1.PodFailed
					}
					status.Message = nodeShutdownMessage
					status.Reason = nodeShutdownReason
					if utilfeature.DefaultFeatureGate.Enabled(features.PodDisruptionConditions) {
						podutil.UpdatePodCondition(status, &v1.PodCondition{
							Type:    v1.DisruptionTarget,
							Status:  v1.ConditionTrue,
							Reason:  v1.PodReasonTerminationByKubelet,
							Message: nodeShutdownMessage,
						})
					}
				}); err != nil {
					m.logger.V(1).Info("Shutdown manager failed killing pod", "pod", klog.KObj(pod), "err", err)
				} else {
					m.logger.V(1).Info("Shutdown manager finished killing pod", "pod", klog.KObj(pod))
				}
			}(pod, group)
		}

		var (
			doneCh = make(chan struct{})
			timer  = m.clock.NewTimer(time.Duration(group.ShutdownGracePeriodSeconds) * time.Second)
		)
		go func() {
			defer close(doneCh)
			wg.Wait()
		}()

		select {
		case <-doneCh:
			timer.Stop()
		case <-timer.C():
			m.logger.V(1).Info("Shutdown manager pod killing time out", "gracePeriod", group.ShutdownGracePeriodSeconds, "priority", group.Priority)
		}
	}

	return nil
}

func (m *managerImpl) periodRequested() time.Duration {
	var sum int64
	for _, period := range m.shutdownGracePeriodByPodPriority {
		sum += period.ShutdownGracePeriodSeconds
	}
	return time.Duration(sum) * time.Second
}

func migrateConfig(shutdownGracePeriodRequested, shutdownGracePeriodCriticalPods time.Duration) []kubeletconfig.ShutdownGracePeriodByPodPriority {
	if shutdownGracePeriodRequested == 0 {
		return nil
	}
	defaultPriority := shutdownGracePeriodRequested - shutdownGracePeriodCriticalPods
	if defaultPriority < 0 {
		return nil
	}
	criticalPriority := shutdownGracePeriodRequested - defaultPriority
	if criticalPriority < 0 {
		return nil
	}
	return []kubeletconfig.ShutdownGracePeriodByPodPriority{
		{
			Priority:                   scheduling.DefaultPriorityWhenNoDefaultClassExists,
			ShutdownGracePeriodSeconds: int64(defaultPriority / time.Second),
		},
		{
			Priority:                   scheduling.SystemCriticalPriority,
			ShutdownGracePeriodSeconds: int64(criticalPriority / time.Second),
		},
	}
}

func groupByPriority(shutdownGracePeriodByPodPriority []kubeletconfig.ShutdownGracePeriodByPodPriority, pods []*v1.Pod) []podShutdownGroup {
	groups := make([]podShutdownGroup, 0, len(shutdownGracePeriodByPodPriority))
	for _, period := range shutdownGracePeriodByPodPriority {
		groups = append(groups, podShutdownGroup{
			ShutdownGracePeriodByPodPriority: period,
		})
	}

	for _, pod := range pods {
		var priority int32
		if pod.Spec.Priority != nil {
			priority = *pod.Spec.Priority
		}

		// Find the group index according to the priority.
		index := sort.Search(len(groups), func(i int) bool {
			return groups[i].Priority >= priority
		})

		// 1. Those higher than the highest priority default to the highest priority
		// 2. Those lower than the lowest priority default to the lowest priority
		// 3. Those boundary priority default to the lower priority
		// if priority of pod is:
		//   groups[index-1].Priority <= pod priority < groups[index].Priority
		// in which case we want to pick lower one (i.e index-1)
		if index == len(groups) {
			index = len(groups) - 1
		} else if index < 0 {
			index = 0
		} else if index > 0 && groups[index].Priority > priority {
			index--
		}

		groups[index].Pods = append(groups[index].Pods, pod)
	}
	return groups
}

type podShutdownGroup struct {
	kubeletconfig.ShutdownGracePeriodByPodPriority
	Pods []*v1.Pod
}
