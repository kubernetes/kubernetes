//go:build linux

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
	"context"
	"fmt"
	"path/filepath"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	kubeletevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/nodeshutdown/systemd"
)

const (
	dbusReconnectPeriod = 1 * time.Second
)

var systemDbus = func() (dbusInhibiter, error) {
	return systemd.NewDBusCon()
}

type dbusInhibiter interface {
	CurrentInhibitDelay() (time.Duration, error)
	InhibitShutdown() (systemd.InhibitLock, error)
	ReleaseInhibitLock(lock systemd.InhibitLock) error
	ReloadLogindConf() error
	MonitorShutdown(klog.Logger) (<-chan bool, error)
	OverrideInhibitDelay(inhibitDelayMax time.Duration) error
	Close() error
}

// managerImpl has functions that can be used to interact with the Node Shutdown Manager.
type managerImpl struct {
	logger   klog.Logger
	recorder record.EventRecorder
	nodeRef  *v1.ObjectReference

	getPods        eviction.ActivePodsFunc
	syncNodeStatus func(context.Context)

	newDBusConn func() (dbusInhibiter, error)
	inhibitLock systemd.InhibitLock

	nodeShuttingDownMutex sync.Mutex
	nodeShuttingDownNow   bool
	podManager            *podManager

	enableMetrics bool
	storage       storage
}

// NewManager returns a new node shutdown manager.
func NewManager(conf *Config) Manager {
	if !utilfeature.DefaultFeatureGate.Enabled(features.GracefulNodeShutdown) {
		m := managerStub{}
		return m
	}

	podManager := newPodManager(conf)

	// Disable if the configuration is empty
	if len(podManager.shutdownGracePeriodByPodPriority) == 0 {
		m := managerStub{}
		return m
	}

	manager := &managerImpl{
		logger:         conf.Logger,
		recorder:       conf.Recorder,
		nodeRef:        conf.NodeRef,
		getPods:        conf.GetPodsFunc,
		syncNodeStatus: conf.SyncNodeStatusFunc,
		newDBusConn:    systemDbus,
		podManager:     podManager,
		enableMetrics:  utilfeature.DefaultFeatureGate.Enabled(features.GracefulNodeShutdownBasedOnPodPriority),
		storage: localStorage{
			Path: filepath.Join(conf.StateDirectory, localStorageStateFile),
		},
	}
	manager.logger.Info("Creating node shutdown manager",
		"shutdownGracePeriodRequested", conf.ShutdownGracePeriodRequested,
		"shutdownGracePeriodCriticalPods", conf.ShutdownGracePeriodCriticalPods,
		"shutdownGracePeriodByPodPriority", podManager.shutdownGracePeriodByPodPriority,
	)
	return manager
}

// Admit rejects all pods if node is shutting
func (m *managerImpl) Admit(ctx context.Context, attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	nodeShuttingDown := m.ShutdownStatus() != nil

	if nodeShuttingDown {
		return lifecycle.PodAdmitResult{
			Admit:   false,
			Reason:  NodeShutdownNotAdmittedReason,
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
func (m *managerImpl) Start(ctx context.Context) error {
	done, err := m.start(ctx)
	if err != nil {
		return err
	}
	go func() {
		for {
			if done != nil {
				select {
				case <-done:
				case <-ctx.Done():
					return
				}
			}

			t := time.NewTimer(dbusReconnectPeriod)
			select {
			case <-t.C:
			case <-ctx.Done():
				t.Stop()
				return
			}
			m.logger.V(1).Info("Restarting watch for node shutdown events")
			done, err = m.start(ctx)
			if err != nil {
				m.logger.Error(err, "Unable to watch the node for shutdown events")
			}
		}
	}()

	m.setMetrics()
	return nil
}

// start starts a goroutine which owns the dbus connection for its lifetime.
// The returned channel is closed when the goroutine exits.
func (m *managerImpl) start(ctx context.Context) (chan struct{}, error) {
	startResult := make(chan error, 1)
	done := make(chan struct{})

	go func() {
		defer close(done)

		systemBus, err := m.newDBusConn()
		if err != nil {
			startResult <- err
			return
		}

		var startErr error
		defer func() {
			// Report startup errors after closing the connection so callers
			// observe a fully stopped watcher.
			if err := systemBus.Close(); err != nil {
				m.logger.Error(err, "Failed to close dbus connection")
			}
			if startErr != nil {
				startResult <- startErr
			}
		}()

		events, err := m.startWatching(systemBus)
		if err != nil {
			startErr = err
			return
		}

		startResult <- nil
		m.watchShutdownEvents(ctx, systemBus, events)
	}()

	if err := <-startResult; err != nil {
		return nil, err
	}
	return done, nil
}

func (m *managerImpl) startWatching(systemBus dbusInhibiter) (<-chan bool, error) {
	currentInhibitDelay, err := systemBus.CurrentInhibitDelay()
	if err != nil {
		return nil, err
	}

	// If the logind's InhibitDelayMaxUSec as configured in (logind.conf) is less than periodRequested, attempt to update the value to periodRequested.
	if periodRequested := m.podManager.periodRequested(); periodRequested > currentInhibitDelay {
		err := systemBus.OverrideInhibitDelay(periodRequested)
		if err != nil {
			return nil, fmt.Errorf("unable to override inhibit delay by shutdown manager: %v", err)
		}

		err = systemBus.ReloadLogindConf()
		if err != nil {
			return nil, err
		}

		// The ReloadLogindConf call is asynchronous. Poll with exponential backoff until the configuration is updated.
		backoff := wait.Backoff{
			Duration: 100 * time.Millisecond,
			Factor:   2.0,
			Steps:    5,
		}
		var updatedInhibitDelay time.Duration
		attempt := 0
		err = wait.ExponentialBackoff(backoff, func() (bool, error) {
			attempt += 1
			// Read the current inhibitDelay again, if the override was successful, currentInhibitDelay will be equal to shutdownGracePeriodRequested.
			updatedInhibitDelay, err = systemBus.CurrentInhibitDelay()
			if err != nil {
				return false, err
			}
			if periodRequested <= updatedInhibitDelay {
				return true, nil
			}
			if attempt < backoff.Steps {
				m.logger.V(3).Info("InhibitDelayMaxSec still less than requested, retrying", "attempt", attempt, "current", updatedInhibitDelay, "requested", periodRequested)
			}
			return false, nil
		})
		if err != nil {
			if !wait.Interrupted(err) {
				return nil, err
			}
			if periodRequested > updatedInhibitDelay {
				return nil, fmt.Errorf("node shutdown manager was timed out after %d attempts waiting for logind InhibitDelayMaxSec to update to %v (ShutdownGracePeriod), current value is %v", attempt, periodRequested, updatedInhibitDelay)
			}
		}
	}

	if err := m.acquireInhibitLock(systemBus); err != nil {
		return nil, err
	}

	events, err := systemBus.MonitorShutdown(m.logger)
	if err != nil {
		releaseErr := systemBus.ReleaseInhibitLock(m.inhibitLock)
		if releaseErr != nil {
			return nil, fmt.Errorf("failed releasing inhibitLock: %v and failed monitoring shutdown: %v", releaseErr, err)
		}
		return nil, fmt.Errorf("failed to monitor shutdown: %v", err)
	}
	return events, nil
}

func (m *managerImpl) watchShutdownEvents(ctx context.Context, systemBus dbusInhibiter, events <-chan bool) {
	// Monitor for shutdown events. This follows the logind Inhibit Delay pattern described on https://www.freedesktop.org/wiki/Software/systemd/inhibit/
	// 1. When shutdown manager starts, an inhibit lock is taken.
	// 2. When shutdown(true) event is received, process the shutdown and release the inhibit lock.
	// 3. When shutdown(false) event is received, this indicates a previous shutdown was cancelled. In this case, acquire the inhibit lock again.
	for {
		select {
		case <-ctx.Done():
			return
		case isShuttingDown, ok := <-events:
			if !ok {
				m.logger.Error(nil, "Ended to watching the node for shutdown events")
				return
			}
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
				nodeStatusCtx := klog.NewContext(ctx, m.logger)
				go m.syncNodeStatus(nodeStatusCtx)

				if err := m.processShutdownEvent(ctx, systemBus); err != nil {
					m.logger.Error(err, "Shutdown manager failed to process shutdown event")
				}
			} else {
				_ = m.acquireInhibitLock(systemBus)
			}
		}
	}
}

func (m *managerImpl) acquireInhibitLock(systemBus dbusInhibiter) error {
	lock, err := systemBus.InhibitShutdown()
	if err != nil {
		return err
	}
	if m.inhibitLock != 0 {
		systemBus.ReleaseInhibitLock(m.inhibitLock)
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

func (m *managerImpl) processShutdownEvent(ctx context.Context, systemBus dbusInhibiter) error {
	m.logger.V(1).Info("Shutdown manager processing shutdown event")
	activePods := m.getPods()

	defer func() {
		systemBus.ReleaseInhibitLock(m.inhibitLock)
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
			metrics.GracefulShutdownEndTime.Set(timestamp(endTime))
		}()
	}

	return m.podManager.killPods(ctx, activePods)
}
