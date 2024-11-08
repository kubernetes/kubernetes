//go:build linux
// +build linux

/*
Copyright 2024 The Kubernetes Authors.

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

package watchdog

import (
	"fmt"
	"time"

	"github.com/coreos/go-systemd/v22/daemon"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/klog/v2"
)

// WatchdogClient defines the interface for interacting with the systemd watchdog.
type WatchdogClient interface {
	SdWatchdogEnabled(unsetEnvironment bool) (time.Duration, error)
	SdNotify(unsetEnvironment bool) (bool, error)
}

// DefaultWatchdogClient implements the WatchdogClient interface using the actual systemd daemon functions.
type DefaultWatchdogClient struct{}

var _ WatchdogClient = &DefaultWatchdogClient{}

func (d *DefaultWatchdogClient) SdWatchdogEnabled(unsetEnvironment bool) (time.Duration, error) {
	return daemon.SdWatchdogEnabled(unsetEnvironment)
}

func (d *DefaultWatchdogClient) SdNotify(unsetEnvironment bool) (bool, error) {
	return daemon.SdNotify(unsetEnvironment, daemon.SdNotifyWatchdog)
}

// Option defines optional parameters for initializing the healthChecker
// structure.
type Option func(*healthChecker)

func WithWatchdogClient(watchdog WatchdogClient) Option {
	return func(hc *healthChecker) {
		hc.watchdog = watchdog
	}
}

func WithExtendedCheckers(checkers []healthz.HealthChecker) Option {
	return func(hc *healthChecker) {
		hc.checkers = append(hc.checkers, checkers...)
	}
}

type healthChecker struct {
	checkers     []healthz.HealthChecker
	retryBackoff wait.Backoff
	interval     time.Duration
	watchdog     WatchdogClient
}

var _ HealthChecker = &healthChecker{}

const minimalNotifyInterval = time.Second

// NewHealthChecker creates a new HealthChecker instance.
// This function initializes the health checker and configures its behavior based on the status of the systemd watchdog.
// If the watchdog is not enabled, the function returns an error.
func NewHealthChecker(syncLoop syncLoopHealthChecker, opts ...Option) (HealthChecker, error) {
	hc := &healthChecker{
		watchdog: &DefaultWatchdogClient{},
	}
	for _, o := range opts {
		o(hc)
	}

	// get watchdog information
	watchdogVal, err := hc.watchdog.SdWatchdogEnabled(false)
	if err != nil {
		// Failed to get watchdog configuration information.
		// This occurs when we want to start the watchdog but the configuration is incorrect,
		// for example, the time is not configured correctly.
		return nil, fmt.Errorf("configure watchdog: %w", err)
	}
	if watchdogVal == 0 {
		klog.InfoS("Systemd watchdog is not enabled")
		return &healthChecker{}, nil
	}
	if watchdogVal <= minimalNotifyInterval {
		return nil, fmt.Errorf("configure watchdog timeout too small: %v", watchdogVal)
	}

	// The health checks performed by checkers are the same as those for "/healthz".
	checkers := []healthz.HealthChecker{
		healthz.PingHealthz,
		healthz.LogHealthz,
		healthz.NamedCheck("syncloop", syncLoop.SyncLoopHealthCheck),
	}
	retryBackoff := wait.Backoff{
		Duration: time.Second,
		Factor:   2.0,
		Jitter:   0.1,
		Steps:    2,
	}
	hc.checkers = append(hc.checkers, checkers...)
	hc.retryBackoff = retryBackoff
	hc.interval = watchdogVal / 2

	return hc, nil
}

func (hc *healthChecker) Start() {
	if hc.interval <= 0 {
		klog.InfoS("Systemd watchdog is not enabled or the interval is invalid, so health checking will not be started.")
		return
	}
	klog.InfoS("Starting systemd watchdog with interval", "interval", hc.interval)

	go wait.Forever(func() {
		if err := hc.doCheck(); err != nil {
			klog.ErrorS(err, "Do not notify watchdog this iteration as the kubelet is reportedly not healthy")
			return
		}

		err := wait.ExponentialBackoff(hc.retryBackoff, func() (bool, error) {
			ack, err := hc.watchdog.SdNotify(false)
			if err != nil {
				klog.V(5).InfoS("Failed to notify systemd watchdog, retrying", "error", err)
				return false, nil
			}
			if !ack {
				return false, fmt.Errorf("failed to notify systemd watchdog, notification not supported - (i.e. NOTIFY_SOCKET is unset)")
			}

			klog.V(5).InfoS("Watchdog plugin notified", "acknowledgment", ack, "state", daemon.SdNotifyWatchdog)
			return true, nil
		})
		if err != nil {
			klog.ErrorS(err, "Failed to notify watchdog")
		}
	}, hc.interval)
}

func (hc *healthChecker) doCheck() error {
	for _, hc := range hc.checkers {
		if err := hc.Check(nil); err != nil {
			return fmt.Errorf("checker %s failed: %w", hc.Name(), err)
		}
	}
	return nil
}
