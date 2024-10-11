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

type healthChecker struct {
	checkers     []healthz.HealthChecker
	retryBackoff wait.Backoff
	interval     time.Duration
}

var _ HealthChecker = &healthChecker{}

// NewHealthChecker creates a new HealthChecker instance.
// This function initializes the health checker and configures its behavior based on the status of the systemd watchdog.
// If the watchdog is not enabled, the function returns an error.
func NewHealthChecker(syncLoop syncLoopHealthChecker) (HealthChecker, error) {
	// get watchdog information
	watchdogVal, err := daemon.SdWatchdogEnabled(false)
	if err != nil {
		// Failed to get watchdog configuration information.
		// This occurs when we want to start the watchdog but the configuration is incorrect,
		// for example, the time is not configured correctly.
		return nil, fmt.Errorf("configure watchdog: %w", err)
	}
	if watchdogVal == 0 {
		klog.InfoS("Systemd watchdog is not enabled")
		return nil, nil
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

	return &healthChecker{
		checkers:     checkers,
		retryBackoff: retryBackoff,
		interval:     watchdogVal / 2,
	}, nil
}

func (hc *healthChecker) Start() {
	klog.InfoS("Starting systemd watchdog with interval", "interval", hc.interval)

	go wait.Forever(func() {
		if err := hc.doCheck(); err != nil {
			klog.ErrorS(err, "Do not notify watchdog this iteration as the kubelet is reportedly not healthy")
			return
		}

		err := wait.ExponentialBackoff(hc.retryBackoff, func() (bool, error) {
			ack, err := daemon.SdNotify(false, daemon.SdNotifyWatchdog)
			if err != nil {
				klog.V(2).InfoS("Operation failed, retrying", "error", err)
				return false, nil
			}
			klog.V(5).InfoS("Watchdog plugin notified", "acknowledgment", ack, "state", daemon.SdNotifyWatchdog)
			return true, nil
		})
		if err != nil {
			klog.ErrorS(err, "Failed to notify watchdog", "retries", hc.retryBackoff.Steps)
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
