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
	"context"
	"fmt"
	"sync/atomic"
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

type healthChecker struct {
	checkers     atomic.Value
	retryBackoff wait.Backoff
	interval     time.Duration
	watchdog     WatchdogClient
}

var _ HealthChecker = &healthChecker{}

const minimalNotifyInterval = time.Second

// NewHealthChecker creates a new HealthChecker instance.
// This function initializes the health checker and configures its behavior based on the status of the systemd watchdog.
// If the watchdog is not enabled, the function returns an error.
func NewHealthChecker(logger klog.Logger, opts ...Option) (HealthChecker, error) {
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
		logger.Info("Systemd watchdog is not enabled")
		return &healthChecker{}, nil
	}
	if watchdogVal <= minimalNotifyInterval {
		return nil, fmt.Errorf("configure watchdog timeout too small: %v", watchdogVal)
	}

	retryBackoff := wait.Backoff{
		Duration: time.Second,
		Factor:   2.0,
		Jitter:   0.1,
		Steps:    2,
	}
	hc.retryBackoff = retryBackoff
	hc.interval = watchdogVal / 2

	return hc, nil
}

func (hc *healthChecker) SetHealthCheckers(syncLoop syncLoopHealthChecker, checkers []healthz.HealthChecker) {
	// Define the default set of health checkers that should always be present
	defaultCheckers := []healthz.HealthChecker{
		healthz.PingHealthz,
		healthz.LogHealthz,
		healthz.NamedCheck("syncloop", syncLoop.SyncLoopHealthCheck),
	}

	var combined []healthz.HealthChecker
	combined = append(combined, defaultCheckers...)
	combined = append(combined, checkers...)
	hc.checkers.Store(combined)
}

func (hc *healthChecker) getHealthCheckers() []healthz.HealthChecker {
	if v := hc.checkers.Load(); v != nil {
		return v.([]healthz.HealthChecker)
	}
	return []healthz.HealthChecker{}
}

func (hc *healthChecker) Start(ctx context.Context) {
	logger := klog.FromContext(ctx)
	if hc.interval <= 0 {
		logger.Info("Systemd watchdog is not enabled or the interval is invalid, so health checking will not be started.")
		return
	}
	logger.Info("Starting systemd watchdog with interval", "interval", hc.interval)

	go wait.UntilWithContext(ctx, func(ctx context.Context) {
		if err := hc.doCheck(ctx); err != nil {
			logger.Error(err, "Do not notify watchdog this iteration as the kubelet is reportedly not healthy")
			return
		}

		// Use a timeout for SdNotify to prevent blocking if systemd socket is unresponsive
		notifyTimeout := hc.interval / 2
		err := wait.ExponentialBackoffWithContext(ctx, hc.retryBackoff, func(_ context.Context) (bool, error) {
			notifyCtx, cancel := context.WithTimeout(ctx, notifyTimeout)
			defer cancel()

			type notifyResult struct {
				ack bool
				err error
			}
			resultCh := make(chan notifyResult, 1)

			go func() {
				ack, err := hc.watchdog.SdNotify(false)
				resultCh <- notifyResult{ack: ack, err: err}
			}()

			select {
			case result := <-resultCh:
				if result.err != nil {
					logger.Error(result.err, "Failed to notify systemd watchdog, retrying")
					return false, nil
				}
				if !result.ack {
					return false, fmt.Errorf("failed to notify systemd watchdog, notification not supported - (i.e. NOTIFY_SOCKET is unset)")
				}
				logger.V(5).Info("Watchdog plugin notified", "acknowledgment", result.ack, "state", daemon.SdNotifyWatchdog)
				return true, nil
			case <-notifyCtx.Done():
				logger.Error(nil, "SdNotify timed out waiting for systemd socket response", "timeout", notifyTimeout)
				return false, nil
			}
		})
		if err != nil {
			logger.Error(err, "Failed to notify watchdog")
		}
	}, hc.interval)
}

func (hc *healthChecker) doCheck(ctx context.Context) error {
	// Use a fraction of the interval as timeout for each check
	// This ensures all checks complete well before the watchdog interval
	checkTimeout := hc.interval / 4
	logger := klog.FromContext(ctx)

	for _, checker := range hc.getHealthCheckers() {
		checkCtx, cancel := context.WithTimeout(ctx, checkTimeout)
		errCh := make(chan error, 1)

		go func(c healthz.HealthChecker) {
			errCh <- c.Check(nil)
		}(checker)

		select {
		case err := <-errCh:
			cancel()
			if err != nil {
				return fmt.Errorf("checker %s failed: %w", checker.Name(), err)
			}
		case <-checkCtx.Done():
			cancel()
			logger.Error(nil, "Health check timed out", "checker", checker.Name(), "timeout", checkTimeout)
			return fmt.Errorf("checker %s timed out after %v", checker.Name(), checkTimeout)
		}
	}
	return nil
}
