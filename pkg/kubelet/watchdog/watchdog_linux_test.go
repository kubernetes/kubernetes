//go:build linux

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
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/kubernetes/test/utils/ktesting/initoption"
)

// Mock syncLoopHealthChecker
type mockSyncLoopHealthChecker struct {
	healthCheckErr error
}

func (m *mockSyncLoopHealthChecker) SyncLoopHealthCheck(req *http.Request) error {
	return m.healthCheckErr
}

// Mock WatchdogClient
type mockWatchdogClient struct {
	enabledVal time.Duration
	enabledErr error
	notifyAck  bool
	notifyErr  error
}

func (m *mockWatchdogClient) SdWatchdogEnabled(unsetEnvironment bool) (time.Duration, error) {
	return m.enabledVal, m.enabledErr
}

func (m *mockWatchdogClient) SdNotify(unsetEnvironment bool) (bool, error) {
	return m.notifyAck, m.notifyErr
}

const (
	interval      = 4 * time.Second
	intervalSmall = 1 * time.Second
)

// TestNewHealthChecker tests the NewHealthChecker function.
func TestNewHealthChecker(t *testing.T) {
	// Test cases
	tests := []struct {
		name        string
		mockEnabled time.Duration
		mockErr     error
		wantErr     bool
	}{
		{"Watchdog enabled", interval, nil, false},
		{"Watchdog not enabled", 0, nil, false},
		{"Watchdog enabled with error", interval, errors.New("mock error"), true},
		{"Watchdog timeout too small", intervalSmall, nil, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &mockWatchdogClient{
				enabledVal: tt.mockEnabled,
				enabledErr: tt.mockErr,
			}
			logger, _ := ktesting.NewTestContext(t)
			_, err := NewHealthChecker(logger, WithWatchdogClient(mockClient))
			if (err != nil) != tt.wantErr {
				t.Errorf("NewHealthChecker() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestHealthCheckerStart tests the Start method of the healthChecker.
func TestHealthCheckerStart(t *testing.T) {
	// Test cases
	tests := []struct {
		name           string
		enabledVal     time.Duration
		noCheckers     bool
		healthCheckErr error
		notifyAck      bool
		notifyErr      error
		expectedLogs   []string
	}{
		{
			name:           "Watchdog enabled and notify succeeds",
			enabledVal:     interval,
			healthCheckErr: nil,
			notifyAck:      true,
			notifyErr:      nil,
			expectedLogs:   []string{"Starting systemd watchdog with interval", "Watchdog plugin notified"},
		},
		{
			name:           "Watchdog enabled and notify fails, notification not supported",
			enabledVal:     interval,
			healthCheckErr: nil,
			notifyAck:      false,
			notifyErr:      nil,
			expectedLogs:   []string{"Starting systemd watchdog with interval", "Failed to notify watchdog", "notification not supported"},
		},
		{
			name:           "Watchdog enabled and notify fails, transmission failed",
			enabledVal:     interval,
			healthCheckErr: nil,
			notifyAck:      false,
			notifyErr:      errors.New("mock notify error"),
			expectedLogs:   []string{"Starting systemd watchdog with interval", "Failed to notify watchdog"},
		},
		{
			name:           "Watchdog enabled and health check fails",
			enabledVal:     interval,
			healthCheckErr: errors.New("mock healthy error"),
			notifyAck:      true,
			notifyErr:      nil,
			expectedLogs:   []string{"Starting systemd watchdog with interval", "Do not notify watchdog this iteration as the kubelet is reportedly not healthy"},
		},
		{
			name:       "Watchdog enabled and no checkers added, notify succeeds",
			enabledVal: interval,
			noCheckers: true,
			notifyAck:  true,
			notifyErr:  nil,
			expectedLogs: []string{
				"Starting systemd watchdog with interval", "Watchdog plugin notified"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tCtx := ktesting.Init(t, initoption.BufferLogs(true))
			logger := tCtx.Logger()
			defer func() {
				tCtx.Cancel("test has completed")
			}()

			// Mock SdWatchdogEnabled to return a valid value
			mockClient := &mockWatchdogClient{
				enabledVal: tt.enabledVal,
				notifyAck:  tt.notifyAck,
				notifyErr:  tt.notifyErr,
			}

			// Create a healthChecker
			hc, err := NewHealthChecker(logger, WithWatchdogClient(mockClient))
			if err != nil {
				t.Fatalf("NewHealthChecker() failed: %v", err)
			}
			if !tt.noCheckers {
				hc.SetHealthCheckers(&mockSyncLoopHealthChecker{healthCheckErr: tt.healthCheckErr}, nil)
			}

			// Start the health checker
			hc.Start(tCtx)

			// Wait for a short period to allow the health check to run
			time.Sleep(2 * interval)

			// Check logs to verify the health check ran
			underlier, ok := logger.GetSink().(ktesting.Underlier)
			if !ok {
				t.Fatalf("Should have had a ktesting LogSink, got %T", logger.GetSink())
			}
			logs := underlier.GetBuffer().String()
			for _, expectedLog := range tt.expectedLogs {
				if !strings.Contains(logs, expectedLog) {
					t.Errorf("Expected log '%s' not found in logs: %s", expectedLog, logs)
				}
			}
		})
	}
}
