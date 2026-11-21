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
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/healthz"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	klogtesting "k8s.io/klog/v2/ktesting"
	"k8s.io/ktesting"
	"k8s.io/ktesting/initoption"
	"k8s.io/kubernetes/pkg/features"
)

func TestWatchdogDiagnosticsCheckerError(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletWatchdogDiagnostics, true)
	logger := klogtesting.NewLogger(t, klogtesting.NewConfig(klogtesting.BufferLogs(true), klogtesting.Verbosity(0)))
	ctx := klog.NewContext(context.Background(), logger)
	client := &mockWatchdogClient{notifyAck: true}
	hc := &healthChecker{watchdog: client, interval: time.Second, retryBackoff: wait.Backoff{Steps: 2}}
	hc.checkers.Store([]healthz.HealthChecker{healthz.NamedCheck("broken-checker", func(*http.Request) error {
		return errors.New("checker failed")
	})})

	hc.runOnce(ctx)

	logs := logger.GetSink().(klogtesting.Underlier).GetBuffer().String()
	for _, want := range []string{"watchdog_health_check", "broken-checker", "checker failed", "result=\"error\""} {
		if !strings.Contains(logs, want) {
			t.Errorf("expected %q in logs: %s", want, logs)
		}
	}
}

// Some of these tests capture log output. Don't reduce the verbosity or they will fail!
func init() {
	ktesting.SetDefaultVerbosity(5)
}

// Mock syncLoopHealthChecker
type mockSyncLoopHealthChecker struct {
	healthCheckErr error
}

func (m *mockSyncLoopHealthChecker) SyncLoopHealthCheck(req *http.Request) error {
	return m.healthCheckErr
}

// Mock WatchdogClient
type mockWatchdogClient struct {
	enabledVal   time.Duration
	enabledErr   error
	notifyAck    bool
	notifyErr    error
	notifyCalls  int
	notifyAcks   []bool
	notifyErrors []error
}

func (m *mockWatchdogClient) SdWatchdogEnabled(unsetEnvironment bool) (time.Duration, error) {
	return m.enabledVal, m.enabledErr
}

func (m *mockWatchdogClient) SdNotify(unsetEnvironment bool) (bool, error) {
	m.notifyCalls++
	if m.notifyCalls <= len(m.notifyAcks) {
		return m.notifyAcks[m.notifyCalls-1], m.notifyErrors[m.notifyCalls-1]
	}
	return m.notifyAck, m.notifyErr
}

func TestWatchdogDiagnosticsNotify(t *testing.T) {
	tests := []struct {
		name               string
		diagnosticsEnabled bool
		client             mockWatchdogClient
		wantCalls          int
		wantLogs           []string
		unwantedLogs       []string
		wantSummaryCount   int
	}{
		{
			name:               "retry succeeds",
			diagnosticsEnabled: true,
			client: mockWatchdogClient{
				notifyAcks:   []bool{false, true},
				notifyErrors: []error{errors.New("temporary failure"), nil},
			},
			wantCalls: 2,
			wantLogs:  []string{"temporary failure"},
		},
		{
			name:               "notification unsupported",
			diagnosticsEnabled: true,
			wantCalls:          1,
			wantLogs:           []string{"notification not supported"},
		},
		{
			name:         "diagnostics disabled",
			client:       mockWatchdogClient{notifyErr: errors.New("socket failed")},
			wantCalls:    2,
			wantLogs:     []string{"Failed to notify watchdog"},
			unwantedLogs: []string{"operation="},
		},
		{
			name:               "retries exhausted",
			diagnosticsEnabled: true,
			client:             mockWatchdogClient{notifyErr: errors.New("socket failed")},
			wantCalls:          2,
			wantLogs:           []string{"operation=\"watchdog_notify\"", "attempt=1", "socket failed", "attempts=2"},
			wantSummaryCount:   1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletWatchdogDiagnostics, tt.diagnosticsEnabled)
			logger := klogtesting.NewLogger(t, klogtesting.NewConfig(klogtesting.BufferLogs(true), klogtesting.Verbosity(0)))
			client := tt.client
			hc := &healthChecker{watchdog: &client, interval: time.Second, retryBackoff: wait.Backoff{Steps: 2}}

			hc.runOnce(klog.NewContext(context.Background(), logger))

			if client.notifyCalls != tt.wantCalls {
				t.Errorf("SdNotify calls = %d, want %d", client.notifyCalls, tt.wantCalls)
			}
			logs := logger.GetSink().(klogtesting.Underlier).GetBuffer().String()
			for _, want := range tt.wantLogs {
				if !strings.Contains(logs, want) {
					t.Errorf("expected %q in logs: %s", want, logs)
				}
			}
			for _, unwanted := range tt.unwantedLogs {
				if strings.Contains(logs, unwanted) {
					t.Errorf("unexpected %q in logs: %s", unwanted, logs)
				}
			}
			if got := strings.Count(logs, "result=\"retry_exhausted\""); got != tt.wantSummaryCount {
				t.Errorf("retry summaries = %d, want %d: %s", got, tt.wantSummaryCount, logs)
			}
		})
	}
}

func TestWatchdogDiagnosticsGateToggle(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletWatchdogDiagnostics, true)
	logger := klogtesting.NewLogger(t, klogtesting.NewConfig(klogtesting.BufferLogs(true), klogtesting.Verbosity(0)))
	client := &mockWatchdogClient{notifyAck: true}
	hc := &healthChecker{watchdog: client, interval: time.Second}
	hc.checkers.Store([]healthz.HealthChecker{healthz.NamedCheck("broken-checker", func(*http.Request) error {
		return errors.New("checker failed")
	})})
	ctx := klog.NewContext(context.Background(), logger)

	hc.runOnce(ctx)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletWatchdogDiagnostics, false)
	hc.runOnce(ctx)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletWatchdogDiagnostics, true)
	hc.runOnce(ctx)

	logs := logger.GetSink().(klogtesting.Underlier).GetBuffer().String()
	if got := strings.Count(logs, "operation=\"watchdog_health_check\""); got != 2 {
		t.Errorf("enabled diagnostic records = %d, want 2: %s", got, logs)
	}
	if !strings.Contains(logs, "Do not notify watchdog this iteration") || client.notifyCalls != 0 {
		t.Errorf("gate-off behavior changed: calls=%d logs=%s", client.notifyCalls, logs)
	}
}

func TestWatchdogDiagnosticsCheckerRateLimit(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletWatchdogDiagnostics, true)
	logger := klogtesting.NewLogger(t, klogtesting.NewConfig(klogtesting.BufferLogs(true), klogtesting.Verbosity(0)))
	client := &mockWatchdogClient{notifyAck: true}
	hc := &healthChecker{watchdog: client, interval: time.Second}
	hc.checkers.Store([]healthz.HealthChecker{healthz.NamedCheck("broken-checker", func(*http.Request) error {
		return errors.New("checker failed")
	})})
	ctx := klog.NewContext(context.Background(), logger)

	hc.runOnce(ctx)
	hc.runOnce(ctx)

	logs := logger.GetSink().(klogtesting.Underlier).GetBuffer().String()
	if got := strings.Count(logs, "operation=\"watchdog_health_check\""); got != 1 {
		t.Errorf("checker diagnostics = %d, want 1: %s", got, logs)
	}
	if client.notifyCalls != 0 {
		t.Errorf("SdNotify calls = %d, want 0", client.notifyCalls)
	}
}

func TestWatchdogDiagnosticsCancellationDoesNotSummarize(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletWatchdogDiagnostics, true)
	logger := klogtesting.NewLogger(t, klogtesting.NewConfig(klogtesting.BufferLogs(true), klogtesting.Verbosity(0)))
	client := &mockWatchdogClient{notifyErr: errors.New("socket failed")}
	hc := &healthChecker{watchdog: client, interval: time.Second, retryBackoff: wait.Backoff{Steps: 2}}
	ctx, cancel := context.WithCancel(klog.NewContext(context.Background(), logger))
	cancel()

	hc.runOnce(ctx)

	logs := logger.GetSink().(klogtesting.Underlier).GetBuffer().String()
	if client.notifyCalls != 0 || strings.Contains(logs, "retry_exhausted") {
		t.Errorf("cancelled watchdog iteration produced misleading diagnostics: calls=%d logs=%s", client.notifyCalls, logs)
	}
}

func TestWatchdogDiagnosticsRateLimit(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletWatchdogDiagnostics, true)
	logger := klogtesting.NewLogger(t, klogtesting.NewConfig(klogtesting.BufferLogs(true), klogtesting.Verbosity(5)))
	client := &mockWatchdogClient{
		notifyAcks: []bool{false, false, false, false},
		notifyErrors: []error{
			errors.New("socket failed 1"), errors.New("socket failed 2"),
			errors.New("socket failed 3"), errors.New("socket failed 4"),
		},
	}
	hc := &healthChecker{watchdog: client, interval: time.Second, retryBackoff: wait.Backoff{Steps: 2}}
	ctx := klog.NewContext(context.Background(), logger)

	hc.runOnce(ctx)
	hc.runOnce(ctx)

	logs := logger.GetSink().(klogtesting.Underlier).GetBuffer().String()
	if got := strings.Count(logs, "Watchdog notification failed"); got != 1 {
		t.Errorf("notification error records = %d, want 1: %s", got, logs)
	}
	if got := strings.Count(logs, "Watchdog notification retries exhausted"); got != 1 {
		t.Errorf("retry summaries = %d, want 1: %s", got, logs)
	}
	if got := strings.Count(logs, "Failed to notify systemd watchdog, retrying"); got != 4 {
		t.Errorf("verbose retry records = %d, want 4: %s", got, logs)
	}
	for _, want := range []string{"socket failed 1", "socket failed 2", "socket failed 3", "socket failed 4"} {
		if !strings.Contains(logs, want) {
			t.Errorf("missing retry error %q: %s", want, logs)
		}
	}
	if client.notifyCalls != 4 {
		t.Errorf("SdNotify calls = %d, want 4", client.notifyCalls)
	}
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
	tCtx := ktesting.Init(t)

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
		tCtx.SyncTest(tt.name, func(tCtx ktesting.TContext) {
			tCtx = ktesting.Init(tCtx, initoption.BufferLogs(true))
			logger := tCtx.Logger()

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
				// Bypass SetHealthCheckers because it adds LogHealthz, which leaks a background
				// goroutine that doesn't play well with SyncTest.
				hc.(*healthChecker).checkers.Store([]healthz.HealthChecker{
					healthz.NamedCheck("syncloop", (&mockSyncLoopHealthChecker{healthCheckErr: tt.healthCheckErr}).SyncLoopHealthCheck),
				})
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
