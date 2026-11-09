//go:build linux

/*
Copyright 2026 The Kubernetes Authors.

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
	"net"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	klogtesting "k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/watchdog"
)

func TestSystemdWatchdogSendsHeartbeat(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletWatchdogDiagnostics, true)
	setWatchdogEnvironment(t)
	socketPath := filepath.Join(t.TempDir(), "notify.sock")
	listener, err := net.ListenUnixgram("unixgram", &net.UnixAddr{Name: socketPath, Net: "unixgram"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := listener.Close(); err != nil {
			t.Errorf("close watchdog notification socket: %v", err)
		}
	})
	t.Setenv("NOTIFY_SOCKET", socketPath)

	logger := klogtesting.NewLogger(t, klogtesting.NewConfig(klogtesting.BufferLogs(true), klogtesting.Verbosity(0)))
	healthChecker, err := watchdog.NewHealthChecker(logger)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(klog.NewContext(t.Context(), logger))
	defer cancel()
	healthChecker.Start(ctx)

	if err := listener.SetReadDeadline(time.Now().Add(3 * time.Second)); err != nil {
		t.Fatal(err)
	}
	message := make([]byte, 64)
	n, _, err := listener.ReadFromUnix(message)
	if err != nil {
		t.Fatalf("no systemd watchdog heartbeat received: %v", err)
	}
	if got := string(message[:n]); got != "WATCHDOG=1" {
		t.Errorf("watchdog notification = %q, want WATCHDOG=1", got)
	}
}

func TestSystemdWatchdogNotificationFailureDiagnostics(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletWatchdogDiagnostics, true)
	setWatchdogEnvironment(t)
	t.Setenv("NOTIFY_SOCKET", filepath.Join(t.TempDir(), "missing.sock"))

	logger := klogtesting.NewLogger(t, klogtesting.NewConfig(klogtesting.BufferLogs(true), klogtesting.Verbosity(0)))
	healthChecker, err := watchdog.NewHealthChecker(logger)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(klog.NewContext(t.Context(), logger))
	defer cancel()
	healthChecker.Start(ctx)

	buffer := logger.GetSink().(klogtesting.Underlier).GetBuffer()
	err = wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
		return strings.Contains(buffer.String(), "result=\"retry_exhausted\""), nil
	})
	if err != nil {
		t.Fatalf("no retry exhaustion diagnostic from real SdNotify failure: %v; logs=%s", err, buffer.String())
	}
	logs := buffer.String()
	for _, want := range []string{"operation=\"watchdog_notify\"", "attempt=1", "attempts=2", "result=\"retry_exhausted\"", "no such file or directory"} {
		if !strings.Contains(logs, want) {
			t.Errorf("expected %q in watchdog logs: %s", want, logs)
		}
	}
}

func setWatchdogEnvironment(t *testing.T) {
	t.Helper()
	t.Setenv("WATCHDOG_USEC", "1200000")
	t.Setenv("WATCHDOG_PID", "")
}
