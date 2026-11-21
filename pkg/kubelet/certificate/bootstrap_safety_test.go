/*
Copyright The Kubernetes Authors.

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

package certificate

import (
	"context"
	"errors"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"k8s.io/client-go/rest"
	"k8s.io/klog/v2/textlogger"
)

type healthManager struct {
	stubManager
	healthy bool
}

func (m *healthManager) ServerHealthy() bool { return m.healthy }

func TestInitiallyBootstrappedClientRotatesAgain(t *testing.T) {
	f := newFixture(t, true, nil, 20*time.Millisecond)
	f.manager.cert.Store(newCertificate(t, 1))
	done := make(chan result, 1)
	go func() { done <- f.post(context.Background(), "/hold") }()
	if first := receive(t, f.received); first.serial != 1 {
		t.Fatalf("initial certificate not used: %+v", first)
	}
	f.manager.cert.Store(newCertificate(t, 2))
	if got := receive(t, done); got.err == nil {
		t.Fatal("later rotation did not close the old connection")
	}
	receive(t, f.canceled)
	if got := f.post(context.Background(), "/ok"); got.err != nil || got.status != http.StatusCreated {
		t.Fatalf("retry after later rotation failed: %+v", got)
	}
	if next := receive(t, f.received); next.serial != 2 {
		t.Fatalf("later certificate not used: %+v", next)
	}
}

func TestNoCertificateNoUnauthenticatedRequest(t *testing.T) {
	f := newFixture(t, true, nil, 20*time.Millisecond)
	for i := 0; i < 3; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
		got := f.post(ctx, "/ok")
		cancel()
		if !errors.Is(got.err, context.DeadlineExceeded) {
			t.Fatalf("missing certificate hid request timeout: %+v", got)
		}
	}
	select {
	case got := <-f.received:
		t.Fatalf("unauthenticated request reached server: %+v", got)
	default:
	}
	f.manager.cert.Store(newCertificate(t, 1))
	if got := f.post(context.Background(), "/ok"); got.err != nil || got.status != http.StatusCreated {
		t.Fatalf("transport did not recover after enrollment: %+v", got)
	}
}

func TestCertificateExitPolicy(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, scenario := range []struct {
		name        string
		wantExit    bool
		wantMessage string
	}{
		{"missing-healthy", true, "No valid client certificate is found and the server is responsive, exiting"},
		{"expired-healthy", true, "The currently active client certificate has expired and the server is responsive, exiting"},
		{"missing-unhealthy", false, "No valid client certificate is found but the server is not responsive"},
		{"expired-unhealthy", false, "The currently active client certificate has expired, but the server is not responsive"},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			command := exec.CommandContext(ctx, executable, "-test.run=^TestCertificateExitPolicyHelper$")
			command.Env = append(os.Environ(), "BOOTSTRAP_EXIT_SCENARIO="+scenario.name)
			output, err := command.CombinedOutput()
			if ctx.Err() != nil {
				t.Fatalf("exit-policy subprocess hung: %v\n%s", ctx.Err(), output)
			}
			if scenario.wantExit {
				var exitError *exec.ExitError
				if !errors.As(err, &exitError) || exitError.ExitCode() != 1 {
					t.Fatalf("expected process exit 1, got %v\n%s", err, output)
				}
			} else if err != nil {
				t.Fatalf("unhealthy server unexpectedly terminated process: %v\n%s", err, output)
			}
			if !strings.Contains(string(output), scenario.wantMessage) {
				t.Fatalf("expected explicit diagnostic %q:\n%s", scenario.wantMessage, output)
			}
		})
	}
}

func TestCertificateExitPolicyHelper(t *testing.T) {
	scenario := os.Getenv("BOOTSTRAP_EXIT_SCENARIO")
	if scenario == "" {
		return
	}
	manager := &healthManager{healthy: !strings.HasSuffix(scenario, "-unhealthy")}
	if strings.HasPrefix(scenario, "expired-") {
		cert := newCertificate(t, 1)
		cert.Leaf.NotAfter = time.Now().Add(-time.Minute)
		manager.cert.Store(cert)
	}
	stop := make(chan struct{})
	defer close(stop)
	// os.Exit bypasses the testing logger's buffered output.
	logger := textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(os.Stderr)))
	closeConnections, err := updateTransport(logger, stop, 5*time.Millisecond, &rest.Config{}, manager, 30*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	defer closeConnections()
	time.Sleep(100 * time.Millisecond)
	if manager.healthy {
		t.Fatal("healthy-server expiration/missing-certificate policy failed to terminate")
	}
}
