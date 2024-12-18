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

package serving

import (
	"io"
	"net/http"
	"os"
	"strings"
	"testing"

	"k8s.io/klog/v2/ktesting"
	kubeschedulertesting "k8s.io/kubernetes/cmd/kube-scheduler/app/testing"
)

func TestHealthEndpoints(t *testing.T) {
	server, configStr, _, err := startTestAPIServer(t)
	if err != nil {
		t.Fatalf("Failed to start kube-apiserver server: %v", err)
	}
	defer server.TearDownFn()

	apiserverConfig, err := os.CreateTemp("", "kubeconfig")
	if err != nil {
		t.Fatalf("Failed to create config file: %v", err)
	}
	defer func() {
		_ = os.Remove(apiserverConfig.Name())
	}()
	if _, err = apiserverConfig.WriteString(configStr); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	brokenConfigStr := strings.ReplaceAll(configStr, "127.0.0.1", "127.0.0.2")
	brokenConfig, err := os.CreateTemp("", "kubeconfig")
	if err != nil {
		t.Fatalf("Failed to create config file: %v", err)
	}
	if _, err := brokenConfig.WriteString(brokenConfigStr); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}
	defer func() {
		_ = os.Remove(brokenConfig.Name())
	}()

	tests := []struct {
		name             string
		path             string
		useBrokenConfig  bool
		wantResponseCode int
	}{
		{
			"/healthz",
			"/healthz",
			false,
			http.StatusOK,
		},
		{
			"/livez",
			"/livez",
			false,
			http.StatusOK,
		},
		{
			"/livez with ping check",
			"/livez/ping",
			false,
			http.StatusOK,
		},
		{
			"/readyz",
			"/readyz",
			false,
			http.StatusOK,
		},
		{
			"/readyz with sched-handler-sync",
			"/readyz/sched-handler-sync",
			false,
			http.StatusOK,
		},
		{
			"/readyz with shutdown",
			"/readyz/shutdown",
			false,
			http.StatusOK,
		},
		{
			"/readyz with broken apiserver",
			"/readyz",
			true,
			http.StatusInternalServerError,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt := tt
			_, ctx := ktesting.NewTestContext(t)

			configFile := apiserverConfig.Name()
			if tt.useBrokenConfig {
				configFile = brokenConfig.Name()
			}
			result, err := kubeschedulertesting.StartTestServer(
				ctx,
				[]string{"--kubeconfig", configFile, "--leader-elect=false", "--authorization-always-allow-paths", tt.path})

			if err != nil {
				t.Fatalf("Failed to start kube-scheduler server: %v", err)
			}
			if result.TearDownFn != nil {
				defer result.TearDownFn()
			}

			client, base, err := clientAndURLFromTestServer(result)
			if err != nil {
				t.Fatalf("Failed to get client from test server: %v", err)
			}
			req, err := http.NewRequest("GET", base+tt.path, nil)
			if err != nil {
				t.Fatalf("failed to request: %v", err)
			}
			r, err := client.Do(req)
			if err != nil {
				t.Fatalf("failed to GET %s from component: %v", tt.path, err)
			}

			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatalf("failed to read response body: %v", err)
			}
			if err = r.Body.Close(); err != nil {
				t.Fatalf("failed to close response body: %v", err)
			}
			if got, expected := r.StatusCode, tt.wantResponseCode; got != expected {
				t.Fatalf("expected http %d at %s of component, got: %d %q", expected, tt.path, got, string(body))
			}
		})
	}
}
