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

package collectors

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"k8s.io/klog/v2"
)

func testServerAddress(t *testing.T, s *httptest.Server) string {
	t.Helper()
	u, err := url.Parse(s.URL)
	if err != nil {
		t.Fatalf("failed to parse test server URL: %v", err)
	}
	return u.Host
}

func TestCRIStatsHandler(t *testing.T) {
	criResponse := `# HELP container_cpu CPU usage
# TYPE container_cpu gauge
container_cpu{id="c1"} 42
`
	criSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if _, err := w.Write([]byte(criResponse)); err != nil {
			t.Errorf("failed to write CRI response: %v", err)
		}
	}))
	defer criSrv.Close()

	registryOutput := `# HELP machine_cpu_cores Number of cores
# TYPE machine_cpu_cores gauge
machine_cpu_cores 4
`
	registryHandler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if _, err := w.Write([]byte(registryOutput)); err != nil {
			t.Errorf("failed to write registry response: %v", err)
		}
	})

	handler := CRIStatsHandler(klog.Background(), registryHandler, testServerAddress(t, criSrv))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics/cadvisor", nil))

	body := rec.Body.String()
	if !strings.Contains(body, "machine_cpu_cores 4") {
		t.Fatalf("response missing registry metrics: %s", body)
	}
	if !strings.Contains(body, `container_cpu{id="c1"} 42`) {
		t.Fatalf("response missing CRI metrics: %s", body)
	}
}

func TestCRIStatsHandlerNon200Status(t *testing.T) {
	criSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
	}))
	defer criSrv.Close()

	registryOutput := `# HELP machine_cpu_cores Number of cores
# TYPE machine_cpu_cores gauge
machine_cpu_cores 4
`
	registryHandler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if _, err := w.Write([]byte(registryOutput)); err != nil {
			t.Errorf("failed to write registry response: %v", err)
		}
	})

	handler := CRIStatsHandler(klog.Background(), registryHandler, testServerAddress(t, criSrv))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics/cadvisor", nil))

	body := rec.Body.String()
	if !strings.Contains(body, "machine_cpu_cores 4") {
		t.Fatalf("registry metrics should still be served on CRI error: %s", body)
	}
	if strings.Contains(body, "Internal Server Error") {
		t.Fatalf("CRI error body should not be included in response: %s", body)
	}
}

func TestCRIStatsHandlerTimeout(t *testing.T) {
	criSrv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer criSrv.Close()

	registryOutput := `# HELP machine_cpu_cores Number of cores
# TYPE machine_cpu_cores gauge
machine_cpu_cores 4
`
	registryHandler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if _, err := w.Write([]byte(registryOutput)); err != nil {
			t.Errorf("failed to write registry response: %v", err)
		}
	})

	handler := CRIStatsHandler(klog.Background(), registryHandler, testServerAddress(t, criSrv))
	rec := httptest.NewRecorder()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics/cadvisor", nil).WithContext(ctx))

	body := rec.Body.String()
	if !strings.Contains(body, "machine_cpu_cores 4") {
		t.Fatalf("registry metrics should still be served on timeout: %s", body)
	}
}

func TestCRIStatsHandlerEmptyAddress(t *testing.T) {
	registryOutput := `# HELP machine_cpu_cores Number of cores
# TYPE machine_cpu_cores gauge
machine_cpu_cores 4
`
	registryHandler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if _, err := w.Write([]byte(registryOutput)); err != nil {
			t.Errorf("failed to write registry response: %v", err)
		}
	})

	handler := CRIStatsHandler(klog.Background(), registryHandler, "")

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/metrics/cadvisor", nil)
	handler.ServeHTTP(rec, req)

	body := rec.Body.String()
	if !strings.Contains(body, "machine_cpu_cores 4") {
		t.Fatalf("registry metrics should still be served when CRI address is empty: %s", body)
	}
}

func TestCRIStatsHandlerFetchError(t *testing.T) {
	registryOutput := `# HELP machine_cpu_cores Number of cores
# TYPE machine_cpu_cores gauge
machine_cpu_cores 4
`
	registryHandler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if _, err := w.Write([]byte(registryOutput)); err != nil {
			t.Errorf("failed to write registry response: %v", err)
		}
	})

	// Start and immediately close a server to get an address that nothing listens on.
	closedSrv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	closedAddr := testServerAddress(t, closedSrv)
	closedSrv.Close()

	handler := CRIStatsHandler(klog.Background(), registryHandler, closedAddr)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/metrics/cadvisor", nil))

	body := rec.Body.String()
	if !strings.Contains(body, "machine_cpu_cores 4") {
		t.Fatalf("registry metrics should still be served on fetch error: %s", body)
	}
}
