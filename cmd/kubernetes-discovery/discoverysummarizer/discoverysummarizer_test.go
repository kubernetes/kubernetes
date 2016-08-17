/*
Copyright 2016 The Kubernetes Authors.

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

package discoverysummarizer

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"k8s.io/kubernetes/examples/apiserver"
)

func waitForServerUp(serverURL string) error {
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		_, err := http.Get(serverURL)
		if err == nil {
			return nil
		}
	}
	return fmt.Errorf("waiting for server timed out")
}

func testResponse(t *testing.T, serverURL, path string, expectedStatusCode int) {
	response, err := http.Get(serverURL + path)
	if err != nil {
		t.Errorf("unexpected error in GET %s: %v", path, err)
	}
	if response.StatusCode != expectedStatusCode {
		t.Errorf("unexpected status code: %v, expected: %v", response.StatusCode, expectedStatusCode)
	}
}

func runDiscoverySummarizer(t *testing.T) string {
	configFilePath := "../config.json"
	port := "9090"
	serverURL := "http://localhost:" + port
	s, err := NewDiscoverySummarizer(configFilePath)
	if err != nil {
		t.Errorf("unexpected error: %v\n", err)
	}
	go func() {
		if err := s.Run(port); err != nil {
			t.Fatalf("error in bringing up the server: %v", err)
		}
	}()
	if err := waitForServerUp(serverURL); err != nil {
		t.Fatalf("%v", err)
	}
	return serverURL
}

func runAPIServer(t *testing.T) string {
	serverRunOptions := apiserver.NewServerRunOptions()
	// Change the port, because otherwise it will fail if examples/apiserver/apiserver_test and this are run in parallel.
	serverRunOptions.InsecurePort = 8083
	go func() {
		if err := apiserver.Run(serverRunOptions); err != nil {
			t.Fatalf("Error in bringing up the example apiserver: %v", err)
		}
	}()

	serverURL := fmt.Sprintf("http://localhost:%d", serverRunOptions.InsecurePort)
	if err := waitForServerUp(serverURL); err != nil {
		t.Fatalf("%v", err)
	}
	return serverURL
}

// Runs a discovery summarizer server and tests that all endpoints work as expected.
func TestRunDiscoverySummarizer(t *testing.T) {
	discoveryURL := runDiscoverySummarizer(t)

	// Test /api path.
	// There is no server running at that URL, so we will get a 500.
	testResponse(t, discoveryURL, "/api", http.StatusBadGateway)

	// Test /apis path.
	// There is no server running at that URL, so we will get a 500.
	testResponse(t, discoveryURL, "/apis", http.StatusBadGateway)

	// Test a random path, which should give a 404.
	testResponse(t, discoveryURL, "/randomPath", http.StatusNotFound)

	// Run the APIServer now to test the good case.
	runAPIServer(t)

	// Test /api path.
	// There is no server running at that URL, so we will get a 500.
	testResponse(t, discoveryURL, "/api", http.StatusOK)

	// Test /apis path.
	// There is no server running at that URL, so we will get a 500.
	testResponse(t, discoveryURL, "/apis", http.StatusOK)

	// Test a random path, which should give a 404.
	testResponse(t, discoveryURL, "/randomPath", http.StatusNotFound)
}
