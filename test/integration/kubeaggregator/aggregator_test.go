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

package kubeaggregator

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
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
		t.Errorf("unexpected status code for %q: %v, expected: %v", path, response.StatusCode, expectedStatusCode)
	}
}

func runAPIServer(t *testing.T, stopCh <-chan struct{}) string {
	serverRunOptions := apiserver.NewServerRunOptions()
	// Change the ports, because otherwise it will fail if examples/apiserver/apiserver_test and this are run in parallel.
	serverRunOptions.SecureServing.ServingOptions.BindPort = 6443 + 3
	serverRunOptions.InsecureServing.BindPort = 8080 + 3

	// Avoid default cert-dir of /var/run/kubernetes to allow this to run on darwin
	certDir, _ := ioutil.TempDir("", "test-integration-kubeaggregator")
	defer os.Remove(certDir)
	serverRunOptions.SecureServing.ServerCert.CertDirectory = certDir

	go func() {
		if err := serverRunOptions.Run(stopCh); err != nil {
			t.Fatalf("Error in bringing up the example apiserver: %v", err)
		}
	}()

	serverURL := fmt.Sprintf("http://localhost:%d", serverRunOptions.InsecureServing.BindPort)
	if err := waitForServerUp(serverURL); err != nil {
		t.Fatalf("%v", err)
	}
	return serverURL
}

// Runs a discovery summarizer server and tests that all endpoints work as expected.
func TestRunKubeAggregator(t *testing.T) {
	// Run the APIServer now to test the good case.
	stopCh := make(chan struct{})
	discoveryURL := runAPIServer(t, stopCh)
	defer close(stopCh)

	// Test /api path.
	// There is no server running at that URL, so we will get a 500.
	testResponse(t, discoveryURL, "/api", http.StatusNotFound)

	// Test /apis path.
	// There is no server running at that URL, so we will get a 500.
	testResponse(t, discoveryURL, "/apis", http.StatusOK)

	// Test a random path, which should give a 404.
	testResponse(t, discoveryURL, "/randomPath", http.StatusNotFound)
}
