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
	"bytes"
	"io"
	"net/http"
	"os"
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/zpages/features"
	"k8s.io/klog/v2/ktesting"
	kubeschedulertesting "k8s.io/kubernetes/cmd/kube-scheduler/app/testing"
)

func TestFlagzEndpoints(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ComponentFlagz, true)
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

	_, ctx := ktesting.NewTestContext(t)

	configFile := apiserverConfig.Name()
	testSchedulerServer, err := kubeschedulertesting.StartTestServer(
		ctx,
		[]string{"--kubeconfig", configFile, "--leader-elect=false", "--authorization-always-allow-paths", "/flagz"})

	if err != nil {
		t.Fatalf("Failed to start kube-scheduler server: %v", err)
	}
	if testSchedulerServer.TearDownFn != nil {
		defer testSchedulerServer.TearDownFn()
	}

	client, base, err := clientAndURLFromTestServer(testSchedulerServer)
	if err != nil {
		t.Fatalf("Failed to get client from test server: %v", err)
	}
	req, err := http.NewRequest(http.MethodGet, base+"/flagz", nil)
	if err != nil {
		t.Fatalf("failed to request: %v", err)
	}
	req.Header.Set("Accept", "text/plain")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("failed to GET %s from component: %v", "/flagz", err)
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			t.Fatalf("failed to close response body: %v", err)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("flagz/ should be healthy, got %v", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("failed to read response body: %v", err)
	}
	expectedHeader := `
kube-scheduler flags
Warning: This endpoint is not meant to be machine parseable, has no formatting compatibility guarantees and is for debugging purposes only.`
	if !bytes.HasPrefix(body, []byte(expectedHeader)) {
		t.Fatalf("Header mismatch!\nExpected:\n%s\n\nGot:\n%s", expectedHeader, string(body))
	}
}
