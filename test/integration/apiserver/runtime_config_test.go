// +build integration,!no-etcd

/*
Copyright 2017 The Kubernetes Authors.

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

package apiserver

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/pborman/uuid"

	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/test/integration/framework"
)

// Starts the apiserver. It takes an index to start the apiserver on a different port each time.
// Returns the server IP and a channel to stop it.
func run(t *testing.T, runtimeConfig string, index int) (string, chan struct{}) {
	certDir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Failed to create temporary certificate directory: %v", err)
	}
	defer os.RemoveAll(certDir)

	s := options.NewServerRunOptions()
	_, serviceClusterIPRange, err := net.ParseCIDR("10.0.0.0/24")
	if err != nil {
		t.Fatalf("unexpected error in generation service cluster IP range: %v", err)
	}
	s.ServiceClusterIPRange = *serviceClusterIPRange
	securePort := 6450 + index
	insecurePort := 8090 + index
	serverIP := fmt.Sprintf("http://localhost:%v", insecurePort)
	s.SecureServing.ServingOptions.BindPort = securePort
	s.InsecureServing.BindPort = insecurePort
	s.Etcd.StorageConfig.ServerList = []string{framework.GetEtcdURLFromEnv()}
	// Use a unique prefix to ensure isolation from other tests using the same etcd instance
	s.Etcd.StorageConfig.Prefix = uuid.New()
	s.SecureServing.ServerCert.CertDirectory = certDir
	s.APIEnablement.RuntimeConfig.Set(runtimeConfig)

	stopCh := make(chan struct{})
	go func() {
		if err := app.Run(s, stopCh); err != nil {
			t.Fatalf("Error in bringing up the server: %v", err)
		}
	}()
	if err := waitForApiserverUp(serverIP); err != nil {
		t.Fatalf("%v", err)
	}
	return serverIP, stopCh
}

func waitForApiserverUp(serverIP string) error {
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		_, err := http.Get(serverIP)
		if err == nil {
			return nil
		}
	}
	return fmt.Errorf("waiting for apiserver timed out")
}

func corePath(resource, namespace, name string) string {
	return testapi.Default.ResourcePath(resource, namespace, name)
}
func extensionsPath(resource, namespace, name string) string {
	return testapi.Extensions.ResourcePath(resource, namespace, name)
}
func batchPath(resource, namespace, name string) string {
	return testapi.Batch.ResourcePath(resource, namespace, name)
}

// Tests that the apiserver enables/disables the API as per runtime-config.
func TestRuntimeConfig(t *testing.T) {
	testCases := []struct {
		runtimeConfig string
		statusCodes   map[string]int
	}{
		{
			// Verify that expected resources are enabled by default.
			"",
			map[string]int{
				corePath("pods", "default", ""):              200,
				extensionsPath("replicasets", "default", ""): 200,
				batchPath("jobs", "default", ""):             200,
			},
		},
		{
			// Setting all=true should enable all resources.
			"api/all=true",
			map[string]int{
				corePath("pods", "default", ""):              200,
				extensionsPath("replicasets", "default", ""): 200,
				batchPath("jobs", "default", ""):             200,
			},
		},
		{
			// Setting all=false should disable all resources.
			"api/all=false",
			map[string]int{
				corePath("pods", "default", ""):              404,
				extensionsPath("replicasets", "default", ""): 404,
				batchPath("jobs", "default", ""):             404,
			},
		},
		{
			// Should be able to disable all but one group version.
			"api/all=false,api/v1=true",
			map[string]int{
				corePath("pods", "default", ""):              200,
				extensionsPath("replicasets", "default", ""): 404,
				batchPath("jobs", "default", ""):             404,
			},
		},
		{
			// Should be able to disable a group version.
			"batch/v1=false",
			map[string]int{
				corePath("pods", "default", ""):              200,
				extensionsPath("replicasets", "default", ""): 200,
				batchPath("jobs", "default", ""):             404,
			},
		},
		{
			// Should be able to disable a specific resource.
			"extensions/v1beta1/ingresses=false",
			map[string]int{
				corePath("pods", "default", ""):              200,
				extensionsPath("replicasets", "default", ""): 200,
				extensionsPath("ingresses", "default", ""):   404,
				batchPath("jobs", "default", ""):             200,
			},
		},
		{
			// Disabling specific resources should not work with core group.
			"api/v1/secrets=false",
			map[string]int{
				corePath("pods", "default", ""):              200,
				corePath("secrets", "default", ""):           200,
				extensionsPath("replicasets", "default", ""): 200,
				batchPath("jobs", "default", ""):             200,
			},
		},
	}
	for index, test := range testCases {
		serverIP, stopCh := run(t, test.runtimeConfig, index)
		for path, code := range test.statusCodes {
			verifyStatusCode(t, "GET", serverIP+path, "", code)
		}
		close(stopCh)
	}
}
