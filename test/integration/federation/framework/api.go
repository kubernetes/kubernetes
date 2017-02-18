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

package framework

import (
	"fmt"
	"net"
	"net/http"
	"strconv"
	"testing"
	"time"

	"github.com/pborman/uuid"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/federation/cmd/federation-apiserver/app"
	"k8s.io/kubernetes/federation/cmd/federation-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
)

const apiNoun = "federation apiserver"

func getRunOptions() *options.ServerRunOptions {
	r := options.NewServerRunOptions()
	r.Etcd.StorageConfig.ServerList = []string{framework.GetEtcdURLFromEnv()}
	// Use a unique prefix to ensure isolation from other tests using the same etcd instance
	r.Etcd.StorageConfig.Prefix = uuid.New()
	// Disable secure serving
	r.SecureServing.ServingOptions.BindPort = 0
	// The insecure port will be set to an ephemeral port before starting the api
	r.InsecureServing.BindPort = 0
	return r
}

// FederationAPIFixture manages a federation api server
type FederationAPIFixture struct {
	Host     string
	stopChan chan struct{}
}

func (f *FederationAPIFixture) Setup(t *testing.T) {
	if f.stopChan != nil {
		t.Fatal("Setup() already called")
	}
	f.stopChan = make(chan struct{})

	runOptions := getRunOptions()

	maxAttempts := 3
	err := startServer(t, runOptions, f.stopChan, maxAttempts)
	if err != nil {
		t.Fatal(err)
	}

	f.Host = fmt.Sprintf("http://%s:%d", runOptions.InsecureServing.BindAddress, runOptions.InsecureServing.BindPort)

	err = waitForServer(t, f.Host)
	if err != nil {
		t.Fatal(err)
	}
}

func (f *FederationAPIFixture) Teardown(t *testing.T) {
	close(f.stopChan)
}

func startServer(t *testing.T, runOptions *options.ServerRunOptions, stopChan <-chan struct{}, maxAttempts int) error {
	// Try to bind to an ephemeral port, making multiple attempts if necessary.
	// It would be preferable to use httptest.NewServer to ensure a port was
	// bound in one step, but that will require refactoring the Run method to
	// not require the port be known before the server is started.
	attempts := 0
	for attempts < maxAttempts {
		port, err := getEphemeralPort(runOptions.InsecureServing.BindAddress)
		if err != nil {
			// Only count attempts to start the api
			t.Logf("Error allocating an ephemeral port: %v", err)
			continue
		}

		runOptions.InsecureServing.BindPort = port

		err = app.NonBlockingRun(runOptions, stopChan)
		if err == nil {
			// Server was successfully started
			return nil
		}

		t.Logf("Error starting the %s: %v", apiNoun, err)
		time.Sleep(100 * time.Millisecond)
		attempts++
	}
	return fmt.Errorf("Failed to start the %s after %d attempts", apiNoun, maxAttempts)
}

func getEphemeralPort(bindAddress net.IP) (int, error) {
	l, err := net.Listen("tcp", fmt.Sprintf("%s:0", bindAddress))
	if err != nil {
		return -1, err
	}
	defer l.Close()
	_, portStr, err := net.SplitHostPort(l.Addr().String())
	if err != nil {
		return -1, err
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return -1, err
	}
	return port, nil
}

func waitForServer(t *testing.T, host string) error {
	err := wait.PollImmediate(50*time.Millisecond, 60*time.Second, func() (bool, error) {
		_, err := http.Get(host)
		if err != nil {
			t.Logf("Error when trying to contact the API: %v", err)
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		return fmt.Errorf("Timed out waiting for the %s", apiNoun)
	}
	return nil
}
