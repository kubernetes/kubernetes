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

package fixture

import (
	"fmt"
	"net"
	"net/http"
	"strconv"
	"testing"
	"time"

	"github.com/pborman/uuid"

	"k8s.io/apimachinery/pkg/util/wait"
	restclient "k8s.io/client-go/rest"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/cmd/federation-apiserver/app"
	"k8s.io/kubernetes/federation/cmd/federation-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
)

const PortNotSet = 0

func getRunOptions() *options.ServerRunOptions {
	r := options.NewServerRunOptions()
	r.Etcd.StorageConfig.ServerList = []string{framework.GetEtcdURLFromEnv()}
	// Use a unique prefix to ensure isolation from other tests using the same etcd instance
	r.Etcd.StorageConfig.Prefix = uuid.New()
	// Disable secure serving
	r.SecureServing.ServingOptions.BindPort = 0
	// The insecure port will be set to an ephemeral port before starting the api
	r.InsecureServing.BindPort = PortNotSet
	return r
}

// FederationAPIFixture manages a federation api server
type FederationAPIFixture struct {
	Host     string
	Client   federationclientset.Interface
	stopChan chan struct{}
}

func (f *FederationAPIFixture) Setup(t *testing.T) {
	if len(f.Host) > 0 {
		t.Fatal("Setup() already called")
	}

	runOptions := getRunOptions()
	f.stopChan = make(chan struct{})

	// Try to bind to an ephemeral port, making multiple attempts if necessary.
	// It would be preferable to use httptest.NewServer to ensure a port was
	// bound in one step, but that will require refactoring the Run method to
	// not require the port be known before the server is started.
	go func() {
		attempts := 0
		maxAttempts := 3
		var err error
		for attempts < maxAttempts {
			port, err := getEphemeralPort(runOptions.InsecureServing.BindAddress)
			if err != nil {
				// Only count attempts to start the api
				t.Logf("Error allocating an ephemeral port: %v", err)
				continue
			}
			runOptions.InsecureServing.BindPort = port
			// If successful, RunWithChannel blocks until the channel is closed
			err = app.RunWithChannel(runOptions, f.stopChan)
			if err != nil {
				t.Logf("Error starting the federation api: %v", err)
				attempts++
			}
		}
		t.Fatalf("Error starting api server: %v", err)
	}()

	// Wait for the API to be available via http
	err := wait.PollImmediate(1*time.Second, 60*time.Second, func() (bool, error) {
		if runOptions.InsecureServing.BindPort == PortNotSet {
			return false, nil
		}
		// Synchronization with the goroutine's modification of BindPort
		// shouldn't be necessary since this method will catch up to the new
		// value eventually.
		f.Host = fmt.Sprintf("http://%s:%d", runOptions.InsecureServing.BindAddress, runOptions.InsecureServing.BindPort)
		_, err := http.Get(f.Host)
		if err != nil {
			t.Logf("Error when trying to contact the API: %v", err)
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Timed out waiting for federation API")
	}

	f.Client = federationclientset.NewForConfigOrDie(f.NewRestConfig("test"))
}

func (f *FederationAPIFixture) Teardown(t *testing.T) {
	close(f.stopChan)
}

func (f *FederationAPIFixture) NewRestConfig(userAgent string) *restclient.Config {
	config := &restclient.Config{Host: f.Host}
	restclient.AddUserAgent(config, userAgent)
	return config
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
