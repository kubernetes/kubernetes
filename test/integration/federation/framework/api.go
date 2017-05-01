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
	"net/http"
	"testing"

	"github.com/pborman/uuid"

	"k8s.io/apimachinery/pkg/util/wait"
	restclient "k8s.io/client-go/rest"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/cmd/federation-apiserver/app"
	"k8s.io/kubernetes/federation/cmd/federation-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
)

const apiNoun = "federation apiserver"

// GetRunOptions returns the default run options that can be used to run a test federation apiserver.
func GetRunOptions() *options.ServerRunOptions {
	r := options.NewServerRunOptions()
	r.Etcd.StorageConfig.ServerList = []string{framework.GetEtcdURLFromEnv()}
	// Use a unique prefix to ensure isolation from other tests using the same etcd instance
	r.Etcd.StorageConfig.Prefix = uuid.New()
	// Disable secure serving
	r.SecureServing.BindPort = 0
	return r
}

// FederationAPIFixture manages a federation api server
type FederationAPIFixture struct {
	Host     string
	stopChan chan struct{}
}

// SetUp runs federation apiserver with default run options.
func (f *FederationAPIFixture) SetUp(t *testing.T) {
	f.SetUpWithRunOptions(t, GetRunOptions())
}

// SetUpWithRunOptions runs federation apiserver with the given run options.
// Uses default run options if runOptions is nil.
func (f *FederationAPIFixture) SetUpWithRunOptions(t *testing.T, runOptions *options.ServerRunOptions) {
	if f.stopChan != nil {
		t.Fatal("SetUp() already called")
	}
	defer TearDownOnPanic(t, f)

	f.stopChan = make(chan struct{})

	err := startServer(t, runOptions, f.stopChan)
	if err != nil {
		t.Fatal(err)
	}

	f.Host = fmt.Sprintf("http://%s:%d", runOptions.InsecureServing.BindAddress, runOptions.InsecureServing.BindPort)

	err = waitForServer(t, f.Host)
	if err != nil {
		t.Fatal(err)
	}
}

func (f *FederationAPIFixture) TearDown(t *testing.T) {
	if f.stopChan != nil {
		close(f.stopChan)
		f.stopChan = nil
	}
}

func (f *FederationAPIFixture) NewConfig() *restclient.Config {
	return &restclient.Config{Host: f.Host}
}

func (f *FederationAPIFixture) NewClient(userAgent string) federationclientset.Interface {
	config := f.NewConfig()
	restclient.AddUserAgent(config, userAgent)
	return federationclientset.NewForConfigOrDie(config)
}

func startServer(t *testing.T, runOptions *options.ServerRunOptions, stopChan <-chan struct{}) error {
	err := wait.PollImmediate(DefaultWaitInterval, wait.ForeverTestTimeout, func() (bool, error) {
		port, err := framework.FindFreeLocalPort()
		if err != nil {
			t.Logf("Error allocating an ephemeral port: %v", err)
			return false, nil
		}

		runOptions.InsecureServing.BindPort = port
		err = app.NonBlockingRun(runOptions, stopChan)
		if err != nil {
			t.Logf("Error starting the %s: %v", apiNoun, err)
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		return fmt.Errorf("Timed out waiting for the %s: %v", apiNoun, err)
	}
	return nil
}

func waitForServer(t *testing.T, host string) error {
	err := wait.PollImmediate(DefaultWaitInterval, wait.ForeverTestTimeout, func() (bool, error) {
		_, err := http.Get(host)
		if err != nil {
			t.Logf("Error when trying to contact the API: %v", err)
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		return fmt.Errorf("Timed out waiting for the %s: %v", apiNoun, err)
	}
	return nil
}
