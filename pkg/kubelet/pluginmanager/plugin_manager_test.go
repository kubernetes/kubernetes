/*
Copyright 2019 The Kubernetes Authors.

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

package pluginmanager

import (
	"fmt"
	"io/ioutil"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/record"
	registerapi "k8s.io/kubelet/pkg/apis/pluginregistration/v1"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/pluginwatcher"
)

var (
	socketDir         string
	supportedVersions = []string{"v1beta1", "v1beta2"}
)

type fakePluginHandler struct {
	validatePluginCalled   bool
	registerPluginCalled   bool
	deregisterPluginCalled bool
	sync.RWMutex
}

func newFakePluginHandler() *fakePluginHandler {
	return &fakePluginHandler{
		validatePluginCalled:   false,
		registerPluginCalled:   false,
		deregisterPluginCalled: false,
	}
}

// ValidatePlugin is a fake method
func (f *fakePluginHandler) ValidatePlugin(pluginName string, endpoint string, versions []string) error {
	f.Lock()
	defer f.Unlock()
	f.validatePluginCalled = true
	return nil
}

// RegisterPlugin is a fake method
func (f *fakePluginHandler) RegisterPlugin(pluginName, endpoint string, versions []string) error {
	f.Lock()
	defer f.Unlock()
	f.registerPluginCalled = true
	return nil
}

// DeRegisterPlugin is a fake method
func (f *fakePluginHandler) DeRegisterPlugin(pluginName string) {
	f.Lock()
	defer f.Unlock()
	f.deregisterPluginCalled = true
}

func init() {
	d, err := ioutil.TempDir("", "plugin_manager_test")
	if err != nil {
		panic(fmt.Sprintf("Could not create a temp directory: %s", d))
	}

	socketDir = d
}

func cleanup(t *testing.T) {
	require.NoError(t, os.RemoveAll(socketDir))
	os.MkdirAll(socketDir, 0755)
}

func waitForRegistration(t *testing.T, fakePluginHandler *fakePluginHandler) {
	err := retryWithExponentialBackOff(
		time.Duration(500*time.Millisecond),
		func() (bool, error) {
			fakePluginHandler.Lock()
			defer fakePluginHandler.Unlock()
			if fakePluginHandler.validatePluginCalled && fakePluginHandler.registerPluginCalled {
				return true, nil
			}
			return false, nil
		},
	)
	if err != nil {
		t.Fatalf("Timed out waiting for plugin to be added to actual state of world cache.")
	}
}

func retryWithExponentialBackOff(initialDuration time.Duration, fn wait.ConditionFunc) error {
	backoff := wait.Backoff{
		Duration: initialDuration,
		Factor:   3,
		Jitter:   0,
		Steps:    6,
	}
	return wait.ExponentialBackoff(backoff, fn)
}

func TestPluginRegistration(t *testing.T) {
	defer cleanup(t)

	pluginManager := newTestPluginManager(socketDir)

	// Start the plugin manager
	stopChan := make(chan struct{})
	defer close(stopChan)
	go func() {
		sourcesReady := config.NewSourcesReady(func(_ sets.String) bool { return true })
		pluginManager.Run(sourcesReady, stopChan)
	}()

	// Add handler for device plugin
	fakeHandler := newFakePluginHandler()
	pluginManager.AddHandler(registerapi.DevicePlugin, fakeHandler)

	// Add a new plugin
	socketPath := fmt.Sprintf("%s/plugin.sock", socketDir)
	pluginName := "example-plugin"
	p := pluginwatcher.NewTestExamplePlugin(pluginName, registerapi.DevicePlugin, socketPath, supportedVersions...)
	require.NoError(t, p.Serve("v1beta1", "v1beta2"))

	// Verify that the plugin is registered
	waitForRegistration(t, fakeHandler)
}

func newTestPluginManager(
	sockDir string) PluginManager {

	pm := NewPluginManager(
		sockDir,
		&record.FakeRecorder{},
	)
	return pm
}
