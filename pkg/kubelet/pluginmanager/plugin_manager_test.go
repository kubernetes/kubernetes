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
	pluginwatcherapi "k8s.io/kubernetes/pkg/kubelet/apis/pluginregistration/v1"
	registerapi "k8s.io/kubernetes/pkg/kubelet/apis/pluginregistration/v1"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/pluginwatcher"
)

const (
	testHostname = "test-hostname"
)

var (
	socketDir           string
	deprecatedSocketDir string
	supportedVersions   = []string{"v1beta1", "v1beta2"}
)

// fake cache.PluginHandler
type PluginHandler interface {
	ValidatePlugin(pluginName string, endpoint string, versions []string, foundInDeprecatedDir bool) error
	RegisterPlugin(pluginName, endpoint string, versions []string) error
	DeRegisterPlugin(pluginName string)
}

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
func (f *fakePluginHandler) ValidatePlugin(pluginName string, endpoint string, versions []string, foundInDeprecatedDir bool) error {
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
	return
}

func init() {
	d, err := ioutil.TempDir("", "plugin_manager_test")
	if err != nil {
		panic(fmt.Sprintf("Could not create a temp directory: %s", d))
	}

	d2, err := ioutil.TempDir("", "deprecateddir_plugin_manager_test")
	if err != nil {
		panic(fmt.Sprintf("Could not create a temp directory: %s", d))
	}

	socketDir = d
	deprecatedSocketDir = d2
}

func cleanup(t *testing.T) {
	require.NoError(t, os.RemoveAll(socketDir))
	require.NoError(t, os.RemoveAll(deprecatedSocketDir))
	os.MkdirAll(socketDir, 0755)
	os.MkdirAll(deprecatedSocketDir, 0755)
}

func newWatcher(
	t *testing.T, testDeprecatedDir bool,
	desiredStateOfWorldCache cache.DesiredStateOfWorld) *pluginwatcher.Watcher {

	depSocketDir := ""
	if testDeprecatedDir {
		depSocketDir = deprecatedSocketDir
	}
	w := pluginwatcher.NewWatcher(socketDir, depSocketDir, desiredStateOfWorldCache)
	require.NoError(t, w.Start(wait.NeverStop))

	return w
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

	pluginManager := newTestPluginManager(socketDir, deprecatedSocketDir)

	// Start the plugin manager
	stopChan := make(chan struct{})
	defer close(stopChan)
	go func() {
		sourcesReady := config.NewSourcesReady(func(_ sets.String) bool { return true })
		pluginManager.Run(sourcesReady, stopChan)
	}()

	// Add handler for device plugin
	fakeHandler := newFakePluginHandler()
	pluginManager.AddHandler(pluginwatcherapi.DevicePlugin, fakeHandler)

	// Add a new plugin
	socketPath := fmt.Sprintf("%s/plugin.sock", socketDir)
	pluginName := "example-plugin"
	p := pluginwatcher.NewTestExamplePlugin(pluginName, registerapi.DevicePlugin, socketPath, supportedVersions...)
	require.NoError(t, p.Serve("v1beta1", "v1beta2"))

	// Verify that the plugin is registered
	waitForRegistration(t, fakeHandler)
}

func newTestPluginManager(
	sockDir string,
	deprecatedSockDir string) PluginManager {

	pm := NewPluginManager(
		sockDir,
		deprecatedSockDir,
		&record.FakeRecorder{},
	)
	return pm
}
