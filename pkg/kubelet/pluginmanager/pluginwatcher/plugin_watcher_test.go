/*
Copyright 2018 The Kubernetes Authors.

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

package pluginwatcher

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog"
	registerapi "k8s.io/kubernetes/pkg/kubelet/apis/pluginregistration/v1"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
)

var (
	socketDir           string
	deprecatedSocketDir string

	supportedVersions = []string{"v1beta1", "v1beta2"}
)

func init() {
	var logLevel string

	klog.InitFlags(flag.CommandLine)
	flag.Set("alsologtostderr", fmt.Sprintf("%t", true))
	flag.StringVar(&logLevel, "logLevel", "6", "test")
	flag.Lookup("v").Value.Set(logLevel)

	d, err := ioutil.TempDir("", "plugin_test")
	if err != nil {
		panic(fmt.Sprintf("Could not create a temp directory: %s", d))
	}

	d2, err := ioutil.TempDir("", "deprecated_plugin_test")
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

func waitForRegistration(
	t *testing.T,
	socketPath string,
	dsw cache.DesiredStateOfWorld) {
	err := retryWithExponentialBackOff(
		time.Duration(500*time.Millisecond),
		func() (bool, error) {
			if dsw.PluginExists(socketPath) {
				return true, nil
			}
			return false, nil
		},
	)
	if err != nil {
		t.Fatalf("Timed out waiting for plugin to be added to desired state of world cache:\n%s.", socketPath)
	}
}

func waitForUnregistration(
	t *testing.T,
	socketPath string,
	dsw cache.DesiredStateOfWorld) {
	err := retryWithExponentialBackOff(
		time.Duration(500*time.Millisecond),
		func() (bool, error) {
			if !dsw.PluginExists(socketPath) {
				return true, nil
			}
			return false, nil
		},
	)

	if err != nil {
		t.Fatalf("Timed out waiting for plugin to be unregistered:\n%s.", socketPath)
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

	dsw := cache.NewDesiredStateOfWorld()
	newWatcher(t, false /* testDeprecatedDir */, dsw, wait.NeverStop)

	for i := 0; i < 10; i++ {
		socketPath := fmt.Sprintf("%s/plugin-%d.sock", socketDir, i)
		pluginName := fmt.Sprintf("example-plugin-%d", i)

		p := NewTestExamplePlugin(pluginName, registerapi.DevicePlugin, socketPath, supportedVersions...)
		require.NoError(t, p.Serve("v1beta1", "v1beta2"))

		pluginInfo := GetPluginInfo(p, false /* testDeprecatedDir */)
		waitForRegistration(t, pluginInfo.SocketPath, dsw)

		// Check the desired state for plugins
		dswPlugins := dsw.GetPluginsToRegister()
		if len(dswPlugins) != 1 {
			t.Fatalf("TestPluginRegistration: desired state of world length should be 1 but it's %d", len(dswPlugins))
		}

		// Stop the plugin; the plugin should be removed from the desired state of world cache
		require.NoError(t, p.Stop())
		// The following doesn't work when running the unit tests locally: event.Op of plugin watcher won't pick up the delete event
		waitForUnregistration(t, pluginInfo.SocketPath, dsw)
		dswPlugins = dsw.GetPluginsToRegister()
		if len(dswPlugins) != 0 {
			t.Fatalf("TestPluginRegistration: desired state of world length should be 0 but it's %d", len(dswPlugins))
		}
	}
}

func TestPluginRegistrationDeprecated(t *testing.T) {
	defer cleanup(t)

	dsw := cache.NewDesiredStateOfWorld()
	newWatcher(t, true /* testDeprecatedDir */, dsw, wait.NeverStop)

	// Test plugins in deprecated dir
	for i := 0; i < 10; i++ {
		endpoint := fmt.Sprintf("%s/dep-plugin-%d.sock", deprecatedSocketDir, i)
		pluginName := fmt.Sprintf("dep-example-plugin-%d", i)

		p := NewTestExamplePlugin(pluginName, registerapi.DevicePlugin, endpoint, supportedVersions...)
		require.NoError(t, p.Serve("v1beta1", "v1beta2"))

		pluginInfo := GetPluginInfo(p, true /* testDeprecatedDir */)
		waitForRegistration(t, pluginInfo.SocketPath, dsw)

		// Check the desired state for plugins
		dswPlugins := dsw.GetPluginsToRegister()
		if len(dswPlugins) != i+1 {
			t.Fatalf("TestPluginRegistrationDeprecated: desired state of world length should be %d but it's %d", i+1, len(dswPlugins))
		}
	}
}

func TestPluginRegistrationSameName(t *testing.T) {
	defer cleanup(t)

	dsw := cache.NewDesiredStateOfWorld()
	newWatcher(t, false /* testDeprecatedDir */, dsw, wait.NeverStop)

	// Make 10 plugins with the same name and same type but different socket path;
	// all 10 should be in desired state of world cache
	pluginName := "dep-example-plugin"
	for i := 0; i < 10; i++ {
		socketPath := fmt.Sprintf("%s/plugin-%d.sock", socketDir, i)
		p := NewTestExamplePlugin(pluginName, registerapi.DevicePlugin, socketPath, supportedVersions...)
		require.NoError(t, p.Serve("v1beta1", "v1beta2"))

		pluginInfo := GetPluginInfo(p, false /* testDeprecatedDir */)
		waitForRegistration(t, pluginInfo.SocketPath, dsw)

		// Check the desired state for plugins
		dswPlugins := dsw.GetPluginsToRegister()
		if len(dswPlugins) != i+1 {
			t.Fatalf("TestPluginRegistrationSameName: desired state of world length should be %d but it's %d", i+1, len(dswPlugins))
		}
	}
}

func TestPluginReRegistration(t *testing.T) {
	defer cleanup(t)

	dsw := cache.NewDesiredStateOfWorld()
	newWatcher(t, false /* testDeprecatedDir */, dsw, wait.NeverStop)

	// Create a plugin first, we are then going to remove the plugin, update the plugin with a different name
	// and recreate it.
	socketPath := fmt.Sprintf("%s/plugin-reregistration.sock", socketDir)
	pluginName := "reregister-plugin"
	p := NewTestExamplePlugin(pluginName, registerapi.DevicePlugin, socketPath, supportedVersions...)
	require.NoError(t, p.Serve("v1beta1", "v1beta2"))
	pluginInfo := GetPluginInfo(p, false /* testDeprecatedDir */)
	lastTimestamp := time.Now()
	waitForRegistration(t, pluginInfo.SocketPath, dsw)

	// Remove this plugin, then recreate it again with a different name for 10 times
	// The updated plugin should be in the desired state of world cache
	for i := 0; i < 10; i++ {
		// Stop the plugin; the plugin should be removed from the desired state of world cache
		// The plugin removel doesn't work when running the unit tests locally: event.Op of plugin watcher won't pick up the delete event
		require.NoError(t, p.Stop())
		waitForUnregistration(t, pluginInfo.SocketPath, dsw)

		// Add the plugin again
		pluginName := fmt.Sprintf("dep-example-plugin-%d", i)
		p := NewTestExamplePlugin(pluginName, registerapi.DevicePlugin, socketPath, supportedVersions...)
		require.NoError(t, p.Serve("v1beta1", "v1beta2"))
		waitForRegistration(t, pluginInfo.SocketPath, dsw)

		// Check the dsw cache. The updated plugin should be the only plugin in it
		dswPlugins := dsw.GetPluginsToRegister()
		if len(dswPlugins) != 1 {
			t.Fatalf("TestPluginReRegistration: desired state of world length should be 1 but it's %d", len(dswPlugins))
		}
		if !dswPlugins[0].Timestamp.After(lastTimestamp) {
			t.Fatalf("TestPluginReRegistration: for plugin %s timestamp of plugin is not updated", pluginName)
		}
		lastTimestamp = dswPlugins[0].Timestamp
	}
}

func TestPluginRegistrationAtKubeletStart(t *testing.T) {
	defer cleanup(t)

	plugins := make([]*examplePlugin, 10)

	for i := 0; i < len(plugins); i++ {
		socketPath := fmt.Sprintf("%s/plugin-%d.sock", socketDir, i)
		pluginName := fmt.Sprintf("example-plugin-%d", i)

		p := NewTestExamplePlugin(pluginName, registerapi.DevicePlugin, socketPath, supportedVersions...)
		require.NoError(t, p.Serve("v1beta1", "v1beta2"))
		defer func(p *examplePlugin) {
			require.NoError(t, p.Stop())
		}(p)

		plugins[i] = p
	}

	dsw := cache.NewDesiredStateOfWorld()
	newWatcher(t, false /* testDeprecatedDir */, dsw, wait.NeverStop)

	var wg sync.WaitGroup
	for i := 0; i < len(plugins); i++ {
		wg.Add(1)
		go func(p *examplePlugin) {
			defer wg.Done()

			pluginInfo := GetPluginInfo(p, false /* testDeprecatedDir */)
			// Validate that the plugin is in the desired state cache
			waitForRegistration(t, pluginInfo.SocketPath, dsw)
		}(plugins[i])
	}

	c := make(chan struct{})
	go func() {
		defer close(c)
		wg.Wait()
	}()

	select {
	case <-c:
		return
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("Timeout while waiting for the plugin registration status")
	}
}

func waitForPluginRegistrationStatus(t *testing.T, statusChan chan registerapi.RegistrationStatus) bool {
	select {
	case status := <-statusChan:
		return status.PluginRegistered
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("Timed out while waiting for registration status")
	}
	return false
}

func waitForEvent(t *testing.T, expected examplePluginEvent, eventChan chan examplePluginEvent) bool {
	select {
	case event := <-eventChan:
		return event == expected
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("Timed out while waiting for registration status %v", expected)
	}

	return false
}

func newWatcher(t *testing.T, testDeprecatedDir bool, desiredStateOfWorldCache cache.DesiredStateOfWorld, stopCh <-chan struct{}) *Watcher {
	depSocketDir := ""
	if testDeprecatedDir {
		depSocketDir = deprecatedSocketDir
	}
	w := NewWatcher(socketDir, depSocketDir, desiredStateOfWorldCache)
	require.NoError(t, w.Start(stopCh))

	return w
}

func TestFoundInDeprecatedDir(t *testing.T) {
	testCases := []struct {
		sockDir                    string
		deprecatedSockDir          string
		socketPath                 string
		expectFoundInDeprecatedDir bool
	}{
		{
			sockDir:                    "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir:          "/var/lib/kubelet/plugins",
			socketPath:                 "/var/lib/kubelet/plugins_registry/mydriver.foo/csi.sock",
			expectFoundInDeprecatedDir: false,
		},
		{
			sockDir:                    "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir:          "/var/lib/kubelet/plugins",
			socketPath:                 "/var/lib/kubelet/plugins/mydriver.foo/csi.sock",
			expectFoundInDeprecatedDir: true,
		},
		{
			sockDir:                    "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir:          "/var/lib/kubelet/plugins",
			socketPath:                 "/var/lib/kubelet/plugins_registry",
			expectFoundInDeprecatedDir: false,
		},
		{
			sockDir:                    "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir:          "/var/lib/kubelet/plugins",
			socketPath:                 "/var/lib/kubelet/plugins",
			expectFoundInDeprecatedDir: true,
		},
		{
			sockDir:                    "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir:          "/var/lib/kubelet/plugins",
			socketPath:                 "/var/lib/kubelet/plugins/kubernetes.io",
			expectFoundInDeprecatedDir: true,
		},
		{
			sockDir:                    "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir:          "/var/lib/kubelet/plugins",
			socketPath:                 "/var/lib/kubelet/plugins/my.driver.com",
			expectFoundInDeprecatedDir: true,
		},
		{
			sockDir:                    "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir:          "/var/lib/kubelet/plugins",
			socketPath:                 "/var/lib/kubelet/plugins_registry",
			expectFoundInDeprecatedDir: false,
		},
		{
			sockDir:                    "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir:          "/var/lib/kubelet/plugins",
			socketPath:                 "/var/lib/kubelet/plugins_registry/kubernetes.io",
			expectFoundInDeprecatedDir: false,
		},
		{
			sockDir:                    "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir:          "/var/lib/kubelet/plugins",
			socketPath:                 "/var/lib/kubelet/plugins_registry/my.driver.com",
			expectFoundInDeprecatedDir: false,
		},
	}

	for _, tc := range testCases {
		// Arrange & Act
		watcher := NewWatcher(tc.sockDir, tc.deprecatedSockDir, cache.NewDesiredStateOfWorld())

		actualFoundInDeprecatedDir := watcher.foundInDeprecatedDir(tc.socketPath)

		// Assert
		if tc.expectFoundInDeprecatedDir != actualFoundInDeprecatedDir {
			t.Fatalf("expecting actualFoundInDeprecatedDir=%v, but got %v for testcase: %#v", tc.expectFoundInDeprecatedDir, actualFoundInDeprecatedDir, tc)
		}
	}
}

func TestContainsBlacklistedDir(t *testing.T) {
	testCases := []struct {
		sockDir           string
		deprecatedSockDir string
		path              string
		expected          bool
	}{
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins_registry/mydriver.foo/csi.sock",
			expected:          false,
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins/mydriver.foo/csi.sock",
			expected:          false,
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins_registry",
			expected:          false,
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins",
			expected:          false,
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins/kubernetes.io",
			expected:          true,
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins/kubernetes.io/csi.sock",
			expected:          true,
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins/kubernetes.io/my.plugin/csi.sock",
			expected:          true,
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins/kubernetes.io/",
			expected:          true,
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins/my.driver.com",
			expected:          false,
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins_registry",
			expected:          false,
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins_registry/kubernetes.io",
			expected:          false, // New (non-deprecated dir) has no blacklist
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins_registry/my.driver.com",
			expected:          false,
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins/my-kubernetes.io-plugin",
			expected:          false,
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins/my-kubernetes.io-plugin/csi.sock",
			expected:          false,
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins/kubernetes.io-plugin",
			expected:          false,
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins/kubernetes.io-plugin/csi.sock",
			expected:          false,
		},
		{
			sockDir:           "/var/lib/kubelet/plugins_registry",
			deprecatedSockDir: "/var/lib/kubelet/plugins",
			path:              "/var/lib/kubelet/plugins/kubernetes.io-plugin/",
			expected:          false,
		},
	}

	for _, tc := range testCases {
		// Arrange & Act
		watcher := NewWatcher(tc.sockDir, tc.deprecatedSockDir, cache.NewDesiredStateOfWorld())

		actual := watcher.containsBlacklistedDir(tc.path)

		// Assert
		if tc.expected != actual {
			t.Fatalf("expecting %v but got %v for testcase: %#v", tc.expected, actual, tc)
		}
	}
}
