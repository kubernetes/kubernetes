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

	"k8s.io/klog"
	registerapi "k8s.io/kubernetes/pkg/kubelet/apis/pluginregistration/v1"
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

func TestPluginRegistration(t *testing.T) {
	defer cleanup(t)

	hdlr := NewExampleHandler(supportedVersions, false /* permitDeprecatedDir */)
	w := newWatcherWithHandler(t, hdlr, false /* testDeprecatedDir */)
	defer func() { require.NoError(t, w.Stop()) }()

	for i := 0; i < 10; i++ {
		socketPath := fmt.Sprintf("%s/plugin-%d.sock", socketDir, i)
		pluginName := fmt.Sprintf("example-plugin-%d", i)

		hdlr.AddPluginName(pluginName)

		p := NewTestExamplePlugin(pluginName, registerapi.DevicePlugin, socketPath, supportedVersions...)
		require.NoError(t, p.Serve("v1beta1", "v1beta2"))

		require.True(t, waitForEvent(t, exampleEventValidate, hdlr.EventChan(p.pluginName)))
		require.True(t, waitForEvent(t, exampleEventRegister, hdlr.EventChan(p.pluginName)))

		require.True(t, waitForPluginRegistrationStatus(t, p.registrationStatus))

		require.NoError(t, p.Stop())
		require.True(t, waitForEvent(t, exampleEventDeRegister, hdlr.EventChan(p.pluginName)))
	}
}

func TestPluginRegistrationDeprecated(t *testing.T) {
	defer cleanup(t)

	hdlr := NewExampleHandler(supportedVersions, true /* permitDeprecatedDir */)
	w := newWatcherWithHandler(t, hdlr, true /* testDeprecatedDir */)
	defer func() { require.NoError(t, w.Stop()) }()

	// Test plugins in deprecated dir
	for i := 0; i < 10; i++ {
		endpoint := fmt.Sprintf("%s/dep-plugin-%d.sock", deprecatedSocketDir, i)
		pluginName := fmt.Sprintf("dep-example-plugin-%d", i)

		hdlr.AddPluginName(pluginName)

		p := NewTestExamplePlugin(pluginName, registerapi.DevicePlugin, endpoint, supportedVersions...)
		require.NoError(t, p.Serve("v1beta1", "v1beta2"))

		require.True(t, waitForEvent(t, exampleEventValidate, hdlr.EventChan(p.pluginName)))
		require.True(t, waitForEvent(t, exampleEventRegister, hdlr.EventChan(p.pluginName)))

		require.True(t, waitForPluginRegistrationStatus(t, p.registrationStatus))

		require.NoError(t, p.Stop())
		require.True(t, waitForEvent(t, exampleEventDeRegister, hdlr.EventChan(p.pluginName)))
	}
}

func TestPluginReRegistration(t *testing.T) {
	defer cleanup(t)

	pluginName := fmt.Sprintf("example-plugin")
	hdlr := NewExampleHandler(supportedVersions, false /* permitDeprecatedDir */)

	w := newWatcherWithHandler(t, hdlr, false /* testDeprecatedDir */)
	defer func() { require.NoError(t, w.Stop()) }()

	plugins := make([]*examplePlugin, 10)

	for i := 0; i < 10; i++ {
		socketPath := fmt.Sprintf("%s/plugin-%d.sock", socketDir, i)
		hdlr.AddPluginName(pluginName)

		p := NewTestExamplePlugin(pluginName, registerapi.DevicePlugin, socketPath, supportedVersions...)
		require.NoError(t, p.Serve("v1beta1", "v1beta2"))

		require.True(t, waitForEvent(t, exampleEventValidate, hdlr.EventChan(p.pluginName)))
		require.True(t, waitForEvent(t, exampleEventRegister, hdlr.EventChan(p.pluginName)))

		require.True(t, waitForPluginRegistrationStatus(t, p.registrationStatus))

		plugins[i] = p
	}

	plugins[len(plugins)-1].Stop()
	require.True(t, waitForEvent(t, exampleEventDeRegister, hdlr.EventChan(pluginName)))

	close(hdlr.EventChan(pluginName))
	for i := 0; i < len(plugins)-1; i++ {
		plugins[i].Stop()
	}
}

func TestPluginRegistrationAtKubeletStart(t *testing.T) {
	defer cleanup(t)

	hdlr := NewExampleHandler(supportedVersions, false /* permitDeprecatedDir */)
	plugins := make([]*examplePlugin, 10)

	for i := 0; i < len(plugins); i++ {
		socketPath := fmt.Sprintf("%s/plugin-%d.sock", socketDir, i)
		pluginName := fmt.Sprintf("example-plugin-%d", i)
		hdlr.AddPluginName(pluginName)

		p := NewTestExamplePlugin(pluginName, registerapi.DevicePlugin, socketPath, supportedVersions...)
		require.NoError(t, p.Serve("v1beta1", "v1beta2"))
		defer func(p *examplePlugin) { require.NoError(t, p.Stop()) }(p)

		plugins[i] = p
	}

	w := newWatcherWithHandler(t, hdlr, false /* testDeprecatedDir */)
	defer func() { require.NoError(t, w.Stop()) }()

	var wg sync.WaitGroup
	for i := 0; i < len(plugins); i++ {
		wg.Add(1)
		go func(p *examplePlugin) {
			defer wg.Done()

			require.True(t, waitForEvent(t, exampleEventValidate, hdlr.EventChan(p.pluginName)))
			require.True(t, waitForEvent(t, exampleEventRegister, hdlr.EventChan(p.pluginName)))

			require.True(t, waitForPluginRegistrationStatus(t, p.registrationStatus))
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
	case <-time.After(2 * time.Second):
		t.Fatalf("Timeout while waiting for the plugin registration status")
	}
}

func TestPluginRegistrationFailureWithUnsupportedVersion(t *testing.T) {
	defer cleanup(t)

	pluginName := fmt.Sprintf("example-plugin")
	socketPath := socketDir + "/plugin.sock"

	hdlr := NewExampleHandler(supportedVersions, false /* permitDeprecatedDir */)
	hdlr.AddPluginName(pluginName)

	w := newWatcherWithHandler(t, hdlr, false /* testDeprecatedDir */)
	defer func() { require.NoError(t, w.Stop()) }()

	// Advertise v1beta3 but don't serve anything else than the plugin service
	p := NewTestExamplePlugin(pluginName, registerapi.DevicePlugin, socketPath, "v1beta3")
	require.NoError(t, p.Serve())
	defer func() { require.NoError(t, p.Stop()) }()

	require.True(t, waitForEvent(t, exampleEventValidate, hdlr.EventChan(p.pluginName)))
	require.False(t, waitForPluginRegistrationStatus(t, p.registrationStatus))
}

func TestPlugiRegistrationFailureWithUnsupportedVersionAtKubeletStart(t *testing.T) {
	defer cleanup(t)

	pluginName := fmt.Sprintf("example-plugin")
	socketPath := socketDir + "/plugin.sock"

	// Advertise v1beta3 but don't serve anything else than the plugin service
	p := NewTestExamplePlugin(pluginName, registerapi.DevicePlugin, socketPath, "v1beta3")
	require.NoError(t, p.Serve())
	defer func() { require.NoError(t, p.Stop()) }()

	hdlr := NewExampleHandler(supportedVersions, false /* permitDeprecatedDir */)
	hdlr.AddPluginName(pluginName)

	w := newWatcherWithHandler(t, hdlr, false /* testDeprecatedDir */)
	defer func() { require.NoError(t, w.Stop()) }()

	require.True(t, waitForEvent(t, exampleEventValidate, hdlr.EventChan(p.pluginName)))
	require.False(t, waitForPluginRegistrationStatus(t, p.registrationStatus))
}

func waitForPluginRegistrationStatus(t *testing.T, statusChan chan registerapi.RegistrationStatus) bool {
	select {
	case status := <-statusChan:
		return status.PluginRegistered
	case <-time.After(10 * time.Second):
		t.Fatalf("Timed out while waiting for registration status")
	}
	return false
}

func waitForEvent(t *testing.T, expected examplePluginEvent, eventChan chan examplePluginEvent) bool {
	select {
	case event := <-eventChan:
		return event == expected
	case <-time.After(2 * time.Second):
		t.Fatalf("Timed out while waiting for registration status %v", expected)
	}

	return false
}

func newWatcherWithHandler(t *testing.T, hdlr PluginHandler, testDeprecatedDir bool) *Watcher {
	depSocketDir := ""
	if testDeprecatedDir {
		depSocketDir = deprecatedSocketDir
	}
	w := NewWatcher(socketDir, depSocketDir)

	w.AddHandler(registerapi.DevicePlugin, hdlr)
	require.NoError(t, w.Start())

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
		watcher := NewWatcher(tc.sockDir, tc.deprecatedSockDir)

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
		watcher := NewWatcher(tc.sockDir, tc.deprecatedSockDir)

		actual := watcher.containsBlacklistedDir(tc.path)

		// Assert
		if tc.expected != actual {
			t.Fatalf("expecting %v but got %v for testcase: %#v", tc.expected, actual, tc)
		}
	}
}
