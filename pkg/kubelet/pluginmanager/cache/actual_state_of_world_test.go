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

package cache

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// Calls AddPlugin() to add a plugin
// Verifies newly added plugin exists in GetRegisteredPlugins()
// Verifies PluginExistsWithCorrectTimestamp returns true for the plugin
func Test_ASW_AddPlugin_Positive_NewPlugin(t *testing.T) {
	pluginInfo := PluginInfo{
		SocketPath: "/var/lib/kubelet/device-plugins/test-plugin.sock",
		Timestamp:  time.Now(),
	}
	asw := NewActualStateOfWorld()
	err := asw.AddPlugin(pluginInfo)
	// Assert
	if err != nil {
		t.Fatalf("AddPlugin failed. Expected: <no error> Actual: <%v>", err)
	}

	// Get registered plugins and check the newly added plugin is there
	aswPlugins := asw.GetRegisteredPlugins()
	if len(aswPlugins) != 1 {
		t.Fatalf("Actual state of world length should be one but it's %d", len(aswPlugins))
	}
	if aswPlugins[0] != pluginInfo {
		t.Fatalf("Expected\n%v\nin actual state of world, but got\n%v\n", pluginInfo, aswPlugins[0])
	}

	// Check PluginExistsWithCorrectTimestamp returns true
	if !asw.PluginExistsWithCorrectTimestamp(pluginInfo) {
		t.Fatalf("PluginExistsWithCorrectTimestamp returns false for plugin that should be registered")
	}
}

// Calls AddPlugin() to add an empty string for socket path
// Verifies the plugin does not exist in GetRegisteredPlugins()
// Verifies PluginExistsWithCorrectTimestamp returns false
func Test_ASW_AddPlugin_Negative_EmptySocketPath(t *testing.T) {
	asw := NewActualStateOfWorld()
	pluginInfo := PluginInfo{
		SocketPath: "",
		Timestamp:  time.Now(),
	}
	err := asw.AddPlugin(pluginInfo)
	require.EqualError(t, err, "socket path is empty")

	// Get registered plugins and check the newly added plugin is there
	aswPlugins := asw.GetRegisteredPlugins()
	if len(aswPlugins) != 0 {
		t.Fatalf("Actual state of world length should be zero but it's %d", len(aswPlugins))
	}

	// Check PluginExistsWithCorrectTimestamp returns false
	if asw.PluginExistsWithCorrectTimestamp(pluginInfo) {
		t.Fatalf("PluginExistsWithCorrectTimestamp returns true for plugin that's not registered")
	}
}

// Calls RemovePlugin() to remove a plugin
// Verifies newly removed plugin no longer exists in GetRegisteredPlugins()
// Verifies PluginExistsWithCorrectTimestamp returns false
func Test_ASW_RemovePlugin_Positive(t *testing.T) {
	// First, add a plugin
	asw := NewActualStateOfWorld()
	pluginInfo := PluginInfo{
		SocketPath: "/var/lib/kubelet/device-plugins/test-plugin.sock",
		Timestamp:  time.Now(),
	}
	err := asw.AddPlugin(pluginInfo)
	// Assert
	if err != nil {
		t.Fatalf("AddPlugin failed. Expected: <no error> Actual: <%v>", err)
	}

	// Try removing this plugin
	asw.RemovePlugin(pluginInfo.SocketPath)

	// Get registered plugins and check the newly added plugin is not there
	aswPlugins := asw.GetRegisteredPlugins()
	if len(aswPlugins) != 0 {
		t.Fatalf("Actual state of world length should be zero but it's %d", len(aswPlugins))
	}

	// Check PluginExistsWithCorrectTimestamp returns false
	if asw.PluginExistsWithCorrectTimestamp(pluginInfo) {
		t.Fatalf("PluginExistsWithCorrectTimestamp returns true for the removed plugin")
	}
}

// Verifies PluginExistsWithCorrectTimestamp returns false for an existing
// plugin with the wrong timestamp
func Test_ASW_PluginExistsWithCorrectTimestamp_Negative_WrongTimestamp(t *testing.T) {
	// First, add a plugin
	asw := NewActualStateOfWorld()
	pluginInfo := PluginInfo{
		SocketPath: "/var/lib/kubelet/device-plugins/test-plugin.sock",
		Timestamp:  time.Now(),
	}
	err := asw.AddPlugin(pluginInfo)
	// Assert
	if err != nil {
		t.Fatalf("AddPlugin failed. Expected: <no error> Actual: <%v>", err)
	}

	newerPlugin := PluginInfo{
		SocketPath: "/var/lib/kubelet/device-plugins/test-plugin.sock",
		Timestamp:  time.Now(),
	}
	// Check PluginExistsWithCorrectTimestamp returns false
	if asw.PluginExistsWithCorrectTimestamp(newerPlugin) {
		t.Fatalf("PluginExistsWithCorrectTimestamp returns true for a plugin with newer timestamp")
	}
}
