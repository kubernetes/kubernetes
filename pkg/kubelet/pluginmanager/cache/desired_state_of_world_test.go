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
)

// Calls AddOrUpdatePlugin() to add a plugin
// Verifies newly added plugin exists in GetPluginsToRegister()
// Verifies newly added plugin returns true for PluginExists()
func Test_DSW_AddOrUpdatePlugin_Positive_NewPlugin(t *testing.T) {
	dsw := NewDesiredStateOfWorld()
	socketPath := "/var/lib/kubelet/device-plugins/test-plugin.sock"
	err := dsw.AddOrUpdatePlugin(socketPath, false /* foundInDeprecatedDir */)
	// Assert
	if err != nil {
		t.Fatalf("AddOrUpdatePlugin failed. Expected: <no error> Actual: <%v>", err)
	}

	// Get pluginsToRegister and check the newly added plugin is there
	dswPlugins := dsw.GetPluginsToRegister()
	if len(dswPlugins) != 1 {
		t.Fatalf("Desired state of world length should be one but it's %d", len(dswPlugins))
	}
	if dswPlugins[0].SocketPath != socketPath {
		t.Fatalf("Expected\n%s\nin desired state of world, but got\n%v\n", socketPath, dswPlugins[0])
	}

	// Check PluginExists returns true
	if !dsw.PluginExists(socketPath) {
		t.Fatalf("PluginExists returns false for the newly added plugin")
	}
}

// Calls AddOrUpdatePlugin() to update timestamp of an existing plugin
// Verifies the timestamp the existing plugin is updated
// Verifies newly added plugin returns true for PluginExists()
func Test_DSW_AddOrUpdatePlugin_Positive_ExistingPlugin(t *testing.T) {
	dsw := NewDesiredStateOfWorld()
	socketPath := "/var/lib/kubelet/device-plugins/test-plugin.sock"
	// Adding the plugin for the first time
	err := dsw.AddOrUpdatePlugin(socketPath, false /* foundInDeprecatedDir */)
	if err != nil {
		t.Fatalf("AddOrUpdatePlugin failed. Expected: <no error> Actual: <%v>", err)
	}

	// Get pluginsToRegister and check the newly added plugin is there, and get the old timestamp
	dswPlugins := dsw.GetPluginsToRegister()
	if len(dswPlugins) != 1 {
		t.Fatalf("Desired state of world length should be one but it's %d", len(dswPlugins))
	}
	if dswPlugins[0].SocketPath != socketPath {
		t.Fatalf("Expected\n%s\nin desired state of world, but got\n%v\n", socketPath, dswPlugins[0])
	}
	oldTimestamp := dswPlugins[0].Timestamp

	// Adding the plugin again so that the timestamp will be updated
	err = dsw.AddOrUpdatePlugin(socketPath, false /* foundInDeprecatedDir */)
	if err != nil {
		t.Fatalf("AddOrUpdatePlugin failed. Expected: <no error> Actual: <%v>", err)
	}
	newDswPlugins := dsw.GetPluginsToRegister()
	if len(newDswPlugins) != 1 {
		t.Fatalf("Desired state of world length should be one but it's %d", len(newDswPlugins))
	}
	if newDswPlugins[0].SocketPath != socketPath {
		t.Fatalf("Expected\n%s\nin desired state of world, but got\n%v\n", socketPath, newDswPlugins[0])
	}

	// Verify that the new timestamp is newer than the old timestamp
	if !newDswPlugins[0].Timestamp.After(oldTimestamp) {
		t.Fatal("New timestamp is not newer than the old timestamp", newDswPlugins[0].Timestamp, oldTimestamp)
	}

}

// Calls AddOrUpdatePlugin() to add an empty string for socket path
// Verifies the plugin does not exist in GetPluginsToRegister() after AddOrUpdatePlugin()
// Verifies the plugin returns false for PluginExists()
func Test_DSW_AddOrUpdatePlugin_Negative_PluginMissingInfo(t *testing.T) {
	dsw := NewDesiredStateOfWorld()
	socketPath := ""
	err := dsw.AddOrUpdatePlugin(socketPath, false /* foundInDeprecatedDir */)
	// Assert
	if err == nil || err.Error() != "Socket path is empty" {
		t.Fatalf("AddOrUpdatePlugin failed. Expected: <Socket path is empty> Actual: <%v>", err)
	}

	// Get pluginsToRegister and check the newly added plugin is there
	dswPlugins := dsw.GetPluginsToRegister()
	if len(dswPlugins) != 0 {
		t.Fatalf("Desired state of world length should be zero but it's %d", len(dswPlugins))
	}

	// Check PluginExists returns false
	if dsw.PluginExists(socketPath) {
		t.Fatalf("PluginExists returns true for the plugin that should not have been registered")
	}
}

// Calls RemovePlugin() to remove a plugin
// Verifies newly removed plugin no longer exists in GetPluginsToRegister()
// Verifies newly removed plugin returns false for PluginExists()
func Test_DSW_RemovePlugin_Positive(t *testing.T) {
	// First, add a plugin
	dsw := NewDesiredStateOfWorld()
	socketPath := "/var/lib/kubelet/device-plugins/test-plugin.sock"
	err := dsw.AddOrUpdatePlugin(socketPath, false /* foundInDeprecatedDir */)
	// Assert
	if err != nil {
		t.Fatalf("AddOrUpdatePlugin failed. Expected: <no error> Actual: <%v>", err)
	}

	// Try removing this plugin
	dsw.RemovePlugin(socketPath)

	// Get pluginsToRegister and check the newly added plugin is there
	dswPlugins := dsw.GetPluginsToRegister()
	if len(dswPlugins) != 0 {
		t.Fatalf("Desired state of world length should be zero but it's %d", len(dswPlugins))
	}

	// Check PluginExists returns false
	if dsw.PluginExists(socketPath) {
		t.Fatalf("PluginExists returns true for the removed plugin")
	}
}
