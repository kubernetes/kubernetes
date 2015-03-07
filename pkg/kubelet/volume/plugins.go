/*
Copyright 2014 Google Inc. All rights reserved.

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

package volume

import (
	"fmt"
	"strings"
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
	"github.com/golang/glog"
)

// Plugin is an interface to volume plugins.
type Plugin interface {
	// Init initializes the plugin.  This will be called exactly once
	// before any New* calls are made - implementations of plugins may
	// depend on this.
	Init(host Host)

	// Name returns the plugin's name.  Plugins should use namespaced names
	// such as "example.com/volume".  The "kubernetes.io" namespace is
	// reserved for plugins which are bundled with kubernetes.
	Name() string

	// CanSupport tests whether the Plugin supports a given volume
	// specification from the API.  The spec pointer should be considered
	// const.
	CanSupport(spec *api.Volume) bool

	// NewBuilder creates a new volume.Builder from an API specification.
	// Ownership of the spec pointer in *not* transferred.
	// - spec: The api.Volume spec
	// - podUID: The UID of the enclosing pod
	NewBuilder(spec *api.Volume, podUID types.UID) (Builder, error)

	// NewCleaner creates a new volume.Cleaner from recoverable state.
	// - name: The volume name, as per the api.Volume spec.
	// - podUID: The UID of the enclosing pod
	NewCleaner(name string, podUID types.UID) (Cleaner, error)
}

// Host is an interface that plugins can use to access the kubelet.
//
// TODO: this interface should become more general once we have consensus
// around the characteristic matching that has been discussed.
type Host interface {
	// GetPluginDir returns the absolute path to a directory under which
	// a given plugin may store data.  This directory might not actually
	// exist on disk yet.  For plugin data that is per-pod, see
	// GetPodPluginDir().
	GetPluginDir(pluginName string) string

	// GetPodVolumeDir returns the absolute path to a directory which
	// represents the named volume under the named plugin for the given
	// pod.  If the specified pod does not exist, the result of this call
	// might not exist.
	GetPodVolumeDir(podUID types.UID, pluginName, volumeName string) string

	// GetTmpfsPodVolumeDir returns the absolute path to a directory which
	// represents the tmpfs storage for the named volume under the named
	// plugin for the given pod.  If the specified pod does not exist, the result
	// of this call might not exist.
	GetTmpfsPodVolumeDir(podUID types.UID, pluginName, volumeName string) string

	// GetPodPluginDir returns the absolute path to a directory under which
	// a given plugin may store data for a given pod.  If the specified pod
	// does not exist, the result of this call might not exist.  This
	// directory might not actually exist on disk yet.
	GetPodPluginDir(podUID types.UID, pluginName string) string

	// GetKubeClient returns a client interface
	GetKubeClient() client.Interface
}

// PluginMgr tracks registered plugins.
type PluginMgr struct {
	mutex   sync.Mutex
	plugins map[string]Plugin
}

// InitPlugins initializes each plugin.  All plugins must have unique names.
// This must be called exactly once before any New* methods are called on any
// plugins.
func (pm *PluginMgr) InitPlugins(plugins []Plugin, host Host) error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	if pm.plugins == nil {
		pm.plugins = map[string]Plugin{}
	}

	allErrs := []error{}
	for _, plugin := range plugins {
		name := plugin.Name()
		if !util.IsQualifiedName(name) {
			allErrs = append(allErrs, fmt.Errorf("volume plugin has invalid name: %#v", plugin))
			continue
		}

		if _, found := pm.plugins[name]; found {
			allErrs = append(allErrs, fmt.Errorf("volume plugin %q was registered more than once", name))
			continue
		}
		plugin.Init(host)
		pm.plugins[name] = plugin
		glog.V(1).Infof("Loaded volume plugin %q", name)
	}
	return errors.NewAggregate(allErrs)
}

// FindPluginBySpec looks for a plugin that can support a given volume
// specification.  If no plugins can support or more than one plugin can
// support it, return error.
func (pm *PluginMgr) FindPluginBySpec(spec *api.Volume) (Plugin, error) {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	matches := []string{}
	for k, v := range pm.plugins {
		if v.CanSupport(spec) {
			matches = append(matches, k)
		}
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("no volume plugin matched")
	}
	if len(matches) > 1 {
		return nil, fmt.Errorf("multiple volume plugins matched: %s", strings.Join(matches, ","))
	}
	return pm.plugins[matches[0]], nil
}

// FindPluginByName fetches a plugin by name or by legacy name.  If no plugin
// is found, returns error.
func (pm *PluginMgr) FindPluginByName(name string) (Plugin, error) {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	// Once we can get rid of legacy names we can reduce this to a map lookup.
	matches := []string{}
	for k, v := range pm.plugins {
		if v.Name() == name {
			matches = append(matches, k)
		}
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("no volume plugin matched")
	}
	if len(matches) > 1 {
		return nil, fmt.Errorf("multiple volume plugins matched: %s", strings.Join(matches, ","))
	}
	return pm.plugins[matches[0]], nil
}

// EscapePluginName converts a plugin name, which might contain a / into a
// string that is safe to use on-disk.  This assumes that the input has already
// been validates as a qualified name.  we use "~" rather than ":" here in case
// we ever use a filesystem that doesn't allow ":".
func EscapePluginName(in string) string {
	return strings.Replace(in, "/", "~", -1)
}

// UnescapePluginName converts an escaped plugin name (as per EscapePluginName)
// back to its normal form.  This assumes that the input has already been
// validates as a qualified name.
func UnescapePluginName(in string) string {
	return strings.Replace(in, "~", "/", -1)
}
