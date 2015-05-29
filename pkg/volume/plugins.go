/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/golang/glog"
)

// VolumeOptions contains option information about a volume.
//
// Currently, this struct containers only a single field for the
// rootcontext of the volume.  This is a temporary measure in order
// to set the rootContext of tmpfs mounts correctly; it will be replaced
// and expanded on by future SecurityContext work.
type VolumeOptions struct {
	// The rootcontext to use when performing mounts for a volume.
	RootContext string
}

// VolumePlugin is an interface to volume plugins that can be used on a
// kubernetes node (e.g. by kubelet) to instantiate and manage volumes.
type VolumePlugin interface {
	// Init initializes the plugin.  This will be called exactly once
	// before any New* calls are made - implementations of plugins may
	// depend on this.
	Init(host VolumeHost)

	// Name returns the plugin's name.  Plugins should use namespaced names
	// such as "example.com/volume".  The "kubernetes.io" namespace is
	// reserved for plugins which are bundled with kubernetes.
	Name() string

	// CanSupport tests whether the plugin supports a given volume
	// specification from the API.  The spec pointer should be considered
	// const.
	CanSupport(spec *Spec) bool

	// NewBuilder creates a new volume.Builder from an API specification.
	// Ownership of the spec pointer in *not* transferred.
	// - spec: The api.Volume spec
	// - pod: The enclosing pod
	NewBuilder(spec *Spec, podRef *api.Pod, opts VolumeOptions, mounter mount.Interface) (Builder, error)

	// NewCleaner creates a new volume.Cleaner from recoverable state.
	// - name: The volume name, as per the api.Volume spec.
	// - podUID: The UID of the enclosing pod
	NewCleaner(name string, podUID types.UID, mounter mount.Interface) (Cleaner, error)
}

// PersistentVolumePlugin is an extended interface of VolumePlugin and is used
// by volumes that want to provide long term persistence of data
type PersistentVolumePlugin interface {
	VolumePlugin
	// GetAccessModes describes the ways a given volume can be accessed/mounted.
	GetAccessModes() []api.PersistentVolumeAccessMode
}

// RecyclableVolumePlugin is an extended interface of VolumePlugin and is used
// by persistent volumes that want to be recycled before being made available again to new claims
type RecyclableVolumePlugin interface {
	VolumePlugin
	// NewRecycler creates a new volume.Recycler which knows how to reclaim this resource
	// after the volume's release from a PersistentVolumeClaim
	NewRecycler(spec *Spec) (Recycler, error)
}

// VolumeHost is an interface that plugins can use to access the kubelet.
type VolumeHost interface {
	// GetPluginDir returns the absolute path to a directory under which
	// a given plugin may store data.  This directory might not actually
	// exist on disk yet.  For plugin data that is per-pod, see
	// GetPodPluginDir().
	GetPluginDir(pluginName string) string

	// GetPodVolumeDir returns the absolute path a directory which
	// represents the named volume under the named plugin for the given
	// pod.  If the specified pod does not exist, the result of this call
	// might not exist.
	GetPodVolumeDir(podUID types.UID, pluginName string, volumeName string) string

	// GetPodPluginDir returns the absolute path to a directory under which
	// a given plugin may store data for a given pod.  If the specified pod
	// does not exist, the result of this call might not exist.  This
	// directory might not actually exist on disk yet.
	GetPodPluginDir(podUID types.UID, pluginName string) string

	// GetKubeClient returns a client interface
	GetKubeClient() client.Interface

	// NewWrapperBuilder finds an appropriate plugin with which to handle
	// the provided spec.  This is used to implement volume plugins which
	// "wrap" other plugins.  For example, the "secret" volume is
	// implemented in terms of the "emptyDir" volume.
	NewWrapperBuilder(spec *Spec, pod *api.Pod, opts VolumeOptions, mounter mount.Interface) (Builder, error)

	// NewWrapperCleaner finds an appropriate plugin with which to handle
	// the provided spec.  See comments on NewWrapperBuilder for more
	// context.
	NewWrapperCleaner(spec *Spec, podUID types.UID, mounter mount.Interface) (Cleaner, error)
}

// VolumePluginMgr tracks registered plugins.
type VolumePluginMgr struct {
	mutex   sync.Mutex
	plugins map[string]VolumePlugin
}

// Spec is an internal representation of a volume.  All API volume types translate to Spec.
type Spec struct {
	Name                   string
	VolumeSource           api.VolumeSource
	PersistentVolumeSource api.PersistentVolumeSource
}

// NewSpecFromVolume creates an Spec from an api.Volume
func NewSpecFromVolume(vs *api.Volume) *Spec {
	return &Spec{
		Name:         vs.Name,
		VolumeSource: vs.VolumeSource,
	}
}

// NewSpecFromPersistentVolume creates an Spec from an api.PersistentVolume
func NewSpecFromPersistentVolume(pv *api.PersistentVolume) *Spec {
	return &Spec{
		Name: pv.Name,
		PersistentVolumeSource: pv.Spec.PersistentVolumeSource,
	}
}

// InitPlugins initializes each plugin.  All plugins must have unique names.
// This must be called exactly once before any New* methods are called on any
// plugins.
func (pm *VolumePluginMgr) InitPlugins(plugins []VolumePlugin, host VolumeHost) error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	if pm.plugins == nil {
		pm.plugins = map[string]VolumePlugin{}
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
func (pm *VolumePluginMgr) FindPluginBySpec(spec *Spec) (VolumePlugin, error) {
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
func (pm *VolumePluginMgr) FindPluginByName(name string) (VolumePlugin, error) {
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

// FindPersistentPluginBySpec looks for a persistent volume plugin that can support a given volume
// specification.  If no plugin is found, return an error
func (pm *VolumePluginMgr) FindPersistentPluginBySpec(spec *Spec) (PersistentVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, fmt.Errorf("Could not find volume plugin for spec: %+v", spec)
	}
	if persistentVolumePlugin, ok := volumePlugin.(PersistentVolumePlugin); ok {
		return persistentVolumePlugin, nil
	}
	return nil, fmt.Errorf("no persistent volume plugin matched")
}

// FindPersistentPluginByName fetches a persistent volume plugin by name.  If no plugin
// is found, returns error.
func (pm *VolumePluginMgr) FindPersistentPluginByName(name string) (PersistentVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginByName(name)
	if err != nil {
		return nil, err
	}
	if persistentVolumePlugin, ok := volumePlugin.(PersistentVolumePlugin); ok {
		return persistentVolumePlugin, nil
	}
	return nil, fmt.Errorf("no persistent volume plugin matched: %+v")
}

// FindRecyclablePluginByName fetches a persistent volume plugin by name.  If no plugin
// is found, returns error.
func (pm *VolumePluginMgr) FindRecyclablePluginBySpec(spec *Spec) (RecyclableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	if recyclableVolumePlugin, ok := volumePlugin.(RecyclableVolumePlugin); ok {
		return recyclableVolumePlugin, nil
	}
	return nil, fmt.Errorf("no recyclable volume plugin matched")
}
