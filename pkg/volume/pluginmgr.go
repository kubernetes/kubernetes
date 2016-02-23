/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

	"github.com/golang/glog"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/validation"
)

// VolumePluginMgr tracks registered plugins.
type VolumePluginMgr struct {
	mutex   sync.Mutex
	plugins map[string]VolumePlugin
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
		if !validation.IsQualifiedName(name) {
			allErrs = append(allErrs, fmt.Errorf("volume plugin has invalid name: %#v", plugin))
			continue
		}

		if _, found := pm.plugins[name]; found {
			allErrs = append(allErrs, fmt.Errorf("volume plugin %q was registered more than once", name))
			continue
		}
		err := plugin.Init(host)
		if err != nil {
			glog.Errorf("Failed to load volume plugin %s, error: %s", plugin, err.Error())
			allErrs = append(allErrs, err)
			continue
		}
		pm.plugins[name] = plugin
		glog.V(1).Infof("Loaded volume plugin %q", name)
	}
	return utilerrors.NewAggregate(allErrs)
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

// FindMountablePluginBySpec looks for a persistent volume plugin that can support a given volume
// specification.  If no plugin is found, return an error
func (pm *VolumePluginMgr) FindMountablePluginBySpec(spec *Spec) (MountableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, fmt.Errorf("Could not find volume plugin for spec: %+v", spec)
	}
	if mountableVolumePlugin, ok := volumePlugin.(MountableVolumePlugin); ok {
		return mountableVolumePlugin, nil
	}
	return nil, fmt.Errorf("no mountable volume plugin matched")
}

// FindMountablePluginByName fetches a persistent volume plugin by name.  If no plugin
// is found, returns error.
func (pm *VolumePluginMgr) FindMountablePluginByName(name string) (MountableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginByName(name)
	if err != nil {
		return nil, err
	}
	if mountableVolumePlugin, ok := volumePlugin.(MountableVolumePlugin); ok {
		return mountableVolumePlugin, nil
	}
	return nil, fmt.Errorf("no mountable volume plugin matched")
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
	return nil, fmt.Errorf("no persistent volume plugin matched")
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

// FindDeletablePluginByName fetches a persistent volume plugin by name.  If no plugin
// is found, returns error.
func (pm *VolumePluginMgr) FindDeletablePluginBySpec(spec *Spec) (DeletableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	if deletableVolumePlugin, ok := volumePlugin.(DeletableVolumePlugin); ok {
		return deletableVolumePlugin, nil
	}
	return nil, fmt.Errorf("no deletable volume plugin matched")
}

// FindCreatablePluginBySpec fetches a persistent volume plugin by name.  If no plugin
// is found, returns error.
func (pm *VolumePluginMgr) FindCreatablePluginBySpec(spec *Spec) (ProvisionableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	if provisionableVolumePlugin, ok := volumePlugin.(ProvisionableVolumePlugin); ok {
		return provisionableVolumePlugin, nil
	}
	return nil, fmt.Errorf("no creatable volume plugin matched")
}

// FindCreatablePluginByName fetches a persistent volume plugin by name.  If no plugin
// is found, returns error.
func (pm *VolumePluginMgr) FindCreatablePluginByName(name string) (ProvisionableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginByName(name)
	if err != nil {
		return nil, err
	}
	if provisionableVolumePlugin, ok := volumePlugin.(ProvisionableVolumePlugin); ok {
		return provisionableVolumePlugin, nil
	}
	return nil, fmt.Errorf("no provisionable volume plugin matched")
}

// FindAttachablePluginBySpec fetches a persistent volume plugin by name.  Unlike the other "FindPlugin" methods, this
// does not return error if no plugin is found.  All volumes require a builder and cleaner, but not every volume will
// have an attacher/detacher.
func (pm *VolumePluginMgr) FindAttachablePluginBySpec(spec *Spec) (AttachableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginBySpec(spec)
	if err != nil {
		return nil, err
	}
	if attachableVolumePlugin, ok := volumePlugin.(AttachableVolumePlugin); ok {
		return attachableVolumePlugin, nil
	}
	return nil, nil
}

// FindAttachablePluginByName fetches an attachable volume plugin by name. Unlike the other "FindPlugin" methods, this
// does not return error if no plugin is found.  All volumes require a builder and cleaner, but not every volume will
// have an attacher/detacher.
func (pm *VolumePluginMgr) FindAttachablePluginByName(name string) (AttachableVolumePlugin, error) {
	volumePlugin, err := pm.FindPluginByName(name)
	if err != nil {
		return nil, err
	}
	if attachablePlugin, ok := volumePlugin.(AttachableVolumePlugin); ok {
		return attachablePlugin, nil
	}
	return nil, nil
}
