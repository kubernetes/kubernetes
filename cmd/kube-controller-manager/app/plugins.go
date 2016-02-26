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

package app

import (
	// This file exists to force the desired plugin implementations to be linked.
	// This should probably be part of some configuration fed into the build for a
	// given binary target.

	"fmt"

	// Cloud providers
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	_ "k8s.io/kubernetes/pkg/cloudprovider/providers"

	// Volume plugins
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/host_path"
	"k8s.io/kubernetes/pkg/volume/nfs"
)

// ProbeRecyclableVolumePlugins collects all persistent volume plugins into an easy to use list.
func ProbeRecyclableVolumePlugins(config componentconfig.VolumeConfiguration) []volume.VolumePlugin {
	// Each plugin can make use of VolumeConfig.  The single arg to this func contains *all* enumerated
	// options meant to configure volume plugins.  From that single config, create an instance of volume.VolumeConfig
	// for a specific plugin and pass that instance to the plugin's ProbeVolumePlugins(config) func.
	// HostPath recycling is for testing and development purposes only!
	hostPathConfig := volume.VolumeConfig{
		RecyclerMinimumTimeout:   config.PersistentVolumeRecyclerConfiguration.MinimumTimeoutHostPath,
		RecyclerTimeoutIncrement: config.PersistentVolumeRecyclerConfiguration.IncrementTimeoutHostPath,
		RecyclerPodTemplate:      volume.NewPersistentVolumeRecyclerPodTemplate(),
	}
	nfsConfig := volume.VolumeConfig{
		RecyclerMinimumTimeout:   config.PersistentVolumeRecyclerConfiguration.MinimumTimeoutNFS,
		RecyclerTimeoutIncrement: config.PersistentVolumeRecyclerConfiguration.IncrementTimeoutNFS,
		RecyclerPodTemplate:      volume.NewPersistentVolumeRecyclerPodTemplate(),
	}
	configs := map[string]volume.VolumeConfig{
		host_path.HostPathPluginName: hostPathConfig,
		nfs.NfsPluginName:            nfsConfig,
	}

	// Instantiate all plugins and filter out any non-recyclable
	plugins := volume.CreateVolumePlugins(configs)
	recyclablePlugins := []volume.VolumePlugin{}
	for _, plugin := range plugins {
		_, ok := plugin.(volume.RecyclableVolumePlugin)
		if ok {
			recyclablePlugins = append(recyclablePlugins, plugin)
		}
	}
	return recyclablePlugins
}

// NewVolumeProvisioner returns a volume provisioner to use when running in a cloud or development environment.
// The beta implementation of provisioning allows 1 implied provisioner per cloud, until we allow configuration of many.
// We explicitly map clouds to volume plugins here which allows us to configure many later without backwards compatibility issues.
// Not all cloudproviders have provisioning capability, which is the reason for the bool in the return to tell the caller to expect one or not.
func NewVolumeProvisioner(cloud cloudprovider.Interface, config componentconfig.VolumeConfiguration) (volume.ProvisionableVolumePlugin, error) {
	if cloud != nil && config.EnableHostPathProvisioning {
		return nil, fmt.Errorf("host-path provisioner is not supported when running in a cloud")
	}
	// Instantiate all plugins and find the first provisionable one
	// TODO: make all the provisioners available to the user
	plugins := volume.CreateVolumePlugins(map[string]volume.VolumeConfig{})
	for _, plugin := range plugins {
		provisionablePlugin, ok := plugin.(volume.ProvisionableVolumePlugin)
		if !ok {
			continue
		}
		return provisionablePlugin, nil
	}
	return nil, fmt.Errorf("Cannot find any provisionable volume plugin")
}

func getProvisionablePluginFromVolumePlugins(plugins []volume.VolumePlugin) (volume.ProvisionableVolumePlugin, error) {
	for _, plugin := range plugins {
		if provisonablePlugin, ok := plugin.(volume.ProvisionableVolumePlugin); ok {
			return provisonablePlugin, nil
		}
	}
	return nil, fmt.Errorf("ProvisionablePlugin expected but not found in %#v: ", plugins)
}

// AttemptToLoadRecycler tries decoding a pod from a filepath for use as a recycler for a volume.
// If successful, this method will set the recycler on the config.
// If unsuccessful, an error is returned. Function is exported for reuse downstream.
func AttemptToLoadRecycler(path string, config *volume.VolumeConfig) error {
	if path != "" {
		recyclerPod, err := io.LoadPodFromFile(path)
		if err != nil {
			return err
		}
		config.RecyclerPodTemplate = recyclerPod
	}
	return nil
}
