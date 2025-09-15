/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/volume/csi"
	"k8s.io/kubernetes/pkg/volume/iscsi"

	// Volume plugins
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/fc"
	"k8s.io/kubernetes/pkg/volume/flexvolume"
	"k8s.io/kubernetes/pkg/volume/hostpath"
	"k8s.io/kubernetes/pkg/volume/nfs"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	persistentvolumeconfig "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/config"
	"k8s.io/utils/exec"
)

// ProbeAttachableVolumePlugins collects all volume plugins for the attach/
// detach controller.
func ProbeAttachableVolumePlugins(logger klog.Logger, config persistentvolumeconfig.VolumeConfiguration) ([]volume.VolumePlugin, error) {
	return probeControllerVolumePlugins(logger, config, func(plugin volume.VolumePlugin) bool {
		_, ok := plugin.(volume.AttachableVolumePlugin)
		return ok
	})
}

// GetDynamicPluginProber gets the probers of dynamically discoverable plugins
// for the attach/detach controller.
// Currently only Flexvolume plugins are dynamically discoverable.
func GetDynamicPluginProber(config persistentvolumeconfig.VolumeConfiguration) volume.DynamicPluginProber {
	return flexvolume.GetDynamicPluginProber(config.FlexVolumePluginDir, exec.New() /*exec.Interface*/)
}

// ProbeExpandableVolumePlugins returns volume plugins which are expandable
func ProbeExpandableVolumePlugins(logger klog.Logger, config persistentvolumeconfig.VolumeConfiguration) ([]volume.VolumePlugin, error) {
	return probeControllerVolumePlugins(logger, config, func(plugin volume.VolumePlugin) bool {
		_, ok := plugin.(volume.ExpandableVolumePlugin)
		return ok
	})
}

func ProbeProvisionableRecyclableVolumePlugins(logger klog.Logger, config persistentvolumeconfig.VolumeConfiguration) ([]volume.VolumePlugin, error) {
	return probeControllerVolumePlugins(logger, config, func(plugin volume.VolumePlugin) bool {
		if _, ok := plugin.(volume.ProvisionableVolumePlugin); ok {
			return true
		}
		if _, ok := plugin.(volume.DeletableVolumePlugin); ok {
			return true
		}
		if _, ok := plugin.(volume.RecyclableVolumePlugin); ok {
			return true
		}
		return false
	})
}

// ProbePersistentVolumePlugins collects all volume plugins that are actually persistent.
func ProbePersistentVolumePlugins(logger klog.Logger, config persistentvolumeconfig.VolumeConfiguration) ([]volume.VolumePlugin, error) {
	return probeControllerVolumePlugins(logger, config, nil)
}

// probeControllerVolumePlugins collects all persistent volume plugins
// used by KCM controllers into an easy to use list.
func probeControllerVolumePlugins(logger klog.Logger, config persistentvolumeconfig.VolumeConfiguration, filter func(plugin volume.VolumePlugin) bool) ([]volume.VolumePlugin, error) {
	var allPlugins []volume.VolumePlugin

	// The list of plugins to probe is decided by this binary, not
	// by dynamic linking or other "magic".  Plugins will be analyzed and
	// initialized later.

	// Each plugin can make use of VolumeConfig.  The single arg to this func contains *all* enumerated
	// options meant to configure volume plugins.  From that single config, create an instance of volume.VolumeConfig
	// for a specific plugin and pass that instance to the plugin's ProbeVolumePlugins(config) func.

	// HostPath recycling is for testing and development purposes only!
	hostPathConfig := volume.VolumeConfig{
		RecyclerMinimumTimeout:   int(config.PersistentVolumeRecyclerConfiguration.MinimumTimeoutHostPath),
		RecyclerTimeoutIncrement: int(config.PersistentVolumeRecyclerConfiguration.IncrementTimeoutHostPath),
		RecyclerPodTemplate:      volume.NewPersistentVolumeRecyclerPodTemplate(),
		ProvisioningEnabled:      config.EnableHostPathProvisioning,
	}
	if err := AttemptToLoadRecycler(config.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathHostPath, &hostPathConfig); err != nil {
		logger.Error(err, "Could not create hostpath recycler pod from file", "path", config.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathHostPath)
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
	allPlugins = append(allPlugins, hostpath.ProbeVolumePlugins(hostPathConfig)...)

	nfsConfig := volume.VolumeConfig{
		RecyclerMinimumTimeout:   int(config.PersistentVolumeRecyclerConfiguration.MinimumTimeoutNFS),
		RecyclerTimeoutIncrement: int(config.PersistentVolumeRecyclerConfiguration.IncrementTimeoutNFS),
		RecyclerPodTemplate:      volume.NewPersistentVolumeRecyclerPodTemplate(),
	}
	if err := AttemptToLoadRecycler(config.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathNFS, &nfsConfig); err != nil {
		logger.Error(err, "Could not create NFS recycler pod from file", "path", config.PersistentVolumeRecyclerConfiguration.PodTemplateFilePathNFS)
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}
	allPlugins = append(allPlugins, nfs.ProbeVolumePlugins(nfsConfig)...)
	allPlugins = append(allPlugins, fc.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, iscsi.ProbeVolumePlugins()...)
	allPlugins = append(allPlugins, csi.ProbeVolumePlugins()...)

	var err error
	allPlugins, err = appendLegacyControllerProviders(logger, allPlugins, utilfeature.DefaultFeatureGate)
	if err != nil {
		return allPlugins, err
	}

	var filteredPlugins []volume.VolumePlugin
	if filter == nil {
		filteredPlugins = allPlugins
	} else {
		for _, plugin := range allPlugins {
			if filter(plugin) {
				filteredPlugins = append(filteredPlugins, plugin)
			}
		}
	}

	return filteredPlugins, nil
}

// AttemptToLoadRecycler tries decoding a pod from a filepath for use as a recycler for a volume.
// If successful, this method will set the recycler on the config.
// If unsuccessful, an error is returned. Function is exported for reuse downstream.
func AttemptToLoadRecycler(path string, config *volume.VolumeConfig) error {
	if path != "" {
		recyclerPod, err := volumeutil.LoadPodFromFile(path)
		if err != nil {
			return err
		}
		if err = volume.ValidateRecyclerPodTemplate(recyclerPod); err != nil {
			return fmt.Errorf("pod specification (%v): %v", path, err)
		}
		config.RecyclerPodTemplate = recyclerPod
	}
	return nil
}
