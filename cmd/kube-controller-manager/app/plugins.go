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

	//Cloud providers
	_ "k8s.io/kubernetes/pkg/cloudprovider/providers"

	// Volume plugins
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/host_path"
	"k8s.io/kubernetes/pkg/volume/nfs"

	"github.com/golang/glog"
)

// ProbeRecyclableVolumePlugins collects all persistent volume plugins into an easy to use list.
func ProbeRecyclableVolumePlugins(flags VolumeConfigFlags) []volume.VolumePlugin {
	allPlugins := []volume.VolumePlugin{}

	// The list of plugins to probe is decided by this binary, not
	// by dynamic linking or other "magic".  Plugins will be analyzed and
	// initialized later.

	// Each plugin can make use of VolumeConfig.  The single arg to this func contains *all* enumerated
	// CLI flags meant to configure volume plugins.  From that single config, create an instance of volume.VolumeConfig
	// for a specific plugin and pass that instance to the plugin's ProbeVolumePlugins(config) func.

	// HostPath recycling is for testing and development purposes only!
	hostPathConfig := volume.VolumeConfig{
		RecyclerMinimumTimeout:   flags.PersistentVolumeRecyclerMinimumTimeoutHostPath,
		RecyclerTimeoutIncrement: flags.PersistentVolumeRecyclerIncrementTimeoutHostPath,
		RecyclerPodTemplate:      volume.NewPersistentVolumeRecyclerPodTemplate(),
	}
	if err := attemptToLoadRecycler(flags.PersistentVolumeRecyclerPodTemplateFilePathHostPath, &hostPathConfig); err != nil {
		glog.Fatalf("Could not create hostpath recycler pod from file %s: %+v", err)
	}
	allPlugins = append(allPlugins, host_path.ProbeVolumePlugins(hostPathConfig)...)

	nfsConfig := volume.VolumeConfig{
		RecyclerMinimumTimeout:   flags.PersistentVolumeRecyclerMinimumTimeoutNFS,
		RecyclerTimeoutIncrement: flags.PersistentVolumeRecyclerIncrementTimeoutNFS,
		RecyclerPodTemplate:      volume.NewPersistentVolumeRecyclerPodTemplate(),
	}
	if err := attemptToLoadRecycler(flags.PersistentVolumeRecyclerPodTemplateFilePathNFS, &nfsConfig); err != nil {
		glog.Fatalf("Could not create NFS recycler pod from file %s: %+v", err)
	}
	allPlugins = append(allPlugins, nfs.ProbeVolumePlugins(nfsConfig)...)

	return allPlugins
}

// attemptToLoadRecycler tries decoding a pod from a filepath for use as a recycler for a volume.
// If successful, this method will set the recycler on the config.
// If unsucessful, an error is returned.
func attemptToLoadRecycler(path string, config *volume.VolumeConfig) error {
	if path != "" {
		recyclerPod, err := io.LoadPodFromFile(path)
		if err != nil {
			return err
		}
		config.RecyclerPodTemplate = recyclerPod
	}
	return nil
}
