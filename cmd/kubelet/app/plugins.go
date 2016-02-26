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

// This file exists to force the desired plugin implementations to be linked.
import (
	// Credential providers
	_ "k8s.io/kubernetes/pkg/credentialprovider/aws"
	_ "k8s.io/kubernetes/pkg/credentialprovider/gcp"
	// Network plugins
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/network/cni"
	"k8s.io/kubernetes/pkg/kubelet/network/exec"
	"k8s.io/kubernetes/pkg/kubelet/network/kubenet"
	// Volume plugins
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/flexvolume"
	// Cloud providers
	_ "k8s.io/kubernetes/pkg/cloudprovider/providers"
)

// ProbeVolumePlugins collects all volume plugins into an easy to use
// list. PluginDir specifies the directory to search for additional third party
// volume plugins.
func ProbeVolumePlugins(pluginDir string) []volume.VolumePlugin {

	// Pass pluginDir to the flex plugin through VolumeConfig
	configs := map[string]volume.VolumeConfig{
		flexvolume.FlexPluginName: {
			VolumePluginDir: pluginDir,
		},
	}

	plugins := volume.CreateVolumePlugins(configs)
	return plugins
}

// ProbeNetworkPlugins collects all compiled-in plugins
func ProbeNetworkPlugins(pluginDir string) []network.NetworkPlugin {
	allPlugins := []network.NetworkPlugin{}

	// for each existing plugin, add to the list
	allPlugins = append(allPlugins, exec.ProbeNetworkPlugins(pluginDir)...)
	allPlugins = append(allPlugins, cni.ProbeNetworkPlugins(pluginDir)...)
	allPlugins = append(allPlugins, kubenet.NewPlugin())

	return allPlugins
}
