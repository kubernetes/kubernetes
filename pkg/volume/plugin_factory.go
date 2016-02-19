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
	"sync"

	"k8s.io/kubernetes/pkg/api"

	"github.com/golang/glog"
)

// VolumePluginFactory is a function that returns array of
// uninitialized VolumePlugins.
type VolumePluginFactory func(config VolumeConfig) []VolumePlugin

// VolumeConfig is how volume plugins receive configuration.  An instance specific to the plugin will be passed to
// the plugin's factory func.  Reasonable defaults will be provided by the binary hosting
// the plugins while allowing override of those default values.  Those config values are then set to an instance of
// VolumeConfig and passed to the plugin.
//
// Values in VolumeConfig are intended to be relevant to several plugins, but not necessarily all plugins.  The
// preference is to leverage strong typing in this struct.  All config items must have a descriptive but non-specific
// name (i.e, RecyclerMinimumTimeout is OK but RecyclerMinimumTimeoutForNFS is !OK).  An instance of config will be
// given directly to the plugin, so config names specific to plugins are unneeded and wrongly expose plugins
// in this VolumeConfig struct.
//
// OtherAttributes is a map of string values intended for one-off configuration of a plugin or config that is only
// relevant to a single plugin.  All values are passed by string and require interpretation by the plugin.
// Passing config as strings is the least desirable option but can be used for truly one-off configuration.
// The binary should still use strong typing for this value when binding CLI values before they are passed as strings
// in OtherAttributes.
type VolumeConfig struct {
	// RecyclerPodTemplate is pod template that understands how to scrub clean a persistent volume after its release.
	// The template is used by plugins which override specific properties of the pod in accordance with that plugin.
	// See NewPersistentVolumeRecyclerPodTemplate for the properties that are expected to be overridden.
	RecyclerPodTemplate *api.Pod

	// RecyclerMinimumTimeout is the minimum amount of time in seconds for the recycler pod's ActiveDeadlineSeconds attribute.
	// Added to the minimum timeout is the increment per Gi of capacity.
	RecyclerMinimumTimeout int

	// RecyclerTimeoutIncrement is the number of seconds added to the recycler pod's ActiveDeadlineSeconds for each
	// Gi of capacity in the persistent volume.
	// Example: 5Gi volume x 30s increment = 150s + 30s minimum = 180s ActiveDeadlineSeconds for recycler pod
	RecyclerTimeoutIncrement int

	// VolumePluginDir specifies a directory where the volume plugin factory should look for additional plugins, if the
	// factory supports such functionality (e.g. FlexVolume)
	VolumePluginDir string

	// OtherAttributes stores config as strings.  These strings are opaque to the system and only understood by the binary
	// hosting the plugin and the plugin itself.
	OtherAttributes map[string]string
}

var (
	factories   map[string]VolumePluginFactory = make(map[string]VolumePluginFactory)
	factoryLock sync.Mutex
)

// RegisterFactory registers a plugin VolumePluginFactory by name. This is
// expected to happen during app startup.
func RegisterFactory(name string, factory VolumePluginFactory) {
	factoryLock.Lock()
	defer factoryLock.Unlock()

	_, found := factories[name]
	if found {
		glog.Fatalf("Volume plugin factory %q was registered twice", name)
	}
	glog.V(3).Infof("Registered volume plugin factory %q", name)
	factories[name] = factory
}

// CreateVolumePlugins instantiates volume plugins for given purpose from all
// registered factories. Returned plugins are not initialized!
func CreateVolumePlugins(configs map[string]VolumeConfig) []VolumePlugin {
	factoryLock.Lock()
	defer factoryLock.Unlock()

	allPlugins := []VolumePlugin{}

	for name, factory := range factories {
		glog.V(5).Infof("Running volume plugin factory %s", name)
		cfg, _ := configs[name]
		plugins := factory(cfg)
		allPlugins = append(allPlugins, plugins...)
	}
	return allPlugins
}
