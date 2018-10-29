/*
Copyright 2018 The Kubernetes Authors.

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

/*
TL,DR: This code is only here for a transition phase. We need some changes in
the kubelete, after those this code should go away again and the global stuff
in the csi package is not needed anymore.

The CSIPlugin and the registration handler for the plugin need access to some
shared data (specifically the registered drivers and the node info manager). To
enable this and remove some global state, we decided to unify the plugin and
its registration handler, the plugin also implements the methods for a
regstration handler and thus can be used as such.

With the kubelet as it stands right now, we cannot register the plugin directly
with the plugin watcher, we do not have a reference to the CSI plugin at hand at
this point in time.

To fix that, we plan to change the kubelet, to query the plugin from the volume
plugin manager for the reference to the CSI plugin. We can then use that for
registering it with the plugin watcher without relying on global state in the
csi package -- we can remove this singleton then.

This means, we need 3 PRs to achieve that:
1.) This code change, unifying The Plugin & the registration handler, remove all
    global state in the csi package except ensure a singleton plugin, regardless if
    we go in via the plugin.Init() or the pluginHandler.RegisterPlugin() route.
2.) Make kubelet changes: allow to register the CSI plugin (we already have a
    reference to in the volume plugin manager) with the plugin watcher.
3.) Remove this singleton again.

*/

package csi

import (
	"sync"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

type regHandler struct{}

func (r regHandler) ValidatePlugin(pluginName string, endpoint string, versions []string) error {
	return getPluginSingleton().ValidatePlugin(pluginName, endpoint, versions)
}

func (r regHandler) RegisterPlugin(pluginName string, endpoint string) error {
	return getPluginSingleton().RegisterPlugin(pluginName, endpoint)
}

func (r regHandler) DeRegisterPlugin(pluginName string) {
	getPluginSingleton().DeRegisterPlugin(pluginName)
}

var pluginInitOnce sync.Once
var instance *plugin

func getPluginSingleton() *plugin {
	pluginInitOnce.Do(func() {
		instance = &plugin{
			host:         nil,
			blockEnabled: utilfeature.DefaultFeatureGate.Enabled(features.CSIBlockVolume),
		}
	})
	return instance
}

var PluginHandler = &regHandler{}
