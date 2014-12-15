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

package registry

import (
	_ "io" //FIXME:
	_ "os" //FIXME:
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/golang/glog"
)

// Plugin is a registry plugin.
type Plugin interface {
	// Name returns the top-level REST name of this plugin, e.g.
	// "replicationControllers".
	Name() string

	// Path is the desired path for this plugin within the backing
	// store.
	Path() string

	// New instantiates the plugin into the provided backing store.
	New(store Store) (apiserver.RESTStorage, error)
}

// All registered registries.
var pluginsMutex sync.Mutex
var pluginsMap = make(map[string]Plugin)

// RegisterPlugin registers a Plugin by name.  This is expected to happen
// during app startup.
func RegisterPlugin(plugin Plugin) {
	pluginsMutex.Lock()
	defer pluginsMutex.Unlock()

	name := plugin.Name()
	if name == "" {
		glog.Fatalf("Plugin has no name: %#v", plugin)
	}
	if plugin.Path() == "" {
		glog.Fatalf("Plugin %q was has no Path: %#v", name, plugin)
	}

	_, found := pluginsMap[name]
	if found {
		glog.Fatalf("Plugin %q was registered twice", name)
	}
	glog.V(1).Infof("Registered plugin %q", name)
	pluginsMap[name] = plugin
}

//FIXME: comment
func ForEachPlugin(fn func(plugin Plugin)) {
	pluginsMutex.Lock()
	defer pluginsMutex.Unlock()

	for _, v := range pluginsMap {
		fn(v)
	}
}
