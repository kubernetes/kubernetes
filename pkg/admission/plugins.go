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

package admission

import (
	"io"
	"os"
	"sync"

	"github.com/golang/glog"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

// Factory is a function that returns an Interface for admission decisions.
// The config parameter provides an io.Reader handler to the factory in
// order to load specific configurations. If no configuration is provided
// the parameter is nil.
type Factory func(client client.Interface, config io.Reader) (Interface, error)

// All registered admission options.
var (
	pluginsMutex sync.Mutex
	plugins      = make(map[string]Factory)
)

// GetPlugins enumerates the names of all registered plugins.
func GetPlugins() []string {
	pluginsMutex.Lock()
	defer pluginsMutex.Unlock()
	keys := []string{}
	for k := range plugins {
		keys = append(keys, k)
	}
	return keys
}

// RegisterPlugin registers a plugin Factory by name. This
// is expected to happen during app startup.
func RegisterPlugin(name string, plugin Factory) {
	pluginsMutex.Lock()
	defer pluginsMutex.Unlock()
	_, found := plugins[name]
	if found {
		glog.Fatalf("Admission plugin %q was registered twice", name)
	}
	glog.V(1).Infof("Registered admission plugin %q", name)
	plugins[name] = plugin
}

// GetPlugin creates an instance of the named plugin, or nil if the name is not
// known. The error is returned only when the named provider was known but failed
// to initialize. The config parameter specifies the io.Reader handler of the
// configuration file for the cloud provider, or nil for no configuration.
func GetPlugin(name string, client client.Interface, config io.Reader) (Interface, error) {
	pluginsMutex.Lock()
	defer pluginsMutex.Unlock()
	f, found := plugins[name]
	if !found {
		return nil, nil
	}
	return f(client, config)
}

// InitPlugin creates an instance of the named interface.
func InitPlugin(name string, client client.Interface, configFilePath string) Interface {
	var (
		config *os.File
		err    error
	)

	if name == "" {
		glog.Info("No admission plugin specified.")
		return nil
	}

	if configFilePath != "" {
		config, err = os.Open(configFilePath)
		if err != nil {
			glog.Fatalf("Couldn't open admission plugin configuration %s: %#v",
				configFilePath, err)
		}

		defer config.Close()
	}

	plugin, err := GetPlugin(name, client, config)
	if err != nil {
		glog.Fatalf("Couldn't init admission plugin %q: %v", name, err)
	}
	if plugin == nil {
		glog.Fatalf("Unknown admission plugin: %s", name)
	}

	return plugin
}
