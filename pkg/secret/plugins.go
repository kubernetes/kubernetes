/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package secret

import (
  "io"
  "os"
  "sort"
  "sync"

  "github.com/golang/glog"
  client "k8s.io/kubernetes/pkg/client/unversioned"
)

// Factory is a function that returns an Interface for a secret generator.
// The config parameter provides an io.Reader handler to the factory in
// order to load specific configurations. If no configuration is provided
// the parameter is nil.
type Factory func(client client.Interface, config io.Reader) (Interface, error)

// All registered admission options.
var (
  pluginsMutex    sync.Mutex
  pluginFactories = make(map[string]Factory)
  plugins         = make(map[string]Interface)
)

// GetPlugins enumerates the names of all registered plugins.
func GetPlugins() []string {
  pluginsMutex.Lock()
  defer pluginsMutex.Unlock()
  keys := []string{}
  for k := range plugins {
    keys = append(keys, k)
  }
  sort.Strings(keys)
  return keys
}

// RegisterPlugin registers a plugin Factory by name. This
// is expected to happen during app startup.
func RegisterPlugin(name string, factory Factory) {
  pluginsMutex.Lock()
  defer pluginsMutex.Unlock()
  _, found := pluginFactories[name]
  if found {
    glog.Fatalf("Secret generator plugin %q was registered twice", name)
  }
  glog.V(1).Infof("Registered secret generator plugin %q", name)
  pluginFactories[name] = factory
}

// GetPlugin returns an instance of the named plugin, or nil if the name is not
// known.
func GetPlugin(name string) Interface {
  pluginsMutex.Lock()
  defer pluginsMutex.Unlock()
  return plugins[name]
}

// InitPlugins initializes all requested plugins.
func InitPlugins(names []string, configFilePath string, client client.Interface) {
  var (
    config *os.File
    err    error
  )

  if len(names) == 0 {
    glog.Info("No secret generator plugin specified.")
    return
  }

  if configFilePath != "" {
    config, err = os.Open(configFilePath)
    if err != nil {
      glog.Fatalf("Couldn't open secret generator plugin configuration %s: %#v",
        configFilePath, err)
    }

    defer config.Close()
  }

  pluginsMutex.Lock()
  defer pluginsMutex.Unlock()
  for _, name := range names {
    if name != "" {
      f, found := pluginFactories[name]
      if !found {
        glog.Fatalf("Unknown secret generator plugin: %s", name)
      }
      plugin, err := f(client, config)
      if err != nil {
        glog.Fatalf("Couldn't init secret generator plugin %q: %v", name, err)
      }

      plugins[name] = plugin
    }
  }
}
