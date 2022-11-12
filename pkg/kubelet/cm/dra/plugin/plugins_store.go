/*
Copyright 2019 The Kubernetes Authors.

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

package plugin

import (
	"sync"

	utilversion "k8s.io/apimachinery/pkg/util/version"
)

// Plugin is a description of a DRA Plugin, defined by an endpoint
// and the highest DRA version supported.
type Plugin struct {
	endpoint                string
	highestSupportedVersion *utilversion.Version
}

// PluginsStore holds a list of DRA Plugins.
type PluginsStore struct {
	sync.RWMutex
	store map[string]*Plugin
}

// Get lets you retrieve a DRA Plugin by name.
// This method is protected by a mutex.
func (s *PluginsStore) Get(pluginName string) *Plugin {
	s.RLock()
	defer s.RUnlock()

	return s.store[pluginName]
}

// Set lets you save a DRA Plugin to the list and give it a specific name.
// This method is protected by a mutex.
func (s *PluginsStore) Set(pluginName string, plugin *Plugin) {
	s.Lock()
	defer s.Unlock()

	if s.store == nil {
		s.store = make(map[string]*Plugin)
	}

	s.store[pluginName] = plugin
}

// Delete lets you delete a DRA Plugin by name.
// This method is protected by a mutex.
func (s *PluginsStore) Delete(pluginName string) {
	s.Lock()
	defer s.Unlock()

	delete(s.store, pluginName)
}

// Clear deletes all entries in the store.
// This methiod is protected by a mutex.
func (s *PluginsStore) Clear() {
	s.Lock()
	defer s.Unlock()

	s.store = make(map[string]*Plugin)
}
