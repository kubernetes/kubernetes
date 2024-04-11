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
	"errors"
	"sync"
)

// PluginsStore holds a list of DRA Plugins.
type pluginsStore struct {
	sync.RWMutex
	store map[string]*Plugin
}

// draPlugins map keeps track of all registered DRA plugins on the node
// and their corresponding sockets.
var draPlugins = &pluginsStore{}

// Get lets you retrieve a DRA Plugin by name.
// This method is protected by a mutex.
func (s *pluginsStore) get(pluginName string) *Plugin {
	s.RLock()
	defer s.RUnlock()

	return s.store[pluginName]
}

// Set lets you save a DRA Plugin to the list and give it a specific name.
// This method is protected by a mutex.
func (s *pluginsStore) add(pluginName string, p *Plugin) (replaced bool) {
	s.Lock()
	defer s.Unlock()

	if s.store == nil {
		s.store = make(map[string]*Plugin)
	}

	_, exists := s.store[pluginName]
	s.store[pluginName] = p
	return exists
}

// Delete lets you delete a DRA Plugin by name.
// This method is protected by a mutex.
func (s *pluginsStore) delete(pluginName string) *Plugin {
	s.Lock()
	defer s.Unlock()

	p, exists := s.store[pluginName]
	if !exists {
		return nil
	}
	if p.cancel != nil {
		p.cancel(errors.New("plugin got removed"))
	}
	delete(s.store, pluginName)

	return p
}
