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
	"fmt"
	"slices"
	"sync"
)

// Store keeps track of how to reach plugins registered for DRA drivers.
// Each plugin has a gRPC endpoint. There may be more than one plugin per driver.
//
// The null Store is usable.
type Store struct {
	sync.RWMutex
	// plugin name -> Plugin in the order in which they got added
	store map[string][]*Plugin
}

// NewDRAPluginClient returns a wrapper around those gRPC methods of a DRA
// driver kubelet plugin which need to be called by kubelet. The wrapper
// handles gRPC connection management and logging. Connections are reused
// across different NewDRAPluginClient calls.
//
// It returns an informative error message including the driver name
// with an explanation why the driver is not usable.
func (s *Store) NewDRAPluginClient(driverName string) (*Plugin, error) {
	if driverName == "" {
		return nil, errors.New("DRA driver name is empty")
	}
	client := s.get(driverName)
	if client == nil {
		return nil, fmt.Errorf("DRA driver %s is not registered", driverName)
	}
	return client, nil
}

// Get lets you retrieve a DRA Plugin by name.
// This method is protected by a mutex.
func (s *Store) get(pluginName string) *Plugin {
	s.RLock()
	defer s.RUnlock()

	instances := s.store[pluginName]
	if len(instances) == 0 {
		return nil
	}
	// Heuristic: pick the most recent one. It's most likely
	// the newest, except when kubelet got restarted and registered
	// all running plugins in random order.
	return instances[len(instances)-1]
}

// Set lets you save a DRA Plugin to the list and give it a specific name.
// This method is protected by a mutex.
func (s *Store) add(p *Plugin) error {
	s.Lock()
	defer s.Unlock()

	if s.store == nil {
		s.store = make(map[string][]*Plugin)
	}
	for _, oldP := range s.store[p.name] {
		if oldP.endpoint == p.endpoint {
			// One plugin instance cannot hijack the endpoint of another instance.
			return fmt.Errorf("endpoint %s already registered for plugin %s", p.endpoint, p.name)
		}
	}
	s.store[p.name] = append(s.store[p.name], p)
	return nil
}

// remove lets you remove one endpoint for a DRA Plugin.
// This method is protected by a mutex. It returns the
// plugin if found and true if that was the last instance
func (s *Store) remove(pluginName, endpoint string) (*Plugin, bool) {
	s.Lock()
	defer s.Unlock()

	instances := s.store[pluginName]
	i := slices.IndexFunc(instances, func(p *Plugin) bool { return p.endpoint == endpoint })
	if i == -1 {
		return nil, false
	}
	p := instances[i]
	last := len(instances) == 1
	if last {
		delete(s.store, pluginName)
	} else {
		s.store[pluginName] = slices.Delete(instances, i, i+1)
	}

	if p.cancel != nil {
		p.cancel(errors.New("plugin got removed"))
	}
	return p, last
}
