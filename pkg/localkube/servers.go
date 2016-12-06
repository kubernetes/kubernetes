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

package localkube

import (
	"fmt"
)

// Servers allows operations to be performed on many servers at once.
// Uses slice to preserve ordering.
type Servers []Server

// Get returns a server matching name, returns nil if server doesn't exit.
func (servers Servers) Get(name string) (Server, error) {
	for _, server := range servers {
		if server.Name() == name {
			return server, nil
		}
	}
	return nil, fmt.Errorf("server '%s' does not exist", name)
}

// StartAll starts all services, starting from 0th item and ascending.
func (servers Servers) StartAll() {
	for _, server := range servers {
		fmt.Printf("Starting %s...\n", server.Name())
		server.Start()
	}
}

// StopAll stops all services, starting with the last item.
func (servers Servers) StopAll() {
	for i := len(servers) - 1; i >= 0; i-- {
		server := servers[i]
		fmt.Printf("Stopping %s...\n", server.Name())
		server.Stop()
	}
}

// Start is a helper method to start the Server specified, returns error if server doesn't exist.
func (servers Servers) Start(serverName string) error {
	server, err := servers.Get(serverName)
	if err != nil {
		return err
	}

	server.Start()
	return nil
}

// Stop is a helper method to start the Server specified, returns error if server doesn't exist.
func (servers Servers) Stop(serverName string) error {
	server, err := servers.Get(serverName)
	if err != nil {
		return err
	}

	server.Stop()
	return nil
}
