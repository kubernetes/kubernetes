// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package container

import (
	"fmt"
	"log"
	"sync"
)

type ContainerHandlerFactory interface {
	// Create a new ContainerHandler using this factory. CanHandle() must have returned true.
	NewContainerHandler(name string) (ContainerHandler, error)

	// Returns whether this factory can handle the specified container.
	CanHandle(name string) bool

	// Name of the factory.
	String() string
}

// TODO(vmarmol): Consider not making this global.
// Global list of factories.
var (
	factories     []ContainerHandlerFactory
	factoriesLock sync.RWMutex
)

// Register a ContainerHandlerFactory. These should be registered from least general to most general
// as they will be asked in order whether they can handle a particular container.
func RegisterContainerHandlerFactory(factory ContainerHandlerFactory) {
	factoriesLock.Lock()
	defer factoriesLock.Unlock()

	factories = append(factories, factory)
}

// Create a new ContainerHandler for the specified container.
func NewContainerHandler(name string) (ContainerHandler, error) {
	factoriesLock.RLock()
	defer factoriesLock.RUnlock()

	// Create the ContainerHandler with the first factory that supports it.
	for _, factory := range factories {
		if factory.CanHandle(name) {
			log.Printf("Using factory %q for container %q", factory.String(), name)
			return factory.NewContainerHandler(name)
		}
	}

	return nil, fmt.Errorf("no known factory can handle creation of container")
}

// Clear the known factories.
func ClearContainerHandlerFactories() {
	factoriesLock.Lock()
	defer factoriesLock.Unlock()

	factories = make([]ContainerHandlerFactory, 0, 4)
}
