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

package csi

import (
	"sync"

	utilversion "k8s.io/apimachinery/pkg/util/version"
)

// Driver is a description of a CSI Driver, defined by an endpoint and the
// highest CSI version supported
type Driver struct {
	endpoint                string
	highestSupportedVersion *utilversion.Version
}

// DriversStore holds a list of CSI Drivers
type DriversStore struct {
	store
	sync.RWMutex
}

type store map[string]Driver

// Get lets you retrieve a CSI Driver by name.
// This method is protected by a mutex.
func (s *DriversStore) Get(driverName string) (Driver, bool) {
	s.RLock()
	defer s.RUnlock()

	driver, ok := s.store[driverName]
	return driver, ok
}

// Set lets you save a CSI Driver to the list and give it a specific name.
// This method is protected by a mutex.
func (s *DriversStore) Set(driverName string, driver Driver) {
	s.Lock()
	defer s.Unlock()

	if s.store == nil {
		s.store = store{}
	}

	s.store[driverName] = driver
}

// Delete lets you delete a CSI Driver by name.
// This method is protected by a mutex.
func (s *DriversStore) Delete(driverName string) {
	s.Lock()
	defer s.Unlock()

	delete(s.store, driverName)
}

// Clear deletes all entries in the store.
// This methiod is protected by a mutex.
func (s *DriversStore) Clear() {
	s.Lock()
	defer s.Unlock()

	s.store = store{}
}
