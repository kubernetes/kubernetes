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

package storage

import (
	"fmt"
	"sort"

	info "github.com/google/cadvisor/lib/model"
)

type StorageDriver interface {
	AddStats(cInfo *info.ContainerInfo, stats *info.ContainerStats) error

	// Close will clear the state of the storage driver. The elements
	// stored in the underlying storage may or may not be deleted depending
	// on the implementation of the storage driver.
	Close() error
}

// The backend-storage-driver registry. The kubelet registers no drivers (it
// uses the in-memory cache only); the full cAdvisor binary's storage drivers
// (influxdb, kafka, …) register here. Keeping the registry in this single
// storage package means the library cache and the binary share one
// StorageDriver type and one set of storage_driver_* flags — no second package
// and no duplicate flag registration.

type StorageDriverFunc func() (StorageDriver, error)

var registeredPlugins = map[string](StorageDriverFunc){}

func RegisterStorageDriver(name string, f StorageDriverFunc) {
	registeredPlugins[name] = f
}

func New(name string) (StorageDriver, error) {
	if name == "" {
		return nil, nil
	}
	f, ok := registeredPlugins[name]
	if !ok {
		return nil, fmt.Errorf("unknown backend storage driver: %s", name)
	}
	return f()
}

func ListDrivers() []string {
	drivers := make([]string, 0, len(registeredPlugins))
	for name := range registeredPlugins {
		drivers = append(drivers, name)
	}
	sort.Strings(drivers)
	return drivers
}
