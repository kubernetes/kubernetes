// Copyright (c) 2025 Bart Venter <bartventer@proton.me>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package registry provides a registry for cache drivers.
package registry

import (
	"errors"
	"fmt"
	"maps"
	"net/url"
	"slices"
	"sync"

	"github.com/bartventer/httpcache/store/driver"
)

var ErrUnknownDriver = errors.New("store: unknown driver")

type driverRegistry struct {
	mu      sync.RWMutex
	drivers map[string]driver.Driver
}

func (dr *driverRegistry) RegisterDriver(name string, driver driver.Driver) {
	dr.mu.Lock()
	defer dr.mu.Unlock()
	if driver == nil {
		panic("store: Register driver is nil")
	}
	if _, dup := dr.drivers[name]; dup {
		panic("store: Register called twice for driver " + name)
	}
	dr.drivers[name] = driver
}

func (dr *driverRegistry) OpenConn(dsn string) (driver.Conn, error) {
	u, err := url.Parse(dsn)
	if err != nil {
		return nil, err
	}

	dr.mu.RLock()
	driver, ok := dr.drivers[u.Scheme]
	dr.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrUnknownDriver, u.Scheme)
	}

	return driver.Open(u)
}

func (dr *driverRegistry) Drivers() []string {
	dr.mu.RLock()
	defer dr.mu.RUnlock()
	return slices.Sorted(maps.Keys(dr.drivers))
}

var defaultRegistry = New()

func Default() *driverRegistry {
	return defaultRegistry
}

func New() *driverRegistry {
	return &driverRegistry{drivers: make(map[string]driver.Driver)}
}
