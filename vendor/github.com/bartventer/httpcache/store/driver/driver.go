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

// Package driver defines interfaces to be implemented by cache backends
// as used by the github.com/bartventer/httpcache/store package.
//
// # Implementing a Cache Backend
//
// To implement a custom cache backend, provide a type that implements the [Conn] interface.
//
// The [Driver] interface is used to create a [Conn] from a URL.
//
// The URL scheme determines which driver is used.
//
// Implementations must be safe for concurrent use by multiple goroutines.
// Example implementations can be found in sub-packages such as store/memcache
// and store/fscache.
package driver

import (
	"errors"
	"net/url"
)

// ErrNotExist is returned when a cache entry does not exist.
//
// Methods such as [Cache.Get] and [Cache.Delete] should return an error
// that satisfies errors.Is(err, store.ErrNotExist) if the entry is not found.
var ErrNotExist = errors.New("driver: entry does not exist")

// Driver is the interface implemented by cache backends that can create a [Conn]
// from a URL. The URL scheme determines which driver is used.
type Driver interface {
	Open(u *url.URL) (Conn, error)
}

type DriverFunc func(u *url.URL) (Conn, error)

func (f DriverFunc) Open(u *url.URL) (Conn, error) {
	return f(u)
}

// Conn describes the interface implemented by types that provide
// a connection to a cache backend. It allows for basic operations such as
// getting, setting, and deleting cache entries by key.
type Conn interface {
	// Get retrieves the cached value for the given key.
	// If the key does not exist, it should return an error satisfying
	// errors.Is(err, store.ErrNotExist).
	Get(key string) ([]byte, error)

	// Set stores the value for the given key.
	// If the key already exists, it should overwrite the existing value.
	Set(key string, value []byte) error

	// Delete removes the cached value for the given key.
	// If the key does not exist, it should return an error satisfying
	// errors.Is(err, store.ErrNotExist).
	Delete(key string) error
}
