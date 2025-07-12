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

// Package expapi provides HTTP handlers for managing and interacting with cache backends.
//
// WARNING: This package is intended for debugging, development, or administrative use only.
// It is NOT recommended to expose these endpoints in production environments, as they
// allow direct access to cache contents and deletion.
//
// Endpoints:
//
//	GET    /debug/httpcache           -- List cache keys (if supported)
//	GET    /debug/httpcache/{key}     -- Retrieve a cache entry
//	DELETE /debug/httpcache/{key}     -- Delete a cache entry
//
// Backends that implement the [KeyLister] interface will support key listing.
// All handlers expect a "dsn" query parameter to select the cache backend.
package expapi

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/bartventer/httpcache/store/driver"
	"github.com/bartventer/httpcache/store/internal/registry"
)

type connOpener interface {
	OpenConn(dsn string) (driver.Conn, error)
}

type storeService struct {
	co connOpener
}

// KeyLister is an optional interface implemented by cache backends that
// support listing keys. It provides a method to retrieve all keys in the
// cache that match a given prefix.
type KeyLister interface {
	Keys(prefix string) ([]string, error)
}

func keyFromRequest(r *http.Request) string { return r.PathValue("key") }

func connHandler(co connOpener, handler func(driver.Conn) http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		dsn := r.URL.Query().Get("dsn")
		conn, err := co.OpenConn(dsn)
		if err != nil {
			http.Error(w, "failed to open cache: "+err.Error(), http.StatusInternalServerError)
			return
		}
		handler(conn).ServeHTTP(w, r)
	})
}

func list(conn driver.Conn) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		kl, ok := conn.(KeyLister)
		if !ok {
			http.Error(w, "cache does not support listing keys", http.StatusNotImplemented)
			return
		}
		prefix := r.URL.Query().Get("prefix")
		keys, err := kl.Keys(prefix)
		if err != nil {
			http.Error(
				w,
				fmt.Sprintf("failed to list keys: %v", err),
				http.StatusInternalServerError,
			)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string][]string{"keys": keys})
	})
}

func retrieve(conn driver.Conn) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := keyFromRequest(r)
		value, err := conn.Get(key)
		if err != nil {
			if errors.Is(err, driver.ErrNotExist) {
				http.Error(w, fmt.Sprintf("key %q not found", key), http.StatusNotFound)
			} else {
				http.Error(w, fmt.Sprintf("failed to get value for key %q: %v", key, err), http.StatusInternalServerError)
			}
			return
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		w.WriteHeader(http.StatusOK)
		if _, err := w.Write(value); err != nil {
			http.Error(
				w,
				fmt.Sprintf("failed to write response: %v", err),
				http.StatusInternalServerError,
			)
		}
	})
}

func destroy(conn driver.Conn) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := keyFromRequest(r)
		if err := conn.Delete(key); err != nil {
			if errors.Is(err, driver.ErrNotExist) {
				http.Error(w, fmt.Sprintf("key %q not found", key), http.StatusNotFound)
			} else {
				http.Error(w, fmt.Sprintf("failed to delete value for key %q: %v", key, err), http.StatusInternalServerError)
			}
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})
}

type registerConfig struct {
	Mux *http.ServeMux
}

type RegisterOption interface {
	apply(*registerConfig)
}

type registerOptionFunc func(*registerConfig)

func (f registerOptionFunc) apply(cfg *registerConfig) {
	f(cfg)
}

// WithServeMux allows specifying a custom http.ServeMux for the HTTP cache API
// handlers; default: [http.DefaultServeMux].
func WithServeMux(mux *http.ServeMux) RegisterOption {
	return registerOptionFunc(func(cfg *registerConfig) {
		cfg.Mux = mux
	})
}

func (m *storeService) Register(opts ...RegisterOption) {
	cfg := &registerConfig{
		Mux: http.DefaultServeMux,
	}
	for _, opt := range opts {
		opt.apply(cfg)
	}
	mux := cfg.Mux
	mux.Handle("GET /debug/httpcache", connHandler(m.co, list))
	mux.Handle("GET /debug/httpcache/{key}", connHandler(m.co, retrieve))
	mux.Handle("DELETE /debug/httpcache/{key}", connHandler(m.co, destroy))
}

var defaultService = &storeService{co: registry.Default()}

// Register registers the HTTP cache API handlers with the provided options.
func Register(opts ...RegisterOption) { defaultService.Register(opts...) }

// ListHandler returns the list handler for the HTTP cache API.
//
// This is only needed to install the handler in a non-standard location.
func ListHandler() http.Handler { return connHandler(defaultService.co, list) }

// RetrieveHandler returns the retrieve handler for the HTTP cache API.
//
// This is only needed to install the handler in a non-standard location.
func RetrieveHandler() http.Handler { return connHandler(defaultService.co, retrieve) }

// DestroyHandler returns the destroy handler for the HTTP cache API.
//
// This is only needed to install the handler in a non-standard location.
func DestroyHandler() http.Handler { return connHandler(defaultService.co, destroy) }
