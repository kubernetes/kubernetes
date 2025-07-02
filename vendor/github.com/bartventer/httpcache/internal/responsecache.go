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

package internal

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/bartventer/httpcache/store/driver"
)

type Cache = driver.Conn

// ResponseCache is an interface for caching HTTP responses.
// It provides methods to delete any cached item by its key,
// retrieve and set full cached responses, and manage references associated
// with a given URL key.
type ResponseCache interface {
	Get(key string, req *http.Request) (*Response, error)
	Set(key string, entry *Response) error
	Delete(key string) error
	GetRefs(key string) (ResponseRefs, error)
	SetRefs(key string, refs ResponseRefs) error
}

type responseCache struct {
	cache Cache
}

func NewResponseCache(cache Cache) *responseCache {
	return &responseCache{cache}
}

var _ ResponseCache = (*responseCache)(nil)

type CacheError struct {
	Op      string
	Message string
	Err     error
}

func (c *CacheError) Error() string { return fmt.Sprintf("%s: %s: %v", c.Op, c.Message, c.Err) }
func (c *CacheError) Unwrap() error { return c.Err }

func (c CacheError) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("op", c.Op),
		slog.String("message", c.Message),
		slog.Any("error", c.Err),
	)
}

func newCacheError(err error, op, message string) *CacheError {
	return &CacheError{Op: op, Message: message, Err: err}
}

var _ slog.LogValuer = (*CacheError)(nil)

func (r *responseCache) Get(responseKey string, req *http.Request) (*Response, error) {
	data, err := r.cache.Get(responseKey)
	if err != nil {
		return nil, err
	}
	entry, err := ParseResponse(data, req)
	if err != nil {
		return nil, newCacheError(
			err,
			"Get",
			fmt.Sprintf("failed to unmarshal cached entry for key %q", responseKey),
		)
	}
	return entry, nil
}

func (r *responseCache) Set(responseKey string, entry *Response) error {
	data, err := entry.MarshalBinary()
	if err != nil {
		return newCacheError(
			err,
			"Set",
			fmt.Sprintf("failed to marshal entry for key %q", responseKey),
		)
	}
	return r.cache.Set(responseKey, data)
}

func (r *responseCache) Delete(key string) error {
	return r.cache.Delete(key)
}

func (r *responseCache) GetRefs(urlKey string) (ResponseRefs, error) {
	data, err := r.cache.Get(urlKey)
	if err != nil {
		return nil, err
	}
	var refs ResponseRefs
	if unmarshalErr := json.Unmarshal(data, &refs); unmarshalErr != nil {
		return nil, newCacheError(
			unmarshalErr,
			"GetRefs",
			fmt.Sprintf("failed to unmarshal cached refs for key %q", urlKey),
		)
	}
	return refs, nil
}

func (r *responseCache) SetRefs(urlKey string, refs ResponseRefs) error {
	data, err := json.Marshal(refs)
	if err != nil {
		return newCacheError(
			err,
			"SetRefs",
			fmt.Sprintf("failed to marshal refs for key %q", urlKey),
		)
	}
	return r.cache.Set(urlKey, data)
}
