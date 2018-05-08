/*
Copyright 2018 The Kubernetes Authors.

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

package openapi

import (
	"net/http"
	"sync"
)

// specCacher fetches the openapi spec once and then caches it in memory
type specCacher struct {
	rwMutex    sync.RWMutex
	cachedSpec Resources
	lastEtag   string

	// These will never change once the cacher is set
	downloader SpecDownloader
	parser     SpecParser
}

var _ SpecSource = &specCacher{}

// NewSpecCacher creates a new SpecCacher from a downloader and parser
func NewSpecCacher(downloader SpecDownloader, parser SpecParser) SpecSource {
	return &specCacher{
		downloader: downloader,
		parser:     parser,
	}
}

// Get implements SpecSource
func (s *specCacher) Get() (Resources, error) {
	err := s.updateCache()
	if err != nil {
		return nil, err
	}

	s.rwMutex.RLock()
	defer s.rwMutex.RUnlock()
	return s.cachedSpec, nil
}

// getLastEtag will take the read lock and return a copy of lastEtag
func (s *specCacher) getLastEtag() string {
	s.rwMutex.RLock()
	defer s.rwMutex.RUnlock()
	return s.lastEtag
}

// updateCache parses the OpenAPI spec if it has changed since the last
// time updateCache was called, and cache it.
func (s *specCacher) updateCache() error {
	lastEtag := s.getLastEtag()
	specBytes, newEtag, httpStatus, err := s.downloader.Download(lastEtag)
	if err != nil {
		return err
	}

	// The spec hasn't changed so there's nothing to update
	if httpStatus == http.StatusNotModified {
		return nil
	}

	s.rwMutex.Lock()
	defer s.rwMutex.Unlock()

	// Check etag again after taking the write lock in case lastEtag
	// was updated while waiting for the write lock
	if newEtag == s.lastEtag {
		return nil
	}

	// Convert raw spec to our Resource type
	spec, err := s.parser.Parse(specBytes)
	if err != nil {
		return err
	}

	s.cachedSpec = spec
	s.lastEtag = newEtag
	return nil
}
