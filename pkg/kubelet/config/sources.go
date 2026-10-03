/*
Copyright 2016 The Kubernetes Authors.

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

// Package config implements the pod configuration readers.
package config

import (
	"sync"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
)

// SourcesReadyFn is function that returns true if the specified sources have been seen.
type SourcesReadyFn func(sourcesSeen sets.Set[string]) bool

// SourceForPodReadyFn is a function that returns true if the source for the specified pod UID is ready.
type SourceForPodReadyFn func(uid types.UID) bool

// SourcesReady tracks the set of configured sources seen by the kubelet.
type SourcesReady interface {
	// AddSource adds the specified source to the set of sources managed.
	AddSource(source string)
	// AllReady returns true if the currently configured sources have all been seen.
	AllReady() bool

	// SourceForPodReady returns true if the source for the specified pod UID is ready
	SourceForPodReady(uid types.UID) bool
}

// NewSourcesReady returns a SourcesReady with the specified function.
func NewSourcesReady(sourcesReadyFn SourcesReadyFn, sourceForPodReadyFn SourceForPodReadyFn) SourcesReady {
	return &sourcesImpl{
		sourcesSeen:         sets.New[string](),
		sourcesReadyFn:      sourcesReadyFn,
		sourceForPodReadyFn: sourceForPodReadyFn,
	}
}

// sourcesImpl implements SourcesReady.  It is thread-safe.
type sourcesImpl struct {
	// lock protects access to sources seen.
	lock sync.RWMutex
	// set of sources seen.
	sourcesSeen sets.Set[string]
	// sourcesReady is a function that evaluates if the sources are ready.
	sourcesReadyFn SourcesReadyFn
	// sourceForPodReadyFn is a function that evaluates if the sources for a specific pod is ready
	sourceForPodReadyFn SourceForPodReadyFn
}

// Add adds the specified source to the set of sources managed.
func (s *sourcesImpl) AddSource(source string) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.sourcesSeen.Insert(source)
}

// AllReady returns true if each configured source is ready.
func (s *sourcesImpl) AllReady() bool {
	s.lock.RLock()
	defer s.lock.RUnlock()
	return s.sourcesReadyFn(s.sourcesSeen)
}

// SourceForPodReady returns true if the source for the specified pod UID is ready
func (s *sourcesImpl) SourceForPodReady(uid types.UID) bool {
	s.lock.RLock()
	defer s.lock.RUnlock()
	if s.sourcesReadyFn(s.sourcesSeen) {
		return true
	}
	return s.sourceForPodReadyFn(uid)
}
