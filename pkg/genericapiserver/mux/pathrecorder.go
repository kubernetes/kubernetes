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

package mux

import (
	"net/http"
)

// Mux is an object that can register http handlers.
type Mux interface {
	Handle(pattern string, handler http.Handler)
	HandleFunc(pattern string, handler func(http.ResponseWriter, *http.Request))
}

// PathRecorderMux wraps a mux object and records the registered paths. It is _not_ go routine safe.
type PathRecorderMux struct {
	mux   Mux
	paths []string
}

// NewPathRecorderMux creates a new PathRecorderMux with the given mux as the base mux.
func NewPathRecorderMux(mux Mux) *PathRecorderMux {
	return &PathRecorderMux{
		mux: mux,
	}
}

// BaseMux returns the underlying mux.
func (m *PathRecorderMux) BaseMux() Mux {
	return m.mux
}

// HandledPaths returns the registered handler paths.
func (m *PathRecorderMux) HandledPaths() []string {
	return append([]string{}, m.paths...)
}

// Handle registers the handler for the given pattern.
// If a handler already exists for pattern, Handle panics.
func (m *PathRecorderMux) Handle(path string, handler http.Handler) {
	m.paths = append(m.paths, path)
	m.mux.Handle(path, handler)
}

// HandleFunc registers the handler function for the given pattern.
func (m *PathRecorderMux) HandleFunc(path string, handler func(http.ResponseWriter, *http.Request)) {
	m.paths = append(m.paths, path)
	m.mux.HandleFunc(path, handler)
}
