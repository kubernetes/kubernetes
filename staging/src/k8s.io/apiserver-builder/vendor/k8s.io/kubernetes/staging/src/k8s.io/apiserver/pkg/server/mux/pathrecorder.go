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
	"fmt"
	"net/http"
	"runtime/debug"
	"sort"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

// PathRecorderMux wraps a mux object and records the registered exposedPaths. It is _not_ go routine safe.
type PathRecorderMux struct {
	mux          *http.ServeMux
	exposedPaths []string

	// pathStacks holds the stacks of all registered paths.  This allows us to show a more helpful message
	// before the "http: multiple registrations for %s" panic.
	pathStacks map[string]string
}

// NewPathRecorderMux creates a new PathRecorderMux with the given mux as the base mux.
func NewPathRecorderMux() *PathRecorderMux {
	return &PathRecorderMux{
		mux:        http.NewServeMux(),
		pathStacks: map[string]string{},
	}
}

// ListedPaths returns the registered handler exposedPaths.
func (m *PathRecorderMux) ListedPaths() []string {
	handledPaths := append([]string{}, m.exposedPaths...)
	sort.Strings(handledPaths)

	return handledPaths
}

func (m *PathRecorderMux) trackCallers(path string) {
	if existingStack, ok := m.pathStacks[path]; ok {
		utilruntime.HandleError(fmt.Errorf("registered %q from %v", path, existingStack))
	}
	m.pathStacks[path] = string(debug.Stack())
}

// Handle registers the handler for the given pattern.
// If a handler already exists for pattern, Handle panics.
func (m *PathRecorderMux) Handle(path string, handler http.Handler) {
	m.trackCallers(path)

	m.exposedPaths = append(m.exposedPaths, path)
	m.mux.Handle(path, handler)
}

// HandleFunc registers the handler function for the given pattern.
// If a handler already exists for pattern, Handle panics.
func (m *PathRecorderMux) HandleFunc(path string, handler func(http.ResponseWriter, *http.Request)) {
	m.trackCallers(path)

	m.exposedPaths = append(m.exposedPaths, path)
	m.mux.HandleFunc(path, handler)
}

// UnlistedHandle registers the handler for the given pattern, but doesn't list it.
// If a handler already exists for pattern, Handle panics.
func (m *PathRecorderMux) UnlistedHandle(path string, handler http.Handler) {
	m.trackCallers(path)

	m.mux.Handle(path, handler)
}

// UnlistedHandleFunc registers the handler function for the given pattern, but doesn't list it.
// If a handler already exists for pattern, Handle panics.
func (m *PathRecorderMux) UnlistedHandleFunc(path string, handler func(http.ResponseWriter, *http.Request)) {
	m.trackCallers(path)

	m.mux.HandleFunc(path, handler)
}

// ServeHTTP makes it an http.Handler
func (m *PathRecorderMux) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	m.mux.ServeHTTP(w, r)
}
