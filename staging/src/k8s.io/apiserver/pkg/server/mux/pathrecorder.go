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
	"sync"
	"sync/atomic"

	"github.com/gorilla/mux"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

// PathRecorderMux wraps a mux object and records the registered exposedPaths.
type PathRecorderMux struct {
	lock            sync.Mutex
	notFoundHandler http.Handler

	pathToHandler   map[string]http.Handler
	prefixToHandler map[string]http.Handler

	// mux stores a gorilla router and is used to handle the actual serving.
	// Turns out, we want to accept trailing slashes, BUT we don't care about handling
	// everything under them.  This does exactly what we need.
	mux atomic.Value

	// exposedPaths is the list of paths that should be shown at /
	exposedPaths []string

	// pathStacks holds the stacks of all registered paths.  This allows us to show a more helpful message
	// before the "http: multiple registrations for %s" panic.
	pathStacks map[string]string
}

// NewPathRecorderMux creates a new PathRecorderMux with the given mux as the base mux.
func NewPathRecorderMux() *PathRecorderMux {
	ret := &PathRecorderMux{
		pathToHandler:   map[string]http.Handler{},
		prefixToHandler: map[string]http.Handler{},
		mux:             atomic.Value{},
		exposedPaths:    []string{},
		pathStacks:      map[string]string{},
	}

	ret.mux.Store(mux.NewRouter().StrictSlash(true))
	return ret
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

// refreshMuxLocked creates a new mux and must be called while locked.  Otherwise the view of handlers may
// not be consistent
func (m *PathRecorderMux) refreshMuxLocked() {
	newMux := mux.NewRouter().StrictSlash(true)
	newMux.NotFoundHandler = m.notFoundHandler
	for path, handler := range m.pathToHandler {
		newMux.Handle(path, handler)
	}
	for pathPrefix, handler := range m.prefixToHandler {
		newMux.PathPrefix(pathPrefix).Handler(handler)
	}

	m.mux.Store(newMux)
}

// NotFoundHandler sets the handler to use if there's no match for a give path
func (m *PathRecorderMux) NotFoundHandler(notFoundHandler http.Handler) {
	m.lock.Lock()
	defer m.lock.Unlock()

	m.notFoundHandler = notFoundHandler

	m.refreshMuxLocked()
}

// Unregister removes a path from the mux.
func (m *PathRecorderMux) Unregister(path string) {
	m.lock.Lock()
	defer m.lock.Unlock()

	delete(m.pathToHandler, path)
	delete(m.prefixToHandler, path)
	delete(m.pathStacks, path)
	for i := range m.exposedPaths {
		if m.exposedPaths[i] == path {
			m.exposedPaths = append(m.exposedPaths[:i], m.exposedPaths[i+1:]...)
			break
		}
	}

	m.refreshMuxLocked()
}

// Handle registers the handler for the given pattern.
// If a handler already exists for pattern, Handle panics.
func (m *PathRecorderMux) Handle(path string, handler http.Handler) {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.trackCallers(path)

	m.exposedPaths = append(m.exposedPaths, path)
	m.pathToHandler[path] = handler
	m.refreshMuxLocked()
}

// HandleFunc registers the handler function for the given pattern.
// If a handler already exists for pattern, Handle panics.
func (m *PathRecorderMux) HandleFunc(path string, handler func(http.ResponseWriter, *http.Request)) {
	m.Handle(path, http.HandlerFunc(handler))
}

// UnlistedHandle registers the handler for the given pattern, but doesn't list it.
// If a handler already exists for pattern, Handle panics.
func (m *PathRecorderMux) UnlistedHandle(path string, handler http.Handler) {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.trackCallers(path)

	m.pathToHandler[path] = handler
	m.refreshMuxLocked()
}

// UnlistedHandleFunc registers the handler function for the given pattern, but doesn't list it.
// If a handler already exists for pattern, Handle panics.
func (m *PathRecorderMux) UnlistedHandleFunc(path string, handler func(http.ResponseWriter, *http.Request)) {
	m.UnlistedHandle(path, http.HandlerFunc(handler))
}

// HandlePrefix is like Handle, but matches for anything under the path.  Like a standard golang trailing slash.
func (m *PathRecorderMux) HandlePrefix(path string, handler http.Handler) {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.trackCallers(path)

	m.exposedPaths = append(m.exposedPaths, path)
	m.prefixToHandler[path] = handler
	m.refreshMuxLocked()
}

// UnlistedHandlePrefix is like UnlistedHandle, but matches for anything under the path.  Like a standard golang trailing slash.
func (m *PathRecorderMux) UnlistedHandlePrefix(path string, handler http.Handler) {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.trackCallers(path)

	m.prefixToHandler[path] = handler
	m.refreshMuxLocked()
}

// ServeHTTP makes it an http.Handler
func (m *PathRecorderMux) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	m.mux.Load().(*mux.Router).ServeHTTP(w, r)
}
