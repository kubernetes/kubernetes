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
	"strings"
	"sync"
	"sync/atomic"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
)

// PathRecorderMux wraps a mux object and records the registered exposedPaths.
type PathRecorderMux struct {
	lock            sync.Mutex
	notFoundHandler http.Handler

	pathToHandler   map[string]http.Handler
	prefixToHandler map[string]http.Handler

	// mux stores a custom router and is used to handle the actual serving.
	// Turns out, we want to accept trailing slashes, BUT we don't care about handling
	// everything under them.  This does exactly matches only unless its explicitly requested to
	// do something different
	mux atomic.Value

	// exposedPaths is the list of paths that should be shown at /
	exposedPaths []string

	// pathStacks holds the stacks of all registered paths.  This allows us to show a more helpful message
	// before the "http: multiple registrations for %s" panic.
	pathStacks map[string]string
}

type pathHandler struct {
	pathToHandler map[string]http.Handler

	// this has to be sorted by most slashes then by length
	prefixHandlers []prefixHandler

	notFoundHandler http.Handler
}

type prefixHandler struct {
	prefix  string
	handler http.Handler
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

	ret.mux.Store(pathHandler{notFoundHandler: http.NotFoundHandler()})
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
	newMux := pathHandler{
		pathToHandler:   map[string]http.Handler{},
		prefixHandlers:  []prefixHandler{},
		notFoundHandler: http.NotFoundHandler(),
	}
	if m.notFoundHandler != nil {
		newMux.notFoundHandler = m.notFoundHandler
	}
	for path, handler := range m.pathToHandler {
		newMux.pathToHandler[path] = handler
	}

	keys := sets.StringKeySet(m.prefixToHandler).List()
	sort.Sort(byPriority(keys))
	for _, prefix := range keys {
		newMux.prefixHandlers = append(newMux.prefixHandlers, prefixHandler{
			prefix:  prefix,
			handler: m.prefixToHandler[prefix],
		})
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
	m.mux.Load().(pathHandler).ServeHTTP(w, r)
}

// ServeHTTP makes it an http.Handler
func (h pathHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if exactHandler, ok := h.pathToHandler[r.URL.Path]; ok {
		exactHandler.ServeHTTP(w, r)
		return
	}

	for _, prefixHandler := range h.prefixHandlers {
		if strings.HasPrefix(r.URL.Path, prefixHandler.prefix) {
			prefixHandler.handler.ServeHTTP(w, r)
			return
		}
	}

	h.notFoundHandler.ServeHTTP(w, r)
}

// byPriority sorts url prefixes by the order in which they should be tested by the mux
type byPriority []string

func (s byPriority) Len() int      { return len(s) }
func (s byPriority) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s byPriority) Less(i, j int) bool {
	lhsNumParts := strings.Count(s[i], "/")
	rhsNumParts := strings.Count(s[j], "/")
	if lhsNumParts < rhsNumParts {
		return true
	}
	if lhsNumParts > rhsNumParts {
		return false
	}

	lhsLen := len(s[i])
	rhsLen := len(s[j])
	if lhsLen < rhsLen {
		return true
	}
	if lhsLen > rhsLen {
		return false
	}

	return strings.Compare(s[i], s[j]) < 0
}
