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

	"k8s.io/klog/v2"

	"github.com/kcp-dev/logicalcluster/v3"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
)

// PathRecorderMux wraps a mux object and records the registered exposedPaths.
type PathRecorderMux struct {
	// name is used for logging so you can trace requests through
	name string

	lock            sync.Mutex
	notFoundHandler http.Handler
	pathToHandler   map[string]http.Handler
	prefixToHandler map[string]http.Handler

	// mux stores a pathHandler and is used to handle the actual serving.
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

// pathHandler is an http.Handler that will satisfy requests first by exact match, then by prefix,
// then by notFoundHandler
type pathHandler struct {
	// muxName is used for logging so you can trace requests through
	muxName string

	// pathToHandler is a map of exactly matching request to its handler
	pathToHandler map[string]http.Handler

	// this has to be sorted by most slashes then by length
	prefixHandlers []prefixHandler

	// notFoundHandler is the handler to use for satisfying requests with no other match
	notFoundHandler http.Handler
}

// prefixHandler holds the prefix it should match and the handler to use
type prefixHandler struct {
	// prefix is the prefix to test for a request match
	prefix string
	// handler is used to satisfy matching requests
	handler http.Handler
}

// NewPathRecorderMux creates a new PathRecorderMux
func NewPathRecorderMux(name string) *PathRecorderMux {
	ret := &PathRecorderMux{
		name:            name,
		pathToHandler:   map[string]http.Handler{},
		prefixToHandler: map[string]http.Handler{},
		mux:             atomic.Value{},
		exposedPaths:    []string{},
		pathStacks:      map[string]string{},
	}

	ret.mux.Store(&pathHandler{notFoundHandler: http.NotFoundHandler()})
	return ret
}

// ListedPaths returns the registered handler exposedPaths.
func (m *PathRecorderMux) ListedPaths(clusterName logicalcluster.Name) []string {
	m.lock.Lock()
	handledPaths := append([]string{}, m.exposedPaths...)
	m.lock.Unlock()

	sort.Strings(handledPaths)
	return handledPaths
}

func (m *PathRecorderMux) trackCallers(path string) {
	stack := string(debug.Stack())
	if existingStack, ok := m.pathStacks[path]; ok {
		utilruntime.HandleError(fmt.Errorf("duplicate path registration of %q: original registration from %v\n\nnew registration from %v", path, existingStack, stack))
	}
	m.pathStacks[path] = stack
}

// refreshMuxLocked creates a new mux and must be called while locked.  Otherwise the view of handlers may
// not be consistent
func (m *PathRecorderMux) refreshMuxLocked() {
	newMux := &pathHandler{
		muxName:         m.name,
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
	sort.Sort(sort.Reverse(byPrefixPriority(keys)))
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
	if !strings.HasSuffix(path, "/") {
		panic(fmt.Sprintf("%q must end in a trailing slash", path))
	}

	m.lock.Lock()
	defer m.lock.Unlock()
	m.trackCallers(path)

	m.exposedPaths = append(m.exposedPaths, path)
	m.prefixToHandler[path] = handler
	m.refreshMuxLocked()
}

// UnlistedHandlePrefix is like UnlistedHandle, but matches for anything under the path.  Like a standard golang trailing slash.
func (m *PathRecorderMux) UnlistedHandlePrefix(path string, handler http.Handler) {
	if !strings.HasSuffix(path, "/") {
		panic(fmt.Sprintf("%q must end in a trailing slash", path))
	}

	m.lock.Lock()
	defer m.lock.Unlock()
	m.trackCallers(path)

	m.prefixToHandler[path] = handler
	m.refreshMuxLocked()
}

// ServeHTTP makes it an http.Handler
func (m *PathRecorderMux) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	m.mux.Load().(*pathHandler).ServeHTTP(w, r)
}

// ServeHTTP makes it an http.Handler
func (h *pathHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if exactHandler, ok := h.pathToHandler[r.URL.Path]; ok {
		klog.V(5).Infof("%v: %q satisfied by exact match", h.muxName, r.URL.Path)
		exactHandler.ServeHTTP(w, r)
		return
	}

	for _, prefixHandler := range h.prefixHandlers {
		if strings.HasPrefix(r.URL.Path, prefixHandler.prefix) {
			klog.V(5).Infof("%v: %q satisfied by prefix %v", h.muxName, r.URL.Path, prefixHandler.prefix)
			prefixHandler.handler.ServeHTTP(w, r)
			return
		}
	}

	klog.V(5).Infof("%v: %q satisfied by NotFoundHandler", h.muxName, r.URL.Path)
	h.notFoundHandler.ServeHTTP(w, r)
}

// byPrefixPriority sorts url prefixes by the order in which they should be tested by the mux
// this has to be sorted by most slashes then by length so that we can iterate straight
// through to match the "best" one first.
type byPrefixPriority []string

func (s byPrefixPriority) Len() int      { return len(s) }
func (s byPrefixPriority) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s byPrefixPriority) Less(i, j int) bool {
	lhsNumParts := strings.Count(s[i], "/")
	rhsNumParts := strings.Count(s[j], "/")
	if lhsNumParts != rhsNumParts {
		return lhsNumParts < rhsNumParts
	}

	lhsLen := len(s[i])
	rhsLen := len(s[j])
	if lhsLen != rhsLen {
		return lhsLen < rhsLen
	}

	return strings.Compare(s[i], s[j]) < 0
}
