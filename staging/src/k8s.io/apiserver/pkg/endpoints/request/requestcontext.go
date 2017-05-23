/*
Copyright 2014 The Kubernetes Authors.

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

package request

import (
	"errors"
	"net/http"
	"sync"

	"github.com/golang/glog"
)

// LongRunningRequestCheck is a predicate which is true for long-running http requests.
type LongRunningRequestCheck func(r *http.Request, requestInfo *RequestInfo) bool

// RequestContextMapper keeps track of the context associated with a particular request
type RequestContextMapper interface {
	// Get returns the context associated with the given request (if any), and true if the request has an associated context, and false if it does not.
	Get(req *http.Request) (Context, bool)
	// Update maps the request to the given context. If no context was previously associated with the request, an error is returned.
	// Update should only be called with a descendant context of the previously associated context.
	// Updating to an unrelated context may return an error in the future.
	// The context associated with a request should only be updated by a limited set of callers.
	// Valid examples include the authentication layer, or an audit/tracing layer.
	Update(req *http.Request, context Context) error
}

type requestContextMap struct {
	contexts map[*http.Request]Context
	lock     sync.Mutex
}

// NewRequestContextMapper returns a new RequestContextMapper.
// The returned mapper must be added as a request filter using NewRequestContextFilter.
func NewRequestContextMapper() RequestContextMapper {
	return &requestContextMap{
		contexts: make(map[*http.Request]Context),
	}
}

// Get returns the context associated with the given request (if any), and true if the request has an associated context, and false if it does not.
// Get will only return a valid context when called from inside the filter chain set up by NewRequestContextFilter()
func (c *requestContextMap) Get(req *http.Request) (Context, bool) {
	c.lock.Lock()
	defer c.lock.Unlock()
	context, ok := c.contexts[req]
	return context, ok
}

// Update maps the request to the given context.
// If no context was previously associated with the request, an error is returned and the context is ignored.
func (c *requestContextMap) Update(req *http.Request, context Context) error {
	c.lock.Lock()
	defer c.lock.Unlock()
	if _, ok := c.contexts[req]; !ok {
		return errors.New("No context associated")
	}
	// TODO: ensure the new context is a descendant of the existing one
	c.contexts[req] = context
	return nil
}

// init maps the request to the given context and returns true if there was no context associated with the request already.
// if a context was already associated with the request, it ignores the given context and returns false.
// init is intentionally unexported to ensure that all init calls are paired with a remove after a request is handled
func (c *requestContextMap) init(req *http.Request, context Context) bool {
	c.lock.Lock()
	defer c.lock.Unlock()
	if _, exists := c.contexts[req]; exists {
		return false
	}
	c.contexts[req] = context
	return true
}

// remove is intentionally unexported to ensure that the context is not removed until a request is handled
func (c *requestContextMap) remove(req *http.Request) {
	c.lock.Lock()
	defer c.lock.Unlock()
	delete(c.contexts, req)
}

// WithRequestContext ensures there is a Context object associated with the request before calling the passed handler.
// After the passed handler runs, the context is cleaned up.
func WithRequestContext(handler http.Handler, mapper RequestContextMapper) http.Handler {
	rcMap, ok := mapper.(*requestContextMap)
	if !ok {
		glog.Fatal("Unknown RequestContextMapper implementation.")
	}

	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if rcMap.init(req, NewContext()) {
			// If we were the ones to successfully initialize, pair with a remove
			defer rcMap.remove(req)
		}
		handler.ServeHTTP(w, req)
	})
}

// IsEmpty returns true if there are no contexts registered, or an error if it could not be determined. Intended for use by tests.
func IsEmpty(requestsToContexts RequestContextMapper) (bool, error) {
	if requestsToContexts, ok := requestsToContexts.(*requestContextMap); ok {
		return len(requestsToContexts.contexts) == 0, nil
	}
	return true, errors.New("Unknown RequestContextMapper implementation")
}
