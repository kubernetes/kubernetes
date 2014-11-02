/*
Copyright 2014 Google Inc. All rights reserved.

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

package apiserver

import (
	"fmt"
	"net/http"
	"regexp"
	"runtime/debug"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authorizer"
	authhandlers "github.com/GoogleCloudPlatform/kubernetes/pkg/auth/handlers"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/httplog"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

// specialVerbs contains just strings which are used in REST paths for special actions that don't fall under the normal
// CRUDdy GET/POST/PUT/DELETE actions on REST objects.
// TODO: find a way to keep this up to date automatically.  Maybe dynamically populate list as handlers added to
// master's Mux.
var specialVerbs = map[string]bool{
	"proxy":    true,
	"redirect": true,
	"watch":    true,
}

// KindFromRequest returns Kind if Kind can be extracted from the request.  Otherwise, the empty string.
func KindFromRequest(req http.Request) string {
	// TODO: find a way to keep this code's assumptions about paths up to date with changes in the code.  Maybe instead
	// of directly adding handler's code to the master's Mux, have a function which forces the structure when adding
	// them.
	parts := splitPath(req.URL.Path)
	if len(parts) > 2 && parts[0] == "api" {
		if _, ok := specialVerbs[parts[2]]; ok {
			if len(parts) > 3 {
				return parts[3]
			}
		} else {
			return parts[2]
		}
	}
	return ""
}

// IsReadOnlyReq() is true for any (or at least many) request which has no observable
// side effects on state of apiserver (though there may be internal side effects like
// caching and logging).
func IsReadOnlyReq(req http.Request) bool {
	if req.Method == "GET" {
		// TODO: add OPTIONS and HEAD if we ever support those.
		return true
	}
	return false
}

// ReadOnly passes all GET requests on to handler, and returns an error on all other requests.
func ReadOnly(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if IsReadOnlyReq(*req) {
			handler.ServeHTTP(w, req)
			return
		}
		w.WriteHeader(http.StatusForbidden)
		fmt.Fprintf(w, "This is a read-only endpoint.")
	})
}

// RateLimit uses rl to rate limit accepting requests to 'handler'.
func RateLimit(rl util.RateLimiter, handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if rl.CanAccept() {
			handler.ServeHTTP(w, req)
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprintf(w, "Rate limit exceeded.")
	})
}

// RecoverPanics wraps an http Handler to recover and log panics.
func RecoverPanics(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer func() {
			if x := recover(); x != nil {
				w.WriteHeader(http.StatusInternalServerError)
				fmt.Fprint(w, "apis panic. Look in log for details.")
				glog.Infof("APIServer panic'd on %v %v: %v\n%s\n", req.Method, req.RequestURI, x, debug.Stack())
			}
		}()
		defer httplog.NewLogged(req, &w).StacktraceWhen(
			httplog.StatusIsNot(
				http.StatusOK,
				http.StatusCreated,
				http.StatusAccepted,
				http.StatusMovedPermanently,
				http.StatusTemporaryRedirect,
				http.StatusConflict,
				http.StatusNotFound,
				StatusUnprocessableEntity,
			),
		).Log()

		// Dispatch to the internal handler
		handler.ServeHTTP(w, req)
	})
}

// Simple CORS implementation that wraps an http Handler
// For a more detailed implementation use https://github.com/martini-contrib/cors
// or implement CORS at your proxy layer
// Pass nil for allowedMethods and allowedHeaders to use the defaults
func CORS(handler http.Handler, allowedOriginPatterns []*regexp.Regexp, allowedMethods []string, allowedHeaders []string, allowCredentials string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		origin := req.Header.Get("Origin")
		if origin != "" {
			allowed := false
			for _, pattern := range allowedOriginPatterns {
				if allowed = pattern.MatchString(origin); allowed {
					break
				}
			}
			if allowed {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				// Set defaults for methods and headers if nothing was passed
				if allowedMethods == nil {
					allowedMethods = []string{"POST", "GET", "OPTIONS", "PUT", "DELETE"}
				}
				if allowedHeaders == nil {
					allowedHeaders = []string{"Content-Type", "Content-Length", "Accept-Encoding", "X-CSRF-Token", "Authorization", "X-Requested-With", "If-Modified-Since"}
				}
				w.Header().Set("Access-Control-Allow-Methods", strings.Join(allowedMethods, ", "))
				w.Header().Set("Access-Control-Allow-Headers", strings.Join(allowedHeaders, ", "))
				w.Header().Set("Access-Control-Allow-Credentials", allowCredentials)

				// Stop here if its a preflight OPTIONS request
				if req.Method == "OPTIONS" {
					w.WriteHeader(http.StatusNoContent)
					return
				}
			}
		}
		// Dispatch to the next handler
		handler.ServeHTTP(w, req)
	})
}

// RequestAttributeGetter is a function that extracts authorizer.Attributes from an http.Request
type RequestAttributeGetter interface {
	GetAttribs(req *http.Request) (attribs authorizer.Attributes)
}

type requestAttributeGetter struct {
	userContexts authhandlers.RequestContext
}

// NewAttributeGetter returns an object which implements the RequestAttributeGetter interface.
func NewRequestAttributeGetter(userContexts authhandlers.RequestContext) RequestAttributeGetter {
	return &requestAttributeGetter{userContexts}
}

func (r *requestAttributeGetter) GetAttribs(req *http.Request) authorizer.Attributes {
	attribs := authorizer.AttributesRecord{}

	user, ok := r.userContexts.Get(req)
	if ok {
		attribs.User = user
	}

	attribs.ReadOnly = IsReadOnlyReq(*req)

	// If a path follows the conventions of the REST object store, then
	// we can extract the object Kind.  Otherwise, not.
	attribs.Kind = KindFromRequest(*req)

	// If the request specifies a namespace, then the namespace is filled in.
	// Assumes there is no empty string namespace.  Unspecified results
	// in empty (does not understand defaulting rules.)
	attribs.Namespace = req.URL.Query().Get("namespace")

	return &attribs
}

// WithAuthorizationCheck passes all authorized requests on to handler, and returns a forbidden error otherwise.
func WithAuthorizationCheck(handler http.Handler, getAttribs RequestAttributeGetter, a authorizer.Authorizer) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		err := a.Authorize(getAttribs.GetAttribs(req))
		if err == nil {
			handler.ServeHTTP(w, req)
			return
		}
		forbidden(w, req)
	})
}
