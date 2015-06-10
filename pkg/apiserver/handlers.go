/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authorizer"
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

// Constant for the retry-after interval on rate limiting.
// TODO: maybe make this dynamic? or user-adjustable?
const RetryAfter = "1"

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

// MaxInFlight limits the number of in-flight requests to buffer size of the passed in channel.
func MaxInFlightLimit(c chan bool, longRunningRequestRE *regexp.Regexp, handler http.Handler) http.Handler {
	if c == nil {
		return handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if longRunningRequestRE.MatchString(r.URL.Path) {
			// Skip tracking long running events.
			handler.ServeHTTP(w, r)
			return
		}
		select {
		case c <- true:
			defer func() { <-c }()
			handler.ServeHTTP(w, r)
		default:
			tooManyRequests(w)
		}
	})
}

// RateLimit uses rl to rate limit accepting requests to 'handler'.
func RateLimit(rl util.RateLimiter, handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if rl.CanAccept() {
			handler.ServeHTTP(w, req)
			return
		}
		tooManyRequests(w)
	})
}

func tooManyRequests(w http.ResponseWriter) {
	// Return a 429 status indicating "Too Many Requests"
	w.Header().Set("Retry-After", RetryAfter)
	http.Error(w, "Too many requests, please try again later.", errors.StatusTooManyRequests)
}

// RecoverPanics wraps an http Handler to recover and log panics.
func RecoverPanics(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer func() {
			if x := recover(); x != nil {
				http.Error(w, "apis panic. Look in log for details.", http.StatusInternalServerError)
				glog.Errorf("APIServer panic'd on %v %v: %v\n%s\n", req.Method, req.RequestURI, x, debug.Stack())
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
				errors.StatusUnprocessableEntity,
				http.StatusSwitchingProtocols,
			),
		).Log()

		// Dispatch to the internal handler
		handler.ServeHTTP(w, req)
	})
}

// TODO: use restful.CrossOriginResourceSharing
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
	requestContextMapper   api.RequestContextMapper
	apiRequestInfoResolver *APIRequestInfoResolver
}

// NewAttributeGetter returns an object which implements the RequestAttributeGetter interface.
func NewRequestAttributeGetter(requestContextMapper api.RequestContextMapper, restMapper meta.RESTMapper, apiRoots ...string) RequestAttributeGetter {
	return &requestAttributeGetter{requestContextMapper, &APIRequestInfoResolver{util.NewStringSet(apiRoots...), restMapper}}
}

func (r *requestAttributeGetter) GetAttribs(req *http.Request) authorizer.Attributes {
	attribs := authorizer.AttributesRecord{}

	ctx, ok := r.requestContextMapper.Get(req)
	if ok {
		user, ok := api.UserFrom(ctx)
		if ok {
			attribs.User = user
		}
	}

	attribs.ReadOnly = IsReadOnlyReq(*req)

	apiRequestInfo, _ := r.apiRequestInfoResolver.GetAPIRequestInfo(req)

	// If a path follows the conventions of the REST object store, then
	// we can extract the resource.  Otherwise, not.
	attribs.Resource = apiRequestInfo.Resource

	// If the request specifies a namespace, then the namespace is filled in.
	// Assumes there is no empty string namespace.  Unspecified results
	// in empty (does not understand defaulting rules.)
	attribs.Namespace = apiRequestInfo.Namespace

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

// APIRequestInfo holds information parsed from the http.Request
type APIRequestInfo struct {
	// Verb is the kube verb associated with the request, not the http verb.  This includes things like list and watch.
	Verb       string
	APIVersion string
	Namespace  string
	// Resource is the name of the resource being requested.  This is not the kind.  For example: pods
	Resource string
	// Subresource is the name of the subresource being requested.  This is a different resource, scoped to the parent resource, but it may have a different kind.
	// For instance, /pods has the resource "pods" and the kind "Pod", while /pods/foo/status has the resource "pods", the sub resource "status", and the kind "Pod"
	// (because status operates on pods). The binding resource for a pod though may be /pods/foo/binding, which has resource "pods", subresource "binding", and kind "Binding".
	Subresource string
	// Kind is the type of object being manipulated.  For example: Pod
	Kind string
	// Name is empty for some verbs, but if the request directly indicates a name (not in body content) then this field is filled in.
	Name string
	// Parts are the path parts for the request, always starting with /{resource}/{name}
	Parts []string
	// Raw is the unparsed form of everything other than parts.
	// Raw + Parts = complete URL path
	Raw []string
}

type APIRequestInfoResolver struct {
	APIPrefixes util.StringSet
	RestMapper  meta.RESTMapper
}

// TODO write an integration test against the swagger doc to test the APIRequestInfo and match up behavior to responses
// GetAPIRequestInfo returns the information from the http request.  If error is not nil, APIRequestInfo holds the information as best it is known before the failure
// Valid Inputs:
// Storage paths
// /namespaces
// /namespaces/{namespace}
// /namespaces/{namespace}/{resource}
// /namespaces/{namespace}/{resource}/{resourceName}
// /{resource}
// /{resource}/{resourceName}
//
// Special verbs:
// /proxy/{resource}/{resourceName}
// /proxy/namespaces/{namespace}/{resource}/{resourceName}
// /redirect/namespaces/{namespace}/{resource}/{resourceName}
// /redirect/{resource}/{resourceName}
// /watch/{resource}
// /watch/namespaces/{namespace}/{resource}
//
// Fully qualified paths for above:
// /api/{version}/*
// /api/{version}/*
func (r *APIRequestInfoResolver) GetAPIRequestInfo(req *http.Request) (APIRequestInfo, error) {
	requestInfo := APIRequestInfo{
		Raw: splitPath(req.URL.Path),
	}

	currentParts := requestInfo.Raw
	if len(currentParts) < 1 {
		return requestInfo, fmt.Errorf("Unable to determine kind and namespace from an empty URL path")
	}

	for _, currPrefix := range r.APIPrefixes.List() {
		// handle input of form /api/{version}/* by adjusting special paths
		if currentParts[0] == currPrefix {
			if len(currentParts) > 1 {
				requestInfo.APIVersion = currentParts[1]
			}

			if len(currentParts) > 2 {
				currentParts = currentParts[2:]
			} else {
				return requestInfo, fmt.Errorf("Unable to determine kind and namespace from url, %v", req.URL)
			}
		}
	}

	// handle input of form /{specialVerb}/*
	if _, ok := specialVerbs[currentParts[0]]; ok {
		requestInfo.Verb = currentParts[0]

		if len(currentParts) > 1 {
			currentParts = currentParts[1:]
		} else {
			return requestInfo, fmt.Errorf("Unable to determine kind and namespace from url, %v", req.URL)
		}
	} else {
		switch req.Method {
		case "POST":
			requestInfo.Verb = "create"
		case "GET":
			requestInfo.Verb = "get"
		case "PUT":
			requestInfo.Verb = "update"
		case "DELETE":
			requestInfo.Verb = "delete"
		}

	}

	// URL forms: /namespaces/{namespace}/{kind}/*, where parts are adjusted to be relative to kind
	if currentParts[0] == "namespaces" {
		if len(currentParts) > 1 {
			requestInfo.Namespace = currentParts[1]

			// if there is another step after the namespace name and it is not a known namespace subresource
			// move currentParts to include it as a resource in its own right
			if len(currentParts) > 2 {
				currentParts = currentParts[2:]
			}
		}
	} else {
		requestInfo.Namespace = api.NamespaceNone
	}

	// parsing successful, so we now know the proper value for .Parts
	requestInfo.Parts = currentParts
	// Raw should have everything not in Parts
	requestInfo.Raw = requestInfo.Raw[:len(requestInfo.Raw)-len(currentParts)]

	// parts look like: resource/resourceName/subresource/other/stuff/we/don't/interpret
	switch {
	case len(requestInfo.Parts) >= 3:
		requestInfo.Subresource = requestInfo.Parts[2]
		fallthrough
	case len(requestInfo.Parts) == 2:
		requestInfo.Name = requestInfo.Parts[1]
		fallthrough
	case len(requestInfo.Parts) == 1:
		requestInfo.Resource = requestInfo.Parts[0]
	}

	// if there's no name on the request and we thought it was a get before, then the actual verb is a list
	if len(requestInfo.Name) == 0 && requestInfo.Verb == "get" {
		requestInfo.Verb = "list"
	}

	// if we have a resource, we have a good shot at being able to determine kind
	if len(requestInfo.Resource) > 0 {
		_, requestInfo.Kind, _ = r.RestMapper.VersionAndKindForResource(requestInfo.Resource)
	}

	return requestInfo, nil
}
