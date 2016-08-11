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

package apiserver

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"regexp"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/httplog"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

// specialVerbs contains just strings which are used in REST paths for special actions that don't fall under the normal
// CRUDdy GET/POST/PUT/DELETE actions on REST objects.
// TODO: find a way to keep this up to date automatically.  Maybe dynamically populate list as handlers added to
// master's Mux.
var specialVerbs = sets.NewString("proxy", "redirect", "watch")

// specialVerbsNoSubresources contains root verbs which do not allow subresources
var specialVerbsNoSubresources = sets.NewString("proxy", "redirect")

// namespaceSubresources contains subresources of namespace
// this list allows the parser to distinguish between a namespace subresource, and a namespaced resource
var namespaceSubresources = sets.NewString("status", "finalize")

// NamespaceSubResourcesForTest exports namespaceSubresources for testing in pkg/master/master_test.go, so we never drift
var NamespaceSubResourcesForTest = sets.NewString(namespaceSubresources.List()...)

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

type LongRunningRequestCheck func(r *http.Request) bool

// BasicLongRunningRequestCheck pathRegex operates against the url path, the queryParams match is case insensitive.
// Any one match flags the request.
// TODO tighten this check to eliminate the abuse potential by malicious clients that start setting queryParameters
// to bypass the rate limitter.  This could be done using a full parse and special casing the bits we need.
func BasicLongRunningRequestCheck(pathRegex *regexp.Regexp, queryParams map[string]string) LongRunningRequestCheck {
	return func(r *http.Request) bool {
		if pathRegex.MatchString(r.URL.Path) {
			return true
		}

		for key, expectedValue := range queryParams {
			if strings.ToLower(expectedValue) == strings.ToLower(r.URL.Query().Get(key)) {
				return true
			}
		}

		return false
	}
}

// MaxInFlight limits the number of in-flight requests to buffer size of the passed in channel.
func MaxInFlightLimit(c chan bool, longRunningRequestCheck LongRunningRequestCheck, handler http.Handler) http.Handler {
	if c == nil {
		return handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if longRunningRequestCheck(r) {
			// Skip tracking long running events.
			handler.ServeHTTP(w, r)
			return
		}
		select {
		case c <- true:
			defer func() { <-c }()
			handler.ServeHTTP(w, r)
		default:
			tooManyRequests(r, w)
		}
	})
}

func tooManyRequests(req *http.Request, w http.ResponseWriter) {
	// "Too Many Requests" response is returned before logger is setup for the request.
	// So we need to explicitly log it here.
	defer httplog.NewLogged(req, &w).Log()

	// Return a 429 status indicating "Too Many Requests"
	w.Header().Set("Retry-After", RetryAfter)
	http.Error(w, "Too many requests, please try again later.", errors.StatusTooManyRequests)
}

// RecoverPanics wraps an http Handler to recover and log panics.
func RecoverPanics(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer runtime.HandleCrash(func(err interface{}) {
			http.Error(w, "This request caused apisever to panic. Look in log for details.", http.StatusInternalServerError)
			glog.Errorf("APIServer panic'd on %v %v: %v\n%s\n", req.Method, req.RequestURI, err, debug.Stack())
		})
		defer httplog.NewLogged(req, &w).StacktraceWhen(
			httplog.StatusIsNot(
				http.StatusOK,
				http.StatusCreated,
				http.StatusAccepted,
				http.StatusBadRequest,
				http.StatusMovedPermanently,
				http.StatusTemporaryRedirect,
				http.StatusConflict,
				http.StatusNotFound,
				http.StatusUnauthorized,
				http.StatusForbidden,
				errors.StatusUnprocessableEntity,
				http.StatusSwitchingProtocols,
			),
		).Log()

		// Dispatch to the internal handler
		handler.ServeHTTP(w, req)
	})
}

var errConnKilled = fmt.Errorf("kill connection/stream")

// TimeoutHandler returns an http.Handler that runs h with a timeout
// determined by timeoutFunc. The new http.Handler calls h.ServeHTTP to handle
// each request, but if a call runs for longer than its time limit, the
// handler responds with a 503 Service Unavailable error and the message
// provided. (If msg is empty, a suitable default message will be sent.) After
// the handler times out, writes by h to its http.ResponseWriter will return
// http.ErrHandlerTimeout. If timeoutFunc returns a nil timeout channel, no
// timeout will be enforced.
func TimeoutHandler(h http.Handler, timeoutFunc func(*http.Request) (timeout <-chan time.Time, msg string)) http.Handler {
	return &timeoutHandler{h, timeoutFunc}
}

type timeoutHandler struct {
	handler http.Handler
	timeout func(*http.Request) (<-chan time.Time, string)
}

func (t *timeoutHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	after, msg := t.timeout(r)
	if after == nil {
		t.handler.ServeHTTP(w, r)
		return
	}

	done := make(chan struct{})
	tw := newTimeoutWriter(w)
	go func() {
		t.handler.ServeHTTP(tw, r)
		close(done)
	}()
	select {
	case <-done:
		return
	case <-after:
		tw.timeout(msg)
	}
}

type timeoutWriter interface {
	http.ResponseWriter
	timeout(string)
}

func newTimeoutWriter(w http.ResponseWriter) timeoutWriter {
	base := &baseTimeoutWriter{w: w}

	_, notifiable := w.(http.CloseNotifier)
	_, hijackable := w.(http.Hijacker)

	switch {
	case notifiable && hijackable:
		return &closeHijackTimeoutWriter{base}
	case notifiable:
		return &closeTimeoutWriter{base}
	case hijackable:
		return &hijackTimeoutWriter{base}
	default:
		return base
	}
}

type baseTimeoutWriter struct {
	w http.ResponseWriter

	mu sync.Mutex
	// if the timeout handler has timedout
	timedOut bool
	// if this timeout writer has wrote header
	wroteHeader bool
	// if this timeout writer has been hijacked
	hijacked bool
}

func (tw *baseTimeoutWriter) Header() http.Header {
	tw.mu.Lock()
	defer tw.mu.Unlock()

	if tw.timedOut {
		return http.Header{}
	}

	return tw.w.Header()
}

func (tw *baseTimeoutWriter) Write(p []byte) (int, error) {
	tw.mu.Lock()
	defer tw.mu.Unlock()

	if tw.timedOut {
		return 0, http.ErrHandlerTimeout
	}
	if tw.hijacked {
		return 0, http.ErrHijacked
	}

	tw.wroteHeader = true
	return tw.w.Write(p)
}

func (tw *baseTimeoutWriter) Flush() {
	tw.mu.Lock()
	defer tw.mu.Unlock()

	if tw.timedOut {
		return
	}

	if flusher, ok := tw.w.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (tw *baseTimeoutWriter) WriteHeader(code int) {
	tw.mu.Lock()
	defer tw.mu.Unlock()

	if tw.timedOut || tw.wroteHeader || tw.hijacked {
		return
	}

	tw.wroteHeader = true
	tw.w.WriteHeader(code)
}

func (tw *baseTimeoutWriter) timeout(msg string) {
	tw.mu.Lock()
	defer tw.mu.Unlock()

	tw.timedOut = true

	// The timeout writer has not been used by the inner handler.
	// We can safely timeout the HTTP request by sending by a timeout
	// handler
	if !tw.wroteHeader && !tw.hijacked {
		tw.w.WriteHeader(http.StatusGatewayTimeout)
		if msg != "" {
			tw.w.Write([]byte(msg))
		} else {
			enc := json.NewEncoder(tw.w)
			enc.Encode(errors.NewServerTimeout(api.Resource(""), "", 0))
		}
	} else {
		// The timeout writer has been used by the inner handler. There is
		// no way to timeout the HTTP request at the point. We have to shutdown
		// the connection for HTTP1 or reset stream for HTTP2.
		//
		// Note from: Brad Fitzpatrick
		// if the ServeHTTP goroutine panics, that will do the best possible thing for both
		// HTTP/1 and HTTP/2. In HTTP/1, assuming you're replying with at least HTTP/1.1 and
		// you've already flushed the headers so it's using HTTP chunking, it'll kill the TCP
		// connection immediately without a proper 0-byte EOF chunk, so the peer will recognize
		// the response as bogus. In HTTP/2 the server will just RST_STREAM the stream, leaving
		// the TCP connection open, but resetting the stream to the peer so it'll have an error,
		// like the HTTP/1 case.
		panic(errConnKilled)
	}
}

func (tw *baseTimeoutWriter) closeNotify() <-chan bool {
	tw.mu.Lock()
	defer tw.mu.Unlock()

	if tw.timedOut {
		done := make(chan bool)
		close(done)
		return done
	}

	return tw.w.(http.CloseNotifier).CloseNotify()
}

func (tw *baseTimeoutWriter) hijack() (net.Conn, *bufio.ReadWriter, error) {
	tw.mu.Lock()
	defer tw.mu.Unlock()

	if tw.timedOut {
		return nil, nil, http.ErrHandlerTimeout
	}
	conn, rw, err := tw.w.(http.Hijacker).Hijack()
	if err == nil {
		tw.hijacked = true
	}
	return conn, rw, err
}

type closeTimeoutWriter struct {
	*baseTimeoutWriter
}

func (tw *closeTimeoutWriter) CloseNotify() <-chan bool {
	return tw.closeNotify()
}

type hijackTimeoutWriter struct {
	*baseTimeoutWriter
}

func (tw *hijackTimeoutWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return tw.hijack()
}

type closeHijackTimeoutWriter struct {
	*baseTimeoutWriter
}

func (tw *closeHijackTimeoutWriter) CloseNotify() <-chan bool {
	return tw.closeNotify()
}

func (tw *closeHijackTimeoutWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return tw.hijack()
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
	requestContextMapper api.RequestContextMapper
	requestInfoResolver  *RequestInfoResolver
}

// NewAttributeGetter returns an object which implements the RequestAttributeGetter interface.
func NewRequestAttributeGetter(requestContextMapper api.RequestContextMapper, requestInfoResolver *RequestInfoResolver) RequestAttributeGetter {
	return &requestAttributeGetter{requestContextMapper, requestInfoResolver}
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

	requestInfo, _ := r.requestInfoResolver.GetRequestInfo(req)

	// Start with common attributes that apply to resource and non-resource requests
	attribs.ResourceRequest = requestInfo.IsResourceRequest
	attribs.Path = requestInfo.Path
	attribs.Verb = requestInfo.Verb

	attribs.APIGroup = requestInfo.APIGroup
	attribs.APIVersion = requestInfo.APIVersion
	attribs.Resource = requestInfo.Resource
	attribs.Subresource = requestInfo.Subresource
	attribs.Namespace = requestInfo.Namespace
	attribs.Name = requestInfo.Name

	return &attribs
}

func WithImpersonation(handler http.Handler, requestContextMapper api.RequestContextMapper, a authorizer.Authorizer) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		requestedSubject := req.Header.Get("Impersonate-User")
		if len(requestedSubject) == 0 {
			handler.ServeHTTP(w, req)
			return
		}

		ctx, exists := requestContextMapper.Get(req)
		if !exists {
			forbidden(w, req)
			return
		}
		requestor, exists := api.UserFrom(ctx)
		if !exists {
			forbidden(w, req)
			return
		}

		actingAsAttributes := &authorizer.AttributesRecord{
			User:            requestor,
			Verb:            "impersonate",
			APIGroup:        api.GroupName,
			Resource:        "users",
			Name:            requestedSubject,
			ResourceRequest: true,
		}
		if namespace, name, err := serviceaccount.SplitUsername(requestedSubject); err == nil {
			actingAsAttributes.Resource = "serviceaccounts"
			actingAsAttributes.Namespace = namespace
			actingAsAttributes.Name = name
		}

		authorized, reason, err := a.Authorize(actingAsAttributes)
		if err != nil {
			internalError(w, req, err)
			return
		}
		if !authorized {
			glog.V(4).Infof("Forbidden: %#v, Reason: %s", req.RequestURI, reason)
			forbidden(w, req)
			return
		}

		switch {
		case strings.HasPrefix(requestedSubject, serviceaccount.ServiceAccountUsernamePrefix):
			namespace, name, err := serviceaccount.SplitUsername(requestedSubject)
			if err != nil {
				forbidden(w, req)
				return
			}
			requestContextMapper.Update(req, api.WithUser(ctx, serviceaccount.UserInfo(namespace, name, "")))

		default:
			newUser := &user.DefaultInfo{
				Name: requestedSubject,
			}
			requestContextMapper.Update(req, api.WithUser(ctx, newUser))
		}

		newCtx, _ := requestContextMapper.Get(req)
		oldUser, _ := api.UserFrom(ctx)
		newUser, _ := api.UserFrom(newCtx)
		httplog.LogOf(req, w).Addf("%v is acting as %v", oldUser, newUser)

		handler.ServeHTTP(w, req)
	})
}

// WithAuthorizationCheck passes all authorized requests on to handler, and returns a forbidden error otherwise.
func WithAuthorizationCheck(handler http.Handler, getAttribs RequestAttributeGetter, a authorizer.Authorizer) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		authorized, reason, err := a.Authorize(getAttribs.GetAttribs(req))
		if err != nil {
			internalError(w, req, err)
			return
		}
		if !authorized {
			glog.V(4).Infof("Forbidden: %#v, Reason: %s", req.RequestURI, reason)
			forbidden(w, req)
			return
		}
		handler.ServeHTTP(w, req)
	})
}

// RequestInfo holds information parsed from the http.Request
type RequestInfo struct {
	// IsResourceRequest indicates whether or not the request is for an API resource or subresource
	IsResourceRequest bool
	// Path is the URL path of the request
	Path string
	// Verb is the kube verb associated with the request for API requests, not the http verb.  This includes things like list and watch.
	// for non-resource requests, this is the lowercase http verb
	Verb string

	APIPrefix  string
	APIGroup   string
	APIVersion string
	Namespace  string
	// Resource is the name of the resource being requested.  This is not the kind.  For example: pods
	Resource string
	// Subresource is the name of the subresource being requested.  This is a different resource, scoped to the parent resource, but it may have a different kind.
	// For instance, /pods has the resource "pods" and the kind "Pod", while /pods/foo/status has the resource "pods", the sub resource "status", and the kind "Pod"
	// (because status operates on pods). The binding resource for a pod though may be /pods/foo/binding, which has resource "pods", subresource "binding", and kind "Binding".
	Subresource string
	// Name is empty for some verbs, but if the request directly indicates a name (not in body content) then this field is filled in.
	Name string
	// Parts are the path parts for the request, always starting with /{resource}/{name}
	Parts []string
}

type RequestInfoResolver struct {
	APIPrefixes          sets.String
	GrouplessAPIPrefixes sets.String
}

// TODO write an integration test against the swagger doc to test the RequestInfo and match up behavior to responses
// GetRequestInfo returns the information from the http request.  If error is not nil, RequestInfo holds the information as best it is known before the failure
// It handles both resource and non-resource requests and fills in all the pertinent information for each.
// Valid Inputs:
// Resource paths
// /apis/{api-group}/{version}/namespaces
// /api/{version}/namespaces
// /api/{version}/namespaces/{namespace}
// /api/{version}/namespaces/{namespace}/{resource}
// /api/{version}/namespaces/{namespace}/{resource}/{resourceName}
// /api/{version}/{resource}
// /api/{version}/{resource}/{resourceName}
//
// Special verbs without subresources:
// /api/{version}/proxy/{resource}/{resourceName}
// /api/{version}/proxy/namespaces/{namespace}/{resource}/{resourceName}
// /api/{version}/redirect/namespaces/{namespace}/{resource}/{resourceName}
// /api/{version}/redirect/{resource}/{resourceName}
//
// Special verbs with subresources:
// /api/{version}/watch/{resource}
// /api/{version}/watch/namespaces/{namespace}/{resource}
//
// NonResource paths
// /apis/{api-group}/{version}
// /apis/{api-group}
// /apis
// /api/{version}
// /api
// /healthz
// /
func (r *RequestInfoResolver) GetRequestInfo(req *http.Request) (RequestInfo, error) {
	// start with a non-resource request until proven otherwise
	requestInfo := RequestInfo{
		IsResourceRequest: false,
		Path:              req.URL.Path,
		Verb:              strings.ToLower(req.Method),
	}

	currentParts := splitPath(req.URL.Path)
	if len(currentParts) < 3 {
		// return a non-resource request
		return requestInfo, nil
	}

	if !r.APIPrefixes.Has(currentParts[0]) {
		// return a non-resource request
		return requestInfo, nil
	}
	requestInfo.APIPrefix = currentParts[0]
	currentParts = currentParts[1:]

	if !r.GrouplessAPIPrefixes.Has(requestInfo.APIPrefix) {
		// one part (APIPrefix) has already been consumed, so this is actually "do we have four parts?"
		if len(currentParts) < 3 {
			// return a non-resource request
			return requestInfo, nil
		}

		requestInfo.APIGroup = currentParts[0]
		currentParts = currentParts[1:]
	}

	requestInfo.IsResourceRequest = true
	requestInfo.APIVersion = currentParts[0]
	currentParts = currentParts[1:]

	// handle input of form /{specialVerb}/*
	if specialVerbs.Has(currentParts[0]) {
		if len(currentParts) < 2 {
			return requestInfo, fmt.Errorf("unable to determine kind and namespace from url, %v", req.URL)
		}

		requestInfo.Verb = currentParts[0]
		currentParts = currentParts[1:]

	} else {
		switch req.Method {
		case "POST":
			requestInfo.Verb = "create"
		case "GET", "HEAD":
			requestInfo.Verb = "get"
		case "PUT":
			requestInfo.Verb = "update"
		case "PATCH":
			requestInfo.Verb = "patch"
		case "DELETE":
			requestInfo.Verb = "delete"
		default:
			requestInfo.Verb = ""
		}
	}

	// URL forms: /namespaces/{namespace}/{kind}/*, where parts are adjusted to be relative to kind
	if currentParts[0] == "namespaces" {
		if len(currentParts) > 1 {
			requestInfo.Namespace = currentParts[1]

			// if there is another step after the namespace name and it is not a known namespace subresource
			// move currentParts to include it as a resource in its own right
			if len(currentParts) > 2 && !namespaceSubresources.Has(currentParts[2]) {
				currentParts = currentParts[2:]
			}
		}
	} else {
		requestInfo.Namespace = api.NamespaceNone
	}

	// parsing successful, so we now know the proper value for .Parts
	requestInfo.Parts = currentParts

	// parts look like: resource/resourceName/subresource/other/stuff/we/don't/interpret
	switch {
	case len(requestInfo.Parts) >= 3 && !specialVerbsNoSubresources.Has(requestInfo.Verb):
		requestInfo.Subresource = requestInfo.Parts[2]
		fallthrough
	case len(requestInfo.Parts) >= 2:
		requestInfo.Name = requestInfo.Parts[1]
		fallthrough
	case len(requestInfo.Parts) >= 1:
		requestInfo.Resource = requestInfo.Parts[0]
	}

	// if there's no name on the request and we thought it was a get before, then the actual verb is a list
	if len(requestInfo.Name) == 0 && requestInfo.Verb == "get" {
		requestInfo.Verb = "list"
	}
	// if there's no name on the request and we thought it was a delete before, then the actual verb is deletecollection
	if len(requestInfo.Name) == 0 && requestInfo.Verb == "delete" {
		requestInfo.Verb = "deletecollection"
	}

	return requestInfo, nil
}
