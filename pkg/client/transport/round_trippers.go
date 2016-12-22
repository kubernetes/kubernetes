/*
Copyright 2015 The Kubernetes Authors.

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

package transport

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/golang/glog"
)

// HTTPWrappersForConfig wraps a round tripper with any relevant layered
// behavior from the config. Exposed to allow more clients that need HTTP-like
// behavior but then must hijack the underlying connection (like WebSocket or
// HTTP2 clients). Pure HTTP clients should use the RoundTripper returned from
// New.
func HTTPWrappersForConfig(config *Config, rt http.RoundTripper) (http.RoundTripper, error) {
	if config.WrapTransport != nil {
		rt = config.WrapTransport(rt)
	}

	rt = DebugWrappers(rt)

	// Set authentication wrappers
	switch {
	case config.HasBasicAuth() && config.HasTokenAuth():
		return nil, fmt.Errorf("username/password or bearer token may be set, but not both")
	case config.HasTokenAuth():
		rt = NewBearerAuthRoundTripper(config.BearerToken, rt)
	case config.HasBasicAuth():
		rt = NewBasicAuthRoundTripper(config.Username, config.Password, rt)
	}
	if len(config.UserAgent) > 0 {
		rt = NewUserAgentRoundTripper(config.UserAgent, rt)
	}
	if len(config.Impersonate.UserName) > 0 ||
		len(config.Impersonate.Groups) > 0 ||
		len(config.Impersonate.Extra) > 0 {
		rt = NewImpersonatingRoundTripper(config.Impersonate, rt)
	}
	return rt, nil
}

// DebugWrappers wraps a round tripper and logs based on the current log level.
func DebugWrappers(rt http.RoundTripper) http.RoundTripper {
	switch {
	case bool(glog.V(9)):
		rt = newDebuggingRoundTripper(rt, debugCurlCommand, debugURLTiming, debugResponseHeaders)
	case bool(glog.V(8)):
		rt = newDebuggingRoundTripper(rt, debugJustURL, debugRequestHeaders, debugResponseStatus, debugResponseHeaders)
	case bool(glog.V(7)):
		rt = newDebuggingRoundTripper(rt, debugJustURL, debugRequestHeaders, debugResponseStatus)
	case bool(glog.V(6)):
		rt = newDebuggingRoundTripper(rt, debugURLTiming)
	}

	return rt
}

type requestCanceler interface {
	CancelRequest(*http.Request)
}

type authProxyRoundTripper struct {
	username string
	groups   []string
	extra    map[string][]string

	rt http.RoundTripper
}

// NewAuthProxyRoundTripper provides a roundtripper which will add auth proxy fields to requests for
// authentication terminating proxy cases
// assuming you pull the user from the context:
// username is the user.Info.GetName() of the user
// groups is the user.Info.GetGroups() of the user
// extra is the user.Info.GetExtra() of the user
// extra can contain any additional information that the authenticator
// thought was interesting, for example authorization scopes.
// In order to faithfully round-trip through an impersonation flow, these keys
// MUST be lowercase.
func NewAuthProxyRoundTripper(username string, groups []string, extra map[string][]string, rt http.RoundTripper) http.RoundTripper {
	return &authProxyRoundTripper{
		username: username,
		groups:   groups,
		extra:    extra,
		rt:       rt,
	}
}

func (rt *authProxyRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	req = cloneRequest(req)
	SetAuthProxyHeaders(req, rt.username, rt.groups, rt.extra)

	return rt.rt.RoundTrip(req)
}

// SetAuthProxyHeaders stomps the auth proxy header fields.  It mutates its argument.
func SetAuthProxyHeaders(req *http.Request, username string, groups []string, extra map[string][]string) {
	req.Header.Del("X-Remote-User")
	req.Header.Del("X-Remote-Group")
	for key := range req.Header {
		if strings.HasPrefix(strings.ToLower(key), strings.ToLower("X-Remote-Extra-")) {
			req.Header.Del(key)
		}
	}

	req.Header.Set("X-Remote-User", username)
	for _, group := range groups {
		req.Header.Add("X-Remote-Group", group)
	}
	for key, values := range extra {
		for _, value := range values {
			req.Header.Add("X-Remote-Extra-"+key, value)
		}
	}
}

func (rt *authProxyRoundTripper) CancelRequest(req *http.Request) {
	if canceler, ok := rt.rt.(requestCanceler); ok {
		canceler.CancelRequest(req)
	} else {
		glog.Errorf("CancelRequest not implemented")
	}
}

func (rt *authProxyRoundTripper) WrappedRoundTripper() http.RoundTripper { return rt.rt }

type userAgentRoundTripper struct {
	agent string
	rt    http.RoundTripper
}

func NewUserAgentRoundTripper(agent string, rt http.RoundTripper) http.RoundTripper {
	return &userAgentRoundTripper{agent, rt}
}

func (rt *userAgentRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(req.Header.Get("User-Agent")) != 0 {
		return rt.rt.RoundTrip(req)
	}
	req = cloneRequest(req)
	req.Header.Set("User-Agent", rt.agent)
	return rt.rt.RoundTrip(req)
}

func (rt *userAgentRoundTripper) CancelRequest(req *http.Request) {
	if canceler, ok := rt.rt.(requestCanceler); ok {
		canceler.CancelRequest(req)
	} else {
		glog.Errorf("CancelRequest not implemented")
	}
}

func (rt *userAgentRoundTripper) WrappedRoundTripper() http.RoundTripper { return rt.rt }

type basicAuthRoundTripper struct {
	username string
	password string
	rt       http.RoundTripper
}

// NewBasicAuthRoundTripper will apply a BASIC auth authorization header to a
// request unless it has already been set.
func NewBasicAuthRoundTripper(username, password string, rt http.RoundTripper) http.RoundTripper {
	return &basicAuthRoundTripper{username, password, rt}
}

func (rt *basicAuthRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(req.Header.Get("Authorization")) != 0 {
		return rt.rt.RoundTrip(req)
	}
	req = cloneRequest(req)
	req.SetBasicAuth(rt.username, rt.password)
	return rt.rt.RoundTrip(req)
}

func (rt *basicAuthRoundTripper) CancelRequest(req *http.Request) {
	if canceler, ok := rt.rt.(requestCanceler); ok {
		canceler.CancelRequest(req)
	} else {
		glog.Errorf("CancelRequest not implemented")
	}
}

func (rt *basicAuthRoundTripper) WrappedRoundTripper() http.RoundTripper { return rt.rt }

// These correspond to the headers used in pkg/apis/authentication.  We don't want the package dependency,
// but you must not change the values.
const (
	// ImpersonateUserHeader is used to impersonate a particular user during an API server request
	ImpersonateUserHeader = "Impersonate-User"

	// ImpersonateGroupHeader is used to impersonate a particular group during an API server request.
	// It can be repeated multiplied times for multiple groups.
	ImpersonateGroupHeader = "Impersonate-Group"

	// ImpersonateUserExtraHeaderPrefix is a prefix for a header used to impersonate an entry in the
	// extra map[string][]string for user.Info.  The key for the `extra` map is suffix.
	// The same key can be repeated multiple times to have multiple elements in the slice under a single key.
	// For instance:
	// Impersonate-Extra-Foo: one
	// Impersonate-Extra-Foo: two
	// results in extra["Foo"] = []string{"one", "two"}
	ImpersonateUserExtraHeaderPrefix = "Impersonate-Extra-"
)

type impersonatingRoundTripper struct {
	impersonate ImpersonationConfig
	delegate    http.RoundTripper
}

// NewImpersonatingRoundTripper will add an Act-As header to a request unless it has already been set.
func NewImpersonatingRoundTripper(impersonate ImpersonationConfig, delegate http.RoundTripper) http.RoundTripper {
	return &impersonatingRoundTripper{impersonate, delegate}
}

func (rt *impersonatingRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	// use the user header as marker for the rest.
	if len(req.Header.Get(ImpersonateUserHeader)) != 0 {
		return rt.delegate.RoundTrip(req)
	}
	req = cloneRequest(req)
	req.Header.Set(ImpersonateUserHeader, rt.impersonate.UserName)

	for _, group := range rt.impersonate.Groups {
		req.Header.Add(ImpersonateGroupHeader, group)
	}
	for k, vv := range rt.impersonate.Extra {
		for _, v := range vv {
			req.Header.Add(ImpersonateUserExtraHeaderPrefix+k, v)
		}
	}

	return rt.delegate.RoundTrip(req)
}

func (rt *impersonatingRoundTripper) CancelRequest(req *http.Request) {
	if canceler, ok := rt.delegate.(requestCanceler); ok {
		canceler.CancelRequest(req)
	} else {
		glog.Errorf("CancelRequest not implemented")
	}
}

func (rt *impersonatingRoundTripper) WrappedRoundTripper() http.RoundTripper { return rt.delegate }

type bearerAuthRoundTripper struct {
	bearer string
	rt     http.RoundTripper
}

// NewBearerAuthRoundTripper adds the provided bearer token to a request
// unless the authorization header has already been set.
func NewBearerAuthRoundTripper(bearer string, rt http.RoundTripper) http.RoundTripper {
	return &bearerAuthRoundTripper{bearer, rt}
}

func (rt *bearerAuthRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(req.Header.Get("Authorization")) != 0 {
		return rt.rt.RoundTrip(req)
	}

	req = cloneRequest(req)
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", rt.bearer))
	return rt.rt.RoundTrip(req)
}

func (rt *bearerAuthRoundTripper) CancelRequest(req *http.Request) {
	if canceler, ok := rt.rt.(requestCanceler); ok {
		canceler.CancelRequest(req)
	} else {
		glog.Errorf("CancelRequest not implemented")
	}
}

func (rt *bearerAuthRoundTripper) WrappedRoundTripper() http.RoundTripper { return rt.rt }

// cloneRequest returns a clone of the provided *http.Request.
// The clone is a shallow copy of the struct and its Header map.
func cloneRequest(r *http.Request) *http.Request {
	// shallow copy of the struct
	r2 := new(http.Request)
	*r2 = *r
	// deep copy of the Header
	r2.Header = make(http.Header)
	for k, s := range r.Header {
		r2.Header[k] = s
	}
	return r2
}

// requestInfo keeps track of information about a request/response combination
type requestInfo struct {
	RequestHeaders http.Header
	RequestVerb    string
	RequestURL     string

	ResponseStatus  string
	ResponseHeaders http.Header
	ResponseErr     error

	Duration time.Duration
}

// newRequestInfo creates a new RequestInfo based on an http request
func newRequestInfo(req *http.Request) *requestInfo {
	return &requestInfo{
		RequestURL:     req.URL.String(),
		RequestVerb:    req.Method,
		RequestHeaders: req.Header,
	}
}

// complete adds information about the response to the requestInfo
func (r *requestInfo) complete(response *http.Response, err error) {
	if err != nil {
		r.ResponseErr = err
		return
	}
	r.ResponseStatus = response.Status
	r.ResponseHeaders = response.Header
}

// toCurl returns a string that can be run as a command in a terminal (minus the body)
func (r *requestInfo) toCurl() string {
	headers := ""
	for key, values := range r.RequestHeaders {
		for _, value := range values {
			headers += fmt.Sprintf(` -H %q`, fmt.Sprintf("%s: %s", key, value))
		}
	}

	return fmt.Sprintf("curl -k -v -X%s %s %s", r.RequestVerb, headers, r.RequestURL)
}

// debuggingRoundTripper will display information about the requests passing
// through it based on what is configured
type debuggingRoundTripper struct {
	delegatedRoundTripper http.RoundTripper

	levels map[debugLevel]bool
}

type debugLevel int

const (
	debugJustURL debugLevel = iota
	debugURLTiming
	debugCurlCommand
	debugRequestHeaders
	debugResponseStatus
	debugResponseHeaders
)

func newDebuggingRoundTripper(rt http.RoundTripper, levels ...debugLevel) *debuggingRoundTripper {
	drt := &debuggingRoundTripper{
		delegatedRoundTripper: rt,
		levels:                make(map[debugLevel]bool, len(levels)),
	}
	for _, v := range levels {
		drt.levels[v] = true
	}
	return drt
}

func (rt *debuggingRoundTripper) CancelRequest(req *http.Request) {
	if canceler, ok := rt.delegatedRoundTripper.(requestCanceler); ok {
		canceler.CancelRequest(req)
	} else {
		glog.Errorf("CancelRequest not implemented")
	}
}

func (rt *debuggingRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	reqInfo := newRequestInfo(req)

	if rt.levels[debugJustURL] {
		glog.Infof("%s %s", reqInfo.RequestVerb, reqInfo.RequestURL)
	}
	if rt.levels[debugCurlCommand] {
		glog.Infof("%s", reqInfo.toCurl())

	}
	if rt.levels[debugRequestHeaders] {
		glog.Infof("Request Headers:")
		for key, values := range reqInfo.RequestHeaders {
			for _, value := range values {
				glog.Infof("    %s: %s", key, value)
			}
		}
	}

	startTime := time.Now()
	response, err := rt.delegatedRoundTripper.RoundTrip(req)
	reqInfo.Duration = time.Since(startTime)

	reqInfo.complete(response, err)

	if rt.levels[debugURLTiming] {
		glog.Infof("%s %s %s in %d milliseconds", reqInfo.RequestVerb, reqInfo.RequestURL, reqInfo.ResponseStatus, reqInfo.Duration.Nanoseconds()/int64(time.Millisecond))
	}
	if rt.levels[debugResponseStatus] {
		glog.Infof("Response Status: %s in %d milliseconds", reqInfo.ResponseStatus, reqInfo.Duration.Nanoseconds()/int64(time.Millisecond))
	}
	if rt.levels[debugResponseHeaders] {
		glog.Infof("Response Headers:")
		for key, values := range reqInfo.ResponseHeaders {
			for _, value := range values {
				glog.Infof("    %s: %s", key, value)
			}
		}
	}

	return response, err
}

func (rt *debuggingRoundTripper) WrappedRoundTripper() http.RoundTripper {
	return rt.delegatedRoundTripper
}
