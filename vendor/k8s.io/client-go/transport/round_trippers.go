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
	"crypto/tls"
	"fmt"
	"net/http"
	"net/http/httptrace"
	"strings"
	"sync"
	"time"

	"golang.org/x/oauth2"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/klog/v2"
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
		var err error
		rt, err = NewBearerAuthWithRefreshRoundTripper(config.BearerToken, config.BearerTokenFile, rt)
		if err != nil {
			return nil, err
		}
	case config.HasBasicAuth():
		rt = NewBasicAuthRoundTripper(config.Username, config.Password, rt)
	}
	if len(config.UserAgent) > 0 {
		rt = NewUserAgentRoundTripper(config.UserAgent, rt)
	}
	if len(config.Impersonate.UserName) > 0 ||
		len(config.Impersonate.UID) > 0 ||
		len(config.Impersonate.Groups) > 0 ||
		len(config.Impersonate.Extra) > 0 {
		rt = NewImpersonatingRoundTripper(config.Impersonate, rt)
	}
	return rt, nil
}

// DebugWrappers wraps a round tripper and logs based on the current log level.
func DebugWrappers(rt http.RoundTripper) http.RoundTripper {
	switch {
	case bool(klog.V(9).Enabled()):
		rt = NewDebuggingRoundTripper(rt, DebugCurlCommand, DebugURLTiming, DebugDetailedTiming, DebugResponseHeaders)
	case bool(klog.V(8).Enabled()):
		rt = NewDebuggingRoundTripper(rt, DebugJustURL, DebugRequestHeaders, DebugResponseStatus, DebugResponseHeaders)
	case bool(klog.V(7).Enabled()):
		rt = NewDebuggingRoundTripper(rt, DebugJustURL, DebugRequestHeaders, DebugResponseStatus)
	case bool(klog.V(6).Enabled()):
		rt = NewDebuggingRoundTripper(rt, DebugURLTiming)
	}

	return rt
}

type authProxyRoundTripper struct {
	username string
	groups   []string
	extra    map[string][]string

	rt http.RoundTripper
}

var _ utilnet.RoundTripperWrapper = &authProxyRoundTripper{}

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
	req = utilnet.CloneRequest(req)
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
			req.Header.Add("X-Remote-Extra-"+headerKeyEscape(key), value)
		}
	}
}

func (rt *authProxyRoundTripper) CancelRequest(req *http.Request) {
	tryCancelRequest(rt.WrappedRoundTripper(), req)
}

func (rt *authProxyRoundTripper) WrappedRoundTripper() http.RoundTripper { return rt.rt }

type userAgentRoundTripper struct {
	agent string
	rt    http.RoundTripper
}

var _ utilnet.RoundTripperWrapper = &userAgentRoundTripper{}

// NewUserAgentRoundTripper will add User-Agent header to a request unless it has already been set.
func NewUserAgentRoundTripper(agent string, rt http.RoundTripper) http.RoundTripper {
	return &userAgentRoundTripper{agent, rt}
}

func (rt *userAgentRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(req.Header.Get("User-Agent")) != 0 {
		return rt.rt.RoundTrip(req)
	}
	req = utilnet.CloneRequest(req)
	req.Header.Set("User-Agent", rt.agent)
	return rt.rt.RoundTrip(req)
}

func (rt *userAgentRoundTripper) CancelRequest(req *http.Request) {
	tryCancelRequest(rt.WrappedRoundTripper(), req)
}

func (rt *userAgentRoundTripper) WrappedRoundTripper() http.RoundTripper { return rt.rt }

type basicAuthRoundTripper struct {
	username string
	password string `datapolicy:"password"`
	rt       http.RoundTripper
}

var _ utilnet.RoundTripperWrapper = &basicAuthRoundTripper{}

// NewBasicAuthRoundTripper will apply a BASIC auth authorization header to a
// request unless it has already been set.
func NewBasicAuthRoundTripper(username, password string, rt http.RoundTripper) http.RoundTripper {
	return &basicAuthRoundTripper{username, password, rt}
}

func (rt *basicAuthRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(req.Header.Get("Authorization")) != 0 {
		return rt.rt.RoundTrip(req)
	}
	req = utilnet.CloneRequest(req)
	req.SetBasicAuth(rt.username, rt.password)
	return rt.rt.RoundTrip(req)
}

func (rt *basicAuthRoundTripper) CancelRequest(req *http.Request) {
	tryCancelRequest(rt.WrappedRoundTripper(), req)
}

func (rt *basicAuthRoundTripper) WrappedRoundTripper() http.RoundTripper { return rt.rt }

// These correspond to the headers used in pkg/apis/authentication.  We don't want the package dependency,
// but you must not change the values.
const (
	// ImpersonateUserHeader is used to impersonate a particular user during an API server request
	ImpersonateUserHeader = "Impersonate-User"

	// ImpersonateUIDHeader is used to impersonate a particular UID during an API server request
	ImpersonateUIDHeader = "Impersonate-Uid"

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

var _ utilnet.RoundTripperWrapper = &impersonatingRoundTripper{}

// NewImpersonatingRoundTripper will add an Act-As header to a request unless it has already been set.
func NewImpersonatingRoundTripper(impersonate ImpersonationConfig, delegate http.RoundTripper) http.RoundTripper {
	return &impersonatingRoundTripper{impersonate, delegate}
}

func (rt *impersonatingRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	// use the user header as marker for the rest.
	if len(req.Header.Get(ImpersonateUserHeader)) != 0 {
		return rt.delegate.RoundTrip(req)
	}
	req = utilnet.CloneRequest(req)
	req.Header.Set(ImpersonateUserHeader, rt.impersonate.UserName)
	if rt.impersonate.UID != "" {
		req.Header.Set(ImpersonateUIDHeader, rt.impersonate.UID)
	}
	for _, group := range rt.impersonate.Groups {
		req.Header.Add(ImpersonateGroupHeader, group)
	}
	for k, vv := range rt.impersonate.Extra {
		for _, v := range vv {
			req.Header.Add(ImpersonateUserExtraHeaderPrefix+headerKeyEscape(k), v)
		}
	}

	return rt.delegate.RoundTrip(req)
}

func (rt *impersonatingRoundTripper) CancelRequest(req *http.Request) {
	tryCancelRequest(rt.WrappedRoundTripper(), req)
}

func (rt *impersonatingRoundTripper) WrappedRoundTripper() http.RoundTripper { return rt.delegate }

type bearerAuthRoundTripper struct {
	bearer string
	source oauth2.TokenSource
	rt     http.RoundTripper
}

var _ utilnet.RoundTripperWrapper = &bearerAuthRoundTripper{}

// NewBearerAuthRoundTripper adds the provided bearer token to a request
// unless the authorization header has already been set.
func NewBearerAuthRoundTripper(bearer string, rt http.RoundTripper) http.RoundTripper {
	return &bearerAuthRoundTripper{bearer, nil, rt}
}

// NewBearerAuthWithRefreshRoundTripper adds the provided bearer token to a request
// unless the authorization header has already been set.
// If tokenFile is non-empty, it is periodically read,
// and the last successfully read content is used as the bearer token.
// If tokenFile is non-empty and bearer is empty, the tokenFile is read
// immediately to populate the initial bearer token.
func NewBearerAuthWithRefreshRoundTripper(bearer string, tokenFile string, rt http.RoundTripper) (http.RoundTripper, error) {
	if len(tokenFile) == 0 {
		return &bearerAuthRoundTripper{bearer, nil, rt}, nil
	}
	source := NewCachedFileTokenSource(tokenFile)
	if len(bearer) == 0 {
		token, err := source.Token()
		if err != nil {
			return nil, err
		}
		bearer = token.AccessToken
	}
	return &bearerAuthRoundTripper{bearer, source, rt}, nil
}

func (rt *bearerAuthRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(req.Header.Get("Authorization")) != 0 {
		return rt.rt.RoundTrip(req)
	}

	req = utilnet.CloneRequest(req)
	token := rt.bearer
	if rt.source != nil {
		if refreshedToken, err := rt.source.Token(); err == nil {
			token = refreshedToken.AccessToken
		}
	}
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
	return rt.rt.RoundTrip(req)
}

func (rt *bearerAuthRoundTripper) CancelRequest(req *http.Request) {
	tryCancelRequest(rt.WrappedRoundTripper(), req)
}

func (rt *bearerAuthRoundTripper) WrappedRoundTripper() http.RoundTripper { return rt.rt }

// requestInfo keeps track of information about a request/response combination
type requestInfo struct {
	RequestHeaders http.Header `datapolicy:"token"`
	RequestVerb    string
	RequestURL     string

	ResponseStatus  string
	ResponseHeaders http.Header
	ResponseErr     error

	muTrace          sync.Mutex // Protect trace fields
	DNSLookup        time.Duration
	Dialing          time.Duration
	GetConnection    time.Duration
	TLSHandshake     time.Duration
	ServerProcessing time.Duration
	ConnectionReused bool

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
			value = maskValue(key, value)
			headers += fmt.Sprintf(` -H %q`, fmt.Sprintf("%s: %s", key, value))
		}
	}

	return fmt.Sprintf("curl -v -X%s %s '%s'", r.RequestVerb, headers, r.RequestURL)
}

// debuggingRoundTripper will display information about the requests passing
// through it based on what is configured
type debuggingRoundTripper struct {
	delegatedRoundTripper http.RoundTripper
	levels                map[DebugLevel]bool
}

var _ utilnet.RoundTripperWrapper = &debuggingRoundTripper{}

// DebugLevel is used to enable debugging of certain
// HTTP requests and responses fields via the debuggingRoundTripper.
type DebugLevel int

const (
	// DebugJustURL will add to the debug output HTTP requests method and url.
	DebugJustURL DebugLevel = iota
	// DebugURLTiming will add to the debug output the duration of HTTP requests.
	DebugURLTiming
	// DebugCurlCommand will add to the debug output the curl command equivalent to the
	// HTTP request.
	DebugCurlCommand
	// DebugRequestHeaders will add to the debug output the HTTP requests headers.
	DebugRequestHeaders
	// DebugResponseStatus will add to the debug output the HTTP response status.
	DebugResponseStatus
	// DebugResponseHeaders will add to the debug output the HTTP response headers.
	DebugResponseHeaders
	// DebugDetailedTiming will add to the debug output the duration of the HTTP requests events.
	DebugDetailedTiming
)

// NewDebuggingRoundTripper allows to display in the logs output debug information
// on the API requests performed by the client.
func NewDebuggingRoundTripper(rt http.RoundTripper, levels ...DebugLevel) http.RoundTripper {
	drt := &debuggingRoundTripper{
		delegatedRoundTripper: rt,
		levels:                make(map[DebugLevel]bool, len(levels)),
	}
	for _, v := range levels {
		drt.levels[v] = true
	}
	return drt
}

func (rt *debuggingRoundTripper) CancelRequest(req *http.Request) {
	tryCancelRequest(rt.WrappedRoundTripper(), req)
}

var knownAuthTypes = map[string]bool{
	"bearer":    true,
	"basic":     true,
	"negotiate": true,
}

// maskValue masks credential content from authorization headers
// See https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Authorization
func maskValue(key string, value string) string {
	if !strings.EqualFold(key, "Authorization") {
		return value
	}
	if len(value) == 0 {
		return ""
	}
	var authType string
	if i := strings.Index(value, " "); i > 0 {
		authType = value[0:i]
	} else {
		authType = value
	}
	if !knownAuthTypes[strings.ToLower(authType)] {
		return "<masked>"
	}
	if len(value) > len(authType)+1 {
		value = authType + " <masked>"
	} else {
		value = authType
	}
	return value
}

func (rt *debuggingRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	reqInfo := newRequestInfo(req)

	if rt.levels[DebugJustURL] {
		klog.Infof("%s %s", reqInfo.RequestVerb, reqInfo.RequestURL)
	}
	if rt.levels[DebugCurlCommand] {
		klog.Infof("%s", reqInfo.toCurl())
	}
	if rt.levels[DebugRequestHeaders] {
		klog.Info("Request Headers:")
		for key, values := range reqInfo.RequestHeaders {
			for _, value := range values {
				value = maskValue(key, value)
				klog.Infof("    %s: %s", key, value)
			}
		}
	}

	startTime := time.Now()

	if rt.levels[DebugDetailedTiming] {
		var getConn, dnsStart, dialStart, tlsStart, serverStart time.Time
		var host string
		trace := &httptrace.ClientTrace{
			// DNS
			DNSStart: func(info httptrace.DNSStartInfo) {
				reqInfo.muTrace.Lock()
				defer reqInfo.muTrace.Unlock()
				dnsStart = time.Now()
				host = info.Host
			},
			DNSDone: func(info httptrace.DNSDoneInfo) {
				reqInfo.muTrace.Lock()
				defer reqInfo.muTrace.Unlock()
				reqInfo.DNSLookup = time.Now().Sub(dnsStart)
				klog.Infof("HTTP Trace: DNS Lookup for %s resolved to %v", host, info.Addrs)
			},
			// Dial
			ConnectStart: func(network, addr string) {
				reqInfo.muTrace.Lock()
				defer reqInfo.muTrace.Unlock()
				dialStart = time.Now()
			},
			ConnectDone: func(network, addr string, err error) {
				reqInfo.muTrace.Lock()
				defer reqInfo.muTrace.Unlock()
				reqInfo.Dialing = time.Now().Sub(dialStart)
				if err != nil {
					klog.Infof("HTTP Trace: Dial to %s:%s failed: %v", network, addr, err)
				} else {
					klog.Infof("HTTP Trace: Dial to %s:%s succeed", network, addr)
				}
			},
			// TLS
			TLSHandshakeStart: func() {
				tlsStart = time.Now()
			},
			TLSHandshakeDone: func(_ tls.ConnectionState, _ error) {
				reqInfo.muTrace.Lock()
				defer reqInfo.muTrace.Unlock()
				reqInfo.TLSHandshake = time.Now().Sub(tlsStart)
			},
			// Connection (it can be DNS + Dial or just the time to get one from the connection pool)
			GetConn: func(hostPort string) {
				getConn = time.Now()
			},
			GotConn: func(info httptrace.GotConnInfo) {
				reqInfo.muTrace.Lock()
				defer reqInfo.muTrace.Unlock()
				reqInfo.GetConnection = time.Now().Sub(getConn)
				reqInfo.ConnectionReused = info.Reused
			},
			// Server Processing (time since we wrote the request until first byte is received)
			WroteRequest: func(info httptrace.WroteRequestInfo) {
				reqInfo.muTrace.Lock()
				defer reqInfo.muTrace.Unlock()
				serverStart = time.Now()
			},
			GotFirstResponseByte: func() {
				reqInfo.muTrace.Lock()
				defer reqInfo.muTrace.Unlock()
				reqInfo.ServerProcessing = time.Now().Sub(serverStart)
			},
		}
		req = req.WithContext(httptrace.WithClientTrace(req.Context(), trace))
	}

	response, err := rt.delegatedRoundTripper.RoundTrip(req)
	reqInfo.Duration = time.Since(startTime)

	reqInfo.complete(response, err)

	if rt.levels[DebugURLTiming] {
		klog.Infof("%s %s %s in %d milliseconds", reqInfo.RequestVerb, reqInfo.RequestURL, reqInfo.ResponseStatus, reqInfo.Duration.Nanoseconds()/int64(time.Millisecond))
	}
	if rt.levels[DebugDetailedTiming] {
		stats := ""
		if !reqInfo.ConnectionReused {
			stats += fmt.Sprintf(`DNSLookup %d ms Dial %d ms TLSHandshake %d ms`,
				reqInfo.DNSLookup.Nanoseconds()/int64(time.Millisecond),
				reqInfo.Dialing.Nanoseconds()/int64(time.Millisecond),
				reqInfo.TLSHandshake.Nanoseconds()/int64(time.Millisecond),
			)
		} else {
			stats += fmt.Sprintf(`GetConnection %d ms`, reqInfo.GetConnection.Nanoseconds()/int64(time.Millisecond))
		}
		if reqInfo.ServerProcessing != 0 {
			stats += fmt.Sprintf(` ServerProcessing %d ms`, reqInfo.ServerProcessing.Nanoseconds()/int64(time.Millisecond))
		}
		stats += fmt.Sprintf(` Duration %d ms`, reqInfo.Duration.Nanoseconds()/int64(time.Millisecond))
		klog.Infof("HTTP Statistics: %s", stats)
	}

	if rt.levels[DebugResponseStatus] {
		klog.Infof("Response Status: %s in %d milliseconds", reqInfo.ResponseStatus, reqInfo.Duration.Nanoseconds()/int64(time.Millisecond))
	}
	if rt.levels[DebugResponseHeaders] {
		klog.Info("Response Headers:")
		for key, values := range reqInfo.ResponseHeaders {
			for _, value := range values {
				klog.Infof("    %s: %s", key, value)
			}
		}
	}

	return response, err
}

func (rt *debuggingRoundTripper) WrappedRoundTripper() http.RoundTripper {
	return rt.delegatedRoundTripper
}

func legalHeaderByte(b byte) bool {
	return int(b) < len(legalHeaderKeyBytes) && legalHeaderKeyBytes[b]
}

func shouldEscape(b byte) bool {
	// url.PathUnescape() returns an error if any '%' is not followed by two
	// hexadecimal digits, so we'll intentionally encode it.
	return !legalHeaderByte(b) || b == '%'
}

func headerKeyEscape(key string) string {
	buf := strings.Builder{}
	for i := 0; i < len(key); i++ {
		b := key[i]
		if shouldEscape(b) {
			// %-encode bytes that should be escaped:
			// https://tools.ietf.org/html/rfc3986#section-2.1
			fmt.Fprintf(&buf, "%%%02X", b)
			continue
		}
		buf.WriteByte(b)
	}
	return buf.String()
}

// legalHeaderKeyBytes was copied from net/http/lex.go's isTokenTable.
// See https://httpwg.github.io/specs/rfc7230.html#rule.token.separators
var legalHeaderKeyBytes = [127]bool{
	'%':  true,
	'!':  true,
	'#':  true,
	'$':  true,
	'&':  true,
	'\'': true,
	'*':  true,
	'+':  true,
	'-':  true,
	'.':  true,
	'0':  true,
	'1':  true,
	'2':  true,
	'3':  true,
	'4':  true,
	'5':  true,
	'6':  true,
	'7':  true,
	'8':  true,
	'9':  true,
	'A':  true,
	'B':  true,
	'C':  true,
	'D':  true,
	'E':  true,
	'F':  true,
	'G':  true,
	'H':  true,
	'I':  true,
	'J':  true,
	'K':  true,
	'L':  true,
	'M':  true,
	'N':  true,
	'O':  true,
	'P':  true,
	'Q':  true,
	'R':  true,
	'S':  true,
	'T':  true,
	'U':  true,
	'W':  true,
	'V':  true,
	'X':  true,
	'Y':  true,
	'Z':  true,
	'^':  true,
	'_':  true,
	'`':  true,
	'a':  true,
	'b':  true,
	'c':  true,
	'd':  true,
	'e':  true,
	'f':  true,
	'g':  true,
	'h':  true,
	'i':  true,
	'j':  true,
	'k':  true,
	'l':  true,
	'm':  true,
	'n':  true,
	'o':  true,
	'p':  true,
	'q':  true,
	'r':  true,
	's':  true,
	't':  true,
	'u':  true,
	'v':  true,
	'w':  true,
	'x':  true,
	'y':  true,
	'z':  true,
	'|':  true,
	'~':  true,
}
