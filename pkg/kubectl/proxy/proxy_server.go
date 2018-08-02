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

package proxy

import (
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/golang/glog"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/proxy"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/transport"
	"k8s.io/kubernetes/pkg/kubectl/util"
)

const (
	// DefaultHostAcceptRE is the default value for which hosts to accept.
	DefaultHostAcceptRE = "^localhost$,^127\\.0\\.0\\.1$,^\\[::1\\]$"
	// DefaultPathAcceptRE is the default path to accept.
	DefaultPathAcceptRE = "^.*"
	// DefaultPathRejectRE is the default set of paths to reject.
	DefaultPathRejectRE = "^/api/.*/pods/.*/exec,^/api/.*/pods/.*/attach"
	// DefaultMethodRejectRE is the set of HTTP methods to reject by default.
	DefaultMethodRejectRE = "^$"
)

var (
	// ReverseProxyFlushInterval is the frequency to flush the reverse proxy.
	// Only matters for long poll connections like the one used to watch. With an
	// interval of 0 the reverse proxy will buffer content sent on any connection
	// with transfer-encoding=chunked.
	// TODO: Flush after each chunk so the client doesn't suffer a 100ms latency per
	// watch event.
	ReverseProxyFlushInterval = 100 * time.Millisecond
)

// FilterServer rejects requests which don't match one of the specified regular expressions
type FilterServer struct {
	// Only paths that match this regexp will be accepted
	AcceptPaths []*regexp.Regexp
	// Paths that match this regexp will be rejected, even if they match the above
	RejectPaths []*regexp.Regexp
	// Hosts are required to match this list of regexp
	AcceptHosts []*regexp.Regexp
	// Methods that match this regexp are rejected
	RejectMethods []*regexp.Regexp
	// The delegate to call to handle accepted requests.
	delegate http.Handler
}

// MakeRegexpArray splits a comma separated list of regexps into an array of Regexp objects.
func MakeRegexpArray(str string) ([]*regexp.Regexp, error) {
	parts := strings.Split(str, ",")
	result := make([]*regexp.Regexp, len(parts))
	for ix := range parts {
		re, err := regexp.Compile(parts[ix])
		if err != nil {
			return nil, err
		}
		result[ix] = re
	}
	return result, nil
}

// MakeRegexpArrayOrDie creates an array of regular expression objects from a string or exits.
func MakeRegexpArrayOrDie(str string) []*regexp.Regexp {
	result, err := MakeRegexpArray(str)
	if err != nil {
		glog.Fatalf("Error compiling re: %v", err)
	}
	return result
}

func matchesRegexp(str string, regexps []*regexp.Regexp) bool {
	for _, re := range regexps {
		if re.MatchString(str) {
			glog.V(6).Infof("%v matched %s", str, re)
			return true
		}
	}
	return false
}

func (f *FilterServer) accept(method, path, host string) bool {
	if matchesRegexp(path, f.RejectPaths) {
		return false
	}
	if matchesRegexp(method, f.RejectMethods) {
		return false
	}
	if matchesRegexp(path, f.AcceptPaths) && matchesRegexp(host, f.AcceptHosts) {
		return true
	}
	return false
}

// HandlerFor makes a shallow copy of f which passes its requests along to the
// new delegate.
func (f *FilterServer) HandlerFor(delegate http.Handler) *FilterServer {
	f2 := *f
	f2.delegate = delegate
	return &f2
}

// Get host from a host header value like "localhost" or "localhost:8080"
func extractHost(header string) (host string) {
	host, _, err := net.SplitHostPort(header)
	if err != nil {
		host = header
	}
	return host
}

func (f *FilterServer) ServeHTTP(rw http.ResponseWriter, req *http.Request) {
	host := extractHost(req.Host)
	if f.accept(req.Method, req.URL.Path, host) {
		glog.V(3).Infof("Filter accepting %v %v %v", req.Method, req.URL.Path, host)
		f.delegate.ServeHTTP(rw, req)
		return
	}
	glog.V(3).Infof("Filter rejecting %v %v %v", req.Method, req.URL.Path, host)
	rw.WriteHeader(http.StatusForbidden)
	rw.Write([]byte("<h3>Unauthorized</h3>"))
}

// Server is a http.Handler which proxies Kubernetes APIs to remote API server.
type Server struct {
	handler http.Handler
}

type responder struct{}

func (r *responder) Error(w http.ResponseWriter, req *http.Request, err error) {
	glog.Errorf("Error while proxying request: %v", err)
	http.Error(w, err.Error(), http.StatusInternalServerError)
}

// makeUpgradeTransport creates a transport that explicitly bypasses HTTP2 support
// for proxy connections that must upgrade.
func makeUpgradeTransport(config *rest.Config, keepalive time.Duration) (proxy.UpgradeRequestRoundTripper, error) {
	transportConfig, err := config.TransportConfig()
	if err != nil {
		return nil, err
	}
	tlsConfig, err := transport.TLSConfigFor(transportConfig)
	if err != nil {
		return nil, err
	}
	rt := utilnet.SetOldTransportDefaults(&http.Transport{
		TLSClientConfig: tlsConfig,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: keepalive,
		}).DialContext,
	})

	upgrader, err := transport.HTTPWrappersForConfig(transportConfig, proxy.MirrorRequest)
	if err != nil {
		return nil, err
	}
	return proxy.NewUpgradeRequestRoundTripper(rt, upgrader), nil
}

// NewServer creates and installs a new Server.
// 'filter', if non-nil, protects requests to the api only.
func NewServer(filebase string, apiProxyPrefix string, staticPrefix string, filter *FilterServer, cfg *rest.Config, keepalive time.Duration) (*Server, error) {
	host := cfg.Host
	if !strings.HasSuffix(host, "/") {
		host = host + "/"
	}
	target, err := url.Parse(host)
	if err != nil {
		return nil, err
	}

	responder := &responder{}
	transport, err := rest.TransportFor(cfg)
	if err != nil {
		return nil, err
	}
	upgradeTransport, err := makeUpgradeTransport(cfg, keepalive)
	if err != nil {
		return nil, err
	}
	proxy := proxy.NewUpgradeAwareHandler(target, transport, false, false, responder)
	proxy.UpgradeTransport = upgradeTransport
	proxy.UseRequestLocation = true

	proxyServer := http.Handler(proxy)
	if filter != nil {
		proxyServer = filter.HandlerFor(proxyServer)
	}

	if !strings.HasPrefix(apiProxyPrefix, "/api") {
		proxyServer = stripLeaveSlash(apiProxyPrefix, proxyServer)
	}

	mux := http.NewServeMux()
	mux.Handle(apiProxyPrefix, proxyServer)
	if filebase != "" {
		// Require user to explicitly request this behavior rather than
		// serving their working directory by default.
		mux.Handle(staticPrefix, newFileHandler(staticPrefix, filebase))
	}
	return &Server{handler: mux}, nil
}

// Listen is a simple wrapper around net.Listen.
func (s *Server) Listen(address string, port int) (net.Listener, error) {
	return net.Listen("tcp", fmt.Sprintf("%s:%d", address, port))
}

// ListenUnix does net.Listen for a unix socket
func (s *Server) ListenUnix(path string) (net.Listener, error) {
	// Remove any socket, stale or not, but fall through for other files
	fi, err := os.Stat(path)
	if err == nil && (fi.Mode()&os.ModeSocket) != 0 {
		os.Remove(path)
	}
	// Default to only user accessible socket, caller can open up later if desired
	oldmask, _ := util.Umask(0077)
	l, err := net.Listen("unix", path)
	util.Umask(oldmask)
	return l, err
}

// ServeOnListener starts the server using given listener, loops forever.
func (s *Server) ServeOnListener(l net.Listener) error {
	server := http.Server{
		Handler: s.handler,
	}
	return server.Serve(l)
}

func newFileHandler(prefix, base string) http.Handler {
	return http.StripPrefix(prefix, http.FileServer(http.Dir(base)))
}

// like http.StripPrefix, but always leaves an initial slash. (so that our
// regexps will work.)
func stripLeaveSlash(prefix string, h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		p := strings.TrimPrefix(req.URL.Path, prefix)
		if len(p) >= len(req.URL.Path) {
			http.NotFound(w, req)
			return
		}
		if len(p) > 0 && p[:1] != "/" {
			p = "/" + p
		}
		req.URL.Path = p
		h.ServeHTTP(w, req)
	})
}
