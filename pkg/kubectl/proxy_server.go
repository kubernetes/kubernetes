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

package kubectl

import (
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"regexp"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/golang/glog"
)

const (
	DefaultHostAcceptRE   = "^localhost$,^127\\.0\\.0\\.1$,^\\[::1\\]$"
	DefaultPathAcceptRE   = "^/api/.*"
	DefaultPathRejectRE   = "^/api/.*/exec,^/api/.*/run"
	DefaultMethodRejectRE = "POST,PUT,PATCH"
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

// Splits a comma separated list of regexps into a array of Regexp objects.
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

func (f *FilterServer) ServeHTTP(rw http.ResponseWriter, req *http.Request) {
	if f.accept(req.Method, req.URL.Path, req.Host) {
		f.delegate.ServeHTTP(rw, req)
	}
	rw.WriteHeader(http.StatusForbidden)
	rw.Write([]byte("<h3>Unauthorized</h3>"))
}

// ProxyServer is a http.Handler which proxies Kubernetes APIs to remote API server.
type ProxyServer struct {
	mux *http.ServeMux
	httputil.ReverseProxy
}

// NewProxyServer creates and installs a new ProxyServer.
// It automatically registers the created ProxyServer to http.DefaultServeMux.
func NewProxyServer(filebase string, apiProxyPrefix string, staticPrefix string, filter *FilterServer, cfg *client.Config) (*ProxyServer, error) {
	host := cfg.Host
	if !strings.HasSuffix(host, "/") {
		host = host + "/"
	}
	target, err := url.Parse(host)
	if err != nil {
		return nil, err
	}
	proxy := newProxyServer(target)
	if proxy.Transport, err = client.TransportFor(cfg); err != nil {
		return nil, err
	}

	var server http.Handler
	if strings.HasPrefix(apiProxyPrefix, "/api") {
		server = proxy
	} else {
		server = http.StripPrefix(apiProxyPrefix, proxy)
	}
	if filter != nil {
		filter.delegate = server
		server = filter
	}

	proxy.mux.Handle(apiProxyPrefix, server)
	proxy.mux.Handle(staticPrefix, newFileHandler(staticPrefix, filebase))
	return proxy, nil
}

// Serve starts the server (http.DefaultServeMux) on given port, loops forever.
func (s *ProxyServer) Serve(port int) error {
	server := http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: s.mux,
	}
	return server.ListenAndServe()
}

func newProxyServer(target *url.URL) *ProxyServer {
	director := func(req *http.Request) {
		req.URL.Scheme = target.Scheme
		req.URL.Host = target.Host
		req.URL.Path = singleJoiningSlash(target.Path, req.URL.Path)
	}
	return &ProxyServer{
		ReverseProxy: httputil.ReverseProxy{Director: director},
		mux:          http.NewServeMux(),
	}
}

func newFileHandler(prefix, base string) http.Handler {
	return http.StripPrefix(prefix, http.FileServer(http.Dir(base)))
}

func singleJoiningSlash(a, b string) string {
	aslash := strings.HasSuffix(a, "/")
	bslash := strings.HasPrefix(b, "/")
	switch {
	case aslash && bslash:
		return a + b[1:]
	case !aslash && !bslash:
		return a + "/" + b
	}
	return a + b
}
