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

package kubectl

import (
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

// ProxyServer is a http.Handler which proxies Kubernetes APIs to remote API server.
type ProxyServer struct {
	httputil.ReverseProxy
	Port int
}

// NewProxyServer creates and installs a new ProxyServer.
// It automatically registers the created ProxyServer to http.DefaultServeMux.
func NewProxyServer(filebase string, cfg *client.Config, port int) (*ProxyServer, error) {
	prefix := cfg.Prefix
	if prefix == "" {
		prefix = "/api"
	}
	target, err := url.Parse(singleJoiningSlash(cfg.Host, prefix))
	if err != nil {
		return nil, err
	}
	proxy := newProxyServer(target)
	if proxy.Transport, err = client.TransportFor(cfg); err != nil {
		return nil, err
	}
	http.Handle("/api/", http.StripPrefix("/api/", proxy))
	http.Handle("/static/", newFileHandler("/static/", filebase))
	return proxy, nil
}

// Serve starts the server (http.DefaultServeMux) on TCP port 8001, loops forever.
func (s *ProxyServer) Serve() error {
	addr := fmt.Sprintf(":%d", s.Port)
	return http.ListenAndServe(addr, nil)
}

func newProxyServer(target *url.URL) *ProxyServer {
	director := func(req *http.Request) {
		req.URL.Scheme = target.Scheme
		req.URL.Host = target.Host
		req.URL.Path = singleJoiningSlash(target.Path, req.URL.Path)
	}
	return &ProxyServer{ReverseProxy: httputil.ReverseProxy{Director: director}}
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
