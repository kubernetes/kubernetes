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
	"io"
	"math/rand"
	"net/http"
	"net/http/httputil"
	"net/url"
	"path"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apiserver/metrics"
	"k8s.io/kubernetes/pkg/httplog"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/httpstream"
	proxyutil "k8s.io/kubernetes/pkg/util/proxy"

	"github.com/golang/glog"
)

// ProxyHandler provides a http.Handler which will proxy traffic to locations
// specified by items implementing Redirector.
type ProxyHandler struct {
	prefix              string
	storage             map[string]rest.Storage
	codec               runtime.Codec
	context             api.RequestContextMapper
	requestInfoResolver *RequestInfoResolver
}

func (r *ProxyHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	proxyHandlerTraceID := rand.Int63()

	var verb string
	var apiResource string
	var httpCode int
	reqStart := time.Now()
	defer metrics.Monitor(&verb, &apiResource, util.GetClient(req), &httpCode, reqStart)

	requestInfo, err := r.requestInfoResolver.GetRequestInfo(req)
	if err != nil || !requestInfo.IsResourceRequest {
		notFound(w, req)
		httpCode = http.StatusNotFound
		return
	}
	verb = requestInfo.Verb
	namespace, resource, parts := requestInfo.Namespace, requestInfo.Resource, requestInfo.Parts

	ctx, ok := r.context.Get(req)
	if !ok {
		ctx = api.NewContext()
	}
	ctx = api.WithNamespace(ctx, namespace)
	if len(parts) < 2 {
		notFound(w, req)
		httpCode = http.StatusNotFound
		return
	}
	id := parts[1]
	remainder := ""
	if len(parts) > 2 {
		proxyParts := parts[2:]
		remainder = strings.Join(proxyParts, "/")
		if strings.HasSuffix(req.URL.Path, "/") {
			// The original path had a trailing slash, which has been stripped
			// by KindAndNamespace(). We should add it back because some
			// servers (like etcd) require it.
			remainder = remainder + "/"
		}
	}
	storage, ok := r.storage[resource]
	if !ok {
		httplog.LogOf(req, w).Addf("'%v' has no storage object", resource)
		notFound(w, req)
		httpCode = http.StatusNotFound
		return
	}
	apiResource = resource

	redirector, ok := storage.(rest.Redirector)
	if !ok {
		httplog.LogOf(req, w).Addf("'%v' is not a redirector", resource)
		httpCode = errorJSON(errors.NewMethodNotSupported(resource, "proxy"), r.codec, w)
		return
	}

	location, roundTripper, err := redirector.ResourceLocation(ctx, id)
	if err != nil {
		httplog.LogOf(req, w).Addf("Error getting ResourceLocation: %v", err)
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w, true)
		httpCode = status.Code
		return
	}
	if location == nil {
		httplog.LogOf(req, w).Addf("ResourceLocation for %v returned nil", id)
		notFound(w, req)
		httpCode = http.StatusNotFound
		return
	}

	if roundTripper != nil {
		glog.V(5).Infof("[%x: %v] using transport %T...", proxyHandlerTraceID, req.URL, roundTripper)
	}

	// Default to http
	if location.Scheme == "" {
		location.Scheme = "http"
	}
	// Add the subpath
	if len(remainder) > 0 {
		location.Path = singleJoiningSlash(location.Path, remainder)
	}
	// Start with anything returned from the storage, and add the original request's parameters
	values := location.Query()
	for k, vs := range req.URL.Query() {
		for _, v := range vs {
			values.Add(k, v)
		}
	}
	location.RawQuery = values.Encode()

	newReq, err := http.NewRequest(req.Method, location.String(), req.Body)
	if err != nil {
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w, true)
		notFound(w, req)
		httpCode = status.Code
		return
	}
	httpCode = http.StatusOK
	newReq.Header = req.Header

	// TODO convert this entire proxy to an UpgradeAwareProxy similar to
	// https://github.com/openshift/origin/blob/master/pkg/util/httpproxy/upgradeawareproxy.go.
	// That proxy needs to be modified to support multiple backends, not just 1.
	if r.tryUpgrade(w, req, newReq, location, roundTripper) {
		return
	}

	// Redirect requests of the form "/{resource}/{name}" to "/{resource}/{name}/"
	// This is essentially a hack for http://issue.k8s.io/4958.
	// Note: Keep this code after tryUpgrade to not break that flow.
	if len(parts) == 2 && !strings.HasSuffix(req.URL.Path, "/") {
		var queryPart string
		if len(req.URL.RawQuery) > 0 {
			queryPart = "?" + req.URL.RawQuery
		}
		w.Header().Set("Location", req.URL.Path+"/"+queryPart)
		w.WriteHeader(http.StatusMovedPermanently)
		return
	}

	start := time.Now()
	glog.V(4).Infof("[%x] Beginning proxy %s...", proxyHandlerTraceID, req.URL)
	defer func() {
		glog.V(4).Infof("[%x] Proxy %v finished %v.", proxyHandlerTraceID, req.URL, time.Now().Sub(start))
	}()

	proxy := httputil.NewSingleHostReverseProxy(&url.URL{Scheme: location.Scheme, Host: location.Host})
	alreadyRewriting := false
	if roundTripper != nil {
		_, alreadyRewriting = roundTripper.(*proxyutil.Transport)
		glog.V(5).Infof("[%x] Not making a reriting transport for proxy %s...", proxyHandlerTraceID, req.URL)
	}
	if !alreadyRewriting {
		glog.V(5).Infof("[%x] making a transport for proxy %s...", proxyHandlerTraceID, req.URL)
		prepend := path.Join(r.prefix, resource, id)
		if len(namespace) > 0 {
			prepend = path.Join(r.prefix, "namespaces", namespace, resource, id)
		}
		pTransport := &proxyutil.Transport{
			Scheme:       req.URL.Scheme,
			Host:         req.URL.Host,
			PathPrepend:  prepend,
			RoundTripper: roundTripper,
		}
		roundTripper = pTransport
	}
	proxy.Transport = roundTripper
	proxy.FlushInterval = 200 * time.Millisecond
	proxy.ServeHTTP(w, newReq)
}

// tryUpgrade returns true if the request was handled.
func (r *ProxyHandler) tryUpgrade(w http.ResponseWriter, req, newReq *http.Request, location *url.URL, transport http.RoundTripper) bool {
	if !httpstream.IsUpgradeRequest(req) {
		return false
	}
	backendConn, err := proxyutil.DialURL(location, transport)
	if err != nil {
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w, true)
		return true
	}
	defer backendConn.Close()

	// TODO should we use _ (a bufio.ReadWriter) instead of requestHijackedConn
	// when copying between the client and the backend? Docker doesn't when they
	// hijack, just for reference...
	requestHijackedConn, _, err := w.(http.Hijacker).Hijack()
	if err != nil {
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w, true)
		return true
	}
	defer requestHijackedConn.Close()

	if err = newReq.Write(backendConn); err != nil {
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w, true)
		return true
	}

	done := make(chan struct{}, 2)

	go func() {
		_, err := io.Copy(backendConn, requestHijackedConn)
		if err != nil && !strings.Contains(err.Error(), "use of closed network connection") {
			glog.Errorf("Error proxying data from client to backend: %v", err)
		}
		done <- struct{}{}
	}()

	go func() {
		_, err := io.Copy(requestHijackedConn, backendConn)
		if err != nil && !strings.Contains(err.Error(), "use of closed network connection") {
			glog.Errorf("Error proxying data from backend to client: %v", err)
		}
		done <- struct{}{}
	}()

	<-done
	return true
}

// borrowed from net/http/httputil/reverseproxy.go
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
