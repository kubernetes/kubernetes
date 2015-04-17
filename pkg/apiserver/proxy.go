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
	"bytes"
	"compress/gzip"
	"crypto/tls"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"path"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/httplog"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/httpstream"

	"github.com/GoogleCloudPlatform/kubernetes/third_party/golang/netutil"
	"github.com/golang/glog"
	"golang.org/x/net/html"
)

// tagsToAttrs states which attributes of which tags require URL substitution.
// Sources: http://www.w3.org/TR/REC-html40/index/attributes.html
//          http://www.w3.org/html/wg/drafts/html/master/index.html#attributes-1
var tagsToAttrs = map[string]util.StringSet{
	"a":          util.NewStringSet("href"),
	"applet":     util.NewStringSet("codebase"),
	"area":       util.NewStringSet("href"),
	"audio":      util.NewStringSet("src"),
	"base":       util.NewStringSet("href"),
	"blockquote": util.NewStringSet("cite"),
	"body":       util.NewStringSet("background"),
	"button":     util.NewStringSet("formaction"),
	"command":    util.NewStringSet("icon"),
	"del":        util.NewStringSet("cite"),
	"embed":      util.NewStringSet("src"),
	"form":       util.NewStringSet("action"),
	"frame":      util.NewStringSet("longdesc", "src"),
	"head":       util.NewStringSet("profile"),
	"html":       util.NewStringSet("manifest"),
	"iframe":     util.NewStringSet("longdesc", "src"),
	"img":        util.NewStringSet("longdesc", "src", "usemap"),
	"input":      util.NewStringSet("src", "usemap", "formaction"),
	"ins":        util.NewStringSet("cite"),
	"link":       util.NewStringSet("href"),
	"object":     util.NewStringSet("classid", "codebase", "data", "usemap"),
	"q":          util.NewStringSet("cite"),
	"script":     util.NewStringSet("src"),
	"source":     util.NewStringSet("src"),
	"video":      util.NewStringSet("poster", "src"),

	// TODO: css URLs hidden in style elements.
}

// ProxyHandler provides a http.Handler which will proxy traffic to locations
// specified by items implementing Redirector.
type ProxyHandler struct {
	prefix                 string
	storage                map[string]rest.Storage
	codec                  runtime.Codec
	context                api.RequestContextMapper
	apiRequestInfoResolver *APIRequestInfoResolver
}

func (r *ProxyHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	var verb string
	var apiResource string
	var httpCode int
	reqStart := time.Now()
	defer monitor("proxy", &verb, &apiResource, &httpCode, reqStart)

	requestInfo, err := r.apiRequestInfoResolver.GetAPIRequestInfo(req)
	if err != nil {
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

	location, transport, err := redirector.ResourceLocation(ctx, id)
	if err != nil {
		httplog.LogOf(req, w).Addf("Error getting ResourceLocation: %v", err)
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w)
		httpCode = status.Code
		return
	}
	if location == nil {
		httplog.LogOf(req, w).Addf("ResourceLocation for %v returned nil", id)
		notFound(w, req)
		httpCode = http.StatusNotFound
		return
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
		writeJSON(status.Code, r.codec, status, w)
		notFound(w, req)
		httpCode = status.Code
		return
	}
	httpCode = http.StatusOK
	newReq.Header = req.Header

	// TODO convert this entire proxy to an UpgradeAwareProxy similar to
	// https://github.com/openshift/origin/blob/master/pkg/util/httpproxy/upgradeawareproxy.go.
	// That proxy needs to be modified to support multiple backends, not just 1.
	if r.tryUpgrade(w, req, newReq, location, transport) {
		return
	}

	// Redirect requests of the form "/{resource}/{name}" to "/{resource}/{name}/"
	// This is essentially a hack for https://github.com/GoogleCloudPlatform/kubernetes/issues/4958.
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

	proxy := httputil.NewSingleHostReverseProxy(&url.URL{Scheme: location.Scheme, Host: location.Host})
	if transport == nil {
		prepend := path.Join(r.prefix, resource, id)
		if len(namespace) > 0 {
			prepend = path.Join(r.prefix, "namespaces", namespace, resource, id)
		}
		transport = &proxyTransport{
			proxyScheme:      req.URL.Scheme,
			proxyHost:        req.URL.Host,
			proxyPathPrepend: prepend,
		}
	}
	proxy.Transport = transport
	proxy.FlushInterval = 200 * time.Millisecond
	proxy.ServeHTTP(w, newReq)
}

// tryUpgrade returns true if the request was handled.
func (r *ProxyHandler) tryUpgrade(w http.ResponseWriter, req, newReq *http.Request, location *url.URL, transport http.RoundTripper) bool {
	if !httpstream.IsUpgradeRequest(req) {
		return false
	}

	backendConn, err := dialURL(location, transport)
	if err != nil {
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w)
		return true
	}
	defer backendConn.Close()

	// TODO should we use _ (a bufio.ReadWriter) instead of requestHijackedConn
	// when copying between the client and the backend? Docker doesn't when they
	// hijack, just for reference...
	requestHijackedConn, _, err := w.(http.Hijacker).Hijack()
	if err != nil {
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w)
		return true
	}
	defer requestHijackedConn.Close()

	if err = newReq.Write(backendConn); err != nil {
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w)
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

func dialURL(url *url.URL, transport http.RoundTripper) (net.Conn, error) {
	dialAddr := netutil.CanonicalAddr(url)

	switch url.Scheme {
	case "http":
		return net.Dial("tcp", dialAddr)
	case "https":
		// Get the tls config from the transport if we recognize it
		var tlsConfig *tls.Config
		if transport != nil {
			httpTransport, ok := transport.(*http.Transport)
			if ok {
				tlsConfig = httpTransport.TLSClientConfig
			}
		}

		// Dial
		tlsConn, err := tls.Dial("tcp", dialAddr, tlsConfig)
		if err != nil {
			return nil, err
		}

		// Verify
		host, _, _ := net.SplitHostPort(dialAddr)
		if err := tlsConn.VerifyHostname(host); err != nil {
			tlsConn.Close()
			return nil, err
		}

		return tlsConn, nil
	default:
		return nil, fmt.Errorf("Unknown scheme: %s", url.Scheme)
	}
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

type proxyTransport struct {
	proxyScheme      string
	proxyHost        string
	proxyPathPrepend string
}

func (t *proxyTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Add reverse proxy headers.
	req.Header.Set("X-Forwarded-Uri", t.proxyPathPrepend+req.URL.Path)
	req.Header.Set("X-Forwarded-Host", t.proxyHost)
	req.Header.Set("X-Forwarded-Proto", t.proxyScheme)

	resp, err := http.DefaultTransport.RoundTrip(req)

	if err != nil {
		message := fmt.Sprintf("Error: '%s'\nTrying to reach: '%v'", err.Error(), req.URL.String())
		resp = &http.Response{
			StatusCode: http.StatusServiceUnavailable,
			Body:       ioutil.NopCloser(strings.NewReader(message)),
		}
		return resp, nil
	}

	if redirect := resp.Header.Get("Location"); redirect != "" {
		resp.Header.Set("Location", t.rewriteURL(redirect, req.URL))
	}

	cType := resp.Header.Get("Content-Type")
	cType = strings.TrimSpace(strings.SplitN(cType, ";", 2)[0])
	if cType != "text/html" {
		// Do nothing, simply pass through
		return resp, nil
	}

	return t.fixLinks(req, resp)
}

// rewriteURL rewrites a single URL to go through the proxy, if the URL refers
// to the same host as sourceURL, which is the page on which the target URL
// occurred. If any error occurs (e.g. parsing), it returns targetURL.
func (t *proxyTransport) rewriteURL(targetURL string, sourceURL *url.URL) string {
	url, err := url.Parse(targetURL)
	if err != nil {
		return targetURL
	}
	if url.Host != "" && url.Host != sourceURL.Host {
		return targetURL
	}

	url.Scheme = t.proxyScheme
	url.Host = t.proxyHost
	origPath := url.Path

	if strings.HasPrefix(url.Path, "/") {
		// The path is rooted at the host. Just add proxy prepend.
		url.Path = path.Join(t.proxyPathPrepend, url.Path)
	} else {
		// The path is relative to sourceURL.
		url.Path = path.Join(t.proxyPathPrepend, path.Dir(sourceURL.Path), url.Path)
	}

	if strings.HasSuffix(origPath, "/") {
		// Add back the trailing slash, which was stripped by path.Join().
		url.Path += "/"
	}

	return url.String()
}

// updateURLs checks and updates any of n's attributes that are listed in tagsToAttrs.
// Any URLs found are, if they're relative, updated with the necessary changes to make
// a visit to that URL also go through the proxy.
// sourceURL is the URL of the page which we're currently on; it's required to make
// relative links work.
func (t *proxyTransport) updateURLs(n *html.Node, sourceURL *url.URL) {
	if n.Type != html.ElementNode {
		return
	}
	attrs, ok := tagsToAttrs[n.Data]
	if !ok {
		return
	}
	for i, attr := range n.Attr {
		if !attrs.Has(attr.Key) {
			continue
		}
		n.Attr[i].Val = t.rewriteURL(attr.Val, sourceURL)
	}
}

// scan recursively calls f for every n and every subnode of n.
func (t *proxyTransport) scan(n *html.Node, f func(*html.Node)) {
	f(n)
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		t.scan(c, f)
	}
}

// fixLinks modifies links in an HTML file such that they will be redirected through the proxy if needed.
func (t *proxyTransport) fixLinks(req *http.Request, resp *http.Response) (*http.Response, error) {
	origBody := resp.Body
	defer origBody.Close()

	newContent := &bytes.Buffer{}
	var reader io.Reader = origBody
	var writer io.Writer = newContent
	encoding := resp.Header.Get("Content-Encoding")
	switch encoding {
	case "gzip":
		var err error
		reader, err = gzip.NewReader(reader)
		if err != nil {
			return nil, fmt.Errorf("errorf making gzip reader: %v", err)
		}
		gzw := gzip.NewWriter(writer)
		defer gzw.Close()
		writer = gzw
	// TODO: support flate, other encodings.
	case "":
		// This is fine
	default:
		// Some encoding we don't understand-- don't try to parse this
		glog.Errorf("Proxy encountered encoding %v for text/html; can't understand this so not fixing links.", encoding)
		return resp, nil
	}

	doc, err := html.Parse(reader)
	if err != nil {
		glog.Errorf("Parse failed: %v", err)
		return resp, err
	}

	t.scan(doc, func(n *html.Node) { t.updateURLs(n, req.URL) })
	if err := html.Render(writer, doc); err != nil {
		glog.Errorf("Failed to render: %v", err)
	}

	resp.Body = ioutil.NopCloser(newContent)
	// Update header node with new content-length
	// TODO: Remove any hash/signature headers here?
	resp.Header.Del("Content-Length")
	resp.ContentLength = int64(newContent.Len())

	return resp, err
}
