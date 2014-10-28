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
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httputil"
	"net/url"
	"path"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/httplog"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"code.google.com/p/go.net/html"
	"github.com/golang/glog"
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
	prefix  string
	storage map[string]RESTStorage
	codec   runtime.Codec
}

func (r *ProxyHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	// use the default namespace to address the service
	ctx := api.NewDefaultContext()
	// if not in default namespace, provide the query parameter
	// TODO this will need to go in the path in the future and not as a query parameter
	namespace := req.URL.Query().Get("namespace")
	if len(namespace) > 0 {
		ctx = api.WithNamespace(ctx, namespace)
	}
	parts := strings.SplitN(req.URL.Path, "/", 3)
	if len(parts) < 2 {
		notFound(w, req)
		return
	}
	resourceName := parts[0]
	id := parts[1]
	rest := ""
	if len(parts) == 3 {
		rest = parts[2]
	}
	storage, ok := r.storage[resourceName]
	if !ok {
		httplog.LogOf(req, w).Addf("'%v' has no storage object", resourceName)
		notFound(w, req)
		return
	}

	redirector, ok := storage.(Redirector)
	if !ok {
		httplog.LogOf(req, w).Addf("'%v' is not a redirector", resourceName)
		notFound(w, req)
		return
	}

	location, err := redirector.ResourceLocation(ctx, id)
	if err != nil {
		httplog.LogOf(req, w).Addf("Error getting ResourceLocation: %v", err)
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w)
		return
	}
	if location == "" {
		httplog.LogOf(req, w).Addf("ResourceLocation for %v returned ''", id)
		notFound(w, req)
		return
	}

	destURL, err := url.Parse(location)
	if err != nil {
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w)
		return
	}
	if destURL.Scheme == "" {
		// If no scheme was present in location, url.Parse sometimes mistakes
		// hosts for paths.
		destURL.Host = location
	}
	destURL.Path = rest
	destURL.RawQuery = req.URL.RawQuery
	newReq, err := http.NewRequest(req.Method, destURL.String(), req.Body)
	if err != nil {
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w)
		notFound(w, req)
		return
	}
	newReq.Header = req.Header

	proxy := httputil.NewSingleHostReverseProxy(&url.URL{Scheme: "http", Host: destURL.Host})
	proxy.Transport = &proxyTransport{
		proxyScheme:      req.URL.Scheme,
		proxyHost:        req.URL.Host,
		proxyPathPrepend: path.Join(r.prefix, resourceName, id),
	}
	proxy.FlushInterval = 200 * time.Millisecond
	proxy.ServeHTTP(w, newReq)
}

type proxyTransport struct {
	proxyScheme      string
	proxyHost        string
	proxyPathPrepend string
}

func (t *proxyTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := http.DefaultTransport.RoundTrip(req)

	if err != nil {
		message := fmt.Sprintf("Error: '%s'\nTrying to reach: '%v'", err.Error(), req.URL.String())
		resp = &http.Response{
			StatusCode: http.StatusServiceUnavailable,
			Body:       ioutil.NopCloser(strings.NewReader(message)),
		}
		return resp, nil
	}

	if resp.Header.Get("Content-Type") != "text/html" {
		// Do nothing, simply pass through
		return resp, nil
	}

	return t.fixLinks(req, resp)
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
		url, err := url.Parse(attr.Val)
		if err != nil {
			continue
		}
		// Is this URL relative?
		if url.Host == "" {
			url.Scheme = t.proxyScheme
			url.Host = t.proxyHost
			url.Path = path.Join(t.proxyPathPrepend, path.Dir(sourceURL.Path), url.Path, "/")
			n.Attr[i].Val = url.String()
		} else if url.Host == sourceURL.Host {
			url.Scheme = t.proxyScheme
			url.Host = t.proxyHost
			url.Path = path.Join(t.proxyPathPrepend, url.Path)
			n.Attr[i].Val = url.String()
		}
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
	defer resp.Body.Close()

	doc, err := html.Parse(resp.Body)
	if err != nil {
		glog.Errorf("Parse failed: %v", err)
		return resp, err
	}

	newContent := &bytes.Buffer{}
	t.scan(doc, func(n *html.Node) { t.updateURLs(n, req.URL) })
	if err := html.Render(newContent, doc); err != nil {
		glog.Errorf("Failed to render: %v", err)
	}

	resp.Body = ioutil.NopCloser(newContent)
	// Update header node with new content-length
	// TODO: Remove any hash/signature headers here?
	resp.Header.Del("Content-Length")
	resp.ContentLength = int64(newContent.Len())

	return resp, err
}
