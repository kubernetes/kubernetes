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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/httplog"

	"code.google.com/p/go.net/html"
	"code.google.com/p/go.net/html/atom"
	"github.com/golang/glog"
)

type ProxyHandler struct {
	prefix  string
	storage map[string]RESTStorage
	codec   Codec
}

func (r *ProxyHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
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
		httplog.LogOf(w).Addf("'%v' has no storage object", resourceName)
		notFound(w, req)
		return
	}

	redirector, ok := storage.(Redirector)
	if !ok {
		httplog.LogOf(w).Addf("'%v' is not a redirector", resourceName)
		notFound(w, req)
		return
	}

	location, err := redirector.ResourceLocation(id)
	if err != nil {
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w)
		return
	}

	destURL, err := url.Parse(location)
	if err != nil {
		status := errToAPIStatus(err)
		writeJSON(status.Code, r.codec, status, w)
		return
	}
	destURL.Path = rest
	destURL.RawQuery = req.URL.RawQuery
	newReq, err := http.NewRequest(req.Method, destURL.String(), req.Body)
	if err != nil {
		glog.Errorf("Failed to create request: %s", err)
	}
	newReq.Header = req.Header

	proxy := httputil.NewSingleHostReverseProxy(&url.URL{Scheme: "http", Host: destURL.Host})
	proxy.Transport = &proxyTransport{
		proxyScheme:      req.URL.Scheme,
		proxyHost:        req.URL.Host,
		proxyPathPrepend: path.Join(r.prefix, resourceName, id),
	}
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
		message := err.Error() + "\n" + req.URL.String()
		if strings.Contains(err.Error(), "connection refused") {
			message = fmt.Sprintf("Failed to connect to %s (%s)", req.URL.Host, err)
		}
		resp = &http.Response{
			StatusCode: http.StatusServiceUnavailable,
			Body:       ioutil.NopCloser(strings.NewReader(message)),
		}
		return resp, nil
	}

	if resp.Header.Get("Content-Type") != "text/html" {
		// Do nothing, simply pass through
		return resp, err
	}

	resp, err = t.fixLinks(req, resp)
	return resp, err
}

// fixLinks modifies links in an HTML file such that they will be redirected through the proxy if needed.
func (t *proxyTransport) fixLinks(req *http.Request, resp *http.Response) (*http.Response, error) {
	body, err := ioutil.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		// copying the response body did not work
		return nil, err
	}

	bodyNode := &html.Node{
		Type:     html.ElementNode,
		Data:     "body",
		DataAtom: atom.Body,
	}
	nodes, err := html.ParseFragment(bytes.NewBuffer(body), bodyNode)
	if err != nil {
		glog.Errorf("Failed to found <body> node: %v", err)
		return resp, err
	}

	// Define the method to traverse the doc tree and update href node to
	// point to correct minion
	var updateHRef func(*html.Node)
	updateHRef = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "a" {
			for i, attr := range n.Attr {
				if attr.Key == "href" {
					url, err := url.Parse(attr.Val)
					if err != nil {
						continue
					}
					url.Scheme = t.proxyScheme
					url.Host = t.proxyHost
					url.Path = path.Join(t.proxyPathPrepend, path.Dir(req.URL.Path), url.Path, "/")
					n.Attr[i].Val = url.String()
					break
				}
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			updateHRef(c)
		}
	}

	newContent := &bytes.Buffer{}
	for _, n := range nodes {
		updateHRef(n)
		err = html.Render(newContent, n)
		if err != nil {
			glog.Errorf("Failed to render: %v", err)
		}
	}

	resp.Body = ioutil.NopCloser(newContent)
	// Update header node with new content-length
	// TODO: Remove any hash/signature headers here?
	resp.Header.Del("Content-Length")
	resp.ContentLength = int64(newContent.Len())

	return resp, err
}
