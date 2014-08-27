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
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestProxyTransport_fixLinks(t *testing.T) {
	content := string(`<pre><a href="kubelet.log">kubelet.log</a><a href="/google.log">google.log</a></pre>`)
	transport := &proxyTransport{
		proxyScheme:      "http",
		proxyHost:        "foo.com",
		proxyPathPrepend: "/proxy/minion/minion1:10250/",
	}

	// Test /logs/
	request := &http.Request{
		Method: "GET",
		URL: &url.URL{
			Path: "/logs/",
		},
	}
	response := &http.Response{
		Status:     "200 OK",
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(strings.NewReader(content)),
		Close:      true,
	}
	updated_resp, _ := transport.fixLinks(request, response)
	body, _ := ioutil.ReadAll(updated_resp.Body)
	expected := string(`<pre><a href="http://foo.com/proxy/minion/minion1:10250/logs/kubelet.log">kubelet.log</a><a href="http://foo.com/proxy/minion/minion1:10250/logs/google.log">google.log</a></pre>`)
	if !strings.Contains(string(body), expected) {
		t.Errorf("Received wrong content: %s", string(body))
	}

	// Test subdir under /logs/
	request = &http.Request{
		Method: "GET",
		URL: &url.URL{
			Path: "/whatever/apt/somelog.log",
		},
	}
	transport.proxyScheme = "https"
	transport.proxyPathPrepend = "/proxy/minion/minion1:8080/"
	response = &http.Response{
		Status:     "200 OK",
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(strings.NewReader(content)),
		Close:      true,
	}
	updated_resp, _ = transport.fixLinks(request, response)
	body, _ = ioutil.ReadAll(updated_resp.Body)
	expected = string(`<pre><a href="https://foo.com/proxy/minion/minion1:8080/whatever/apt/kubelet.log">kubelet.log</a><a href="https://foo.com/proxy/minion/minion1:8080/whatever/apt/google.log">google.log</a></pre>`)
	if !strings.Contains(string(body), expected) {
		t.Errorf("Received wrong content: %s", string(body))
	}
}

func TestProxy(t *testing.T) {
	table := []struct {
		method   string
		path     string
		reqBody  string
		respBody string
	}{
		{"GET", "/some/dir", "", "answer"},
		{"POST", "/some/other/dir", "question", "answer"},
		{"PUT", "/some/dir/id", "different question", "answer"},
		{"DELETE", "/some/dir/id", "", "ok"},
	}

	for _, item := range table {
		proxyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			gotBody, err := ioutil.ReadAll(req.Body)
			if err != nil {
				t.Errorf("%v - unexpected error %v", item.method, err)
			}
			if e, a := item.reqBody, string(gotBody); e != a {
				t.Errorf("%v - expected %v, got %v", item.method, e, a)
			}
			fmt.Fprint(w, item.respBody)
		}))

		simpleStorage := &SimpleRESTStorage{
			errors:           map[string]error{},
			resourceLocation: proxyServer.URL,
		}
		handler := Handle(map[string]RESTStorage{
			"foo": simpleStorage,
		}, codec, "/prefix/version")
		server := httptest.NewServer(handler)

		req, err := http.NewRequest(
			item.method,
			server.URL+"/prefix/version/proxy/foo/id"+item.path,
			strings.NewReader(item.reqBody),
		)
		if err != nil {
			t.Errorf("%v - unexpected error %v", item.method, err)
			continue
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Errorf("%v - unexpected error %v", item.method, err)
			continue
		}
		gotResp, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("%v - unexpected error %v", item.method, err)
		}
		resp.Body.Close()
		if e, a := item.respBody, string(gotResp); e != a {
			t.Errorf("%v - expected %v, got %v", item.method, e, a)
		}
	}
}
