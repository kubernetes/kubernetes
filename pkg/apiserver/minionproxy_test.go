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
	"bufio"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestMinionTransport(t *testing.T) {
	content := string(`<pre><a href="kubelet.log">kubelet.log</a><a href="google.log">google.log</a></pre>`)
	transport := &minionTransport{}

	// Test /logs/
	request := &http.Request{
		Method: "GET",
		URL: &url.URL{
			Scheme: "http",
			Host:   "minion1:10250",
			Path:   "/logs/",
		},
	}
	response := &http.Response{
		Status:     "200 OK",
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(strings.NewReader(content)),
		Close:      true,
	}
	updated_resp, _ := transport.ProcessResponse(request, response)
	body, _ := ioutil.ReadAll(updated_resp.Body)
	expected := string(`<pre><a href="/proxy/minion/minion1:10250/logs/kubelet.log">kubelet.log</a><a href="/proxy/minion/minion1:10250/logs/google.log">google.log</a></pre>`)
	if !strings.Contains(string(body), expected) {
		t.Errorf("Received wrong content: %s", string(body))
	}

	// Test subdir under /logs/
	request = &http.Request{
		Method: "GET",
		URL: &url.URL{
			Scheme: "http",
			Host:   "minion1:8080",
			Path:   "/whatever/apt/",
		},
	}
	response = &http.Response{
		Status:     "200 OK",
		StatusCode: http.StatusOK,
		Body:       ioutil.NopCloser(strings.NewReader(content)),
		Close:      true,
	}
	updated_resp, _ = transport.ProcessResponse(request, response)
	body, _ = ioutil.ReadAll(updated_resp.Body)
	expected = string(`<pre><a href="/proxy/minion/minion1:8080/whatever/apt/kubelet.log">kubelet.log</a><a href="/proxy/minion/minion1:8080/whatever/apt/google.log">google.log</a></pre>`)
	if !strings.Contains(string(body), expected) {
		t.Errorf("Received wrong content: %s", string(body))
	}
}

func TestMinionProxy(t *testing.T) {
	proxyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Write([]byte(req.URL.Path))
	}))
	server := httptest.NewServer(http.HandlerFunc(handleProxyMinion))
	//client := http.Client{}
	proxy, _ := url.Parse(proxyServer.URL)

	testCases := map[string]string{
		fmt.Sprintf("/%s/", proxy.Host):     "/",
		fmt.Sprintf("/%s/test", proxy.Host): "/test",
	}

	for value, expected := range testCases {
		resp, err := http.Get(fmt.Sprintf("%s%s", server.URL, value))
		if err != nil {
			t.Errorf("unexpected error for %s: %v", value, err)
			continue
		}
		if resp.StatusCode != http.StatusOK {
			t.Errorf("expected successful request for %s: %#v", value, resp)
			continue
		}
		defer resp.Body.Close()
		actual, _ := bufio.NewReader(resp.Body).ReadString('\n')
		if actual != expected {
			t.Errorf("expected %s to become %s, got %s", value, expected, actual)
		}
	}

	failureCases := map[string]string{
		"": "",
		fmt.Sprintf("/%s", proxy.Host): "/",
	}

	for value := range failureCases {
		resp, err := http.Get(fmt.Sprintf("%s%s", server.URL, value))
		if err != nil {
			t.Errorf("unexpected error for %s: %v", value, err)
			continue
		}
		if resp.StatusCode != http.StatusBadGateway {
			t.Errorf("expected bad gateway response for %s: %#v", value, resp)
		}
	}
}

func TestApiServerMinionProxy(t *testing.T) {
	proxyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Write([]byte(req.URL.Path))
	}))
	server := httptest.NewServer(Handle(nil, nil, "/prefix", false))
	proxy, _ := url.Parse(proxyServer.URL)
	resp, err := http.Get(fmt.Sprintf("%s/proxy/minion/%s%s", server.URL, proxy.Host, "/test"))
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected successful request, got %#v", resp)
	}
	defer resp.Body.Close()
	actual, _ := bufio.NewReader(resp.Body).ReadString('\n')
	if actual != "/test" {
		t.Errorf("unexpected response body %s", actual)
	}
}
