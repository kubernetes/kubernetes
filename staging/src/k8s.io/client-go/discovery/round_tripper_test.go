/*
Copyright 2017 The Kubernetes Authors.

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

package discovery

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"testing"
)

// copied from k8s.io/client-go/transport/round_trippers_test.go
type testRoundTripper struct {
	Request  *http.Request
	Response *http.Response
	Err      error
}

func (rt *testRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.Request = req
	return rt.Response, rt.Err
}

func TestCacheRoundTripper(t *testing.T) {
	rt := &testRoundTripper{}
	cacheDir, err := ioutil.TempDir("", "cache-rt")
	defer os.RemoveAll(cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	cache := newCacheRoundTripper(cacheDir, rt)

	// First call, caches the response
	req := &http.Request{
		Method: http.MethodGet,
		URL:    &url.URL{Host: "localhost"},
	}
	rt.Response = &http.Response{
		Header:     http.Header{"ETag": []string{`"123456"`}},
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("Content"))),
		StatusCode: http.StatusOK,
	}
	resp, err := cache.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	content, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "Content" {
		t.Errorf(`Expected Body to be "Content", got %q`, string(content))
	}

	// Second call, returns cached response
	req = &http.Request{
		Method: http.MethodGet,
		URL:    &url.URL{Host: "localhost"},
	}
	rt.Response = &http.Response{
		StatusCode: http.StatusNotModified,
		Body:       ioutil.NopCloser(bytes.NewReader([]byte("Other Content"))),
	}

	resp, err = cache.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}

	// Read body and make sure we have the initial content
	content, err = ioutil.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "Content" {
		t.Errorf("Invalid content read from cache %q", string(content))
	}
}
