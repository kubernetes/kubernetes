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

package disk

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"testing"

	"github.com/gregjones/httpcache/diskcache"
	"github.com/peterbourgon/diskv"
	"github.com/stretchr/testify/assert"
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

// NOTE(negz): We're adding a benchmark for an external dependency in order to
// prove that one that will be added in a subsequent commit improves write
// performance.
func BenchmarkDiskCache(b *testing.B) {
	cacheDir, err := ioutil.TempDir("", "cache-rt")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(cacheDir)

	d := diskv.New(diskv.Options{
		PathPerm: os.FileMode(0750),
		FilePerm: os.FileMode(0660),
		BasePath: cacheDir,
		TempDir:  filepath.Join(cacheDir, ".diskv-temp"),
	})

	k := "localhost:8080/apis/batch/v1.json"
	v, err := ioutil.ReadFile("../../testdata/apis/batch/v1.json")
	if err != nil {
		b.Fatal(err)
	}

	c := diskcache.NewWithDiskv(d)

	for n := 0; n < b.N; n++ {
		c.Set(k, v)
		c.Get(k)
		c.Delete(k)
	}
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

func TestCacheRoundTripperPathPerm(t *testing.T) {
	assert := assert.New(t)

	rt := &testRoundTripper{}
	cacheDir, err := ioutil.TempDir("", "cache-rt")
	os.RemoveAll(cacheDir)
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

	err = filepath.Walk(cacheDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			assert.Equal(os.FileMode(0750), info.Mode().Perm())
		} else {
			assert.Equal(os.FileMode(0660), info.Mode().Perm())
		}
		return nil
	})
	assert.NoError(err)
}
