/*
Copyright 2018 The Kubernetes Authors.

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

package openapi

import (
	"net/http"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	openapiproto "k8s.io/kube-openapi/pkg/util/proto"
)

// TestOpenAPISpecCacher tests a specCacher with a fake downloader and fake parser
// First it gets the spec twice from the same cacher and makes sure that the returned spec is the same.
// to make sure it won't unnecessarily parse the spec every time that someone calls Get()
// Then it changes the value the cacher thinks the Etag is supposed to be to simulate a stale cache
// to makes sure that the cacher parses a new spec
func TestOpenAPISpecCacher(t *testing.T) {
	downloader := fakeDownloader{
		"":  {[]byte(`AAA`), "C", http.StatusOK, nil},
		"B": {[]byte(`AAA`), "C", http.StatusOK, nil},
		"C": {[]byte(``), "C", http.StatusNotModified, nil},
	}
	parser := &fakeParser{}
	specCacher, _ := NewSpecCacher(downloader, parser).(*specCacher)

	firstGet, err := specCacher.Get()
	if err != nil {
		t.Error(err)
	}
	secondGet, err := specCacher.Get()
	if err != nil {
		t.Error(err)
	}
	if firstGet != secondGet {
		t.Errorf("Expected first and second gets from the specCacher to return the same cached value %v", err)
	}

	// Simulate the cache being stale and make sure that the cacher tries to download a new spec
	specCacher.lastEtag = "B"
	thirdGet, err := specCacher.Get()
	if err != nil {
		t.Error(err)
	}
	if firstGet == thirdGet {
		t.Errorf("Expected specCacher to download a and parse a new spec but got %v and %v", firstGet, thirdGet)
	}
}

type fakeDownloader map[string]downloaderResults
type downloaderResults struct {
	specBytes  []byte
	newEtag    string
	httpStatus int
	err        error
}

var _ SpecDownloader = make(fakeDownloader, 0)

// Download implements SpecDownloader
func (f fakeDownloader) Download(lastEtag string) (specBytes []byte, newEtag string, httpStatus int, err error) {
	results, ok := f[lastEtag]
	if !ok {
		// Return value not specified, coder error.
		panic(nil)
	}
	return results.specBytes, results.newEtag, results.httpStatus, results.err
}

type fakeParser struct {
	counter int
}

var _ SpecParser = &fakeParser{}

// Parse implements SpecParser
func (f *fakeParser) Parse(_ []byte) (Resources, error) {
	f.counter = f.counter + 1
	r := fakeResources(f.counter)
	return r, nil
}

type fakeResources int

var _ Resources = fakeResources(0)

// LookupResource implements Resources
func (fakeResources) LookupResource(gvk schema.GroupVersionKind) (openapiproto.Schema, error) {
	panic(nil)
}
