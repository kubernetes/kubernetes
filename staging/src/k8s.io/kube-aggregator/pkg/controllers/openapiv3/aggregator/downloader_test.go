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

package aggregator

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/kube-openapi/pkg/handler3"
)

type handlerTest struct {
	etag string
	data []byte
}

var _ http.Handler = handlerTest{}

func (h handlerTest) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Create an APIService with a handler for one group/version
	if r.URL.Path == "/openapi/v3" {
		group := &handler3.OpenAPIV3Discovery{
			Paths: map[string]handler3.OpenAPIV3DiscoveryGroupVersion{
				"apis/group/version": {
					ServerRelativeURL: "/openapi/v3/apis/group/version?hash=" + h.etag,
				},
			},
		}

		j, _ := json.Marshal(group)
		w.Write(j)
		return
	}

	if r.URL.Path == "/openapi/v3/apis/group/version" {
		if len(h.etag) > 0 {
			w.Header().Add("Etag", h.etag)
		}
		ifNoneMatches := r.Header["If-None-Match"]
		for _, match := range ifNoneMatches {
			if match == h.etag {
				w.WriteHeader(http.StatusNotModified)
				return
			}
		}
		w.Write(h.data)
	}
}

func TestDownloadOpenAPISpec(t *testing.T) {
	s := Downloader{}

	groups, _, err := s.OpenAPIV3Root(
		handlerTest{data: []byte(""), etag: ""})
	assert.NoError(t, err)
	if assert.NotNil(t, groups) {
		assert.Equal(t, len(groups.Paths), 1)
		if assert.Contains(t, groups.Paths, "apis/group/version") {
			assert.NotEmpty(t, groups.Paths["apis/group/version"].ServerRelativeURL)
		}
	}

}
