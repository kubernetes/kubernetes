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
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

func TestRedirect(t *testing.T) {
	simpleStorage := &SimpleRESTStorage{
		errors: map[string]error{},
	}
	handler := Handle(map[string]RESTStorage{
		"foo": simpleStorage,
	}, codec, "/prefix/version")
	server := httptest.NewServer(handler)

	dontFollow := errors.New("don't follow")
	client := http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return dontFollow
		},
	}

	table := []struct {
		id   string
		err  error
		code int
	}{
		{"cozy", nil, http.StatusTemporaryRedirect},
		{"horse", errors.New("no such id"), http.StatusInternalServerError},
	}

	for _, item := range table {
		simpleStorage.errors["resourceLocation"] = item.err
		resp, err := client.Get(server.URL + "/prefix/version/redirect/foo/" + item.id)
		if resp == nil {
			t.Fatalf("Unexpected nil resp")
		}
		resp.Body.Close()
		if e, a := item.code, resp.StatusCode; e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
		if e, a := item.id, simpleStorage.requestedResourceLocationID; e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
		if item.err != nil {
			continue
		}
		if err == nil || err.(*url.Error).Err != dontFollow {
			t.Errorf("Unexpected err %#v", err)
		}
		if e, a := item.id, resp.Header.Get("Location"); e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
	}
}
