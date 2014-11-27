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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type fakeHttpGet struct {
	err  error
	resp *http.Response
	url  string
}

func (f *fakeHttpGet) Get(url string) (*http.Response, error) {
	f.url = url
	return f.resp, f.err
}

func makeFake(data string, statusCode int, err error) *fakeHttpGet {
	return &fakeHttpGet{
		err: err,
		resp: &http.Response{
			Body:       ioutil.NopCloser(bytes.NewBufferString(data)),
			StatusCode: statusCode,
		},
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		err            error
		data           string
		expectedStatus health.Status
		code           int
		expectErr      bool
	}{
		{fmt.Errorf("test error"), "", health.Unknown, 500 /*ignored*/, true},
		{nil, "foo", health.Healthy, 200, false},
		{nil, "foo", health.Unhealthy, 500, true},
	}

	s := Server{Addr: "foo.com", Port: 8080, Path: "/healthz"}

	for _, test := range tests {
		fake := makeFake(test.data, test.code, test.err)
		status, data, err := s.check(fake)
		expect := fmt.Sprintf("http://%s:%d/healthz", s.Addr, s.Port)
		if fake.url != expect {
			t.Errorf("expected %s, got %s", expect, fake.url)
		}
		if test.expectErr && err == nil {
			t.Errorf("unexpected non-error")
		}
		if !test.expectErr && err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if data != test.data {
			t.Errorf("expected empty string, got %s", status)
		}
		if status != test.expectedStatus {
			t.Errorf("expected %s, got %s", test.expectedStatus.String(), status.String())
		}
	}
}

func TestValidator(t *testing.T) {
	fake := makeFake("foo", 200, nil)
	validator, err := makeTestValidator(map[string]string{
		"foo": "foo.com:80",
		"bar": "bar.com:8080",
	}, fake)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	testServer := httptest.NewServer(validator)
	defer testServer.Close()

	resp, err := http.Get(testServer.URL + "/validatez")

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("unexpected response: %v", resp.StatusCode)
	}
	defer resp.Body.Close()
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	status := []ServerStatus{}
	err = json.Unmarshal(data, &status)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	components := util.StringSet{}
	for _, s := range status {
		components.Insert(s.Component)
	}
	if len(status) != 2 || !components.Has("foo") || !components.Has("bar") {
		t.Errorf("unexpected status: %#v", status)
	}
}
