/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"testing"

	"k8s.io/kubernetes/pkg/probe"
)

type fakeRoundTripper struct {
	err  error
	resp *http.Response
	url  string
}

func (f *fakeRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	f.url = req.URL.String()
	return f.resp, f.err
}

func alwaysError([]byte) error { return errors.New("test error") }

func matchError(data []byte) error {
	if string(data) == "bar" {
		return errors.New("match error")
	}
	return nil
}

func TestValidate(t *testing.T) {
	tests := []struct {
		err            error
		data           string
		expectedStatus probe.Result
		code           int
		expectErr      bool
		validator      ValidatorFn
	}{
		{fmt.Errorf("test error"), "", probe.Unknown, 500 /*ignored*/, true, nil},
		{nil, "foo", probe.Success, 200, false, nil},
		{nil, "foo", probe.Failure, 500, true, nil},
		{nil, "foo", probe.Failure, 200, true, alwaysError},
		{nil, "foo", probe.Success, 200, false, matchError},
	}

	s := Server{Addr: "foo.com", Port: 8080, Path: "/healthz"}

	for _, test := range tests {
		fakeRT := &fakeRoundTripper{
			err: test.err,
			resp: &http.Response{
				Body:       ioutil.NopCloser(bytes.NewBufferString(test.data)),
				StatusCode: test.code,
			},
		}
		s.Validate = test.validator
		status, data, err := s.DoServerCheck(fakeRT)
		expect := fmt.Sprintf("http://%s:%d/healthz", s.Addr, s.Port)
		if fakeRT.url != expect {
			t.Errorf("expected %s, got %s", expect, fakeRT.url)
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
