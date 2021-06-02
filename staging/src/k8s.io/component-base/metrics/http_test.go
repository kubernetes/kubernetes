/*
Copyright 2020 The Kubernetes Authors.

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

package metrics

import (
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
)

func TestResetHandler(t *testing.T) {
	currentVersion := apimachineryversion.Info{
		Major:      "1",
		Minor:      "17",
		GitVersion: "v1.17.1-alpha-1.12345",
	}
	registry := newKubeRegistry(currentVersion)
	resetHandler := HandlerWithReset(registry, HandlerOpts{})
	testCases := []struct {
		desc         string
		method       string
		expectedBody string
	}{
		{
			desc:         "Should return empty body on a get",
			method:       http.MethodGet,
			expectedBody: "",
		},
		{
			desc:         "Should return 'metrics reset' in the body on a delete",
			method:       http.MethodDelete,
			expectedBody: "metrics reset\n",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			req, err := http.NewRequest(tc.method, "http://sample.com/metrics", nil)
			if err != nil {
				t.Fatalf("Error creating http request")
			}
			rec := httptest.NewRecorder()
			resetHandler.ServeHTTP(rec, req)
			body, err := ioutil.ReadAll(rec.Result().Body)
			if err != nil {
				t.Fatalf("Error reading response body")
			}
			if string(body) != tc.expectedBody {
				t.Errorf("Got '%s' as the response body, but want '%v'", body, tc.expectedBody)
			}
		})
	}
}
