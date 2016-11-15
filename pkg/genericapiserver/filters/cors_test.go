/*
Copyright 2016 The Kubernetes Authors.

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

package filters

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestCORSAllowedOrigins(t *testing.T) {
	table := []struct {
		allowedOrigins []string
		origin         string
		allowed        bool
	}{
		{[]string{}, "example.com", false},
		{[]string{"example.com"}, "example.com", true},
		{[]string{"example.com"}, "not-allowed.com", false},
		{[]string{"not-matching.com", "example.com"}, "example.com", true},
		{[]string{".*"}, "example.com", true},
	}

	for _, item := range table {
		handler := WithCORS(
			http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}),
			item.allowedOrigins, nil, nil, nil, "true",
		)
		server := httptest.NewServer(handler)
		defer server.Close()
		client := http.Client{}

		request, err := http.NewRequest("GET", server.URL+"/version", nil)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		request.Header.Set("Origin", item.origin)

		response, err := client.Do(request)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		if item.allowed {
			if !reflect.DeepEqual(item.origin, response.Header.Get("Access-Control-Allow-Origin")) {
				t.Errorf("Expected %#v, Got %#v", item.origin, response.Header.Get("Access-Control-Allow-Origin"))
			}

			if response.Header.Get("Access-Control-Allow-Credentials") == "" {
				t.Errorf("Expected Access-Control-Allow-Credentials header to be set")
			}

			if response.Header.Get("Access-Control-Allow-Headers") == "" {
				t.Errorf("Expected Access-Control-Allow-Headers header to be set")
			}

			if response.Header.Get("Access-Control-Allow-Methods") == "" {
				t.Errorf("Expected Access-Control-Allow-Methods header to be set")
			}
			if response.Header.Get("Access-Control-Expose-Headers") != "Date" {
				t.Errorf("Expected Date in Access-Control-Expose-Headers header")
			}
		} else {
			if response.Header.Get("Access-Control-Allow-Origin") != "" {
				t.Errorf("Expected Access-Control-Allow-Origin header to not be set")
			}

			if response.Header.Get("Access-Control-Allow-Credentials") != "" {
				t.Errorf("Expected Access-Control-Allow-Credentials header to not be set")
			}

			if response.Header.Get("Access-Control-Allow-Headers") != "" {
				t.Errorf("Expected Access-Control-Allow-Headers header to not be set")
			}

			if response.Header.Get("Access-Control-Allow-Methods") != "" {
				t.Errorf("Expected Access-Control-Allow-Methods header to not be set")
			}
			if response.Header.Get("Access-Control-Expose-Headers") == "Date" {
				t.Errorf("Expected Date in Access-Control-Expose-Headers header")
			}
		}
	}
}
