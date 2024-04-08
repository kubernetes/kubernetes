/*
Copyright 2024 The Kubernetes Authors.

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

package flagz

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/spf13/pflag"
	cliflag "k8s.io/component-base/cli/flag"
)

func TestInstallHandler(t *testing.T) {
	mux := http.NewServeMux()
	InstallHandler(mux)
	req, err := http.NewRequest("GET", "http://example.com/flagz", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("expected %v, got %v", http.StatusOK, w.Code)
	}
	c := w.Header().Get("Content-Type")
	if c != "text/plain; charset=utf-8" {
		t.Errorf("expected %v, got %v", "text/plain", c)
	}
	if w.Body.String() != "map[]" {
		t.Errorf("expected %v, got %v", "ok", w.Body.String())
	}
}

func TestFlags(t *testing.T) {
	fs1 := pflag.NewFlagSet("test1-set", pflag.ContinueOnError)
	flagValue1 := ""
	fs1.StringVar(&flagValue1, "test1-flag", "test1-value", "test1-usage")

	fs2 := pflag.NewFlagSet("test2-set", pflag.ContinueOnError)
	flagValue2 := ""
	fs2.StringVar(&flagValue2, "test2-flag", "test2-value", "test2-usage")

	flagset := cliflag.NamedFlagSets{
		FlagSets: map[string]*pflag.FlagSet{
			"test-1": fs1,
			"test-2": fs2},
	}

	tests := []struct {
		path             string
		expectedResponse string
		expectedStatus   int
		addBadCheck      bool
	}{
		{"?verbose", fmt.Sprintf("%s\n%s", "test1-flag=test1-value", "test2-flag=test2-value\n"), http.StatusOK, false},
	}

	for i, test := range tests {
		mux := http.NewServeMux()
		flags := []cliflag.NamedFlagSets{flagset}

		InstallHandler(mux, flags...)
		path := "/flagz"

		req, err := http.NewRequest("GET", fmt.Sprintf("http://example.com%s%v", path, test.path), nil)
		if err != nil {
			t.Fatalf("case[%d] Unexpected error: %v", i, err)
		}
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)
		if w.Code != test.expectedStatus {
			t.Errorf("case[%d] Expected: %v, got: %v", i, test.expectedStatus, w.Code)
		}
		c := w.Header().Get("Content-Type")
		if c != "text/plain; charset=utf-8" {
			t.Errorf("case[%d] Expected: %v, got: %v", i, "text/plain", c)
		}
		if w.Body.String() != test.expectedResponse {
			t.Errorf("case[%d] Expected:\n%v\ngot:\n%v\n", i, test.expectedResponse, w.Body.String())
		}
	}
}
