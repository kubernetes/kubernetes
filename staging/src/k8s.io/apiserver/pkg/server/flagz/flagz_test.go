/*
Copyright The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/pflag"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1alpha1 "k8s.io/apiserver/pkg/server/flagz/api/v1alpha1"
	cliflag "k8s.io/component-base/cli/flag"
)

const wantTmpl = `
%s flagz
Warning: This endpoint is not meant to be machine parseable, has no formatting compatibility guarantees and is for debugging purposes only.
`

func TestHandleFlagz(t *testing.T) {
	fakeFlagName := "test-flag"
	fakeFlagValue := "test-value"
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	fs.String(fakeFlagName, fakeFlagValue, "usage")
	fakeReader := NamedFlagSetsReader{
		FlagSets: cliflag.NamedFlagSets{
			FlagSets: map[string]*pflag.FlagSet{
				"test": fs,
			},
		},
	}

	tests := []struct {
		name           string
		acceptHeader   string
		componentName  string
		registry       *registry
		wantStatusCode int
		wantBody       string
		wantJSONBody   *v1alpha1.Flagz
		wantWarning    bool
	}{
		{
			name:          "valid request for text/plain",
			acceptHeader:  "text/plain",
			componentName: "test-server",
			registry: &registry{
				reader:                fakeReader,
				deprecatedVersionsMap: map[string]bool{},
			},
			wantStatusCode: http.StatusOK,
			wantBody: fmt.Sprintf(
				wantTmpl,
				"test-server",
			),
		},
		{
			name:          "valid request for v1alpha1",
			acceptHeader:  "application/json;v=v1alpha1;g=config.k8s.io;as=Flagz",
			componentName: "test-server",
			registry: &registry{
				reader:                fakeReader,
				deprecatedVersionsMap: map[string]bool{},
			},
			wantStatusCode: http.StatusOK,
			wantJSONBody: &v1alpha1.Flagz{
				TypeMeta: metav1.TypeMeta{
					Kind:       Kind,
					APIVersion: fmt.Sprintf("%s/%s", GroupName, Version),
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-server",
				},
				Flags: map[string]string{
					fakeFlagName: fakeFlagValue,
				},
			},
		},
		{
			name:          "deprecated version request",
			acceptHeader:  "application/json;v=v1alpha1;g=config.k8s.io;as=Flagz",
			componentName: "test-server",
			registry: &registry{
				reader:                fakeReader,
				deprecatedVersionsMap: map[string]bool{"v1alpha1": true},
			},
			wantStatusCode: http.StatusOK,
			wantJSONBody: &v1alpha1.Flagz{
				TypeMeta: metav1.TypeMeta{
					Kind:       Kind,
					APIVersion: fmt.Sprintf("%s/%s", GroupName, Version),
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-server",
				},
				Flags: map[string]string{
					fakeFlagName: fakeFlagValue,
				},
			},
			wantWarning: true,
		},
		{
			name:          "no accept header falls back to text/plain",
			acceptHeader:  "",
			componentName: "test-server",
			registry: &registry{
				reader:                fakeReader,
				deprecatedVersionsMap: map[string]bool{},
			},
			wantStatusCode: http.StatusOK,
			wantBody: fmt.Sprintf(
				wantTmpl,
				"test-server",
			),
		},
		{
			name:          "wildcard accept header falls back to text/plain",
			acceptHeader:  "*/*",
			componentName: "test-server",
			registry: &registry{
				reader:                fakeReader,
				deprecatedVersionsMap: map[string]bool{},
			},
			wantStatusCode: http.StatusOK,
			wantBody: fmt.Sprintf(
				wantTmpl,
				"test-server",
			),
		},
		{
			name:          "bad json header falls back to text/plain with wildcard",
			acceptHeader:  "application/json;v=foo;g=config.k8s.io;as=Flagz,*/*",
			componentName: "test-server",
			registry: &registry{
				reader:                fakeReader,
				deprecatedVersionsMap: map[string]bool{},
			},
			wantStatusCode: http.StatusOK,
			wantBody: fmt.Sprintf(
				wantTmpl,
				"test-server",
			),
		},
		{
			name:          "unsupported accept header",
			acceptHeader:  "application/xml",
			componentName: "test-server",
			registry: &registry{
				reader:                fakeReader,
				deprecatedVersionsMap: map[string]bool{},
			},
			wantStatusCode: http.StatusNotAcceptable,
		},
		{
			name:          "unsupported application/json without params",
			acceptHeader:  "application/json",
			componentName: "test-server",
			registry: &registry{
				reader:                fakeReader,
				deprecatedVersionsMap: map[string]bool{},
			},
			wantStatusCode: http.StatusNotAcceptable,
		},
		{
			name:          "unsupported application/json with missing params",
			acceptHeader:  "application/json;v=v1alpha1;g=config.k8s.io",
			componentName: "test-server",
			registry: &registry{
				reader:                fakeReader,
				deprecatedVersionsMap: map[string]bool{},
			},
			wantStatusCode: http.StatusNotAcceptable,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mux := http.NewServeMux()
			var opts []Option
			var capturedReg *registry
			opts = append(opts, func(reg *registry) {
				capturedReg = reg
				if tt.registry != nil {
					reg.deprecatedVersionsMap = tt.registry.deprecatedVersionsMap
				}
			})
			Install(mux, tt.componentName, fakeReader, opts...)
			// Assign the captured registry to tt.registry for consistency with existing test logic
			tt.registry = capturedReg

			path := "/flagz"
			req, err := http.NewRequest(http.MethodGet, fmt.Sprintf("http://example.com%s", path), nil)
			if err != nil {
				t.Fatalf("unexpected error while creating request: %v", err)
			}
			if tt.acceptHeader != "" {
				req.Header.Set("Accept", tt.acceptHeader)
			}

			w := httptest.NewRecorder()
			mux.ServeHTTP(w, req)

			if w.Code != tt.wantStatusCode {
				t.Fatalf("want status code: %v, got: %v", tt.wantStatusCode, w.Code)
			}

			if tt.wantStatusCode == http.StatusOK {
				if tt.wantJSONBody != nil {
					var got v1alpha1.Flagz
					if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
						t.Fatalf("unexpected error while unmarshalling response: %v", err)
					}
					if diff := cmp.Diff(*tt.wantJSONBody, got); diff != "" {
						t.Errorf("Unexpected diff on response (-want,+got):\n%s", diff)
					}
					if tt.wantWarning {
						if !strings.Contains(w.Header().Get("Warning"), "deprecated") {
							t.Errorf("expected deprecation warning in header, but got: %s", w.Header().Get("Warning"))
						}
					}
				} else if !strings.Contains(string(w.Body.String()), tt.wantBody) {
					t.Errorf("Unexpected response body:\n- want: %s\n- got:  %s", tt.wantBody, string(w.Body.String()))
				}
			}
		})
	}
}

func TestCache(t *testing.T) {
	fakeFlagName := "test-flag"
	fakeFlagValue := "test-value"
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	fs.String(fakeFlagName, fakeFlagValue, "usage")
	fakeReader := NamedFlagSetsReader{
		FlagSets: cliflag.NamedFlagSets{
			FlagSets: map[string]*pflag.FlagSet{
				"test": fs,
			},
		},
	}
	mux := http.NewServeMux()
	var capturedReg *registry
	Install(mux, "test-server", fakeReader, func(reg *registry) {
		capturedReg = reg
	})

	path := "/flagz"
	req, err := http.NewRequest(http.MethodGet, fmt.Sprintf("http://example.com%s", path), nil)
	if err != nil {
		t.Fatalf("unexpected error while creating request: %v", err)
	}
	req.Header.Set("Accept", "text/plain")

	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("want status code: %v, got: %v", http.StatusOK, w.Code)
	}
	if capturedReg.cachedPlainTextResponse == nil {
		t.Fatalf("cached response should not be nil")
	}
	cached := capturedReg.cachedPlainTextResponse
	fs.String("new-flag", "new-value", "usage")

	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("want status code: %v, got: %v", http.StatusOK, w.Code)
	}
	if diff := cmp.Diff(cached, capturedReg.cachedPlainTextResponse); diff != "" {
		t.Errorf("Unexpected diff on cached response (-want,+got):\n%s", diff)
	}
}
