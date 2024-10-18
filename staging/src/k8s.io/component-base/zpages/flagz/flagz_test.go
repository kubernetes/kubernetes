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
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/spf13/pflag"
	cliflag "k8s.io/component-base/cli/flag"
)

const wantTmpl = `
----------------------------
title: %s flagz
description: flags enabled in %s
warning: This endpoint is not meant to be machine parseable and is for debugging purposes only.
----------------------------

`

func TestFlags(t *testing.T) {
	componentName := "test-server"
	wantHeader := fmt.Sprintf(wantTmpl, componentName, componentName)
	tests := []struct {
		name       string
		flagset    *cliflag.NamedFlagSets
		wantResp   string
		wantStatus int
	}{
		{
			name: "common flags",
			flagset: &cliflag.NamedFlagSets{
				FlagSets: map[string]*pflag.FlagSet{
					"commonFlags": flagSet(t, map[string]flagValue{
						"test-flag-bar": {
							value:     "test-value-bar",
							sensitive: false,
						},
						"test-flag-foo": {
							value:     "test-value-foo",
							sensitive: false,
						},
					}),
				},
			},
			wantResp: fmt.Sprintf("%s%s", wantHeader,
				`test-flag-bar=test-value-bar
test-flag-foo=test-value-foo
`),
			wantStatus: http.StatusOK,
		},
		{
			name: "secret flags",
			flagset: &cliflag.NamedFlagSets{
				FlagSets: map[string]*pflag.FlagSet{
					"secretFlags": flagSet(t, map[string]flagValue{
						"test-flag-foo": {
							value:     "test-value-foo",
							sensitive: true,
						},
					}),
				},
			},
			wantResp: fmt.Sprintf("%s%s", wantHeader, `test-flag-foo=CLASSIFIED
`),
			wantStatus: http.StatusOK,
		},
	}

	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mux := http.NewServeMux()
			flagzRegistry = &Registry{
				response: bytes.Buffer{},
				once:     sync.Once{},
			}

			flags := []*cliflag.NamedFlagSets{test.flagset}
			Flagz{}.Install(mux, componentName, flags)
			req, err := http.NewRequest("GET", "http://example.com/flagz", nil)
			if err != nil {
				t.Fatalf("case[%d] Unexpected error: %v", i, err)
			}

			w := httptest.NewRecorder()
			mux.ServeHTTP(w, req)
			if w.Code != test.wantStatus {
				t.Errorf("case[%d] Expected: %v, got: %v", i, test.wantStatus, w.Code)
			}

			c := w.Header().Get("Content-Type")
			if c != "text/plain; charset=utf-8" {
				t.Errorf("case[%d] Expected: %v, got: %v", i, "text/plain", c)
			}

			if w.Body.String() != test.wantResp {
				t.Errorf("case[%d] Expected:\n%v\ngot:\n%v\n", i, test.wantResp, w.Body.String())
			}
		})
	}
}

type flagValue struct {
	value     string
	sensitive bool
}

func flagSet(t *testing.T, flags map[string]flagValue) *pflag.FlagSet {
	fs := pflag.NewFlagSet("test-set", pflag.ContinueOnError)
	for flagName, flagVal := range flags {
		flagValue := ""
		fs.StringVar(&flagValue, flagName, flagVal.value, "test-usage")
		if flagVal.sensitive {
			err := fs.SetAnnotation(flagName, "classified", []string{"true"})
			if err != nil {
				t.Fatalf("unexpected error when setting flag annotation: %v", err)
			}
		}
	}

	return fs
}
