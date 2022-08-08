/*
Copyright 2022 The Kubernetes Authors.

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
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/apiserver/pkg/endpoints/request"
)

func TestWithCompressionDisabled(t *testing.T) {
	tests := []struct {
		name              string
		checkDecision     bool
		checkErr          error
		want              bool
		wantHandlerCalled bool
	}{
		{
			name:              "decision=true",
			checkDecision:     true,
			want:              true,
			wantHandlerCalled: true,
		},
		{
			name:              "decision=false",
			checkDecision:     false,
			want:              false,
			wantHandlerCalled: true,
		},
		{
			name:              "check error",
			checkErr:          errors.New("check failed"),
			want:              false,
			wantHandlerCalled: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handlerCalled := false
			handler := http.HandlerFunc(func(_ http.ResponseWriter, req *http.Request) {
				handlerCalled = true
				if got, want := request.CompressionDisabledFrom(req.Context()), tt.checkDecision; got != want {
					t.Errorf("request.CompressionDisabledFrom(req.Context())=%v; want=%v", got, want)
				}
			})
			fake := func(*http.Request) (bool, error) {
				return tt.checkDecision, tt.checkErr
			}
			wrapped := WithCompressionDisabled(handler, fake)
			testRequest, err := http.NewRequest(http.MethodGet, "/path", nil)
			if err != nil {
				t.Fatal(err)
			}
			w := httptest.NewRecorder()
			wrapped.ServeHTTP(w, testRequest)
			if handlerCalled != tt.wantHandlerCalled {
				t.Errorf("expected handlerCalled=%v, got=%v", handlerCalled, tt.wantHandlerCalled)
			}
		})
	}
}
