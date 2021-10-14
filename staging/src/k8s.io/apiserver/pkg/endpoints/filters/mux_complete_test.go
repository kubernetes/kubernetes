/*
Copyright 2021 The Kubernetes Authors.

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
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestWithMuxCompleteProtectionFilter(t *testing.T) {
	scenarios := []struct {
		name                           string
		muxCompleteSignal              <-chan struct{}
		expectMuxCompleteProtectionKey bool
	}{
		{
			name: "no signals, no protection key in the ctx",
		},
		{
			name:              "signal ready, no protection key in the ctx",
			muxCompleteSignal: func() chan struct{} { ch := make(chan struct{}); close(ch); return ch }(),
		},
		{
			name:                           "signal not ready, the protection key in the ctx",
			muxCompleteSignal:              make(chan struct{}),
			expectMuxCompleteProtectionKey: true,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			// setup
			var actualContext context.Context
			delegate := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				actualContext = req.Context()
			})
			target := WithMuxCompleteProtection(delegate, scenario.muxCompleteSignal)

			// act
			req := &http.Request{}
			target.ServeHTTP(httptest.NewRecorder(), req)

			// validate
			if scenario.expectMuxCompleteProtectionKey != HasMuxCompleteProtectionKey(actualContext) {
				t.Fatalf("expectMuxCompleteProtectionKey in the context = %v, does the actual context contain the key = %v", scenario.expectMuxCompleteProtectionKey, HasMuxCompleteProtectionKey(actualContext))
			}
		})
	}
}
