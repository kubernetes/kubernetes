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

func TestWithMuxAndDiscoveryCompleteProtection(t *testing.T) {
	scenarios := []struct {
		name                             string
		muxAndDiscoveryCompleteSignal    <-chan struct{}
		expectNoMuxAndDiscoIncompleteKey bool
	}{
		{
			name:                             "no signals, no key in the ctx",
			expectNoMuxAndDiscoIncompleteKey: true,
		},
		{
			name:                             "signal ready, no key in the ctx",
			muxAndDiscoveryCompleteSignal:    func() chan struct{} { ch := make(chan struct{}); close(ch); return ch }(),
			expectNoMuxAndDiscoIncompleteKey: true,
		},
		{
			name:                          "signal not ready, the key in the ctx",
			muxAndDiscoveryCompleteSignal: make(chan struct{}),
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			// setup
			var actualContext context.Context
			delegate := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				actualContext = req.Context()
			})
			target := WithMuxAndDiscoveryComplete(delegate, scenario.muxAndDiscoveryCompleteSignal)

			// act
			req := &http.Request{}
			target.ServeHTTP(httptest.NewRecorder(), req)

			// validate
			if scenario.expectNoMuxAndDiscoIncompleteKey != NoMuxAndDiscoveryIncompleteKey(actualContext) {
				t.Fatalf("expectNoMuxAndDiscoIncompleteKey in the context = %v, does the actual context contain the key = %v", scenario.expectNoMuxAndDiscoIncompleteKey, NoMuxAndDiscoveryIncompleteKey(actualContext))
			}
		})
	}
}
