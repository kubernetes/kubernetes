/*
Copyright 2019 The Kubernetes Authors.

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
	"net/http"
	"testing"
)

func TestCleanVerb(t *testing.T) {
	testCases := []struct {
		initialVerb  string
		expectedVerb string
	}{
		{
			"",
			"unknown",
		},
		{
			"WATCHLIST",
			"WATCH",
		},
		{
			"notValid",
			"unknown",
		},
	}
	for _, tt := range testCases {
		t.Run(tt.initialVerb, func(t *testing.T) {
			cleansedVerb := cleanVerb(tt.initialVerb, &http.Request{})
			if cleansedVerb != tt.expectedVerb {
				t.Errorf("Got %s, but expected %s", cleansedVerb, tt.expectedVerb)
			}
		})
	}
}

func TestContentType(t *testing.T) {
	testCases := []struct {
		rawContentType      string
		expectedContentType string
	}{
		{
			"application/json",
			"application/json",
		},
		{
			"",
			"other",
		},
		{
			"notValid",
			"other",
		},
	}
	for _, tt := range testCases {
		t.Run(tt.rawContentType, func(t *testing.T) {
			cleansedContentType := cleanContentType(tt.rawContentType)
			if cleansedContentType != tt.expectedContentType {
				t.Errorf("Got %s, but expected %s", cleansedContentType, tt.expectedContentType)
			}
		})
	}
}
