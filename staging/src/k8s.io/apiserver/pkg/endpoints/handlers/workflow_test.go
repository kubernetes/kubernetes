/*
Copyright 2018 The Kubernetes Authors.

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

package handlers

import (
	"fmt"
	"testing"
)

func TestWorkflowID(t *testing.T) {
	tests := []struct {
		subResource string
		verb        string
		contentType string
		userAgent   string
		wanted      string
	}{
		{
			verb:   "PUT",
			wanted: "default-put",
		},
		{
			subResource: "scale",
			verb:        "PUT",
			userAgent:   "kubectl/v1.10.4 (linux/amd64) kubernetes/5ca598b",
			wanted:      "kubectl-scale-put",
		},
		{
			verb:        "PATCH",
			contentType: "application/json-patch+json",
			userAgent:   "Random stuff",
			wanted:      "random-patch",
		},
		{
			verb:        "PATCH",
			contentType: "application/json-patch+json",
			userAgent:   "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.62 Safari/537.36",
			wanted:      "mozilla-patch",
		},
		{
			verb:        "PATCH",
			contentType: "application/apply-patch+yaml",
			userAgent:   "kubectl/v1.10.4 (linux/amd64) kubernetes/5ca598b",
			wanted:      "kubectl",
		},
		{
			verb:        "POST",
			contentType: "application/json",
			userAgent:   "kubectl/v1.10.4 (linux/amd64) kubernetes/5ca598b",
			wanted:      "kubectl",
		},
		{
			verb:        "PATCH",
			contentType: "application/apply-patch+yaml",
			subResource: "scale",
			wanted:      "default-scale",
		},
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf("guessWorkflowID(%q, %q, %q, %q)", test.userAgent, test.subResource, test.verb, test.contentType),
			func(t *testing.T) {
				got := buildDefaultWorkflowID(
					test.subResource,
					test.verb,
					test.contentType,
					test.userAgent,
				)
				if got != test.wanted {
					t.Errorf("Got %v, wanted %v", got, test.wanted)
				}
			},
		)
	}
}
