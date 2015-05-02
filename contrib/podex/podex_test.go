/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package main

import "testing"

func TestSplitDockerImageName(t *testing.T) {
	tests := []struct {
		name              string
		expectedHost      string
		expectedNamespace string
		expectedRepo      string
		expectedTag       string
	}{
		{
			name:         "foo",
			expectedRepo: "foo",
		},
		{
			name:         "foo:bar",
			expectedRepo: "foo",
			expectedTag:  "bar",
		},
		{
			name:              "me/foo",
			expectedNamespace: "me",
			expectedRepo:      "foo",
		},
		{
			name:              "me/foo:bar",
			expectedNamespace: "me",
			expectedRepo:      "foo",
			expectedTag:       "bar",
		},
		{
			name:              "example.com/me/foo",
			expectedHost:      "example.com",
			expectedNamespace: "me",
			expectedRepo:      "foo",
		},
		{
			name:              "example.com/me/foo:bar",
			expectedHost:      "example.com",
			expectedNamespace: "me",
			expectedRepo:      "foo",
			expectedTag:       "bar",
		},
		{
			name:              "localhost:8080/me/foo",
			expectedHost:      "localhost:8080",
			expectedNamespace: "me",
			expectedRepo:      "foo",
		},
		{
			name:              "localhost:8080/me/foo:bar",
			expectedHost:      "localhost:8080",
			expectedNamespace: "me",
			expectedRepo:      "foo",
			expectedTag:       "bar",
		},
	}
	for _, test := range tests {
		host, namespace, repo, tag := splitDockerImageName(test.name)
		if host != test.expectedHost {
			t.Errorf("expected host %q got %q", test.expectedHost, host)
		}
		if namespace != test.expectedNamespace {
			t.Errorf("expected namespace %q got %q", test.expectedNamespace, namespace)
		}
		if repo != test.expectedRepo {
			t.Errorf("expected repo %q got %q", test.expectedRepo, repo)
		}
		if tag != test.expectedTag {
			t.Errorf("expected tag %q got %q", test.expectedTag, tag)
		}
	}
}
