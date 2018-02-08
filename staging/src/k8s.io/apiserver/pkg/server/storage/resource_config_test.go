/*
Copyright 2015 The Kubernetes Authors.

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

package storage

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestDisabledVersion(t *testing.T) {
	g1v1 := schema.GroupVersion{Group: "group1", Version: "version1"}
	g1v2 := schema.GroupVersion{Group: "group1", Version: "version2"}
	g2v1 := schema.GroupVersion{Group: "group2", Version: "version1"}

	config := NewResourceConfig()

	config.DisableVersions(g1v1)
	config.EnableVersions(g1v2, g2v1)

	if config.VersionEnabled(g1v1) {
		t.Errorf("expected disabled for %v, from %v", g1v1, config)
	}
	if !config.VersionEnabled(g1v2) {
		t.Errorf("expected enabled for %v, from %v", g1v1, config)
	}
	if !config.VersionEnabled(g2v1) {
		t.Errorf("expected enabled for %v, from %v", g1v1, config)
	}
}

func TestAnyVersionForGroupEnabled(t *testing.T) {
	tests := []struct {
		name      string
		creator   func() APIResourceConfigSource
		testGroup string

		expectedResult bool
	}{
		{
			name: "empty",
			creator: func() APIResourceConfigSource {
				return NewResourceConfig()
			},
			testGroup: "one",

			expectedResult: false,
		},
		{
			name: "present, but disabled",
			creator: func() APIResourceConfigSource {
				ret := NewResourceConfig()
				ret.DisableVersions(schema.GroupVersion{Group: "one", Version: "version1"})
				return ret
			},
			testGroup: "one",

			expectedResult: false,
		},
		{
			name: "present, and one version enabled",
			creator: func() APIResourceConfigSource {
				ret := NewResourceConfig()
				ret.DisableVersions(schema.GroupVersion{Group: "one", Version: "version1"})
				ret.EnableVersions(schema.GroupVersion{Group: "one", Version: "version2"})
				return ret
			},
			testGroup: "one",

			expectedResult: true,
		},
	}

	for _, tc := range tests {
		if e, a := tc.expectedResult, tc.creator().AnyVersionForGroupEnabled(tc.testGroup); e != a {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
	}
}
