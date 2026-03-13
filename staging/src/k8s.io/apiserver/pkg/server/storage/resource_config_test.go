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

	if config.versionEnabled(g1v1) {
		t.Errorf("expected disabled for %v, from %v", g1v1, config)
	}
	if !config.versionEnabled(g1v2) {
		t.Errorf("expected enabled for %v, from %v", g1v1, config)
	}
	if !config.versionEnabled(g2v1) {
		t.Errorf("expected enabled for %v, from %v", g1v1, config)
	}
}

func TestDisabledResource(t *testing.T) {
	g1v1 := schema.GroupVersion{Group: "group1", Version: "version1"}
	g1v1rUnspecified := g1v1.WithResource("unspecified")
	g1v1rEnabled := g1v1.WithResource("enabled")
	g1v1rDisabled := g1v1.WithResource("disabled")
	g1v2 := schema.GroupVersion{Group: "group1", Version: "version2"}
	g1v2rUnspecified := g1v2.WithResource("unspecified")
	g1v2rEnabled := g1v2.WithResource("enabled")
	g1v2rDisabled := g1v2.WithResource("disabled")
	g2v1 := schema.GroupVersion{Group: "group2", Version: "version1"}
	g2v1rUnspecified := g2v1.WithResource("unspecified")
	g2v1rEnabled := g2v1.WithResource("enabled")
	g2v1rDisabled := g2v1.WithResource("disabled")

	config := NewResourceConfig()

	config.DisableVersions(g1v1)
	config.EnableVersions(g1v2, g2v1)

	config.EnableResources(g1v1rEnabled, g1v2rEnabled, g2v1rEnabled)
	config.DisableResources(g1v1rDisabled, g1v2rDisabled, g2v1rDisabled)

	// all resources not explicitly enabled under g1v1 are disabled because the group-version is disabled
	if config.ResourceEnabled(g1v1rUnspecified) {
		t.Errorf("expected disabled for %v, from %v", g1v1rUnspecified, config)
	}
	if !config.ResourceEnabled(g1v1rEnabled) {
		t.Errorf("expected enabled for %v, from %v", g1v1rEnabled, config)
	}
	if config.ResourceEnabled(g1v1rDisabled) {
		t.Errorf("expected disabled for %v, from %v", g1v1rDisabled, config)
	}
	if config.ResourceEnabled(g1v1rUnspecified) {
		t.Errorf("expected disabled for %v, from %v", g1v1rUnspecified, config)
	}

	// explicitly disabled resources in enabled group-versions are disabled
	if config.ResourceEnabled(g1v2rDisabled) {
		t.Errorf("expected disabled for %v, from %v", g1v2rDisabled, config)
	}
	if config.ResourceEnabled(g2v1rDisabled) {
		t.Errorf("expected disabled for %v, from %v", g2v1rDisabled, config)
	}

	// unspecified and explicitly enabled resources in enabled group-versions are enabled
	if !config.ResourceEnabled(g1v2rUnspecified) {
		t.Errorf("expected enabled for %v, from %v", g1v2rUnspecified, config)
	}
	if !config.ResourceEnabled(g1v2rEnabled) {
		t.Errorf("expected enabled for %v, from %v", g1v2rEnabled, config)
	}
	if !config.ResourceEnabled(g2v1rUnspecified) {
		t.Errorf("expected enabled for %v, from %v", g2v1rUnspecified, config)
	}
	if !config.ResourceEnabled(g2v1rEnabled) {
		t.Errorf("expected enabled for %v, from %v", g2v1rEnabled, config)
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
		{
			name: "present, and one resource enabled",
			creator: func() APIResourceConfigSource {
				ret := NewResourceConfig()
				ret.DisableVersions(schema.GroupVersion{Group: "one", Version: "version1"})
				ret.EnableResources(schema.GroupVersionResource{Group: "one", Version: "version2", Resource: "foo"})
				return ret
			},
			testGroup: "one",

			expectedResult: true,
		},
		{
			name: "present, and one resource under disabled version enabled",
			creator: func() APIResourceConfigSource {
				ret := NewResourceConfig()
				ret.DisableVersions(schema.GroupVersion{Group: "one", Version: "version1"})
				ret.EnableResources(schema.GroupVersionResource{Group: "one", Version: "version1", Resource: "foo"})
				return ret
			},
			testGroup: "one",

			expectedResult: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if e, a := tc.expectedResult, tc.creator().AnyResourceForGroupEnabled(tc.testGroup); e != a {
				t.Errorf("expected %v, got %v", e, a)
			}
		})
	}
}
