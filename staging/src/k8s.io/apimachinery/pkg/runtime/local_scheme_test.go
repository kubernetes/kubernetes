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

package runtime

import (
	"testing"

	"reflect"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestPreferredVersionsAllGroups(t *testing.T) {
	tests := []struct {
		name                string
		versionPriority     map[string][]string
		observedVersions    []schema.GroupVersion
		expectedPrioritized map[string][]schema.GroupVersion
		expectedPreferred   map[schema.GroupVersion]bool
	}{
		{
			name: "observedOnly",
			observedVersions: []schema.GroupVersion{
				{Group: "", Version: "v3"},
				{Group: "foo", Version: "v1"},
				{Group: "foo", Version: "v2"},
				{Group: "", Version: "v1"},
			},
			expectedPrioritized: map[string][]schema.GroupVersion{
				"": {
					{Group: "", Version: "v3"},
					{Group: "", Version: "v1"},
				},
				"foo": {
					{Group: "foo", Version: "v1"},
					{Group: "foo", Version: "v2"},
				},
			},
			expectedPreferred: map[schema.GroupVersion]bool{
				{Group: "", Version: "v3"}:    true,
				{Group: "foo", Version: "v1"}: true,
			},
		},
		{
			name: "specifiedOnly",
			versionPriority: map[string][]string{
				"":    {"v3", "v1"},
				"foo": {"v1", "v2"},
			},
			expectedPrioritized: map[string][]schema.GroupVersion{
				"": {
					{Group: "", Version: "v3"},
					{Group: "", Version: "v1"},
				},
				"foo": {
					{Group: "foo", Version: "v1"},
					{Group: "foo", Version: "v2"},
				},
			},
			expectedPreferred: map[schema.GroupVersion]bool{
				{Group: "", Version: "v3"}:    true,
				{Group: "foo", Version: "v1"}: true,
			},
		},
		{
			name: "both",
			versionPriority: map[string][]string{
				"":    {"v3", "v1"},
				"foo": {"v1", "v2"},
			},
			observedVersions: []schema.GroupVersion{
				{Group: "", Version: "v1"},
				{Group: "", Version: "v3"},
				{Group: "", Version: "v4"},
				{Group: "", Version: "v5"},
				{Group: "bar", Version: "v1"},
				{Group: "bar", Version: "v2"},
			},
			expectedPrioritized: map[string][]schema.GroupVersion{
				"": {
					{Group: "", Version: "v3"},
					{Group: "", Version: "v1"},
					{Group: "", Version: "v4"},
					{Group: "", Version: "v5"},
				},
				"foo": {
					{Group: "foo", Version: "v1"},
					{Group: "foo", Version: "v2"},
				},
				"bar": {
					{Group: "bar", Version: "v1"},
					{Group: "bar", Version: "v2"},
				},
			},
			expectedPreferred: map[schema.GroupVersion]bool{
				{Group: "", Version: "v3"}:    true,
				{Group: "foo", Version: "v1"}: true,
				{Group: "bar", Version: "v1"}: true,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scheme := NewScheme()
			scheme.versionPriority = test.versionPriority
			scheme.observedVersions = test.observedVersions

			for group, expected := range test.expectedPrioritized {
				actual := scheme.PrioritizedVersionsForGroup(group)
				if !reflect.DeepEqual(expected, actual) {
					t.Error(cmp.Diff(expected, actual))
				}
			}

			prioritizedAll := scheme.PrioritizedVersionsAllGroups()
			actualPrioritizedAll := map[string][]schema.GroupVersion{}
			for _, actual := range prioritizedAll {
				actualPrioritizedAll[actual.Group] = append(actualPrioritizedAll[actual.Group], actual)
			}
			if !reflect.DeepEqual(test.expectedPrioritized, actualPrioritizedAll) {
				t.Error(cmp.Diff(test.expectedPrioritized, actualPrioritizedAll))
			}

			preferredAll := scheme.PreferredVersionAllGroups()
			actualPreferredAll := map[schema.GroupVersion]bool{}
			for _, actual := range preferredAll {
				actualPreferredAll[actual] = true
			}
			if !reflect.DeepEqual(test.expectedPreferred, actualPreferredAll) {
				t.Error(cmp.Diff(test.expectedPreferred, actualPreferredAll))
			}
		})
	}
}
