/*
Copyright 2016 The Kubernetes Authors.

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

package discovery

import (
	"reflect"
	"testing"

	"k8s.io/client-go/1.4/pkg/api/unversioned"
)

func TestRESTMapper(t *testing.T) {
	resources := []*APIGroupResources{
		{
			Group: unversioned.APIGroup{
				Versions: []unversioned.GroupVersionForDiscovery{
					{Version: "v1"},
					{Version: "v2"},
				},
				PreferredVersion: unversioned.GroupVersionForDiscovery{Version: "v1"},
			},
			VersionedResources: map[string][]unversioned.APIResource{
				"v1": {
					{Name: "pods", Namespaced: true, Kind: "Pod"},
				},
				"v2": {
					{Name: "pods", Namespaced: true, Kind: "Pod"},
				},
			},
		},
		{
			Group: unversioned.APIGroup{
				Name: "extensions",
				Versions: []unversioned.GroupVersionForDiscovery{
					{Version: "v1beta"},
				},
				PreferredVersion: unversioned.GroupVersionForDiscovery{Version: "v1beta"},
			},
			VersionedResources: map[string][]unversioned.APIResource{
				"v1beta": {
					{Name: "jobs", Namespaced: true, Kind: "Job"},
				},
			},
		},
	}

	restMapper := NewRESTMapper(resources, nil)

	kindTCs := []struct {
		input unversioned.GroupVersionResource
		want  unversioned.GroupVersionKind
	}{
		{
			input: unversioned.GroupVersionResource{
				Version:  "v1",
				Resource: "pods",
			},
			want: unversioned.GroupVersionKind{
				Version: "v1",
				Kind:    "Pod",
			},
		},
		{
			input: unversioned.GroupVersionResource{
				Version:  "v2",
				Resource: "pods",
			},
			want: unversioned.GroupVersionKind{
				Version: "v2",
				Kind:    "Pod",
			},
		},
		{
			input: unversioned.GroupVersionResource{
				Resource: "pods",
			},
			want: unversioned.GroupVersionKind{
				Version: "v1",
				Kind:    "Pod",
			},
		},
		{
			input: unversioned.GroupVersionResource{
				Resource: "jobs",
			},
			want: unversioned.GroupVersionKind{
				Group:   "extensions",
				Version: "v1beta",
				Kind:    "Job",
			},
		},
	}

	for _, tc := range kindTCs {
		got, err := restMapper.KindFor(tc.input)
		if err != nil {
			t.Errorf("KindFor(%#v) unexpected error: %v", tc.input, err)
			continue
		}

		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("KindFor(%#v) = %#v, want %#v", tc.input, got, tc.want)
		}
	}

	resourceTCs := []struct {
		input unversioned.GroupVersionResource
		want  unversioned.GroupVersionResource
	}{
		{
			input: unversioned.GroupVersionResource{
				Version:  "v1",
				Resource: "pods",
			},
			want: unversioned.GroupVersionResource{
				Version:  "v1",
				Resource: "pods",
			},
		},
		{
			input: unversioned.GroupVersionResource{
				Version:  "v2",
				Resource: "pods",
			},
			want: unversioned.GroupVersionResource{
				Version:  "v2",
				Resource: "pods",
			},
		},
		{
			input: unversioned.GroupVersionResource{
				Resource: "pods",
			},
			want: unversioned.GroupVersionResource{
				Version:  "v1",
				Resource: "pods",
			},
		},
		{
			input: unversioned.GroupVersionResource{
				Resource: "jobs",
			},
			want: unversioned.GroupVersionResource{
				Group:    "extensions",
				Version:  "v1beta",
				Resource: "jobs",
			},
		},
	}

	for _, tc := range resourceTCs {
		got, err := restMapper.ResourceFor(tc.input)
		if err != nil {
			t.Errorf("ResourceFor(%#v) unexpected error: %v", tc.input, err)
			continue
		}

		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("ResourceFor(%#v) = %#v, want %#v", tc.input, got, tc.want)
		}
	}
}
