/*
Copyright The Kubernetes Authors.

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

package routes

import (
	"reflect"
	"testing"

	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

func TestGroupVersionFromPath(t *testing.T) {
	tests := []struct {
		path        string
		wantGroup   string
		wantVersion string
	}{
		{"api/v1", "", "v1"},
		{"apis/apps/v1", "apps", "v1"},
		{"apis/networking.k8s.io/v1", "networking.k8s.io", "v1"},
		{"apis/batch/v1", "batch", "v1"},
		{"api", "", ""},
	}
	for _, tt := range tests {
		group, version := groupVersionFromPath(tt.path)
		if group != tt.wantGroup || version != tt.wantVersion {
			t.Errorf("groupVersionFromPath(%q) = (%q, %q), want (%q, %q)", tt.path, group, version, tt.wantGroup, tt.wantVersion)
		}
	}
}

func gvk(group, version, kind string) map[string]interface{} {
	return map[string]interface{}{"group": group, "version": version, "kind": kind}
}

func TestFilterScopedGVKs(t *testing.T) {
	tests := []struct {
		name     string
		gvks     []interface{}
		group    string
		version  string
		wantGVKs []interface{}
	}{
		{
			name: "cross-registered type keeps local and core/v1",
			gvks: []interface{}{
				gvk("", "v1", "DeleteOptions"),
				gvk("apps", "v1", "DeleteOptions"),
				gvk("batch", "v1", "DeleteOptions"),
				gvk("autoscaling", "v1", "DeleteOptions"),
			},
			group:   "apps",
			version: "v1",
			wantGVKs: []interface{}{
				gvk("", "v1", "DeleteOptions"),
				gvk("apps", "v1", "DeleteOptions"),
			},
		},
		{
			name: "cross-registered type with core group keeps only core/v1",
			gvks: []interface{}{
				gvk("", "v1", "DeleteOptions"),
				gvk("apps", "v1", "DeleteOptions"),
				gvk("batch", "v1", "DeleteOptions"),
				gvk("autoscaling", "v1", "DeleteOptions"),
			},
			group:    "",
			version:  "v1",
			wantGVKs: []interface{}{gvk("", "v1", "DeleteOptions")},
		},
		{
			name: "single GVK unchanged",
			gvks: []interface{}{
				gvk("apps", "v1", "Deployment"),
			},
			group:    "apps",
			version:  "v1",
			wantGVKs: []interface{}{gvk("apps", "v1", "Deployment")},
		},
		{
			name: "multi-version same group unchanged",
			gvks: []interface{}{
				gvk("autoscaling", "v1", "HorizontalPodAutoscaler"),
				gvk("autoscaling", "v2", "HorizontalPodAutoscaler"),
			},
			group:   "autoscaling",
			version: "v2",
			wantGVKs: []interface{}{
				gvk("autoscaling", "v1", "HorizontalPodAutoscaler"),
				gvk("autoscaling", "v2", "HorizontalPodAutoscaler"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &spec3.OpenAPI{
				Components: &spec3.Components{
					Schemas: map[string]*spec.Schema{
						"TestType": {
							VendorExtensible: spec.VendorExtensible{
								Extensions: spec.Extensions{
									"x-kubernetes-group-version-kind": tt.gvks,
								},
							},
						},
					},
				},
			}
			filterScopedGVKs(s, tt.group, tt.version)
			got := s.Components.Schemas["TestType"].Extensions["x-kubernetes-group-version-kind"]
			if !reflect.DeepEqual(got, tt.wantGVKs) {
				t.Errorf("got %v, want %v", got, tt.wantGVKs)
			}
		})
	}
}
