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

package runtime

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestResourceMapper(t *testing.T) {
	gvr := func(g, v, r string) schema.GroupVersionResource {
		return schema.GroupVersionResource{Group: g, Version: v, Resource: r}
	}

	gvk := func(g, v, k string) schema.GroupVersionKind {
		return schema.GroupVersionKind{Group: g, Version: v, Kind: k}
	}

	kindsToRegister := []struct {
		gvr         schema.GroupVersionResource
		subresource string
		gvk         schema.GroupVersionKind
	}{
		// pods
		{gvr("", "v1", "pods"), "", gvk("", "v1", "Pod")},
		// pods/status
		{gvr("", "v1", "pods"), "status", gvk("", "v1", "Pod")},
		// deployments
		{gvr("apps", "v1", "deployments"), "", gvk("apps", "v1", "Deployment")},
		{gvr("apps", "v1beta1", "deployments"), "", gvk("apps", "v1beta1", "Deployment")},
		{gvr("apps", "v1alpha1", "deployments"), "", gvk("apps", "v1alpha1", "Deployment")},
		{gvr("extensions", "v1beta1", "deployments"), "", gvk("extensions", "v1beta1", "Deployment")},
		// deployments/scale (omitted for apps/v1alpha1)
		{gvr("apps", "v1", "deployments"), "scale", gvk("", "", "Scale")},
		{gvr("apps", "v1beta1", "deployments"), "scale", gvk("", "", "Scale")},
		{gvr("extensions", "v1beta1", "deployments"), "scale", gvk("", "", "Scale")},
		// deployments/status (omitted for apps/v1alpha1)
		{gvr("apps", "v1", "deployments"), "status", gvk("apps", "v1", "Deployment")},
		{gvr("apps", "v1beta1", "deployments"), "status", gvk("apps", "v1beta1", "Deployment")},
		{gvr("extensions", "v1beta1", "deployments"), "status", gvk("extensions", "v1beta1", "Deployment")},
	}

	testcases := []struct {
		Name                           string
		IdentityFunc                   func(schema.GroupResource) string
		ResourcesForV1Deployment       []schema.GroupVersionResource
		ResourcesForV1DeploymentScale  []schema.GroupVersionResource
		ResourcesForV1DeploymentStatus []schema.GroupVersionResource
	}{
		{
			Name:                           "no identityfunc",
			ResourcesForV1Deployment:       []schema.GroupVersionResource{gvr("apps", "v1", "deployments"), gvr("apps", "v1beta1", "deployments"), gvr("apps", "v1alpha1", "deployments")},
			ResourcesForV1DeploymentScale:  []schema.GroupVersionResource{gvr("apps", "v1", "deployments"), gvr("apps", "v1beta1", "deployments")},
			ResourcesForV1DeploymentStatus: []schema.GroupVersionResource{gvr("apps", "v1", "deployments"), gvr("apps", "v1beta1", "deployments")},
		},
		{
			Name:         "empty identityfunc",
			IdentityFunc: func(schema.GroupResource) string { return "" },
			// same group
			ResourcesForV1Deployment:       []schema.GroupVersionResource{gvr("apps", "v1", "deployments"), gvr("apps", "v1beta1", "deployments"), gvr("apps", "v1alpha1", "deployments")},
			ResourcesForV1DeploymentScale:  []schema.GroupVersionResource{gvr("apps", "v1", "deployments"), gvr("apps", "v1beta1", "deployments")},
			ResourcesForV1DeploymentStatus: []schema.GroupVersionResource{gvr("apps", "v1", "deployments"), gvr("apps", "v1beta1", "deployments")},
		},
		{
			Name:         "common identityfunc",
			IdentityFunc: func(schema.GroupResource) string { return "x" },
			// all resources are seen as equivalent
			ResourcesForV1Deployment: []schema.GroupVersionResource{gvr("", "v1", "pods"), gvr("apps", "v1", "deployments"), gvr("apps", "v1beta1", "deployments"), gvr("apps", "v1alpha1", "deployments"), gvr("extensions", "v1beta1", "deployments")},
			// all resources with scale are seen as equivalent
			ResourcesForV1DeploymentScale: []schema.GroupVersionResource{gvr("apps", "v1", "deployments"), gvr("apps", "v1beta1", "deployments"), gvr("extensions", "v1beta1", "deployments")},
			// all resources with status are seen as equivalent
			ResourcesForV1DeploymentStatus: []schema.GroupVersionResource{gvr("", "v1", "pods"), gvr("apps", "v1", "deployments"), gvr("apps", "v1beta1", "deployments"), gvr("extensions", "v1beta1", "deployments")},
		},
		{
			Name: "colocated deployments",
			IdentityFunc: func(resource schema.GroupResource) string {
				if resource.Resource == "deployments" {
					return "deployments"
				}
				return ""
			},
			// all deployments are seen as equivalent
			ResourcesForV1Deployment: []schema.GroupVersionResource{gvr("apps", "v1", "deployments"), gvr("apps", "v1beta1", "deployments"), gvr("apps", "v1alpha1", "deployments"), gvr("extensions", "v1beta1", "deployments")},
			// all deployments with scale are seen as equivalent
			ResourcesForV1DeploymentScale: []schema.GroupVersionResource{gvr("apps", "v1", "deployments"), gvr("apps", "v1beta1", "deployments"), gvr("extensions", "v1beta1", "deployments")},
			// all deployments with status are seen as equivalent
			ResourcesForV1DeploymentStatus: []schema.GroupVersionResource{gvr("apps", "v1", "deployments"), gvr("apps", "v1beta1", "deployments"), gvr("extensions", "v1beta1", "deployments")},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			mapper := NewEquivalentResourceRegistryWithIdentity(tc.IdentityFunc)

			// register
			for _, data := range kindsToRegister {
				mapper.RegisterKindFor(data.gvr, data.subresource, data.gvk)
			}
			// verify
			for _, data := range kindsToRegister {
				if kind := mapper.KindFor(data.gvr, data.subresource); kind != data.gvk {
					t.Errorf("KindFor(%#v, %v) returned %#v, expected %#v", data.gvr, data.subresource, kind, data.gvk)
				}
			}

			// Verify equivalents to primary resource
			if resources := mapper.EquivalentResourcesFor(gvr("apps", "v1", "deployments"), ""); !reflect.DeepEqual(resources, tc.ResourcesForV1Deployment) {
				t.Errorf("diff:\n%s", cmp.Diff(tc.ResourcesForV1Deployment, resources))
			}
			// Verify equivalents to subresources
			if resources := mapper.EquivalentResourcesFor(gvr("apps", "v1", "deployments"), "scale"); !reflect.DeepEqual(resources, tc.ResourcesForV1DeploymentScale) {
				t.Errorf("diff:\n%s", cmp.Diff(tc.ResourcesForV1DeploymentScale, resources))
			}
			if resources := mapper.EquivalentResourcesFor(gvr("apps", "v1", "deployments"), "status"); !reflect.DeepEqual(resources, tc.ResourcesForV1DeploymentStatus) {
				t.Errorf("diff:\n%s", cmp.Diff(tc.ResourcesForV1DeploymentStatus, resources))
			}
		})
	}
}
