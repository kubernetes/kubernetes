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

package mutatingadmissionpolicybinding

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	admissionregistration "k8s.io/kubernetes/pkg/apis/admissionregistration"
	_ "k8s.io/kubernetes/pkg/apis/admissionregistration/install"
	registry "k8s.io/kubernetes/pkg/registry/admissionregistration/mutatingadmissionpolicybinding"
	"k8s.io/kubernetes/pkg/registry/admissionregistration/resolver"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

// Helper function to create a baseline valid MutatingAdmissionPolicyBinding with optional tweaks
func mkMutatingAdmissionPolicyBinding(tweaks ...func(*admissionregistration.MutatingAdmissionPolicyBinding)) admissionregistration.MutatingAdmissionPolicyBinding {
	obj := admissionregistration.MutatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-resource-name",
		},
		Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
			PolicyName: "some-policy",
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := registry.NewStrategy(nil, nil, resolver.ResourceResolverFunc(func(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
				return schema.GroupVersionResource{}, nil
			}))
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "admissionregistration.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "mutatingadmissionpolicybindings",
				Namespace:         namespace,
				IsResourceRequest: true,
				Verb:              "create",
			})
			obj := mkMutatingAdmissionPolicyBinding(func(o *admissionregistration.MutatingAdmissionPolicyBinding) {
				o.Namespace = namespace
			})
			meta.RunObjectMetaTestCases(t, ctx, &obj, strategy)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := registry.NewStrategy(nil, nil, resolver.ResourceResolverFunc(func(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
				return schema.GroupVersionResource{}, nil
			}))
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "admissionregistration.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "mutatingadmissionpolicybindings",
				Namespace:         namespace,
				IsResourceRequest: true,
				Verb:              "update",
			})
			obj := mkMutatingAdmissionPolicyBinding(func(o *admissionregistration.MutatingAdmissionPolicyBinding) {
				o.Namespace = namespace
			})
			meta.RunObjectMetaUpdateTestCases(t, ctx, &obj, strategy)
		})
	}
}
