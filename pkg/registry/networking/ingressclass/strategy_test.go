/*
Copyright 2020 The Kubernetes Authors.

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

package ingressclass

import (
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/features"
	utilpointer "k8s.io/utils/pointer"
)

func TestIngressClassStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if Strategy.NamespaceScoped() {
		t.Errorf("IngressClass must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("IngressClass should not allow create on update")
	}

	ingressClass := networking.IngressClass{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
		},
		Spec: networking.IngressClassSpec{
			Controller: "example.com/controller",
		},
	}

	Strategy.PrepareForCreate(ctx, &ingressClass)
	if ingressClass.Generation != 1 {
		t.Error("IngressClass generation should be 1")
	}
	errs := Strategy.Validate(ctx, &ingressClass)
	if len(errs) != 0 {
		t.Errorf("Unexpected error from validation for IngressClass: %v", errs)
	}

	newIngressClass := ingressClass.DeepCopy()
	Strategy.PrepareForUpdate(ctx, newIngressClass, &ingressClass)
	errs = Strategy.ValidateUpdate(ctx, newIngressClass, &ingressClass)
	if len(errs) != 0 {
		t.Errorf("Unexpected error from update validation for IngressClass: %v", errs)
	}

	ingressClass.Name = "invalid/name"

	errs = Strategy.Validate(ctx, &ingressClass)
	if len(errs) == 0 {
		t.Errorf("Expected error from validation for IngressClass, got none")
	}
	errs = Strategy.ValidateUpdate(ctx, &ingressClass, &ingressClass)
	if len(errs) == 0 {
		t.Errorf("Expected error from update validation for IngressClass, got none")
	}
}

func TestIngressClassPrepareForCreate(t *testing.T) {
	tests := []struct {
		name                            string
		original                        *networking.IngressClass
		expected                        *networking.IngressClass
		enableNamespaceScopedParamsGate bool
	}{
		{
			name: "cluster scope is removed when feature is not enabled",
			original: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind:  "k",
						Name:  "n",
						Scope: utilpointer.StringPtr(networking.IngressClassParametersReferenceScopeCluster),
					},
				},
			},
			expected: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind: "k",
						Name: "n",
					},
				},
			},
			enableNamespaceScopedParamsGate: false,
		},
		{
			name: "namespace scope and namespace fields are removed when feature is not enabled",
			original: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind:      "k",
						Name:      "n",
						Scope:     utilpointer.StringPtr(networking.IngressClassParametersReferenceScopeNamespace),
						Namespace: utilpointer.StringPtr("foo-ns"),
					},
				},
			},
			expected: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind: "k",
						Name: "n",
					},
				},
			},
			enableNamespaceScopedParamsGate: false,
		},
		{
			name: "cluster scope is not removed when feature is enabled",
			original: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind:  "k",
						Name:  "n",
						Scope: utilpointer.StringPtr(networking.IngressClassParametersReferenceScopeCluster),
					},
				},
			},
			expected: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind:  "k",
						Name:  "n",
						Scope: utilpointer.StringPtr(networking.IngressClassParametersReferenceScopeCluster),
					},
				},
			},
			enableNamespaceScopedParamsGate: true,
		},
		{
			name: "namespace scope and namespace fields are not removed when feature is enabled",
			original: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind:      "k",
						Name:      "n",
						Scope:     utilpointer.StringPtr(networking.IngressClassParametersReferenceScopeNamespace),
						Namespace: utilpointer.StringPtr("foo-ns"),
					},
				},
			},
			expected: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind:      "k",
						Name:      "n",
						Scope:     utilpointer.StringPtr(networking.IngressClassParametersReferenceScopeNamespace),
						Namespace: utilpointer.StringPtr("foo-ns"),
					},
				},
			},
			enableNamespaceScopedParamsGate: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			original := test.original
			expected := test.expected
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IngressClassNamespacedParams, test.enableNamespaceScopedParamsGate)()
			ctx := genericapirequest.NewDefaultContext()
			Strategy.PrepareForCreate(ctx, runtime.Object(original))
			if !apiequality.Semantic.DeepEqual(original.Spec, expected.Spec) {
				t.Errorf("got different than expected\ngot:\n\t%+v\nexpected:\n\t%+v", original.Spec, expected.Spec)
			}
		})
	}
}

func TestIngressClassPrepareForUpdate(t *testing.T) {
	tests := []struct {
		name                            string
		newIngressClass                 *networking.IngressClass
		oldIngressClass                 *networking.IngressClass
		expected                        *networking.IngressClass
		enableNamespaceScopedParamsGate bool
	}{
		{
			name: "scope can be updated if already set when feature is disabled",
			newIngressClass: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind:      "k",
						Name:      "n",
						Scope:     utilpointer.StringPtr(networking.IngressClassParametersReferenceScopeNamespace),
						Namespace: utilpointer.StringPtr("foo-ns"),
					},
				},
			},
			oldIngressClass: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind:  "k",
						Name:  "n",
						Scope: utilpointer.StringPtr(networking.IngressClassParametersReferenceScopeCluster),
					},
				},
			},
			expected: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind:      "k",
						Name:      "n",
						Scope:     utilpointer.StringPtr(networking.IngressClassParametersReferenceScopeNamespace),
						Namespace: utilpointer.StringPtr("foo-ns"),
					},
				},
			},
			enableNamespaceScopedParamsGate: false,
		},
		{
			name: "scope is removed if not already set previously when feature is disabled",
			newIngressClass: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind:      "k",
						Name:      "n",
						Scope:     utilpointer.StringPtr(networking.IngressClassParametersReferenceScopeNamespace),
						Namespace: utilpointer.StringPtr("foo-ns"),
					},
				},
			},
			oldIngressClass: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind: "k",
						Name: "n",
					},
				},
			},
			expected: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind: "k",
						Name: "n",
					},
				},
			},
			enableNamespaceScopedParamsGate: false,
		},
		{
			name: "scope can be set when feature is enabled",
			newIngressClass: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind:      "k",
						Name:      "n",
						Scope:     utilpointer.StringPtr(networking.IngressClassParametersReferenceScopeNamespace),
						Namespace: utilpointer.StringPtr("foo-ns"),
					},
				},
			},
			oldIngressClass: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind: "k",
						Name: "n",
					},
				},
			},
			expected: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind:      "k",
						Name:      "n",
						Scope:     utilpointer.StringPtr(networking.IngressClassParametersReferenceScopeNamespace),
						Namespace: utilpointer.StringPtr("foo-ns"),
					},
				},
			},
			enableNamespaceScopedParamsGate: true,
		},
		{
			name: "scope can be removed when feature is enabled",
			newIngressClass: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind: "k",
						Name: "n",
					},
				},
			},
			oldIngressClass: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind:      "k",
						Name:      "n",
						Scope:     utilpointer.StringPtr(networking.IngressClassParametersReferenceScopeNamespace),
						Namespace: utilpointer.StringPtr("foo-ns"),
					},
				},
			},
			expected: &networking.IngressClass{
				Spec: networking.IngressClassSpec{
					Controller: "controller",
					Parameters: &networking.IngressClassParametersReference{
						Kind: "k",
						Name: "n",
					},
				},
			},
			enableNamespaceScopedParamsGate: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IngressClassNamespacedParams, test.enableNamespaceScopedParamsGate)()
			ctx := genericapirequest.NewDefaultContext()
			Strategy.PrepareForUpdate(ctx, runtime.Object(test.newIngressClass), runtime.Object(test.oldIngressClass))
			if !apiequality.Semantic.DeepEqual(test.newIngressClass.Spec, test.expected.Spec) {
				t.Errorf("got different than expected\ngot:\n\t%+v\nexpected:\n\t%+v", test.newIngressClass.Spec, test.expected.Spec)
			}
		})
	}
}
