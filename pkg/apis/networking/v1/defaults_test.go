/*
Copyright 2017 The Kubernetes Authors.

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

package v1_test

import (
	"reflect"
	"testing"

	networkingv1 "k8s.io/api/networking/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	_ "k8s.io/kubernetes/pkg/apis/networking/install"
	. "k8s.io/kubernetes/pkg/apis/networking/v1"
	utilpointer "k8s.io/utils/pointer"
)

func TestSetDefaultNetworkPolicy(t *testing.T) {
	tests := []struct {
		original *networkingv1.NetworkPolicy
		expected *networkingv1.NetworkPolicy
	}{
		{ // Empty NetworkPolicy should be set to PolicyTypes Ingress
			original: &networkingv1.NetworkPolicy{
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
				},
			},
			expected: &networkingv1.NetworkPolicy{
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress},
				},
			},
		},
		{ // Empty Ingress NetworkPolicy should be set to PolicyTypes Ingress
			original: &networkingv1.NetworkPolicy{
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Ingress: []networkingv1.NetworkPolicyIngressRule{},
				},
			},
			expected: &networkingv1.NetworkPolicy{
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Ingress:     []networkingv1.NetworkPolicyIngressRule{},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress},
				},
			},
		},
		{ // Defined Ingress and Egress should be set to Ingress,Egress
			original: &networkingv1.NetworkPolicy{
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Ingress: []networkingv1.NetworkPolicyIngressRule{
						{
							From: []networkingv1.NetworkPolicyPeer{
								{
									PodSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{"c": "d"},
									},
									NamespaceSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{"c": "d"},
									},
								},
							},
						},
					},
					Egress: []networkingv1.NetworkPolicyEgressRule{
						{
							To: []networkingv1.NetworkPolicyPeer{
								{
									NamespaceSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{"c": "d"},
									},
								},
							},
						},
					},
				},
			},
			expected: &networkingv1.NetworkPolicy{
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Ingress: []networkingv1.NetworkPolicyIngressRule{
						{
							From: []networkingv1.NetworkPolicyPeer{
								{
									PodSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{"c": "d"},
									},
									NamespaceSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{"c": "d"},
									},
								},
							},
						},
					},
					Egress: []networkingv1.NetworkPolicyEgressRule{
						{
							To: []networkingv1.NetworkPolicyPeer{
								{
									NamespaceSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{"c": "d"},
									},
								},
							},
						},
					},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress, networkingv1.PolicyTypeEgress},
				},
			},
		},
		{ // Egress only with unset PolicyTypes should be set to Ingress, Egress
			original: &networkingv1.NetworkPolicy{
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Egress: []networkingv1.NetworkPolicyEgressRule{
						{
							To: []networkingv1.NetworkPolicyPeer{
								{
									NamespaceSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{"c": "d"},
									},
								},
							},
						},
					},
				},
			},
			expected: &networkingv1.NetworkPolicy{
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Egress: []networkingv1.NetworkPolicyEgressRule{
						{
							To: []networkingv1.NetworkPolicyPeer{
								{
									NamespaceSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{"c": "d"},
									},
								},
							},
						},
					},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress, networkingv1.PolicyTypeEgress},
				},
			},
		},
		{ // Egress only with PolicyTypes set to Egress should be set to only Egress
			original: &networkingv1.NetworkPolicy{
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Egress: []networkingv1.NetworkPolicyEgressRule{
						{
							To: []networkingv1.NetworkPolicyPeer{
								{
									NamespaceSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{"Egress": "only"},
									},
								},
							},
						},
					},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeEgress},
				},
			},
			expected: &networkingv1.NetworkPolicy{
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Egress: []networkingv1.NetworkPolicyEgressRule{
						{
							To: []networkingv1.NetworkPolicyPeer{
								{
									NamespaceSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{"Egress": "only"},
									},
								},
							},
						},
					},
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeEgress},
				},
			},
		},
	}

	for i, test := range tests {
		original := test.original
		expected := test.expected
		obj2 := roundTrip(t, runtime.Object(original))
		got, ok := obj2.(*networkingv1.NetworkPolicy)
		if !ok {
			t.Errorf("(%d) unexpected object: %v", i, got)
			t.FailNow()
		}
		if !apiequality.Semantic.DeepEqual(got.Spec, expected.Spec) {
			t.Errorf("(%d) got different than expected\ngot:\n\t%+v\nexpected:\n\t%+v", i, got.Spec, expected.Spec)
		}
	}
}

func TestSetDefaultsForIngressClassParametersReference(t *testing.T) {
	tests := []struct {
		name     string
		original *networkingv1.IngressClass
		expected *networkingv1.IngressClass
	}{
		{
			name: "populated parameters sets the default Scope",
			original: &networkingv1.IngressClass{
				Spec: networkingv1.IngressClassSpec{
					Controller: "controller",
					Parameters: &networkingv1.IngressClassParametersReference{
						Kind: "k",
						Name: "n",
					},
				},
			},
			expected: &networkingv1.IngressClass{
				Spec: networkingv1.IngressClassSpec{
					Controller: "controller",
					Parameters: &networkingv1.IngressClassParametersReference{
						Kind:  "k",
						Name:  "n",
						Scope: utilpointer.String(networkingv1.IngressClassParametersReferenceScopeCluster),
					},
				},
			},
		},
		{
			name: "existing scope is not overridden",
			original: &networkingv1.IngressClass{
				Spec: networkingv1.IngressClassSpec{
					Controller: "controller",
					Parameters: &networkingv1.IngressClassParametersReference{
						Kind:      "k",
						Name:      "n",
						Scope:     utilpointer.String(networkingv1.IngressClassParametersReferenceScopeNamespace),
						Namespace: utilpointer.String("foo-ns"),
					},
				},
			},
			expected: &networkingv1.IngressClass{
				Spec: networkingv1.IngressClassSpec{
					Controller: "controller",
					Parameters: &networkingv1.IngressClassParametersReference{
						Kind:      "k",
						Name:      "n",
						Scope:     utilpointer.String(networkingv1.IngressClassParametersReferenceScopeNamespace),
						Namespace: utilpointer.String("foo-ns"),
					},
				},
			},
		},
		{
			name: "empty Parameters does not set the default Scope",
			original: &networkingv1.IngressClass{
				Spec: networkingv1.IngressClassSpec{
					Controller: "controller",
				},
			},
			expected: &networkingv1.IngressClass{
				Spec: networkingv1.IngressClassSpec{
					Controller: "controller",
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			original := test.original
			expected := test.expected
			obj2 := roundTrip(t, runtime.Object(original))
			got, ok := obj2.(*networkingv1.IngressClass)
			if !ok {
				t.Errorf("unexpected object: %v", got)
				t.FailNow()
			}
			if !apiequality.Semantic.DeepEqual(got.Spec, expected.Spec) {
				t.Errorf("got different than expected\ngot:\n\t%+v\nexpected:\n\t%+v", got.Spec, expected.Spec)
			}
		})
	}
}

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	t.Helper()
	data, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(SchemeGroupVersion), obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := runtime.Decode(legacyscheme.Codecs.UniversalDecoder(), data)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
		return nil
	}
	obj3 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = legacyscheme.Scheme.Convert(obj2, obj3, nil)
	if err != nil {
		t.Errorf("%v\nSource: %#v", err, obj2)
		return nil
	}
	return obj3
}
