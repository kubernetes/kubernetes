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

package create

import (
	"encoding/json"
	"testing"

	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func TestCreateNetworkPolicyValidation(t *testing.T) {
	tests := map[string]struct {
		podSelector  string
		policyTypes  []string
		ingressRules []string
		egressRules  []string
		expected     string
	}{
		"invalid pod separator": {
			podSelector: "app: nginx",
			expected: `couldn't parse the selector string "app: nginx": unable to parse requirement: <nil>: Invalid value: "app:": ` +
				`name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character ` +
				`(e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')`,
		},
		"invalid network policy type": {
			policyTypes: []string{
				"TODO",
			},
			expected: `invalid policy-types value (TODO). Must be "none", "ingress", or "egress"`,
		},
		"invalid ingress rule": {
			ingressRules: []string{
				"ports=53",
			},
			expected: `ingress rule (ports=53) is invalid and should be in format rule=key:value`,
		},
		"invalid ingress rule name": {
			ingressRules: []string{
				"port=udp:53",
			},
			expected: `invalid ingress rule name (port)`,
		},
		"invalid egress rule": {
			egressRules: []string{
				"ports=udp=53",
			},
			expected: `egress rule (ports=udp=53) is invalid and should be in format rule=key:value`,
		},
		"invalid egress rule name": {
			egressRules: []string{
				"port=udp:53",
			},
			expected: `invalid egress rule name (port)`,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			o := &CreateNetworkPolicyOptions{
				PodSelector:  tc.podSelector,
				PolicyTypes:  tc.policyTypes,
				IngressRules: tc.ingressRules,
				EgressRules:  tc.egressRules,
			}

			err := o.Validate()
			if err != nil && err.Error() != tc.expected {
				t.Errorf("unexpected error: %v", err)
			}
			if tc.expected != "" && err == nil {
				t.Errorf("expected error, got no error")
			}
		})
	}
}

func TestCreateNetworkPolicy(t *testing.T) {
	objectMeta := metav1.ObjectMeta{
		Name: "test-ingress",
	}
	typeMeta := metav1.TypeMeta{
		APIVersion: networkingv1.SchemeGroupVersion.String(),
		Kind:       "NetworkPolicy",
	}
	tests := map[string]struct {
		podSelector  string
		policyTypes  []string
		ingressRules []string
		egressRules  []string
		expected     *networkingv1.NetworkPolicy
	}{
		"prevents all ingress AND egress traffic by creating the following NetworkPolicy": {
			policyTypes: []string{"ingress", "egress"},
			expected: &networkingv1.NetworkPolicy{
				TypeMeta:   typeMeta,
				ObjectMeta: objectMeta,
				Spec: networkingv1.NetworkPolicySpec{
					PolicyTypes: []networkingv1.PolicyType{
						networkingv1.PolicyTypeIngress,
						networkingv1.PolicyTypeEgress,
					},
				},
			},
		},
		"assign an ingress to the policy-types": {
			policyTypes: []string{"ingress"},
			expected: &networkingv1.NetworkPolicy{
				TypeMeta:   typeMeta,
				ObjectMeta: objectMeta,
				Spec: networkingv1.NetworkPolicySpec{
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress},
				},
			},
		},
		"assign an egress to the policy-types": {
			policyTypes: []string{"egress"},
			expected: &networkingv1.NetworkPolicy{
				TypeMeta:   typeMeta,
				ObjectMeta: objectMeta,
				Spec: networkingv1.NetworkPolicySpec{
					PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeEgress},
				},
			},
		},
		"assign an none to the policy-types": {
			policyTypes: []string{"none"},
			expected: &networkingv1.NetworkPolicy{
				TypeMeta:   typeMeta,
				ObjectMeta: objectMeta,
				Spec:       networkingv1.NetworkPolicySpec{},
			},
		},
		"assign match labels to a pod-selector": {
			podSelector: "app=nginx",
			expected: &networkingv1.NetworkPolicy{
				TypeMeta:   typeMeta,
				ObjectMeta: objectMeta,
				Spec: networkingv1.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"app": "nginx"},
					},
				},
			},
		},
		"assign ingress rule": {
			ingressRules: []string{"ports=udp:53,tcp:53,pod=app:nginx,namespace=kubernetes.io/metadata.name:default"},
			expected: &networkingv1.NetworkPolicy{
				TypeMeta:   typeMeta,
				ObjectMeta: objectMeta,
				Spec: networkingv1.NetworkPolicySpec{
					Ingress: []networkingv1.NetworkPolicyIngressRule{
						networkingv1.NetworkPolicyIngressRule{
							Ports: []networkingv1.NetworkPolicyPort{
								networkingv1.NetworkPolicyPort{
									Protocol: func() *v1.Protocol { i := v1.ProtocolUDP; return &i }(),
									Port:     func() *intstr.IntOrString { i := intstr.FromInt(53); return &i }(),
								},
								networkingv1.NetworkPolicyPort{
									Protocol: func() *v1.Protocol { i := v1.ProtocolTCP; return &i }(),
									Port:     func() *intstr.IntOrString { i := intstr.FromInt(53); return &i }(),
								},
							},
							From: []networkingv1.NetworkPolicyPeer{
								networkingv1.NetworkPolicyPeer{
									NamespaceSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{"kubernetes.io/metadata.name": "default"},
									},
								},
								networkingv1.NetworkPolicyPeer{
									PodSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{"app": "nginx"},
									},
								},
							},
						},
					},
				},
			},
		},
		"assign egress rule": {
			ingressRules: []string{"ports=udp:53,tcp:53", "pod=app:nginx", "namespace=kubernetes.io/metadata.name:default"},
			expected: &networkingv1.NetworkPolicy{
				TypeMeta:   typeMeta,
				ObjectMeta: objectMeta,
				Spec: networkingv1.NetworkPolicySpec{
					Ingress: []networkingv1.NetworkPolicyIngressRule{
						networkingv1.NetworkPolicyIngressRule{
							Ports: []networkingv1.NetworkPolicyPort{
								networkingv1.NetworkPolicyPort{
									Protocol: func() *v1.Protocol { i := v1.ProtocolUDP; return &i }(),
									Port:     func() *intstr.IntOrString { i := intstr.FromInt(53); return &i }(),
								},
								networkingv1.NetworkPolicyPort{
									Protocol: func() *v1.Protocol { i := v1.ProtocolTCP; return &i }(),
									Port:     func() *intstr.IntOrString { i := intstr.FromInt(53); return &i }(),
								},
							},
						},
						networkingv1.NetworkPolicyIngressRule{
							From: []networkingv1.NetworkPolicyPeer{
								networkingv1.NetworkPolicyPeer{
									PodSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{"app": "nginx"},
									},
								},
							},
						},
						networkingv1.NetworkPolicyIngressRule{
							From: []networkingv1.NetworkPolicyPeer{
								networkingv1.NetworkPolicyPeer{
									NamespaceSelector: &metav1.LabelSelector{
										MatchLabels: map[string]string{"kubernetes.io/metadata.name": "default"},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			o := &CreateNetworkPolicyOptions{
				Name:         objectMeta.Name,
				PodSelector:  tc.podSelector,
				PolicyTypes:  tc.policyTypes,
				IngressRules: tc.ingressRules,
				EgressRules:  tc.egressRules,
			}
			networkPolicy, _ := o.createNetworkPolicy()
			if !apiequality.Semantic.DeepEqual(networkPolicy, tc.expected) {
				j1, e1 := json.Marshal(tc.expected)
				j2, e2 := json.Marshal(networkPolicy)
				if e1 == nil && e2 == nil {
					t.Errorf("expected:\n%#v\ngot:\n%#v", string(j1), string(j2))
				} else {
					t.Errorf("expected:\n%#v\ngot:\n%#v", tc.expected, networkPolicy)
				}
			}
		})
	}
}
