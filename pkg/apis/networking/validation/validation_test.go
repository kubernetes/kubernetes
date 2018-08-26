/*
Copyright 2014 The Kubernetes Authors.

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

package validation

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/features"
)

func TestValidateNetworkPolicy(t *testing.T) {
	protocolTCP := api.ProtocolTCP
	protocolUDP := api.ProtocolUDP
	protocolICMP := api.Protocol("ICMP")
	protocolSCTP := api.ProtocolSCTP

	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SCTPSupport, true)()

	successCases := []networking.NetworkPolicy{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []networking.NetworkPolicyIngressRule{},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						From:  []networking.NetworkPolicyPeer{},
						Ports: []networking.NetworkPolicyPort{},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						Ports: []networking.NetworkPolicyPort{
							{
								Protocol: nil,
								Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 80},
							},
							{
								Protocol: &protocolTCP,
								Port:     nil,
							},
							{
								Protocol: &protocolTCP,
								Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 443},
							},
							{
								Protocol: &protocolUDP,
								Port:     &intstr.IntOrString{Type: intstr.String, StrVal: "dns"},
							},
							{
								Protocol: &protocolSCTP,
								Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 7777},
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						From: []networking.NetworkPolicyPeer{
							{
								PodSelector: &metav1.LabelSelector{
									MatchLabels: map[string]string{"c": "d"},
								},
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						From: []networking.NetworkPolicyPeer{
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
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						From: []networking.NetworkPolicyPeer{
							{
								NamespaceSelector: &metav1.LabelSelector{
									MatchLabels: map[string]string{"c": "d"},
								},
								PodSelector: &metav1.LabelSelector{
									MatchLabels: map[string]string{"e": "f"},
								},
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Egress: []networking.NetworkPolicyEgressRule{
					{
						To: []networking.NetworkPolicyPeer{
							{
								NamespaceSelector: &metav1.LabelSelector{
									MatchLabels: map[string]string{"c": "d"},
								},
							},
						},
					},
				},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						From: []networking.NetworkPolicyPeer{
							{
								IPBlock: &networking.IPBlock{
									CIDR:   "192.168.0.0/16",
									Except: []string{"192.168.3.0/24", "192.168.4.0/24"},
								},
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						From: []networking.NetworkPolicyPeer{
							{
								IPBlock: &networking.IPBlock{
									CIDR:   "192.168.0.0/16",
									Except: []string{"192.168.3.0/24", "192.168.4.0/24"},
								},
							},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Egress: []networking.NetworkPolicyEgressRule{
					{
						To: []networking.NetworkPolicyPeer{
							{
								IPBlock: &networking.IPBlock{
									CIDR:   "192.168.0.0/16",
									Except: []string{"192.168.3.0/24", "192.168.4.0/24"},
								},
							},
						},
					},
				},
				PolicyTypes: []networking.PolicyType{networking.PolicyTypeEgress},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Egress: []networking.NetworkPolicyEgressRule{
					{
						To: []networking.NetworkPolicyPeer{
							{
								IPBlock: &networking.IPBlock{
									CIDR:   "192.168.0.0/16",
									Except: []string{"192.168.3.0/24", "192.168.4.0/24"},
								},
							},
						},
					},
				},
				PolicyTypes: []networking.PolicyType{networking.PolicyTypeIngress, networking.PolicyTypeEgress},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Egress: []networking.NetworkPolicyEgressRule{
					{
						Ports: []networking.NetworkPolicyPort{
							{
								Protocol: nil,
								Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 80},
							},
							{
								Protocol: &protocolTCP,
								Port:     nil,
							},
							{
								Protocol: &protocolTCP,
								Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 443},
							},
							{
								Protocol: &protocolUDP,
								Port:     &intstr.IntOrString{Type: intstr.String, StrVal: "dns"},
							},
							{
								Protocol: &protocolSCTP,
								Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 7777},
							},
						},
					},
				},
			},
		},
	}

	// Success cases are expected to pass validation.

	for k, v := range successCases {
		if errs := ValidateNetworkPolicy(&v); len(errs) != 0 {
			t.Errorf("Expected success for %d, got %v", k, errs)
		}
	}

	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	errorCases := map[string]networking.NetworkPolicy{
		"namespaceSelector and ipBlock": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						From: []networking.NetworkPolicyPeer{
							{
								NamespaceSelector: &metav1.LabelSelector{
									MatchLabels: map[string]string{"c": "d"},
								},
								IPBlock: &networking.IPBlock{
									CIDR:   "192.168.0.0/16",
									Except: []string{"192.168.3.0/24", "192.168.4.0/24"},
								},
							},
						},
					},
				},
			},
		},
		"podSelector and ipBlock": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Egress: []networking.NetworkPolicyEgressRule{
					{
						To: []networking.NetworkPolicyPeer{
							{
								PodSelector: &metav1.LabelSelector{
									MatchLabels: map[string]string{"c": "d"},
								},
								IPBlock: &networking.IPBlock{
									CIDR:   "192.168.0.0/16",
									Except: []string{"192.168.3.0/24", "192.168.4.0/24"},
								},
							},
						},
					},
				},
			},
		},
		"missing from and to type": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						From: []networking.NetworkPolicyPeer{{}},
					},
				},
				Egress: []networking.NetworkPolicyEgressRule{
					{
						To: []networking.NetworkPolicyPeer{{}},
					},
				},
			},
		},
		"invalid spec.podSelector": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: invalidSelector,
				},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						From: []networking.NetworkPolicyPeer{
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
		"invalid ingress.ports.protocol": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						Ports: []networking.NetworkPolicyPort{
							{
								Protocol: &protocolICMP,
								Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 80},
							},
						},
					},
				},
			},
		},
		"invalid ingress.ports.port (int)": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						Ports: []networking.NetworkPolicyPort{
							{
								Protocol: &protocolTCP,
								Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 123456789},
							},
						},
					},
				},
			},
		},
		"invalid ingress.ports.port (str)": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						Ports: []networking.NetworkPolicyPort{
							{
								Protocol: &protocolTCP,
								Port:     &intstr.IntOrString{Type: intstr.String, StrVal: "!@#$"},
							},
						},
					},
				},
			},
		},
		"invalid ingress.from.podSelector": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						From: []networking.NetworkPolicyPeer{
							{
								PodSelector: &metav1.LabelSelector{
									MatchLabels: invalidSelector,
								},
							},
						},
					},
				},
			},
		},
		"invalid egress.to.podSelector": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{},
				Egress: []networking.NetworkPolicyEgressRule{
					{
						To: []networking.NetworkPolicyPeer{
							{
								PodSelector: &metav1.LabelSelector{
									MatchLabels: invalidSelector,
								},
							},
						},
					},
				},
			},
		},
		"invalid egress.ports.protocol": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{},
				Egress: []networking.NetworkPolicyEgressRule{
					{
						Ports: []networking.NetworkPolicyPort{
							{
								Protocol: &protocolICMP,
								Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 80},
							},
						},
					},
				},
			},
		},
		"invalid egress.ports.port (int)": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{},
				Egress: []networking.NetworkPolicyEgressRule{
					{
						Ports: []networking.NetworkPolicyPort{
							{
								Protocol: &protocolTCP,
								Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 123456789},
							},
						},
					},
				},
			},
		},
		"invalid egress.ports.port (str)": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{},
				Egress: []networking.NetworkPolicyEgressRule{
					{
						Ports: []networking.NetworkPolicyPort{
							{
								Protocol: &protocolTCP,
								Port:     &intstr.IntOrString{Type: intstr.String, StrVal: "!@#$"},
							},
						},
					},
				},
			},
		},
		"invalid ingress.from.namespaceSelector": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						From: []networking.NetworkPolicyPeer{
							{
								NamespaceSelector: &metav1.LabelSelector{
									MatchLabels: invalidSelector,
								},
							},
						},
					},
				},
			},
		},
		"missing cidr field": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						From: []networking.NetworkPolicyPeer{
							{
								IPBlock: &networking.IPBlock{
									Except: []string{"192.168.8.0/24", "192.168.9.0/24"},
								},
							},
						},
					},
				},
			},
		},
		"invalid cidr format": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						From: []networking.NetworkPolicyPeer{
							{
								IPBlock: &networking.IPBlock{
									CIDR:   "192.168.5.6",
									Except: []string{"192.168.1.0/24", "192.168.2.0/24"},
								},
							},
						},
					},
				},
			},
		},
		"except field is an empty string": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						From: []networking.NetworkPolicyPeer{
							{
								IPBlock: &networking.IPBlock{
									CIDR:   "192.168.8.0/24",
									Except: []string{"", " "},
								},
							},
						},
					},
				},
			},
		},
		"except IP is outside of CIDR range": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Ingress: []networking.NetworkPolicyIngressRule{
					{
						From: []networking.NetworkPolicyPeer{
							{
								IPBlock: &networking.IPBlock{
									CIDR:   "192.168.8.0/24",
									Except: []string{"192.168.9.1/24"},
								},
							},
						},
					},
				},
			},
		},
		"invalid policyTypes": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Egress: []networking.NetworkPolicyEgressRule{
					{
						To: []networking.NetworkPolicyPeer{
							{
								IPBlock: &networking.IPBlock{
									CIDR:   "192.168.0.0/16",
									Except: []string{"192.168.3.0/24", "192.168.4.0/24"},
								},
							},
						},
					},
				},
				PolicyTypes: []networking.PolicyType{"foo", "bar"},
			},
		},
		"too many policyTypes": {
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Spec: networking.NetworkPolicySpec{
				PodSelector: metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				Egress: []networking.NetworkPolicyEgressRule{
					{
						To: []networking.NetworkPolicyPeer{
							{
								IPBlock: &networking.IPBlock{
									CIDR:   "192.168.0.0/16",
									Except: []string{"192.168.3.0/24", "192.168.4.0/24"},
								},
							},
						},
					},
				},
				PolicyTypes: []networking.PolicyType{"foo", "bar", "baz"},
			},
		},
	}

	// Error cases are not expected to pass validation.
	for testName, networkPolicy := range errorCases {
		if errs := ValidateNetworkPolicy(&networkPolicy); len(errs) == 0 {
			t.Errorf("Expected failure for test: %s", testName)
		}
	}
}

func TestValidateNetworkPolicyUpdate(t *testing.T) {
	type npUpdateTest struct {
		old    networking.NetworkPolicy
		update networking.NetworkPolicy
	}
	successCases := map[string]npUpdateTest{
		"no change": {
			old: networking.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
				Spec: networking.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Ingress: []networking.NetworkPolicyIngressRule{},
				},
			},
			update: networking.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
				Spec: networking.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Ingress: []networking.NetworkPolicyIngressRule{},
				},
			},
		},
		"change spec": {
			old: networking.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
				Spec: networking.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{},
					Ingress:     []networking.NetworkPolicyIngressRule{},
				},
			},
			update: networking.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
				Spec: networking.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{
						MatchLabels: map[string]string{"a": "b"},
					},
					Ingress: []networking.NetworkPolicyIngressRule{},
				},
			},
		},
	}

	for testName, successCase := range successCases {
		successCase.old.ObjectMeta.ResourceVersion = "1"
		successCase.update.ObjectMeta.ResourceVersion = "1"
		if errs := ValidateNetworkPolicyUpdate(&successCase.update, &successCase.old); len(errs) != 0 {
			t.Errorf("expected success (%s): %v", testName, errs)
		}
	}

	errorCases := map[string]npUpdateTest{
		"change name": {
			old: networking.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
				Spec: networking.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{},
					Ingress:     []networking.NetworkPolicyIngressRule{},
				},
			},
			update: networking.NetworkPolicy{
				ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "bar"},
				Spec: networking.NetworkPolicySpec{
					PodSelector: metav1.LabelSelector{},
					Ingress:     []networking.NetworkPolicyIngressRule{},
				},
			},
		},
	}

	for testName, errorCase := range errorCases {
		errorCase.old.ObjectMeta.ResourceVersion = "1"
		errorCase.update.ObjectMeta.ResourceVersion = "1"
		if errs := ValidateNetworkPolicyUpdate(&errorCase.update, &errorCase.old); len(errs) == 0 {
			t.Errorf("expected failure: %s", testName)
		}
	}
}
