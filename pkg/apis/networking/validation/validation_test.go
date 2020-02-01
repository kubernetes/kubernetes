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
	"fmt"
	"strings"
	"testing"

	networkingv1beta1 "k8s.io/api/networking/v1beta1"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/features"
	utilpointer "k8s.io/utils/pointer"
)

func TestValidateNetworkPolicy(t *testing.T) {
	protocolTCP := api.ProtocolTCP
	protocolUDP := api.ProtocolUDP
	protocolICMP := api.Protocol("ICMP")
	protocolSCTP := api.ProtocolSCTP

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SCTPSupport, true)()

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
									CIDR:   "fd00:192:168::/48",
									Except: []string{"fd00:192:168:3::/64", "fd00:192:168:4::/64"},
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
									CIDR:   "fd00:192:168::/48",
									Except: []string{"fd00:192:168:3::/64", "fd00:192:168:4::/64"},
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
									CIDR:   "fd00:192:168::/48",
									Except: []string{"fd00:192:168:3::/64", "fd00:192:168:4::/64"},
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
									CIDR:   "fd00:192:168::/48",
									Except: []string{"fd00:192:168:3::/64", "fd00:192:168:4::/64"},
								},
							},
						},
					},
				},
				PolicyTypes: []networking.PolicyType{networking.PolicyTypeIngress, networking.PolicyTypeEgress},
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
		"invalid ipv6 cidr format": {
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
									CIDR:   "fd00:192:168::",
									Except: []string{"fd00:192:168:3::/64", "fd00:192:168:4::/64"},
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
		"except IP is not strictly within CIDR range": {
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
									CIDR:   "192.168.0.0/24",
									Except: []string{"192.168.0.0/24"},
								},
							},
						},
					},
				},
			},
		},
		"except IPv6 is outside of CIDR range": {
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
									CIDR:   "fd00:192:168:1::/64",
									Except: []string{"fd00:192:168:2::/64"},
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

func TestValidateIngress(t *testing.T) {
	defaultBackend := networking.IngressBackend{
		ServiceName: "default-backend",
		ServicePort: intstr.FromInt(80),
	}
	pathTypePrefix := networking.PathTypePrefix
	pathTypeFoo := networking.PathType("foo")

	newValid := func() networking.Ingress {
		return networking.Ingress{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo",
				Namespace: metav1.NamespaceDefault,
			},
			Spec: networking.IngressSpec{
				DefaultBackend: &networking.IngressBackend{
					ServiceName: "default-backend",
					ServicePort: intstr.FromInt(80),
				},
				Rules: []networking.IngressRule{
					{
						Host: "foo.bar.com",
						IngressRuleValue: networking.IngressRuleValue{
							HTTP: &networking.HTTPIngressRuleValue{
								Paths: []networking.HTTPIngressPath{
									{
										Path:     "/foo",
										PathType: &pathTypePrefix,
										Backend:  defaultBackend,
									},
								},
							},
						},
					},
				},
			},
			Status: networking.IngressStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{IP: "127.0.0.1"},
					},
				},
			},
		}
	}
	servicelessBackend := newValid()
	servicelessBackend.Spec.DefaultBackend.ServiceName = ""
	invalidNameBackend := newValid()
	invalidNameBackend.Spec.DefaultBackend.ServiceName = "defaultBackend"
	noPortBackend := newValid()
	noPortBackend.Spec.DefaultBackend = &networking.IngressBackend{ServiceName: defaultBackend.ServiceName}
	noForwardSlashPath := newValid()
	noForwardSlashPath.Spec.Rules[0].IngressRuleValue.HTTP.Paths = []networking.HTTPIngressPath{
		{
			Path:    "invalid",
			Backend: defaultBackend,
		},
	}
	badPathType := newValid()
	badPathType.Spec.Rules[0].IngressRuleValue.HTTP.Paths[0].PathType = &pathTypeFoo

	noPaths := newValid()
	noPaths.Spec.Rules[0].IngressRuleValue.HTTP.Paths = []networking.HTTPIngressPath{}
	badHost := newValid()
	badHost.Spec.Rules[0].Host = "foobar:80"
	hostIP := "127.0.0.1"
	badHostIP := newValid()
	badHostIP.Spec.Rules[0].Host = hostIP
	badHostIPErr := fmt.Sprintf("spec.rules[0].host: Invalid value: '%v'", hostIP)

	errorCases := map[string]networking.Ingress{
		"spec.defaultBackend.serviceName: Required value": servicelessBackend,
		"spec.defaultBackend.serviceName: Invalid value":  invalidNameBackend,
		"spec.defaultBackend.servicePort: Invalid value":  noPortBackend,
		"spec.rules[0].host: Invalid value":               badHost,
		"spec.rules[0].http.paths: Required value":        noPaths,
		"spec.rules[0].http.paths[0].path: Invalid value": noForwardSlashPath,
		"spec.rules[0].http.paths[0].pathType: Unsupported value: \"foo\": supported values: \"ImplementationSpecific\", \"Prefix\", \"exact\"": badPathType,
	}
	errorCases[badHostIPErr] = badHostIP

	wildcardHost := "foo.*.bar.com"
	badWildcard := newValid()
	badWildcard.Spec.Rules[0].Host = wildcardHost
	badWildcardErr := fmt.Sprintf("spec.rules[0].host: Invalid value: '%v'", wildcardHost)
	errorCases[badWildcardErr] = badWildcard

	for k, v := range errorCases {
		errs := ValidateIngress(&v, networkingv1beta1.SchemeGroupVersion)
		if len(errs) == 0 {
			t.Errorf("expected failure for %q", k)
		} else {
			s := strings.Split(k, ":")
			err := errs[0]
			if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
				t.Errorf("unexpected error: %q, expected: %q", err, k)
			}
		}
	}
}

func TestValidateIngressCreate(t *testing.T) {
	validDefaultBackend := &networking.IngressBackend{
		ServiceName: "default-backend",
		ServicePort: intstr.FromInt(80),
	}
	validObjectMeta := metav1.ObjectMeta{Name: "test123", Namespace: "test123"}
	validObjectMetaAnnotation := metav1.ObjectMeta{
		Name:        "test123",
		Namespace:   "test123",
		Annotations: map[string]string{annotationIngressClass: "foo"},
	}

	testCases := map[string]struct {
		ingress      networking.Ingress
		expectedErrs field.ErrorList
	}{
		"class field set": {
			ingress: networking.Ingress{
				ObjectMeta: validObjectMeta,
				Spec: networking.IngressSpec{
					Class:          utilpointer.StringPtr("bar"),
					DefaultBackend: validDefaultBackend,
				},
			},
			expectedErrs: field.ErrorList{},
		},
		"class annotation set": {
			ingress: networking.Ingress{
				ObjectMeta: validObjectMetaAnnotation,
				Spec: networking.IngressSpec{
					DefaultBackend: validDefaultBackend,
				},
			},
			expectedErrs: field.ErrorList{},
		},
		"class field and annotation set": {
			ingress: networking.Ingress{
				ObjectMeta: validObjectMetaAnnotation,
				Spec: networking.IngressSpec{
					Class:          utilpointer.StringPtr("bar"),
					DefaultBackend: validDefaultBackend,
				},
			},
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("annotations").Child(annotationIngressClass), "foo", "can not be set when the class field is also set")},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateIngressCreate(&testCase.ingress, networkingv1beta1.SchemeGroupVersion)

			if len(errs) != len(testCase.expectedErrs) {
				t.Fatalf("Expected %d errors, got %d (%+v)", len(testCase.expectedErrs), len(errs), errs)
			}

			for i, err := range errs {
				if err.Error() != testCase.expectedErrs[i].Error() {
					t.Fatalf("Expected error: %v, got %v", testCase.expectedErrs[i].Error(), err.Error())
				}
			}
		})
	}
}

func TestValidateIngressRuleValue(t *testing.T) {
	fldPath := field.NewPath("testing.http.paths[0].path")
	testCases := map[string]struct {
		pathType     networking.PathType
		path         string
		expectedErrs field.ErrorList
	}{
		"implementation specific: no leading slash": {
			pathType:     networking.PathTypeImplementationSpecific,
			path:         "foo",
			expectedErrs: field.ErrorList{field.Invalid(fldPath, "foo", "must be an absolute path")},
		},
		"implementation specific: leading slash": {
			pathType:     networking.PathTypeImplementationSpecific,
			path:         "/foo",
			expectedErrs: field.ErrorList{},
		},
		"implementation specific: many slashes": {
			pathType:     networking.PathTypeImplementationSpecific,
			path:         "/foo/bar/foo",
			expectedErrs: field.ErrorList{},
		},
		"implementation specific: repeating slashes": {
			pathType:     networking.PathTypeImplementationSpecific,
			path:         "/foo//bar/foo",
			expectedErrs: field.ErrorList{},
		},
		"prefix: no leading slash": {
			pathType:     networking.PathTypePrefix,
			path:         "foo",
			expectedErrs: field.ErrorList{field.Invalid(fldPath, "foo", "must be an absolute path")},
		},
		"prefix: leading slash": {
			pathType:     networking.PathTypePrefix,
			path:         "/foo",
			expectedErrs: field.ErrorList{},
		},
		"prefix: many slashes": {
			pathType:     networking.PathTypePrefix,
			path:         "/foo/bar/foo",
			expectedErrs: field.ErrorList{},
		},
		"prefix: repeating slashes": {
			pathType:     networking.PathTypePrefix,
			path:         "/foo//bar/foo",
			expectedErrs: field.ErrorList{field.Invalid(fldPath, "/foo//bar/foo", "must not contain repeating '/' characters")},
		},
		"exact: no leading slash": {
			pathType:     networking.PathTypeExact,
			path:         "foo",
			expectedErrs: field.ErrorList{field.Invalid(fldPath, "foo", "must be an absolute path")},
		},
		"exact: leading slash": {
			pathType:     networking.PathTypeExact,
			path:         "/foo",
			expectedErrs: field.ErrorList{},
		},
		"exact: many slashes": {
			pathType:     networking.PathTypeExact,
			path:         "/foo/bar/foo",
			expectedErrs: field.ErrorList{},
		},
		"exact: repeating slashes": {
			pathType:     networking.PathTypeExact,
			path:         "/foo//bar/foo",
			expectedErrs: field.ErrorList{field.Invalid(fldPath, "/foo//bar/foo", "must not contain repeating '/' characters")},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			irv := &networking.IngressRuleValue{
				HTTP: &networking.HTTPIngressRuleValue{
					Paths: []networking.HTTPIngressPath{
						{
							Path:     testCase.path,
							PathType: &testCase.pathType,
							Backend: networking.IngressBackend{
								ServiceName: "default-backend",
								ServicePort: intstr.FromInt(80),
							},
						},
					},
				},
			}

			errs := validateIngressRuleValue(irv, field.NewPath("testing"))

			if len(errs) != len(testCase.expectedErrs) {
				t.Fatalf("Expected %d errors, got %d (%+v)", len(testCase.expectedErrs), len(errs), errs)
			}

			for i, err := range errs {
				if err.Error() != testCase.expectedErrs[i].Error() {
					t.Fatalf("Expected error: %v, got %v", testCase.expectedErrs[i], err)
				}
			}
		})
	}
}

func TestValidateIngressTLS(t *testing.T) {
	defaultBackend := networking.IngressBackend{
		ServiceName: "default-backend",
		ServicePort: intstr.FromInt(80),
	}

	newValid := func() networking.Ingress {
		return networking.Ingress{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo",
				Namespace: metav1.NamespaceDefault,
			},
			Spec: networking.IngressSpec{
				DefaultBackend: &networking.IngressBackend{
					ServiceName: "default-backend",
					ServicePort: intstr.FromInt(80),
				},
				Rules: []networking.IngressRule{
					{
						Host: "foo.bar.com",
						IngressRuleValue: networking.IngressRuleValue{
							HTTP: &networking.HTTPIngressRuleValue{
								Paths: []networking.HTTPIngressPath{
									{
										Path:    "/foo",
										Backend: defaultBackend,
									},
								},
							},
						},
					},
				},
			},
			Status: networking.IngressStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{IP: "127.0.0.1"},
					},
				},
			},
		}
	}

	errorCases := map[string]networking.Ingress{}

	wildcardHost := "foo.*.bar.com"
	badWildcardTLS := newValid()
	badWildcardTLS.Spec.Rules[0].Host = "*.foo.bar.com"
	badWildcardTLS.Spec.TLS = []networking.IngressTLS{
		{
			Hosts: []string{wildcardHost},
		},
	}
	badWildcardTLSErr := fmt.Sprintf("spec.tls[0].hosts: Invalid value: '%v'", wildcardHost)
	errorCases[badWildcardTLSErr] = badWildcardTLS

	for k, v := range errorCases {
		errs := ValidateIngress(&v, networkingv1beta1.SchemeGroupVersion)
		if len(errs) == 0 {
			t.Errorf("expected failure for %q", k)
		} else {
			s := strings.Split(k, ":")
			err := errs[0]
			if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
				t.Errorf("unexpected error: %q, expected: %q", err, k)
			}
		}
	}
}

func TestValidateIngressStatusUpdate(t *testing.T) {
	defaultBackend := networking.IngressBackend{
		ServiceName: "default-backend",
		ServicePort: intstr.FromInt(80),
	}

	newValid := func() networking.Ingress {
		return networking.Ingress{
			ObjectMeta: metav1.ObjectMeta{
				Name:            "foo",
				Namespace:       metav1.NamespaceDefault,
				ResourceVersion: "9",
			},
			Spec: networking.IngressSpec{
				DefaultBackend: &networking.IngressBackend{
					ServiceName: "default-backend",
					ServicePort: intstr.FromInt(80),
				},
				Rules: []networking.IngressRule{
					{
						Host: "foo.bar.com",
						IngressRuleValue: networking.IngressRuleValue{
							HTTP: &networking.HTTPIngressRuleValue{
								Paths: []networking.HTTPIngressPath{
									{
										Path:    "/foo",
										Backend: defaultBackend,
									},
								},
							},
						},
					},
				},
			},
			Status: networking.IngressStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{
						{IP: "127.0.0.1", Hostname: "foo.bar.com"},
					},
				},
			},
		}
	}
	oldValue := newValid()
	newValue := newValid()
	newValue.Status = networking.IngressStatus{
		LoadBalancer: api.LoadBalancerStatus{
			Ingress: []api.LoadBalancerIngress{
				{IP: "127.0.0.2", Hostname: "foo.com"},
			},
		},
	}
	invalidIP := newValid()
	invalidIP.Status = networking.IngressStatus{
		LoadBalancer: api.LoadBalancerStatus{
			Ingress: []api.LoadBalancerIngress{
				{IP: "abcd", Hostname: "foo.com"},
			},
		},
	}
	invalidHostname := newValid()
	invalidHostname.Status = networking.IngressStatus{
		LoadBalancer: api.LoadBalancerStatus{
			Ingress: []api.LoadBalancerIngress{
				{IP: "127.0.0.1", Hostname: "127.0.0.1"},
			},
		},
	}

	errs := ValidateIngressStatusUpdate(&newValue, &oldValue)
	if len(errs) != 0 {
		t.Errorf("Unexpected error %v", errs)
	}

	errorCases := map[string]networking.Ingress{
		"status.loadBalancer.ingress[0].ip: Invalid value":       invalidIP,
		"status.loadBalancer.ingress[0].hostname: Invalid value": invalidHostname,
	}
	for k, v := range errorCases {
		errs := ValidateIngressStatusUpdate(&v, &oldValue)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		} else {
			s := strings.Split(k, ":")
			err := errs[0]
			if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
				t.Errorf("unexpected error: %q, expected: %q", err, k)
			}
		}
	}
}

func TestValidateIngressClass(t *testing.T) {
	testCases := map[string]struct {
		ingressClass networking.IngressClass
		expectedErrs field.ErrorList
	}{
		"valid name, valid controller": {
			ingressClass: networking.IngressClass{
				ObjectMeta: metav1.ObjectMeta{Name: "test123"},
				Spec: networking.IngressClassSpec{
					Controller: "foo.co/bar",
				},
			},
			expectedErrs: field.ErrorList{},
		},
		"invalid name, valid controller": {
			ingressClass: networking.IngressClass{
				ObjectMeta: metav1.ObjectMeta{Name: "test*123"},
				Spec: networking.IngressClassSpec{
					Controller: "foo.co/bar",
				},
			},
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("metadata").Child("name"), "test*123", "a DNS-1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
		},
		"valid name, empty controller": {
			ingressClass: networking.IngressClass{
				ObjectMeta: metav1.ObjectMeta{Name: "test123"},
				Spec: networking.IngressClassSpec{
					Controller: "",
				},
			},
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec").Child("controller"), "")},
		},
		"valid name, controller max length": {
			ingressClass: networking.IngressClass{
				ObjectMeta: metav1.ObjectMeta{Name: "test123"},
				Spec: networking.IngressClassSpec{
					Controller: "foo.co/" + strings.Repeat("a", 243),
				},
			},
			expectedErrs: field.ErrorList{},
		},
		"valid name, controller too long": {
			ingressClass: networking.IngressClass{
				ObjectMeta: metav1.ObjectMeta{Name: "test123"},
				Spec: networking.IngressClassSpec{
					Controller: "foo.co/" + strings.Repeat("a", 244),
				},
			},
			expectedErrs: field.ErrorList{field.TooLong(field.NewPath("spec").Child("controller"), "", 250)},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateIngressClass(&testCase.ingressClass)

			if len(errs) != len(testCase.expectedErrs) {
				t.Fatalf("Expected %d errors, got %d (%+v)", len(testCase.expectedErrs), len(errs), errs)
			}

			for i, err := range errs {
				if err.Error() != testCase.expectedErrs[i].Error() {
					t.Fatalf("Expected error: %v, got %v", testCase.expectedErrs[i].Error(), err.Error())
				}
			}
		})
	}
}

func TestValidateIngressClassUpdate(t *testing.T) {
	testCases := map[string]struct {
		newIngressClass networking.IngressClass
		oldIngressClass networking.IngressClass
		expectedErrs    field.ErrorList
	}{
		"name change": {
			newIngressClass: networking.IngressClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test123",
					ResourceVersion: "2",
				},
				Spec: networking.IngressClassSpec{
					Controller: "foo.co/bar",
				},
			},
			oldIngressClass: networking.IngressClass{
				ObjectMeta: metav1.ObjectMeta{Name: "test123"},
				Spec: networking.IngressClassSpec{
					Controller: "foo.co/different",
				},
			},
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec").Child("controller"), "foo.co/bar", apimachineryvalidation.FieldImmutableErrorMsg)},
		},
		"parameters change": {
			newIngressClass: networking.IngressClass{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test123",
					ResourceVersion: "2",
				},
				Spec: networking.IngressClassSpec{
					Controller: "foo.co/bar",
					Parameters: &api.TypedLocalObjectReference{
						APIGroup: utilpointer.StringPtr("v1"),
						Kind:     "ConfigMap",
						Name:     "foo",
					},
				},
			},
			oldIngressClass: networking.IngressClass{
				ObjectMeta: metav1.ObjectMeta{Name: "test123"},
				Spec: networking.IngressClassSpec{
					Controller: "foo.co/bar",
				},
			},
			expectedErrs: field.ErrorList{},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateIngressClassUpdate(&testCase.newIngressClass, &testCase.oldIngressClass)

			if len(errs) != len(testCase.expectedErrs) {
				t.Fatalf("Expected %d errors, got %d (%+v)", len(testCase.expectedErrs), len(errs), errs)
			}

			for i, err := range errs {
				if err.Error() != testCase.expectedErrs[i].Error() {
					t.Fatalf("Expected error: %v, got %v", testCase.expectedErrs[i].Error(), err.Error())
				}
			}
		})
	}
}
