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

	networkingv1 "k8s.io/api/networking/v1"
	networkingv1beta1 "k8s.io/api/networking/v1beta1"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/networking"
	utilpointer "k8s.io/utils/pointer"
)

func TestValidateNetworkPolicy(t *testing.T) {
	protocolTCP := api.ProtocolTCP
	protocolUDP := api.ProtocolUDP
	protocolICMP := api.Protocol("ICMP")
	protocolSCTP := api.ProtocolSCTP

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
	serviceBackend := &networking.IngressServiceBackend{
		Name: "defaultbackend",
		Port: networking.ServiceBackendPort{
			Name:   "",
			Number: 80,
		},
	}
	defaultBackend := networking.IngressBackend{
		Service: serviceBackend,
	}
	pathTypePrefix := networking.PathTypePrefix
	pathTypeImplementationSpecific := networking.PathTypeImplementationSpecific
	pathTypeFoo := networking.PathType("foo")

	baseIngress := networking.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: networking.IngressSpec{
			DefaultBackend: &defaultBackend,
			Rules: []networking.IngressRule{
				{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{
								{
									Path:     "/foo",
									PathType: &pathTypeImplementationSpecific,
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

	testCases := map[string]struct {
		groupVersion       *schema.GroupVersion
		tweakIngress       func(ing *networking.Ingress)
		expectErrsOnFields []string
	}{
		"empty path (implementation specific)": {
			tweakIngress: func(ing *networking.Ingress) {
				ing.Spec.Rules[0].IngressRuleValue.HTTP.Paths[0].Path = ""
			},
			expectErrsOnFields: []string{},
		},
		"valid path": {
			tweakIngress: func(ing *networking.Ingress) {
				ing.Spec.Rules[0].IngressRuleValue.HTTP.Paths[0].Path = "/valid"
			},
			expectErrsOnFields: []string{},
		},
		// invalid use cases
		"backend (v1beta1) with no service": {
			groupVersion: &networkingv1beta1.SchemeGroupVersion,
			tweakIngress: func(ing *networking.Ingress) {
				ing.Spec.DefaultBackend.Service.Name = ""
			},
			expectErrsOnFields: []string{
				"spec.backend.serviceName",
			},
		},
		"invalid path type": {
			tweakIngress: func(ing *networking.Ingress) {
				ing.Spec.Rules[0].IngressRuleValue.HTTP.Paths[0].PathType = &pathTypeFoo
			},
			expectErrsOnFields: []string{
				"spec.rules[0].http.paths[0].pathType",
			},
		},
		"empty path (prefix)": {
			tweakIngress: func(ing *networking.Ingress) {
				ing.Spec.Rules[0].IngressRuleValue.HTTP.Paths[0].Path = ""
				ing.Spec.Rules[0].IngressRuleValue.HTTP.Paths[0].PathType = &pathTypePrefix
			},
			expectErrsOnFields: []string{
				"spec.rules[0].http.paths[0].path",
			},
		},
		"no paths": {
			tweakIngress: func(ing *networking.Ingress) {
				ing.Spec.Rules[0].IngressRuleValue.HTTP.Paths = []networking.HTTPIngressPath{}
			},
			expectErrsOnFields: []string{
				"spec.rules[0].http.paths",
			},
		},
		"invalid host (foobar:80)": {
			tweakIngress: func(ing *networking.Ingress) {
				ing.Spec.Rules[0].Host = "foobar:80"
			},
			expectErrsOnFields: []string{
				"spec.rules[0].host",
			},
		},
		"invalid host (127.0.0.1)": {
			tweakIngress: func(ing *networking.Ingress) {
				ing.Spec.Rules[0].Host = "127.0.0.1"
			},
			expectErrsOnFields: []string{
				"spec.rules[0].host",
			},
		},
		"valid wildcard host": {
			tweakIngress: func(ing *networking.Ingress) {
				ing.Spec.Rules[0].Host = "*.bar.com"
			},
			expectErrsOnFields: []string{},
		},
		"invalid wildcard host (foo.*.bar.com)": {
			tweakIngress: func(ing *networking.Ingress) {
				ing.Spec.Rules[0].Host = "foo.*.bar.com"
			},
			expectErrsOnFields: []string{
				"spec.rules[0].host",
			},
		},
		"invalid wildcard host (*)": {
			tweakIngress: func(ing *networking.Ingress) {
				ing.Spec.Rules[0].Host = "*"
			},
			expectErrsOnFields: []string{
				"spec.rules[0].host",
			},
		},
		"path resource backend and service name are not allowed together": {
			tweakIngress: func(ing *networking.Ingress) {
				ing.Spec.Rules[0].IngressRuleValue = networking.IngressRuleValue{
					HTTP: &networking.HTTPIngressRuleValue{
						Paths: []networking.HTTPIngressPath{
							{
								Path:     "/foo",
								PathType: &pathTypeImplementationSpecific,
								Backend: networking.IngressBackend{
									Service: serviceBackend,
									Resource: &api.TypedLocalObjectReference{
										APIGroup: utilpointer.StringPtr("example.com"),
										Kind:     "foo",
										Name:     "bar",
									},
								},
							},
						},
					},
				}
			},
			expectErrsOnFields: []string{
				"spec.rules[0].http.paths[0].backend",
			},
		},
		"path resource backend and service port are not allowed together": {
			tweakIngress: func(ing *networking.Ingress) {
				ing.Spec.Rules[0].IngressRuleValue = networking.IngressRuleValue{
					HTTP: &networking.HTTPIngressRuleValue{
						Paths: []networking.HTTPIngressPath{
							{
								Path:     "/foo",
								PathType: &pathTypeImplementationSpecific,
								Backend: networking.IngressBackend{
									Service: serviceBackend,
									Resource: &api.TypedLocalObjectReference{
										APIGroup: utilpointer.StringPtr("example.com"),
										Kind:     "foo",
										Name:     "bar",
									},
								},
							},
						},
					},
				}
			},
			expectErrsOnFields: []string{
				"spec.rules[0].http.paths[0].backend",
			},
		},
		"spec.backend resource and service name are not allowed together": {
			groupVersion: &networkingv1beta1.SchemeGroupVersion,
			tweakIngress: func(ing *networking.Ingress) {
				ing.Spec.DefaultBackend = &networking.IngressBackend{
					Service: serviceBackend,
					Resource: &api.TypedLocalObjectReference{
						APIGroup: utilpointer.StringPtr("example.com"),
						Kind:     "foo",
						Name:     "bar",
					},
				}
			},
			expectErrsOnFields: []string{
				"spec.backend",
			},
		},
		"spec.backend resource and service port are not allowed together": {
			groupVersion: &networkingv1beta1.SchemeGroupVersion,
			tweakIngress: func(ing *networking.Ingress) {
				ing.Spec.DefaultBackend = &networking.IngressBackend{
					Service: serviceBackend,
					Resource: &api.TypedLocalObjectReference{
						APIGroup: utilpointer.StringPtr("example.com"),
						Kind:     "foo",
						Name:     "bar",
					},
				}
			},
			expectErrsOnFields: []string{
				"spec.backend",
			},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			ingress := baseIngress.DeepCopy()
			testCase.tweakIngress(ingress)
			gv := testCase.groupVersion
			if gv == nil {
				gv = &networkingv1.SchemeGroupVersion
			}
			errs := validateIngress(ingress, IngressValidationOptions{}, *gv)
			if len(testCase.expectErrsOnFields) != len(errs) {
				t.Fatalf("Expected %d errors, got %d errors: %v", len(testCase.expectErrsOnFields), len(errs), errs)
			}
			for i, err := range errs {
				if err.Field != testCase.expectErrsOnFields[i] {
					t.Errorf("Expected error on field: %s, got: %s", testCase.expectErrsOnFields[i], err.Error())
				}
			}
		})
	}
}

func TestValidateIngressRuleValue(t *testing.T) {
	serviceBackend := networking.IngressServiceBackend{
		Name: "defaultbackend",
		Port: networking.ServiceBackendPort{
			Name:   "",
			Number: 80,
		},
	}
	fldPath := field.NewPath("testing.http.paths[0].path")
	testCases := map[string]struct {
		groupVersion *schema.GroupVersion
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
			expectedErrs: field.ErrorList{field.Invalid(fldPath, "/foo//bar/foo", "must not contain '//'")},
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
			expectedErrs: field.ErrorList{field.Invalid(fldPath, "/foo//bar/foo", "must not contain '//'")},
		},
		"prefix: with /./": {
			pathType:     networking.PathTypePrefix,
			path:         "/foo/./foo",
			expectedErrs: field.ErrorList{field.Invalid(fldPath, "/foo/./foo", "must not contain '/./'")},
		},
		"exact: with /../": {
			pathType:     networking.PathTypeExact,
			path:         "/foo/../foo",
			expectedErrs: field.ErrorList{field.Invalid(fldPath, "/foo/../foo", "must not contain '/../'")},
		},
		"prefix: with %2f": {
			pathType:     networking.PathTypePrefix,
			path:         "/foo/%2f/foo",
			expectedErrs: field.ErrorList{field.Invalid(fldPath, "/foo/%2f/foo", "must not contain '%2f'")},
		},
		"exact: with %2F": {
			pathType:     networking.PathTypeExact,
			path:         "/foo/%2F/foo",
			expectedErrs: field.ErrorList{field.Invalid(fldPath, "/foo/%2F/foo", "must not contain '%2F'")},
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
								Service: &serviceBackend,
							},
						},
					},
				},
			}
			gv := testCase.groupVersion
			if gv == nil {
				gv = &networkingv1.SchemeGroupVersion
			}
			errs := validateIngressRuleValue(irv, field.NewPath("testing"), IngressValidationOptions{}, *gv)
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

func TestValidateIngressCreate(t *testing.T) {
	implementationPathType := networking.PathTypeImplementationSpecific
	exactPathType := networking.PathTypeExact
	serviceBackend := &networking.IngressServiceBackend{
		Name: "defaultbackend",
		Port: networking.ServiceBackendPort{
			Number: 80,
		},
	}
	defaultBackend := networking.IngressBackend{
		Service: serviceBackend,
	}
	resourceBackend := &api.TypedLocalObjectReference{
		APIGroup: utilpointer.StringPtr("example.com"),
		Kind:     "foo",
		Name:     "bar",
	}
	baseIngress := networking.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test123",
			Namespace:       "test123",
			ResourceVersion: "1234",
		},
		Spec: networking.IngressSpec{
			DefaultBackend: &defaultBackend,
			Rules:          []networking.IngressRule{},
		},
	}

	testCases := map[string]struct {
		groupVersion *schema.GroupVersion
		tweakIngress func(ingress *networking.Ingress)
		expectedErrs field.ErrorList
	}{
		"class field set": {
			tweakIngress: func(ingress *networking.Ingress) {
				ingress.Spec.IngressClassName = utilpointer.StringPtr("bar")
			},
			expectedErrs: field.ErrorList{},
		},
		"class annotation set": {
			tweakIngress: func(ingress *networking.Ingress) {
				ingress.Annotations = map[string]string{annotationIngressClass: "foo"}
			},
			expectedErrs: field.ErrorList{},
		},
		"class field and annotation set": {
			tweakIngress: func(ingress *networking.Ingress) {
				ingress.Spec.IngressClassName = utilpointer.StringPtr("bar")
				ingress.Annotations = map[string]string{annotationIngressClass: "foo"}
			},
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("annotations").Child(annotationIngressClass), "foo", "can not be set when the class field is also set")},
		},
		"valid regex path": {
			tweakIngress: func(ingress *networking.Ingress) {
				ingress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/([a-z0-9]*)",
								PathType: &implementationPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
			expectedErrs: field.ErrorList{},
		},
		"invalid regex path allowed (v1)": {
			tweakIngress: func(ingress *networking.Ingress) {
				ingress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/([a-z0-9]*)[",
								PathType: &implementationPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
			expectedErrs: field.ErrorList{},
		},
		"Spec.Backend.Resource field allowed on create": {
			tweakIngress: func(ingress *networking.Ingress) {
				ingress.Spec.DefaultBackend = &networking.IngressBackend{
					Resource: resourceBackend}
			},
			expectedErrs: field.ErrorList{},
		},
		"Paths.Backend.Resource field allowed on create": {
			tweakIngress: func(ingress *networking.Ingress) {
				ingress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/([a-z0-9]*)",
								PathType: &implementationPathType,
								Backend: networking.IngressBackend{
									Resource: resourceBackend},
							}},
						},
					},
				}}
			},
			expectedErrs: field.ErrorList{},
		},
		"v1: valid secret": {
			groupVersion: &networkingv1.SchemeGroupVersion,
			tweakIngress: func(ingress *networking.Ingress) {
				ingress.Spec.TLS = []networking.IngressTLS{{SecretName: "valid"}}
			},
		},
		"v1: invalid secret": {
			groupVersion: &networkingv1.SchemeGroupVersion,
			tweakIngress: func(ingress *networking.Ingress) {
				ingress.Spec.TLS = []networking.IngressTLS{{SecretName: "invalid name"}}
			},
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec").Child("tls").Index(0).Child("secretName"), "invalid name", `a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`)},
		},
		"v1beta1: valid secret": {
			groupVersion: &networkingv1beta1.SchemeGroupVersion,
			tweakIngress: func(ingress *networking.Ingress) {
				ingress.Spec.TLS = []networking.IngressTLS{{SecretName: "valid"}}
			},
		},
		"v1beta1: invalid secret": {
			groupVersion: &networkingv1beta1.SchemeGroupVersion,
			tweakIngress: func(ingress *networking.Ingress) {
				ingress.Spec.TLS = []networking.IngressTLS{{SecretName: "invalid name 1"}}
			},
		},
		"v1: valid rules with wildcard host": {
			groupVersion: &networkingv1.SchemeGroupVersion,
			tweakIngress: func(ingress *networking.Ingress) {
				ingress.Spec.TLS = []networking.IngressTLS{{Hosts: []string{"*.bar.com"}}}
				ingress.Spec.Rules = []networking.IngressRule{{
					Host: "*.foo.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/foo",
								PathType: &exactPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
		},
		"v1: invalid rules with wildcard host": {
			groupVersion: &networkingv1.SchemeGroupVersion,
			tweakIngress: func(ingress *networking.Ingress) {
				ingress.Spec.TLS = []networking.IngressTLS{{Hosts: []string{"*.bar.com"}}}
				ingress.Spec.Rules = []networking.IngressRule{{
					Host: "*.foo.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "foo",
								PathType: &exactPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec").Child("rules").Index(0).Child("http").Child("paths").Index(0).Child("path"), "foo", `must be an absolute path`)},
		},
		"v1beta1: valid rules with wildcard host": {
			groupVersion: &networkingv1beta1.SchemeGroupVersion,
			tweakIngress: func(ingress *networking.Ingress) {
				ingress.Spec.TLS = []networking.IngressTLS{{Hosts: []string{"*.bar.com"}}}
				ingress.Spec.Rules = []networking.IngressRule{{
					Host: "*.foo.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/foo",
								PathType: &exactPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
		},
		"v1beta1: invalid rules with wildcard host": {
			groupVersion: &networkingv1beta1.SchemeGroupVersion,
			tweakIngress: func(ingress *networking.Ingress) {
				ingress.Spec.TLS = []networking.IngressTLS{{Hosts: []string{"*.bar.com"}}}
				ingress.Spec.Rules = []networking.IngressRule{{
					Host: "*.foo.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "foo",
								PathType: &exactPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			newIngress := baseIngress.DeepCopy()
			testCase.tweakIngress(newIngress)
			gv := testCase.groupVersion
			if gv == nil {
				gv = &networkingv1.SchemeGroupVersion
			}
			errs := ValidateIngressCreate(newIngress, *gv)
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

func TestValidateIngressUpdate(t *testing.T) {
	implementationPathType := networking.PathTypeImplementationSpecific
	exactPathType := networking.PathTypeExact
	serviceBackend := &networking.IngressServiceBackend{
		Name: "defaultbackend",
		Port: networking.ServiceBackendPort{
			Number: 80,
		},
	}
	defaultBackend := networking.IngressBackend{
		Service: serviceBackend,
	}
	resourceBackend := &api.TypedLocalObjectReference{
		APIGroup: utilpointer.StringPtr("example.com"),
		Kind:     "foo",
		Name:     "bar",
	}
	baseIngress := networking.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test123",
			Namespace:       "test123",
			ResourceVersion: "1234",
		},
		Spec: networking.IngressSpec{
			DefaultBackend: &defaultBackend,
		},
	}

	testCases := map[string]struct {
		gv             schema.GroupVersion
		tweakIngresses func(newIngress, oldIngress *networking.Ingress)
		expectedErrs   field.ErrorList
	}{
		"class field set": {
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				newIngress.Spec.IngressClassName = utilpointer.StringPtr("bar")
			},
			expectedErrs: field.ErrorList{},
		},
		"class annotation set": {
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				newIngress.Annotations = map[string]string{annotationIngressClass: "foo"}
			},
			expectedErrs: field.ErrorList{},
		},
		"class field and annotation set": {
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				newIngress.Spec.IngressClassName = utilpointer.StringPtr("bar")
				newIngress.Annotations = map[string]string{annotationIngressClass: "foo"}
			},
			expectedErrs: field.ErrorList{},
		},
		"valid regex path -> valid regex path": {
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/([a-z0-9]*)",
								PathType: &implementationPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
				newIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/([a-z0-9%]*)",
								PathType: &implementationPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
			expectedErrs: field.ErrorList{},
		},
		"valid regex path -> invalid regex path": {
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/([a-z0-9]*)",
								PathType: &implementationPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
				newIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/bar[",
								PathType: &implementationPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
			expectedErrs: field.ErrorList{},
		},
		"invalid regex path -> valid regex path": {
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/bar[",
								PathType: &implementationPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
				newIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/([a-z0-9]*)",
								PathType: &implementationPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
			expectedErrs: field.ErrorList{},
		},
		"invalid regex path -> invalid regex path": {
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/foo[",
								PathType: &implementationPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
				newIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/bar[",
								PathType: &implementationPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
			expectedErrs: field.ErrorList{},
		},
		"new Backend.Resource allowed on update": {
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.DefaultBackend = &defaultBackend
				newIngress.Spec.DefaultBackend = &networking.IngressBackend{
					Resource: resourceBackend}
			},
			expectedErrs: field.ErrorList{},
		},
		"old DefaultBackend.Resource allowed on update": {
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.DefaultBackend = &networking.IngressBackend{
					Resource: resourceBackend}
				newIngress.Spec.DefaultBackend = &networking.IngressBackend{
					Resource: resourceBackend}
			},
			expectedErrs: field.ErrorList{},
		},
		"changing spec.backend from resource -> no resource": {
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.DefaultBackend = &networking.IngressBackend{
					Resource: resourceBackend}
				newIngress.Spec.DefaultBackend = &defaultBackend
			},
			expectedErrs: field.ErrorList{},
		},
		"changing path backend from resource -> no resource": {
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/foo[",
								PathType: &implementationPathType,
								Backend: networking.IngressBackend{
									Resource: resourceBackend},
							}},
						},
					},
				}}
				newIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/bar[",
								PathType: &implementationPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
			expectedErrs: field.ErrorList{},
		},
		"changing path backend from resource -> resource": {
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/foo[",
								PathType: &implementationPathType,
								Backend: networking.IngressBackend{
									Resource: resourceBackend},
							}},
						},
					},
				}}
				newIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/bar[",
								PathType: &implementationPathType,
								Backend: networking.IngressBackend{
									Resource: resourceBackend},
							}},
						},
					},
				}}
			},
			expectedErrs: field.ErrorList{},
		},
		"changing path backend from non-resource -> non-resource": {
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/foo[",
								PathType: &implementationPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
				newIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/bar[",
								PathType: &implementationPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
			expectedErrs: field.ErrorList{},
		},
		"changing path backend from non-resource -> resource": {
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/foo[",
								PathType: &implementationPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
				newIngress.Spec.Rules = []networking.IngressRule{{
					Host: "foo.bar.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/bar[",
								PathType: &implementationPathType,
								Backend: networking.IngressBackend{
									Resource: resourceBackend},
							}},
						},
					},
				}}
			},
			expectedErrs: field.ErrorList{},
		},
		"v1: change valid secret -> invalid secret": {
			gv: networkingv1.SchemeGroupVersion,
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.TLS = []networking.IngressTLS{{SecretName: "valid"}}
				newIngress.Spec.TLS = []networking.IngressTLS{{SecretName: "invalid name"}}
			},
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec").Child("tls").Index(0).Child("secretName"), "invalid name", `a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`)},
		},
		"v1: change invalid secret -> invalid secret": {
			gv: networkingv1.SchemeGroupVersion,
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.TLS = []networking.IngressTLS{{SecretName: "invalid name 1"}}
				newIngress.Spec.TLS = []networking.IngressTLS{{SecretName: "invalid name 2"}}
			},
		},
		"v1beta1: change valid secret -> invalid secret": {
			gv: networkingv1beta1.SchemeGroupVersion,
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.TLS = []networking.IngressTLS{{SecretName: "valid"}}
				newIngress.Spec.TLS = []networking.IngressTLS{{SecretName: "invalid name"}}
			},
		},
		"v1beta1: change invalid secret -> invalid secret": {
			gv: networkingv1beta1.SchemeGroupVersion,
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.TLS = []networking.IngressTLS{{SecretName: "invalid name 1"}}
				newIngress.Spec.TLS = []networking.IngressTLS{{SecretName: "invalid name 2"}}
			},
		},
		"v1: change valid rules with wildcard host -> invalid rules": {
			gv: networkingv1.SchemeGroupVersion,
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.TLS = []networking.IngressTLS{{Hosts: []string{"*.bar.com"}}}
				oldIngress.Spec.Rules = []networking.IngressRule{{
					Host: "*.foo.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/foo",
								PathType: &exactPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
				newIngress.Spec.TLS = []networking.IngressTLS{{Hosts: []string{"*.bar.com"}}}
				newIngress.Spec.Rules = []networking.IngressRule{{
					Host: "*.foo.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "foo",
								PathType: &exactPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec").Child("rules").Index(0).Child("http").Child("paths").Index(0).Child("path"), "foo", `must be an absolute path`)},
		},
		"v1: change invalid rules with wildcard host -> invalid rules": {
			gv: networkingv1.SchemeGroupVersion,
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.TLS = []networking.IngressTLS{{Hosts: []string{"*.bar.com"}}}
				oldIngress.Spec.Rules = []networking.IngressRule{{
					Host: "*.foo.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "foo",
								PathType: &exactPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
				newIngress.Spec.TLS = []networking.IngressTLS{{Hosts: []string{"*.bar.com"}}}
				newIngress.Spec.Rules = []networking.IngressRule{{
					Host: "*.foo.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "bar",
								PathType: &exactPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
		},
		"v1beta1: change valid rules with wildcard host -> invalid rules": {
			gv: networkingv1beta1.SchemeGroupVersion,
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.TLS = []networking.IngressTLS{{Hosts: []string{"*.bar.com"}}}
				oldIngress.Spec.Rules = []networking.IngressRule{{
					Host: "*.foo.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "/foo",
								PathType: &exactPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
				newIngress.Spec.TLS = []networking.IngressTLS{{Hosts: []string{"*.bar.com"}}}
				newIngress.Spec.Rules = []networking.IngressRule{{
					Host: "*.foo.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "foo",
								PathType: &exactPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
		},
		"v1beta1: change invalid rules with wildcard host -> invalid rules": {
			gv: networkingv1beta1.SchemeGroupVersion,
			tweakIngresses: func(newIngress, oldIngress *networking.Ingress) {
				oldIngress.Spec.TLS = []networking.IngressTLS{{Hosts: []string{"*.bar.com"}}}
				oldIngress.Spec.Rules = []networking.IngressRule{{
					Host: "*.foo.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "foo",
								PathType: &exactPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
				newIngress.Spec.TLS = []networking.IngressTLS{{Hosts: []string{"*.bar.com"}}}
				newIngress.Spec.Rules = []networking.IngressRule{{
					Host: "*.foo.com",
					IngressRuleValue: networking.IngressRuleValue{
						HTTP: &networking.HTTPIngressRuleValue{
							Paths: []networking.HTTPIngressPath{{
								Path:     "bar",
								PathType: &exactPathType,
								Backend:  defaultBackend,
							}},
						},
					},
				}}
			},
		},
	}

	for name, testCase := range testCases {
		t.Run(name, func(t *testing.T) {
			newIngress := baseIngress.DeepCopy()
			oldIngress := baseIngress.DeepCopy()
			testCase.tweakIngresses(newIngress, oldIngress)

			gv := testCase.gv
			if gv.Empty() {
				gv = networkingv1beta1.SchemeGroupVersion
			}
			errs := ValidateIngressUpdate(newIngress, oldIngress, gv)

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
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("metadata.name"), "test*123", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
		},
		"valid name, empty controller": {
			ingressClass: networking.IngressClass{
				ObjectMeta: metav1.ObjectMeta{Name: "test123"},
				Spec: networking.IngressClassSpec{
					Controller: "",
				},
			},
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec.controller"), "")},
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
			expectedErrs: field.ErrorList{field.TooLong(field.NewPath("spec.controller"), "", 250)},
		},
		"valid name, valid controller, valid params": {
			ingressClass: networking.IngressClass{
				ObjectMeta: metav1.ObjectMeta{Name: "test123"},
				Spec: networking.IngressClassSpec{
					Controller: "foo.co/bar",
					Parameters: &api.TypedLocalObjectReference{
						APIGroup: utilpointer.StringPtr("example.com"),
						Kind:     "foo",
						Name:     "bar",
					},
				},
			},
			expectedErrs: field.ErrorList{},
		},
		"valid name, valid controller, invalid params (no kind)": {
			ingressClass: networking.IngressClass{
				ObjectMeta: metav1.ObjectMeta{Name: "test123"},
				Spec: networking.IngressClassSpec{
					Controller: "foo.co/bar",
					Parameters: &api.TypedLocalObjectReference{
						APIGroup: utilpointer.StringPtr("example.com"),
						Name:     "bar",
					},
				},
			},
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec.parameters.kind"), "kind is required")},
		},
		"valid name, valid controller, invalid params (no name)": {
			ingressClass: networking.IngressClass{
				ObjectMeta: metav1.ObjectMeta{Name: "test123"},
				Spec: networking.IngressClassSpec{
					Controller: "foo.co/bar",
					Parameters: &api.TypedLocalObjectReference{
						Kind: "foo",
					},
				},
			},
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec.parameters.name"), "name is required")},
		},
		"valid name, valid controller, invalid params (bad kind)": {
			ingressClass: networking.IngressClass{
				ObjectMeta: metav1.ObjectMeta{Name: "test123"},
				Spec: networking.IngressClassSpec{
					Controller: "foo.co/bar",
					Parameters: &api.TypedLocalObjectReference{
						Kind: "foo/",
						Name: "bar",
					},
				},
			},
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec.parameters.kind"), "foo/", "may not contain '/'")},
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

func TestValidateIngressTLS(t *testing.T) {
	pathTypeImplementationSpecific := networking.PathTypeImplementationSpecific
	serviceBackend := &networking.IngressServiceBackend{
		Name: "defaultbackend",
		Port: networking.ServiceBackendPort{
			Number: 80,
		},
	}
	defaultBackend := networking.IngressBackend{
		Service: serviceBackend,
	}
	newValid := func() networking.Ingress {
		return networking.Ingress{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo",
				Namespace: metav1.NamespaceDefault,
			},
			Spec: networking.IngressSpec{
				DefaultBackend: &defaultBackend,
				Rules: []networking.IngressRule{
					{
						Host: "foo.bar.com",
						IngressRuleValue: networking.IngressRuleValue{
							HTTP: &networking.HTTPIngressRuleValue{
								Paths: []networking.HTTPIngressPath{
									{
										Path:     "/foo",
										PathType: &pathTypeImplementationSpecific,
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

	errorCases := map[string]networking.Ingress{}

	wildcardHost := "foo.*.bar.com"
	badWildcardTLS := newValid()
	badWildcardTLS.Spec.Rules[0].Host = "*.foo.bar.com"
	badWildcardTLS.Spec.TLS = []networking.IngressTLS{
		{
			Hosts: []string{wildcardHost},
		},
	}
	badWildcardTLSErr := fmt.Sprintf("spec.tls[0].hosts[0]: Invalid value: '%v'", wildcardHost)
	errorCases[badWildcardTLSErr] = badWildcardTLS

	for k, v := range errorCases {
		errs := validateIngress(&v, IngressValidationOptions{}, networkingv1beta1.SchemeGroupVersion)
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

	// Test for wildcard host and wildcard TLS
	validCases := map[string]networking.Ingress{}
	wildHost := "*.bar.com"
	goodWildcardTLS := newValid()
	goodWildcardTLS.Spec.Rules[0].Host = "*.bar.com"
	goodWildcardTLS.Spec.TLS = []networking.IngressTLS{
		{
			Hosts: []string{wildHost},
		},
	}
	validCases[fmt.Sprintf("spec.tls[0].hosts: Valid value: '%v'", wildHost)] = goodWildcardTLS
	for k, v := range validCases {
		errs := validateIngress(&v, IngressValidationOptions{}, networkingv1beta1.SchemeGroupVersion)
		if len(errs) != 0 {
			t.Errorf("expected success for %q", k)
		}
	}
}

// TestValidateEmptyIngressTLS verifies that an empty TLS configuration can be
// specified, which ingress controllers may interpret to mean that TLS should be
// used with a default certificate that the ingress controller furnishes.
func TestValidateEmptyIngressTLS(t *testing.T) {
	pathTypeImplementationSpecific := networking.PathTypeImplementationSpecific
	serviceBackend := &networking.IngressServiceBackend{
		Name: "defaultbackend",
		Port: networking.ServiceBackendPort{
			Number: 443,
		},
	}
	defaultBackend := networking.IngressBackend{
		Service: serviceBackend,
	}
	newValid := func() networking.Ingress {
		return networking.Ingress{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo",
				Namespace: metav1.NamespaceDefault,
			},
			Spec: networking.IngressSpec{
				Rules: []networking.IngressRule{
					{
						Host: "foo.bar.com",
						IngressRuleValue: networking.IngressRuleValue{
							HTTP: &networking.HTTPIngressRuleValue{
								Paths: []networking.HTTPIngressPath{
									{
										PathType: &pathTypeImplementationSpecific,
										Backend:  defaultBackend,
									},
								},
							},
						},
					},
				},
			},
		}
	}

	validCases := map[string]networking.Ingress{}
	goodEmptyTLS := newValid()
	goodEmptyTLS.Spec.TLS = []networking.IngressTLS{
		{},
	}
	validCases[fmt.Sprintf("spec.tls[0]: Valid value: %v", goodEmptyTLS.Spec.TLS[0])] = goodEmptyTLS
	goodEmptyHosts := newValid()
	goodEmptyHosts.Spec.TLS = []networking.IngressTLS{
		{
			Hosts: []string{},
		},
	}
	validCases[fmt.Sprintf("spec.tls[0]: Valid value: %v", goodEmptyHosts.Spec.TLS[0])] = goodEmptyHosts
	for k, v := range validCases {
		errs := validateIngress(&v, IngressValidationOptions{}, networkingv1beta1.SchemeGroupVersion)
		if len(errs) != 0 {
			t.Errorf("expected success for %q", k)
		}
	}
}

func TestValidateIngressStatusUpdate(t *testing.T) {
	serviceBackend := &networking.IngressServiceBackend{
		Name: "defaultbackend",
		Port: networking.ServiceBackendPort{
			Number: 80,
		},
	}
	defaultBackend := networking.IngressBackend{
		Service: serviceBackend,
	}

	newValid := func() networking.Ingress {
		return networking.Ingress{
			ObjectMeta: metav1.ObjectMeta{
				Name:            "foo",
				Namespace:       metav1.NamespaceDefault,
				ResourceVersion: "9",
			},
			Spec: networking.IngressSpec{
				DefaultBackend: &defaultBackend,
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
