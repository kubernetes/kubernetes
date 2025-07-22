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

package netpol

import (
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type SetFunc func(policy *networkingv1.NetworkPolicy)

func GenNetworkPolicy(fn ...SetFunc) *networkingv1.NetworkPolicy {
	policy := &networkingv1.NetworkPolicy{}
	for _, f := range fn {
		f(policy)
	}
	return policy
}

func GenNetworkPolicyWithNameAndPodMatchLabel(name string, targetLabels map[string]string, otherFunc ...SetFunc) *networkingv1.NetworkPolicy {
	otherFunc = append(otherFunc, SetObjectMetaName(name), SetSpecPodSelectorMatchLabels(targetLabels))
	return GenNetworkPolicy(otherFunc...)
}

func GenNetworkPolicyWithNameAndPodSelector(name string, targetSelector metav1.LabelSelector, otherFunc ...SetFunc) *networkingv1.NetworkPolicy {
	otherFunc = append(otherFunc, SetObjectMetaName(name), SetSpecPodSelector(targetSelector))
	return GenNetworkPolicy(otherFunc...)
}

func SetObjectMetaName(name string) SetFunc {
	return func(policy *networkingv1.NetworkPolicy) {
		policy.ObjectMeta.Name = name
	}
}

func SetGenerateName(name string) SetFunc {
	return func(policy *networkingv1.NetworkPolicy) {
		policy.ObjectMeta.GenerateName = name
	}
}

func SetObjectMetaLabel(targetLabels map[string]string) SetFunc {
	return func(policy *networkingv1.NetworkPolicy) {
		policy.ObjectMeta.Labels = targetLabels
	}
}

func SetSpecPodSelector(targetSelector metav1.LabelSelector) SetFunc {
	return func(policy *networkingv1.NetworkPolicy) {
		policy.Spec.PodSelector = targetSelector
	}
}

func SetSpecPodSelectorMatchLabels(targetLabels map[string]string) SetFunc {
	return func(policy *networkingv1.NetworkPolicy) {
		policy.Spec.PodSelector = metav1.LabelSelector{
			MatchLabels: targetLabels,
		}
	}
}

func SetSpecIngressRules(rules ...networkingv1.NetworkPolicyIngressRule) SetFunc {
	return func(policy *networkingv1.NetworkPolicy) {
		if policy.Spec.Ingress == nil {
			policy.Spec.Ingress = []networkingv1.NetworkPolicyIngressRule{}
			policy.Spec.PolicyTypes = append(policy.Spec.PolicyTypes, networkingv1.PolicyTypeIngress)
		}
		policy.Spec.Ingress = append(policy.Spec.Ingress, rules...)
	}
}

func SetSpecEgressRules(rules ...networkingv1.NetworkPolicyEgressRule) SetFunc {
	return func(policy *networkingv1.NetworkPolicy) {
		if policy.Spec.Egress == nil {
			policy.Spec.Egress = []networkingv1.NetworkPolicyEgressRule{}
			policy.Spec.PolicyTypes = append(policy.Spec.PolicyTypes, networkingv1.PolicyTypeEgress)
		}
		policy.Spec.Egress = append(policy.Spec.Egress, rules...)
	}
}
