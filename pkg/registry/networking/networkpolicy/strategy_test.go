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

package networkpolicy

import (
	"context"
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/features"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func makeNetworkPolicy(isIngress, isEgress, hasEndPort bool) *networking.NetworkPolicy {

	protocolTCP := api.ProtocolTCP
	endPort := int32(32000)
	netPol := &networking.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
		Spec: networking.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{
				MatchLabels: map[string]string{"a": "b"},
			},
		},
	}
	egress := networking.NetworkPolicyEgressRule{
		To: []networking.NetworkPolicyPeer{
			{
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"c": "d"},
				},
			},
		},
	}

	ingress := networking.NetworkPolicyIngressRule{
		From: []networking.NetworkPolicyPeer{
			{
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"c": "d"},
				},
			},
		},
	}

	ports := []networking.NetworkPolicyPort{
		{
			Protocol: &protocolTCP,
			Port:     &intstr.IntOrString{Type: intstr.Int, IntVal: 31000},
		},
	}

	ingress.Ports = ports
	egress.Ports = ports

	if hasEndPort {
		ingress.Ports[0].EndPort = &endPort
		egress.Ports[0].EndPort = &endPort
	}

	if isIngress {
		netPol.Spec.Ingress = append(netPol.Spec.Ingress, ingress)
	}

	if isEgress {
		netPol.Spec.Egress = append(netPol.Spec.Egress, egress)
	}

	return netPol
}

func TestNetworkPolicyStrategy(t *testing.T) {
	for _, tc := range []struct {
		name              string
		hasEndPort        bool
		enableFeatureGate bool
		isIngress         bool
		isEgress          bool
	}{
		{
			name:              "Create Ingress Rule with EndPort Feature Gate enabled and with EndPort defined",
			hasEndPort:        true,
			enableFeatureGate: true,
			isIngress:         true,
			isEgress:          false,
		},
		{
			name:              "Create Ingress Rule with EndPort Feature Gate disabled and with EndPort defined",
			hasEndPort:        true,
			enableFeatureGate: false,
			isIngress:         true,
			isEgress:          false,
		},
		{
			name:              "Create Ingress Rule with EndPort Feature Gate enabled and with endPort undefined",
			hasEndPort:        false,
			enableFeatureGate: true,
			isIngress:         true,
			isEgress:          false,
		},
		{
			name:              "Create Ingress Rule with EndPort Feature Gate disabled and with endPort undefined",
			hasEndPort:        false,
			enableFeatureGate: false,
			isIngress:         true,
			isEgress:          false,
		},
		{
			name:              "Create Egress Rule with EndPort Feature Gate enabled and with endPort defined",
			hasEndPort:        true,
			enableFeatureGate: true,
			isIngress:         false,
			isEgress:          true,
		},
		{
			name:              "Create Egress Rule with EndPort Feature Gate enabled and with endPort defined",
			hasEndPort:        true,
			enableFeatureGate: false,
			isIngress:         false,
			isEgress:          true,
		},
		{
			name:              "Create Egress Rule with EndPort Feature Gate true and with endPort undefined",
			hasEndPort:        false,
			enableFeatureGate: true,
			isIngress:         false,
			isEgress:          true,
		},
		{
			name:              "Create Egress Rule with EndPort Feature Gate disabled and with endPort undefined",
			hasEndPort:        false,
			enableFeatureGate: false,
			isIngress:         false,
			isEgress:          true,
		},
		{
			name:              "Create Ingress and Egress Rule with EndPort Feature Gate enabled and endPort defined",
			hasEndPort:        true,
			enableFeatureGate: true,
			isIngress:         true,
			isEgress:          true,
		},
		{
			name:              "Create Ingress and Egress Rule with EndPort Feature Gate disabled and endPort defined",
			hasEndPort:        true,
			enableFeatureGate: false,
			isIngress:         true,
			isEgress:          true,
		},
		{
			name:              "Create Ingress and Egress Rule with EndPort Feature Gate enabled and endPort undefined",
			hasEndPort:        false,
			enableFeatureGate: true,
			isIngress:         true,
			isEgress:          true,
		},
		{
			name:              "Create Ingress and Egress Rule with EndPort Feature Gate disabled and endPort undefined",
			hasEndPort:        false,
			enableFeatureGate: false,
			isIngress:         true,
			isEgress:          true,
		},
		{
			name:              "Create a null rule with EndPort Feature Gate enabled",
			hasEndPort:        false,
			enableFeatureGate: true,
			isIngress:         false,
			isEgress:          false,
		},
		{
			name:              "Create a null rule with EndPort Feature Gate disabled",
			hasEndPort:        false,
			enableFeatureGate: false,
			isIngress:         false,
			isEgress:          false,
		},
	} {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NetworkPolicyEndPort, tc.enableFeatureGate)()
		netpol := makeNetworkPolicy(tc.isIngress, tc.isEgress, tc.hasEndPort)
		Strategy.PrepareForCreate(context.Background(), netpol)
		err := ValidateNetworkPolicy(netpol, tc.enableFeatureGate, tc.hasEndPort)
		if err != nil {
			t.Errorf("Create Rule test %s failed: %s", tc.name, err)
		}

		oldNetPol := makeNetworkPolicy(tc.isIngress, tc.isEgress, false)
		newNetPol := makeNetworkPolicy(tc.isIngress, tc.isEgress, tc.hasEndPort)
		Strategy.PrepareForUpdate(context.Background(), newNetPol, oldNetPol)
		err = ValidateNetworkPolicy(newNetPol, tc.enableFeatureGate, tc.hasEndPort)
		if err != nil {
			t.Errorf("Update Rule test %s failed: %s", tc.name, err)
		}

	}
}

func ValidateNetworkPolicy(netPol *networking.NetworkPolicy, fgEnabled, hasEndPort bool) error {
	if (fgEnabled && hasEndPort) && !hasNetworkPolicyEndPort(netPol) {
		return fmt.Errorf("rule should contain endPort defined")
	}

	if (!fgEnabled && !hasEndPort) && hasNetworkPolicyEndPort(netPol) {
		return fmt.Errorf("rule should not contain endPort defined")
	}

	if (fgEnabled && !hasEndPort) && hasNetworkPolicyEndPort(netPol) {
		return fmt.Errorf("rule should not contain endPort defined")
	}

	if !fgEnabled && hasNetworkPolicyEndPort(netPol) {
		return fmt.Errorf("rule should not contain endPort defined")
	}
	return nil
}
