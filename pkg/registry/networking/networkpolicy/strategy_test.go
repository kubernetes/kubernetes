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
	"reflect"
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
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar", Generation: 0},
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
		// Create a Network Policy containing EndPort defined to compare with the generated by the tests
		expectedNetPol := makeNetworkPolicy(tc.isIngress, tc.isEgress,
			(tc.hasEndPort && tc.enableFeatureGate))

		netPol := makeNetworkPolicy(tc.isIngress, tc.isEgress, tc.hasEndPort)
		Strategy.PrepareForCreate(context.Background(), netPol)
		if !reflect.DeepEqual(netPol.Spec, expectedNetPol.Spec) {
			t.Errorf("Create: %s failed. Spec from NetworkPolicy is not equal to the expected. \nGot: %+v \nExpected: %+v",
				tc.name, netPol, expectedNetPol)
		}

		if netPol.Generation != 1 {
			t.Errorf("Create: Test %s failed. Network Policy Generation should be 1, got %d",
				tc.name, netPol.Generation)
		}

		errs := Strategy.Validate(context.Background(), netPol)
		if len(errs) != 0 {
			t.Errorf("Unexpected error from validation for created Network Policy: %v", errs)
		}

		oldNetPol := makeNetworkPolicy(tc.isIngress, tc.isEgress, false)
		updatedNetPol := makeNetworkPolicy(tc.isIngress, tc.isEgress, tc.hasEndPort)
		Strategy.PrepareForUpdate(context.Background(), updatedNetPol, oldNetPol)

		if !reflect.DeepEqual(updatedNetPol.Spec, expectedNetPol.Spec) {
			t.Errorf("Update: %s failed. Spec from NetworkPolicy is not equal to the expected. \nGot: %+v \nExpected: %+v",
				tc.name, updatedNetPol, expectedNetPol)
		}

		if updatedNetPol.Generation != 0 && !tc.enableFeatureGate {
			t.Errorf("Update: Test %s failed. Network Policy Generation should be 1, got %d",
				tc.name, updatedNetPol.Generation)
		}

		errs = Strategy.Validate(context.Background(), updatedNetPol)
		if len(errs) != 0 {
			t.Errorf("Unexpected error from validation for updated Network Policy: %v", errs)
		}

	}
}
