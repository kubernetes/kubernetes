/*
Copyright 2019 The Kubernetes Authors.

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
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/diff"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/networking"
	"k8s.io/kubernetes/pkg/features"
)

func TestValidateNetworkPolicySCTP(t *testing.T) {
	sctpProtocol := api.ProtocolSCTP
	tcpProtocol := api.ProtocolTCP
	objectWithValue := func() *networking.NetworkPolicy {
		return &networking.NetworkPolicy{
			Spec: networking.NetworkPolicySpec{
				Ingress: []networking.NetworkPolicyIngressRule{{Ports: []networking.NetworkPolicyPort{{Protocol: &sctpProtocol}}}},
				Egress:  []networking.NetworkPolicyEgressRule{{Ports: []networking.NetworkPolicyPort{{Protocol: &sctpProtocol}}}},
			},
		}
	}
	objectWithoutValue := func() *networking.NetworkPolicy {
		return &networking.NetworkPolicy{
			Spec: networking.NetworkPolicySpec{
				Ingress: []networking.NetworkPolicyIngressRule{{Ports: []networking.NetworkPolicyPort{{Protocol: &tcpProtocol}}}},
				Egress:  []networking.NetworkPolicyEgressRule{{Ports: []networking.NetworkPolicyPort{{Protocol: &tcpProtocol}}}},
			},
		}
	}

	objectInfo := []struct {
		description string
		hasValue    bool
		object      func() *networking.NetworkPolicy
	}{
		{
			description: "has value",
			hasValue:    true,
			object:      objectWithValue,
		},
		{
			description: "does not have value",
			hasValue:    false,
			object:      objectWithoutValue,
		},
		{
			description: "is nil",
			hasValue:    false,
			object:      func() *networking.NetworkPolicy { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldNetworkPolicyInfo := range objectInfo {
			for _, newNetworkPolicyInfo := range objectInfo {
				oldNetworkPolicyHasValue, oldNetworkPolicy := oldNetworkPolicyInfo.hasValue, oldNetworkPolicyInfo.object()
				newNetworkPolicyHasValue, newNetworkPolicy := newNetworkPolicyInfo.hasValue, newNetworkPolicyInfo.object()
				if newNetworkPolicy == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old object %v, new object %v", enabled, oldNetworkPolicyInfo.description, newNetworkPolicyInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SCTPSupport, enabled)()
					errs := ValidateConditionalNetworkPolicy(newNetworkPolicy, oldNetworkPolicy)
					// objects should never be changed
					if !reflect.DeepEqual(oldNetworkPolicy, oldNetworkPolicyInfo.object()) {
						t.Errorf("old object changed: %v", diff.ObjectReflectDiff(oldNetworkPolicy, oldNetworkPolicyInfo.object()))
					}
					if !reflect.DeepEqual(newNetworkPolicy, newNetworkPolicyInfo.object()) {
						t.Errorf("new object changed: %v", diff.ObjectReflectDiff(newNetworkPolicy, newNetworkPolicyInfo.object()))
					}

					switch {
					case enabled || oldNetworkPolicyHasValue || !newNetworkPolicyHasValue:
						if len(errs) > 0 {
							t.Errorf("unexpected errors: %v", errs)
						}
					default:
						if len(errs) != 2 {
							t.Errorf("expected 2 errors, got %v", errs)
						}
					}
				})
			}
		}
	}
}
