/*
Copyright 2025 The Kubernetes Authors.

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

package node

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/core/node"
)

// Smoke test that create requests are wired through declarative validation.
func TestDeclarativeValidateNodeCreateWiring(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "",
		APIVersion:        "v1",
		Resource:          "nodes",
		IsResourceRequest: true,
		Verb:              "create",
	})
	obj := mkValidNode()
	apitesting.VerifyValidationEquivalence(t, ctx, &obj, registry.Strategy, field.ErrorList{})
}

func TestDeclarativeValidateNodeUpdate(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "",
		APIVersion:        "v1",
		Resource:          "nodes",
		IsResourceRequest: true,
		Verb:              "update",
	})
	testCases := map[string]struct {
		old          api.Node
		update       api.Node
		expectedErrs field.ErrorList
	}{
		"no change": {
			old:    mkValidNode(),
			update: mkValidNode(),
		},
		"no change with providerID": {
			old: mkValidNode(func(n *api.Node) {
				n.Spec.ProviderID = "provider:///node-1"
			}),
			update: mkValidNode(func(n *api.Node) {
				n.Spec.ProviderID = "provider:///node-1"
			}),
		},
		"set providerID from empty": {
			old: mkValidNode(),
			update: mkValidNode(func(n *api.Node) {
				n.Spec.ProviderID = "provider:///node-1"
			}),
		},
		"modify providerID": {
			old: mkValidNode(func(n *api.Node) {
				n.Spec.ProviderID = "provider:///node-1"
			}),
			update: mkValidNode(func(n *api.Node) {
				n.Spec.ProviderID = "provider:///node-2"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "providerID"), nil, "field cannot be modified once set").WithOrigin("update").MarkAlpha(),
			},
		},
		"clear providerID": {
			old: mkValidNode(func(n *api.Node) {
				n.Spec.ProviderID = "provider:///node-1"
			}),
			update: mkValidNode(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "providerID"), nil, "field cannot be cleared once set").WithOrigin("update").MarkAlpha(),
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			tc.old.ObjectMeta.ResourceVersion = "1"
			tc.update.ObjectMeta.ResourceVersion = "1"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, registry.Strategy, tc.expectedErrs)
		})
	}
}

// mkValidNode produces a Node which passes validation with no tweaks.
func mkValidNode(tweaks ...func(n *api.Node)) api.Node {
	node := api.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-node",
		},
	}
	for _, tweak := range tweaks {
		tweak(&node)
	}
	return node
}
