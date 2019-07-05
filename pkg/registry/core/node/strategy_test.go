/*
Copyright 2015 The Kubernetes Authors.

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
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/diff"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"

	// install all api groups for testing
	_ "k8s.io/kubernetes/pkg/api/testapi"
)

func TestMatchNode(t *testing.T) {
	testFieldMap := map[bool][]fields.Set{
		true: {
			{"metadata.name": "foo"},
		},
		false: {
			{"foo": "bar"},
		},
	}

	for expectedResult, fieldSet := range testFieldMap {
		for _, field := range fieldSet {
			m := MatchNode(labels.Everything(), field.AsSelector())
			_, matchesSingle := m.MatchesSingle()
			if e, a := expectedResult, matchesSingle; e != a {
				t.Errorf("%+v: expected %v, got %v", fieldSet, e, a)
			}
		}
	}
}

func TestSelectableFieldLabelConversions(t *testing.T) {
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		"v1",
		"Node",
		NodeToSelectableFields(&api.Node{}),
		nil,
	)
}

// helper creates a NodeNode with a set of PodCIDRs
func makeNodeWithCIDRs(podCIDRs []string) *api.Node {
	return &api.Node{
		Spec: api.NodeSpec{
			PodCIDRs: podCIDRs,
		},
	}
}

func TestDropPodCIDRs(t *testing.T) {
	testCases := []struct {
		name            string
		node            *api.Node
		oldNode         *api.Node
		compareNode     *api.Node
		enableDualStack bool
	}{
		{
			name:            "nil pod cidrs",
			enableDualStack: false,
			node:            makeNodeWithCIDRs(nil),
			oldNode:         nil,
			compareNode:     makeNodeWithCIDRs(nil),
		},
		{
			name:            "empty pod ips",
			enableDualStack: false,
			node:            makeNodeWithCIDRs([]string{}),
			oldNode:         nil,
			compareNode:     makeNodeWithCIDRs([]string{}),
		},
		{
			name:            "single family ipv6",
			enableDualStack: false,
			node:            makeNodeWithCIDRs([]string{"2000::/10"}),
			compareNode:     makeNodeWithCIDRs([]string{"2000::/10"}),
		},
		{
			name:            "single family ipv4",
			enableDualStack: false,
			node:            makeNodeWithCIDRs([]string{"10.0.0.0/8"}),
			compareNode:     makeNodeWithCIDRs([]string{"10.0.0.0/8"}),
		},
		{
			name:            "dualstack 4-6",
			enableDualStack: true,
			node:            makeNodeWithCIDRs([]string{"10.0.0.0/8", "2000::/10"}),
			compareNode:     makeNodeWithCIDRs([]string{"10.0.0.0/8", "2000::/10"}),
		},
		{
			name:            "dualstack 6-4",
			enableDualStack: true,
			node:            makeNodeWithCIDRs([]string{"2000::/10", "10.0.0.0/8"}),
			compareNode:     makeNodeWithCIDRs([]string{"2000::/10", "10.0.0.0/8"}),
		},
		{
			name:            "not dualstack 6-4=>4only",
			enableDualStack: false,
			node:            makeNodeWithCIDRs([]string{"2000::/10", "10.0.0.0/8"}),
			oldNode:         nil,
			compareNode:     makeNodeWithCIDRs([]string{"2000::/10"}),
		},
		{
			name:            "not dualstack 6-4=>as is (used in old)",
			enableDualStack: false,
			node:            makeNodeWithCIDRs([]string{"2000::/10", "10.0.0.0/8"}),
			oldNode:         makeNodeWithCIDRs([]string{"2000::/10", "10.0.0.0/8"}),
			compareNode:     makeNodeWithCIDRs([]string{"2000::/10", "10.0.0.0/8"}),
		},
		{
			name:            "not dualstack 6-4=>6only",
			enableDualStack: false,
			node:            makeNodeWithCIDRs([]string{"2000::/10", "10.0.0.0/8"}),
			oldNode:         nil,
			compareNode:     makeNodeWithCIDRs([]string{"2000::/10"}),
		},
		{
			name:            "not dualstack 6-4=>as is (used in old)",
			enableDualStack: false,
			node:            makeNodeWithCIDRs([]string{"2000::/10", "10.0.0.0/8"}),
			oldNode:         makeNodeWithCIDRs([]string{"2000::/10", "10.0.0.0/8"}),
			compareNode:     makeNodeWithCIDRs([]string{"2000::/10", "10.0.0.0/8"}),
		},
	}

	for _, tc := range testCases {
		func() {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, tc.enableDualStack)()
			dropDisabledFields(tc.node, tc.oldNode)

			old := tc.oldNode.DeepCopy()
			// old node  should never be changed
			if !reflect.DeepEqual(tc.oldNode, old) {
				t.Errorf("%v: old node changed: %v", tc.name, diff.ObjectReflectDiff(tc.oldNode, old))
			}

			if !reflect.DeepEqual(tc.node, tc.compareNode) {
				t.Errorf("%v: unexpected node spec: %v", tc.name, diff.ObjectReflectDiff(tc.node, tc.compareNode))
			}
		}()
	}
}
