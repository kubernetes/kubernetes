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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"

	// ensure types are installed
	_ "k8s.io/kubernetes/pkg/apis/core/install"
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

// helper creates a NodeNode with a set of PodCIDRs, Spec.ConfigSource, Status.Config
func makeNode(podCIDRs []string, addSpecDynamicConfig bool, addStatusDynamicConfig bool) *api.Node {
	node := &api.Node{
		Spec: api.NodeSpec{
			PodCIDRs: podCIDRs,
		},
	}
	if addSpecDynamicConfig {
		node.Spec.ConfigSource = &api.NodeConfigSource{}
	}
	if addStatusDynamicConfig {
		node.Status = api.NodeStatus{
			Config: &api.NodeConfigStatus{},
		}
	}

	return node
}

func TestDropFields(t *testing.T) {
	testCases := []struct {
		name                    string
		node                    *api.Node
		oldNode                 *api.Node
		compareNode             *api.Node
		enableDualStack         bool
		enableNodeDynamicConfig bool
	}{
		{
			name:            "nil pod cidrs",
			enableDualStack: false,
			node:            makeNode(nil, false, false),
			oldNode:         nil,
			compareNode:     makeNode(nil, false, false),
		},
		{
			name:            "empty pod ips",
			enableDualStack: false,
			node:            makeNode([]string{}, false, false),
			oldNode:         nil,
			compareNode:     makeNode([]string{}, false, false),
		},
		{
			name:            "single family ipv6",
			enableDualStack: false,
			node:            makeNode([]string{"2000::/10"}, false, false),
			compareNode:     makeNode([]string{"2000::/10"}, false, false),
		},
		{
			name:            "single family ipv4",
			enableDualStack: false,
			node:            makeNode([]string{"10.0.0.0/8"}, false, false),
			compareNode:     makeNode([]string{"10.0.0.0/8"}, false, false),
		},
		{
			name:            "dualstack 4-6",
			enableDualStack: true,
			node:            makeNode([]string{"10.0.0.0/8", "2000::/10"}, false, false),
			compareNode:     makeNode([]string{"10.0.0.0/8", "2000::/10"}, false, false),
		},
		{
			name:            "dualstack 6-4",
			enableDualStack: true,
			node:            makeNode([]string{"2000::/10", "10.0.0.0/8"}, false, false),
			compareNode:     makeNode([]string{"2000::/10", "10.0.0.0/8"}, false, false),
		},
		{
			name:            "not dualstack 6-4=>4only",
			enableDualStack: false,
			node:            makeNode([]string{"2000::/10", "10.0.0.0/8"}, false, false),
			oldNode:         nil,
			compareNode:     makeNode([]string{"2000::/10"}, false, false),
		},
		{
			name:            "not dualstack 6-4=>as is (used in old)",
			enableDualStack: false,
			node:            makeNode([]string{"2000::/10", "10.0.0.0/8"}, false, false),
			oldNode:         makeNode([]string{"2000::/10", "10.0.0.0/8"}, false, false),
			compareNode:     makeNode([]string{"2000::/10", "10.0.0.0/8"}, false, false),
		},
		{
			name:            "not dualstack 6-4=>6only",
			enableDualStack: false,
			node:            makeNode([]string{"2000::/10", "10.0.0.0/8"}, false, false),
			oldNode:         nil,
			compareNode:     makeNode([]string{"2000::/10"}, false, false),
		},
		{
			name:            "not dualstack 6-4=>as is (used in old)",
			enableDualStack: false,
			node:            makeNode([]string{"2000::/10", "10.0.0.0/8"}, false, false),
			oldNode:         makeNode([]string{"2000::/10", "10.0.0.0/8"}, false, false),
			compareNode:     makeNode([]string{"2000::/10", "10.0.0.0/8"}, false, false),
		},
		{
			name:                    "new with no Spec.ConfigSource and no Status.Config , enableNodeDynamicConfig disabled",
			enableDualStack:         false,
			enableNodeDynamicConfig: false,
			node:                    makeNode(nil, false, false),
			oldNode:                 nil,
			compareNode:             makeNode(nil, false, false),
		},
		{
			name:                    "new with Spec.ConfigSource and no Status.Config, enableNodeDynamicConfig disabled",
			enableDualStack:         false,
			enableNodeDynamicConfig: false,
			node:                    makeNode(nil, true, false),
			oldNode:                 nil,
			compareNode:             makeNode(nil, false, false),
		},
		{
			name:                    "new with Spec.ConfigSource and Status.Config, enableNodeDynamicConfig disabled",
			enableDualStack:         false,
			enableNodeDynamicConfig: false,
			node:                    makeNode(nil, true, true),
			oldNode:                 nil,
			compareNode:             makeNode(nil, false, false),
		},
		{
			name:                    "update with Spec.ConfigSource and Status.Config (old has none), enableNodeDynamicConfig disabled",
			enableDualStack:         false,
			enableNodeDynamicConfig: false,
			node:                    makeNode(nil, true, true),
			oldNode:                 makeNode(nil, false, false),
			compareNode:             makeNode(nil, false, true),
		},
		{
			name:                    "update with Spec.ConfigSource and Status.Config (old has them), enableNodeDynamicConfig disabled",
			enableDualStack:         false,
			enableNodeDynamicConfig: false,
			node:                    makeNode(nil, true, true),
			oldNode:                 makeNode(nil, true, true),
			compareNode:             makeNode(nil, true, true),
		},
		{
			name:                    "update with Spec.ConfigSource and Status.Config (old has Status.Config), enableNodeDynamicConfig disabled",
			enableDualStack:         false,
			enableNodeDynamicConfig: false,
			node:                    makeNode(nil, true, true),
			oldNode:                 makeNode(nil, false, true),
			compareNode:             makeNode(nil, false, true),
		},
		{
			name:                    "new with Spec.ConfigSource and Status.Config, enableNodeDynamicConfig enabled",
			enableDualStack:         false,
			enableNodeDynamicConfig: true,
			node:                    makeNode(nil, true, true),
			oldNode:                 nil,
			compareNode:             makeNode(nil, true, true),
		},
	}

	for _, tc := range testCases {
		func() {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, tc.enableDualStack)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicKubeletConfig, tc.enableNodeDynamicConfig)()

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
