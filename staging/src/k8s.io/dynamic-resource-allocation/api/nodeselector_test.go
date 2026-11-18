/*
Copyright 2026 The Kubernetes Authors.

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

package api

import (
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
)

// TestNodeSelectorMatchesNodeAffinity ensures that the compiled selector
// behaves like the one it is derived from.
func TestNodeSelectorMatchesNodeAffinity(t *testing.T) {
	expr := func(key string, op v1.NodeSelectorOperator, values ...string) v1.NodeSelectorRequirement {
		return v1.NodeSelectorRequirement{Key: key, Operator: op, Values: values}
	}
	selector := func(terms ...v1.NodeSelectorTerm) *v1.NodeSelector {
		return &v1.NodeSelector{NodeSelectorTerms: terms}
	}
	labelTerm := func(reqs ...v1.NodeSelectorRequirement) v1.NodeSelectorTerm {
		return v1.NodeSelectorTerm{MatchExpressions: reqs}
	}
	fieldTerm := func(reqs ...v1.NodeSelectorRequirement) v1.NodeSelectorTerm {
		return v1.NodeSelectorTerm{MatchFields: reqs}
	}

	nodes := map[string]*v1.Node{
		"nil":      nil,
		"no-name":  {ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"zone": "a", "gen": "5"}}},
		"zone-a":   {ObjectMeta: metav1.ObjectMeta{Name: "node-a", Labels: map[string]string{"zone": "a", "gen": "5"}}},
		"zone-b":   {ObjectMeta: metav1.ObjectMeta{Name: "node-b", Labels: map[string]string{"zone": "b", "gen": "x"}}},
		"no-label": {ObjectMeta: metav1.ObjectMeta{Name: "node-c"}},
	}

	testcases := map[string]*v1.NodeSelector{
		"empty":               selector(),
		"empty-term":          selector(v1.NodeSelectorTerm{}),
		"in":                  selector(labelTerm(expr("zone", v1.NodeSelectorOpIn, "a", "c"))),
		"not-in":              selector(labelTerm(expr("zone", v1.NodeSelectorOpNotIn, "a"))),
		"exists":              selector(labelTerm(expr("zone", v1.NodeSelectorOpExists))),
		"does-not-exist":      selector(labelTerm(expr("zone", v1.NodeSelectorOpDoesNotExist))),
		"gt":                  selector(labelTerm(expr("gen", v1.NodeSelectorOpGt, "4"))),
		"lt":                  selector(labelTerm(expr("gen", v1.NodeSelectorOpLt, "6"))),
		"and-of-expressions":  selector(labelTerm(expr("zone", v1.NodeSelectorOpIn, "a"), expr("gen", v1.NodeSelectorOpGt, "1"))),
		"or-of-terms":         selector(labelTerm(expr("zone", v1.NodeSelectorOpIn, "a")), labelTerm(expr("zone", v1.NodeSelectorOpIn, "b"))),
		"field-in":            selector(fieldTerm(expr("metadata.name", v1.NodeSelectorOpIn, "node-a"))),
		"field-not-in":        selector(fieldTerm(expr("metadata.name", v1.NodeSelectorOpNotIn, "node-a"))),
		"field-unknown":       selector(fieldTerm(expr("metadata.uid", v1.NodeSelectorOpIn, "x"))),
		"labels-and-fields":   selector(v1.NodeSelectorTerm{MatchExpressions: []v1.NodeSelectorRequirement{expr("zone", v1.NodeSelectorOpIn, "a")}, MatchFields: []v1.NodeSelectorRequirement{expr("metadata.name", v1.NodeSelectorOpIn, "node-a")}}),
		"invalid-operator":    selector(labelTerm(expr("zone", "Bad", "a"))),
		"invalid-key":         selector(labelTerm(expr("not a key", v1.NodeSelectorOpExists))),
		"invalid-gt-value":    selector(labelTerm(expr("gen", v1.NodeSelectorOpGt, "x"))),
		"invalid-field-op":    selector(fieldTerm(expr("metadata.name", v1.NodeSelectorOpExists))),
		"invalid-field-count": selector(fieldTerm(expr("metadata.name", v1.NodeSelectorOpIn, "a", "b"))),
		"invalid-after-match": selector(labelTerm(expr("zone", v1.NodeSelectorOpIn, "a")), labelTerm(expr("gen", v1.NodeSelectorOpGt, "x"))),
	}

	for name, source := range testcases {
		t.Run(name, func(t *testing.T) {
			expected, expectedErr := nodeaffinity.NewNodeSelector(source)
			actual := newNodeSelector(source, MakeUniqueString)
			for nodeName, node := range nodes {
				t.Run(nodeName, func(t *testing.T) {
					match, err := actual.Match(node)
					if expectedErr != nil {
						assert.Error(t, err)
						assert.False(t, match)
						return
					}
					assert.NoError(t, err)
					assert.Equal(t, expected.Match(node), match)
				})
			}
		})
	}
}
