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

package condition

import (
	"testing"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
)

func simpleNodeCondition(t, reason string) api.NodeCondition {
	return api.NodeCondition{
		Type:   api.NodeConditionType(t),
		Reason: reason,
	}
}

// TestAdmission verifies all create requests for Node result in every conditions being set
func TestAdmission(t *testing.T) {
	tests := []struct {
		admission   []string
		preexisting []api.NodeCondition
		expect      []api.NodeCondition
	}{
		{
			admission: []string{"Foo", "Bar"},
			expect: []api.NodeCondition{
				simpleNodeCondition("Foo", ReasonPending),
				simpleNodeCondition("Bar", ReasonPending),
			},
		},
		{
			admission: []string{"Foo", "Bar"},
			preexisting: []api.NodeCondition{
				simpleNodeCondition("Foo", "Foo"),
				simpleNodeCondition("Ready", "Ready"),
			},
			expect: []api.NodeCondition{
				simpleNodeCondition("Foo", "Foo"),
				simpleNodeCondition("Ready", "Ready"),
				simpleNodeCondition("Bar", ReasonPending),
			},
		},
		{
			preexisting: []api.NodeCondition{
				simpleNodeCondition("Foo", "Foo"),
				simpleNodeCondition("Ready", "Ready"),
			},
			expect: []api.NodeCondition{
				simpleNodeCondition("Foo", "Foo"),
				simpleNodeCondition("Ready", "Ready"),
			},
		},
	}

	for number, tc := range tests {
		namespace := "test"
		handler := newNodeCondition(tc.admission)
		node := api.Node{
			ObjectMeta: api.ObjectMeta{Name: "123", Namespace: namespace},
		}

		for _, nc := range tc.preexisting {
			node.Status.Conditions = append(node.Status.Conditions, nc)
		}
		err := handler.Admit(admission.NewAttributesRecord(&node, nil, api.Kind("Node").WithVersion("version"), node.Namespace, node.Name, api.Resource("nodes").WithVersion("version"), "", admission.Create, nil))
		if err != nil {
			t.Errorf("Unexpected error returned from admission handler (#%d)", number)
		}

		if len(node.Status.Conditions) != len(tc.expect) {
			t.Errorf("expected %d conditions, got %d (#%d)", len(node.Status.Conditions), len(tc.expect), number)
		}
		for i := range node.Status.Conditions {
			if node.Status.Conditions[i].Type != tc.expect[i].Type {
				t.Errorf("expected %v, got %v (#%d)", len(node.Status.Conditions), len(tc.expect), number)
			}
			if node.Status.Conditions[i].Reason != tc.expect[i].Reason {
				t.Errorf("expected %v, got %v (#%d)", len(node.Status.Conditions), len(tc.expect), number)
			}
		}
	}
}

// TestOtherResources ensures that this admission controller is a no-op for other resources,
// subresources, and non-pods.
func TestOtherResources(t *testing.T) {
	namespace := "testnamespace"
	name := "testname"
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespace},
	}
	tests := []struct {
		name        string
		kind        string
		resource    string
		subresource string
		object      runtime.Object
		expectError bool
	}{
		{
			name:     "non-node resource",
			kind:     "Foo",
			resource: "foos",
			object:   node,
		},
		{
			name:        "node subresource",
			kind:        "Node",
			resource:    "nodes",
			subresource: "exec",
			object:      node,
		},
		{
			name:        "non-Node object",
			kind:        "Node",
			resource:    "nodes",
			object:      &api.Service{},
			expectError: true,
		},
	}

	for _, tc := range tests {
		handler := &nodeCondition{}

		err := handler.Admit(admission.NewAttributesRecord(tc.object, nil, api.Kind(tc.kind).WithVersion("version"), namespace, name, api.Resource(tc.resource).WithVersion("version"), tc.subresource, admission.Create, nil))

		if tc.expectError {
			if err == nil {
				t.Errorf("%s: unexpected nil error", tc.name)
			}
			continue
		}

		if err != nil {
			t.Errorf("%s: unexpected error: %v", tc.name, err)
			continue
		}

		if len(node.Status.Conditions) != 0 {
			t.Errorf("%s: node conditions were set %v", tc.name, node.Status.Conditions)
		}
	}
}
