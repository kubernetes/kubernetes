/*
Copyright The Kubernetes Authors.

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

package nodetainttimeadded

import (
	"context"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	api "k8s.io/kubernetes/pkg/apis/core"
)

var (
	mynode   = &user.DefaultInfo{Name: "system:node:mynode", Groups: []string{"system:nodes"}}
	nodeGVR  = api.Resource("nodes").WithVersion("v1")
	nodeKind = api.Kind("Node").WithVersion("v1")
	podGVR   = api.Resource("pods").WithVersion("v1")
	podKind  = api.Kind("Pod").WithVersion("v1")
)

func TestHandles(t *testing.T) {
	p := NewPlugin()
	tests := map[admission.Operation]bool{
		admission.Create:  true,
		admission.Update:  true,
		admission.Delete:  false,
		admission.Connect: false,
	}
	for op, expected := range tests {
		result := p.Handles(op)
		if result != expected {
			t.Errorf("Unexpected result for operation %s: %v\n", op, result)
		}
	}
}

func Test_taintTimeAdded(t *testing.T) {
	oldTime := metav1.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	clientTime := metav1.Date(2025, 6, 1, 12, 0, 0, 0, time.UTC)

	nodeWithTaints := func(taints ...api.Taint) api.Node {
		return api.Node{
			ObjectMeta: metav1.ObjectMeta{Name: "mynode"},
			Spec:       api.NodeSpec{Taints: taints},
		}
	}
	taint := func(key string, effect api.TaintEffect, timeAdded *metav1.Time) api.Taint {
		return api.Taint{Key: key, Effect: effect, TimeAdded: timeAdded}
	}

	tests := []struct {
		name        string
		node        api.Node
		oldNode     api.Node
		operation   admission.Operation
		options     runtime.Object
		subresource string
		check       func(t *testing.T, taints []api.Taint)
	}{
		{
			name: "create, timeAdded is defaulted for every effect",
			node: nodeWithTaints(
				taint("no-schedule", api.TaintEffectNoSchedule, nil),
				taint("prefer-no-schedule", api.TaintEffectPreferNoSchedule, nil),
				taint("no-execute", api.TaintEffectNoExecute, nil),
			),
			operation: admission.Create,
			options:   &metav1.CreateOptions{},
			check: func(t *testing.T, taints []api.Taint) {
				for _, taint := range taints {
					expectTimeAddedNow(t, taint)
				}
			},
		},
		{
			name:      "create, timeAdded set by the client is kept",
			node:      nodeWithTaints(taint("no-schedule", api.TaintEffectNoSchedule, &clientTime)),
			operation: admission.Create,
			options:   &metav1.CreateOptions{},
			check: func(t *testing.T, taints []api.Taint) {
				expectTimeAdded(t, taints[0], clientTime)
			},
		},
		{
			name:      "update, timeAdded is defaulted for a taint which was just added",
			oldNode:   nodeWithTaints(),
			node:      nodeWithTaints(taint("no-schedule", api.TaintEffectNoSchedule, nil)),
			operation: admission.Update,
			options:   &metav1.UpdateOptions{},
			check: func(t *testing.T, taints []api.Taint) {
				expectTimeAddedNow(t, taints[0])
			},
		},
		{
			name:      "update, timeAdded of an existing taint is carried over",
			oldNode:   nodeWithTaints(taint("no-schedule", api.TaintEffectNoSchedule, &oldTime)),
			node:      nodeWithTaints(taint("no-schedule", api.TaintEffectNoSchedule, nil)),
			operation: admission.Update,
			options:   &metav1.UpdateOptions{},
			check: func(t *testing.T, taints []api.Taint) {
				expectTimeAdded(t, taints[0], oldTime)
			},
		},
		{
			name:      "update, timeAdded is defaulted again when the effect changes",
			oldNode:   nodeWithTaints(taint("spot", api.TaintEffectNoSchedule, &oldTime)),
			node:      nodeWithTaints(taint("spot", api.TaintEffectNoExecute, nil)),
			operation: admission.Update,
			options:   &metav1.UpdateOptions{},
			check: func(t *testing.T, taints []api.Taint) {
				expectTimeAddedNow(t, taints[0])
			},
		},
		{
			name:      "update, timeAdded set by the client is kept",
			oldNode:   nodeWithTaints(taint("no-schedule", api.TaintEffectNoSchedule, &oldTime)),
			node:      nodeWithTaints(taint("no-schedule", api.TaintEffectNoSchedule, &clientTime)),
			operation: admission.Update,
			options:   &metav1.UpdateOptions{},
			check: func(t *testing.T, taints []api.Taint) {
				expectTimeAdded(t, taints[0], clientTime)
			},
		},
		{
			name:        "update of a subresource is ignored",
			oldNode:     nodeWithTaints(taint("no-schedule", api.TaintEffectNoSchedule, &oldTime)),
			node:        nodeWithTaints(taint("no-schedule", api.TaintEffectNoSchedule, nil)),
			operation:   admission.Update,
			options:     &metav1.UpdateOptions{},
			subresource: "status",
			check: func(t *testing.T, taints []api.Taint) {
				if taints[0].TimeAdded != nil {
					t.Errorf("expected timeAdded to be unset, got %v", taints[0].TimeAdded)
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			attributes := admission.NewAttributesRecord(&tt.node, &tt.oldNode, nodeKind, tt.node.Namespace, tt.node.Name, nodeGVR, tt.subresource, tt.operation, tt.options, false, mynode)
			if err := NewPlugin().Admit(context.TODO(), attributes, nil); err != nil {
				t.Fatalf("Admit() error = %v", err)
			}
			node, _ := attributes.GetObject().(*api.Node)
			tt.check(t, node.Spec.Taints)
		})
	}
}

func Test_otherResourcesAreIgnored(t *testing.T) {
	pod := &api.Pod{ObjectMeta: metav1.ObjectMeta{Name: "mypod", Namespace: "default"}}
	attributes := admission.NewAttributesRecord(pod, nil, podKind, pod.Namespace, pod.Name, podGVR, "", admission.Create, &metav1.CreateOptions{}, false, mynode)
	if err := NewPlugin().Admit(context.TODO(), attributes, nil); err != nil {
		t.Fatalf("Admit() error = %v", err)
	}
}

// expectTimeAdded fails the test if timeAdded does not match the given time.
func expectTimeAdded(t *testing.T, taint api.Taint, want metav1.Time) {
	t.Helper()
	if taint.TimeAdded == nil {
		t.Fatalf("taint %q: expected timeAdded %v, got nil", taint.Key, want)
	}
	if !taint.TimeAdded.Equal(&want) {
		t.Errorf("taint %q: expected timeAdded %v, got %v", taint.Key, want, taint.TimeAdded)
	}
}

// expectTimeAddedNow fails the test if timeAdded is not close to the current
// time, truncated to seconds because that is all which survives round-tripping.
func expectTimeAddedNow(t *testing.T, taint api.Taint) {
	t.Helper()
	if taint.TimeAdded == nil {
		t.Fatalf("taint %q: expected timeAdded to be defaulted, got nil", taint.Key)
	}
	if delta := time.Since(taint.TimeAdded.Time); delta < -time.Minute || delta > time.Minute {
		t.Errorf("taint %q: expected timeAdded close to the current time, got %v", taint.Key, taint.TimeAdded)
	}
	if truncated := taint.TimeAdded.Truncate(time.Second); !taint.TimeAdded.Time.Equal(truncated) {
		t.Errorf("taint %q: expected timeAdded truncated to seconds, got %v", taint.Key, taint.TimeAdded)
	}
}
