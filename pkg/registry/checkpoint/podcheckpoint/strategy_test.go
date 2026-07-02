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

package podcheckpoint

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/apis/checkpoint"
)

func newPodCheckpoint() *checkpoint.PodCheckpoint {
	return &checkpoint.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "checkpoint-1",
			Namespace: "default",
		},
		Spec: checkpoint.PodCheckpointSpec{
			SourcePod: &checkpoint.PodReference{Name: "my-pod"},
		},
		Status: checkpoint.PodCheckpointStatus{
			NodeName: new("node-1"),
		},
	}
}

func TestSelectableFields(t *testing.T) {
	pc := newPodCheckpoint()
	set := SelectableFields(pc)

	if got := set["spec.sourcePod.name"]; got != "my-pod" {
		t.Errorf("spec.sourcePod.name: expected %q, got %q", "my-pod", got)
	}
	if got := set["status.nodeName"]; got != "node-1" {
		t.Errorf("status.nodeName: expected %q, got %q", "node-1", got)
	}
	if got := set["metadata.name"]; got != "checkpoint-1" {
		t.Errorf("metadata.name: expected %q, got %q", "checkpoint-1", got)
	}
	if got := set["metadata.namespace"]; got != "default" {
		t.Errorf("metadata.namespace: expected %q, got %q", "default", got)
	}
}

func TestGetAttrs(t *testing.T) {
	pc := newPodCheckpoint()
	pc.ObjectMeta.Labels = map[string]string{"team": "blue"}

	lbls, flds, err := GetAttrs(pc)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v := lbls["team"]; v != "blue" {
		t.Errorf("label team: expected %q, got %q", "blue", v)
	}
	if v := flds["spec.sourcePod.name"]; v != "my-pod" {
		t.Errorf("spec.sourcePod.name: expected %q, got %q", "my-pod", v)
	}
	if v := flds["status.nodeName"]; v != "node-1" {
		t.Errorf("status.nodeName: expected %q, got %q", "node-1", v)
	}
}

func TestMatchPodCheckpoint(t *testing.T) {
	testCases := []struct {
		name          string
		fieldSelector fields.Selector
		matches       bool
	}{
		{
			name:          "match on spec.sourcePod.name",
			fieldSelector: fields.SelectorFromSet(fields.Set{"spec.sourcePod.name": "my-pod"}),
			matches:       true,
		},
		{
			name:          "match on status.nodeName",
			fieldSelector: fields.SelectorFromSet(fields.Set{"status.nodeName": "node-1"}),
			matches:       true,
		},
		{
			name:          "match on both fields",
			fieldSelector: fields.SelectorFromSet(fields.Set{"spec.sourcePod.name": "my-pod", "status.nodeName": "node-1"}),
			matches:       true,
		},
		{
			name:          "no match on spec.sourcePod.name",
			fieldSelector: fields.SelectorFromSet(fields.Set{"spec.sourcePod.name": "other-pod"}),
			matches:       false,
		},
		{
			name:          "no match on status.nodeName",
			fieldSelector: fields.SelectorFromSet(fields.Set{"status.nodeName": "node-2"}),
			matches:       false,
		},
	}

	pc := newPodCheckpoint()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			predicate := MatchPodCheckpoint(labels.Everything(), tc.fieldSelector)
			got, err := predicate.Matches(pc)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.matches {
				t.Errorf("expected matches=%v, got %v", tc.matches, got)
			}
		})
	}
}
