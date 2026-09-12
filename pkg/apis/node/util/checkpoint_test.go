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

package util

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func sourcePod() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "my-app",
			Namespace:       "team-a",
			UID:             types.UID("abc-123"),
			ResourceVersion: "456",
			Generation:      7,
			Labels:          map[string]string{"app": "my-app"},
			Annotations:     map[string]string{"team": "a"},
			OwnerReferences: []metav1.OwnerReference{{Kind: "ReplicaSet", Name: "rs-1"}},
			ManagedFields:   []metav1.ManagedFieldsEntry{{Manager: "kubelet"}},
		},
		Spec: v1.PodSpec{
			NodeName:     "node-1",
			NodeSelector: map[string]string{"disktype": "ssd"},
			RestoreFrom: &v1.CheckpointReference{
				Name:    "ckpt",
				Options: map[string]string{"example.runtime/target": "new-node"},
			},
			Containers: []v1.Container{{Name: "main", Image: "my-app:latest"}},
		},
		Status: v1.PodStatus{Phase: v1.PodRunning},
	}
}

func TestSanitizePodTemplate_StripsNodeLocalAndIdentityFields(t *testing.T) {
	pod := sourcePod()
	tmpl := SanitizePodTemplate(pod)

	if tmpl == nil {
		t.Fatal("expected non-nil template")
	}

	// Identity / server-set metadata must be cleared.
	if tmpl.UID != "" {
		t.Errorf("UID not cleared: %q", tmpl.UID)
	}
	if tmpl.ResourceVersion != "" {
		t.Errorf("ResourceVersion not cleared: %q", tmpl.ResourceVersion)
	}
	if tmpl.Generation != 0 {
		t.Errorf("Generation not cleared: %d", tmpl.Generation)
	}
	if tmpl.Name != "" || tmpl.Namespace != "" {
		t.Errorf("Name/Namespace not cleared: %q/%q", tmpl.Name, tmpl.Namespace)
	}
	if len(tmpl.ManagedFields) != 0 {
		t.Errorf("ManagedFields not cleared: %+v", tmpl.ManagedFields)
	}

	// Node-local scheduling state and the restore invocation must be cleared.
	if tmpl.Spec.NodeName != "" {
		t.Errorf("spec.NodeName not cleared: %q", tmpl.Spec.NodeName)
	}
	if tmpl.Spec.RestoreFrom != nil {
		t.Errorf("spec.RestoreFrom not cleared: %v", *tmpl.Spec.RestoreFrom)
	}

	// User-meaningful metadata and spec must be preserved.
	if tmpl.Labels["app"] != "my-app" {
		t.Errorf("labels not preserved: %+v", tmpl.Labels)
	}
	if tmpl.Annotations["team"] != "a" {
		t.Errorf("annotations not preserved: %+v", tmpl.Annotations)
	}
	if len(tmpl.OwnerReferences) != 1 || tmpl.OwnerReferences[0].Name != "rs-1" {
		t.Errorf("ownerReferences not preserved: %+v", tmpl.OwnerReferences)
	}
	if len(tmpl.Spec.Containers) != 1 || tmpl.Spec.Containers[0].Image != "my-app:latest" {
		t.Errorf("containers not preserved: %+v", tmpl.Spec.Containers)
	}
	// Non-identity nodeSelector entries are retained; only node-identity
	// constraints are stripped (see the dedicated test below).
	if tmpl.Spec.NodeSelector["disktype"] != "ssd" {
		t.Errorf("nodeSelector not preserved: %+v", tmpl.Spec.NodeSelector)
	}
}

func TestSanitizePodTemplate_StripsNodeIdentityConstraints(t *testing.T) {
	pod := sourcePod()
	pod.Spec.NodeSelector = map[string]string{
		v1.LabelHostname: "node-1",
		"disktype":       "ssd",
	}
	pod.Spec.Affinity = &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{Key: v1.LabelHostname, Operator: v1.NodeSelectorOpIn, Values: []string{"node-1"}},
							{Key: "zone", Operator: v1.NodeSelectorOpIn, Values: []string{"z1"}},
						},
						MatchFields: []v1.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{"node-1"}},
						},
					},
				},
			},
		},
	}

	tmpl := SanitizePodTemplate(pod)

	if _, ok := tmpl.Spec.NodeSelector[v1.LabelHostname]; ok {
		t.Errorf("hostname nodeSelector not stripped: %+v", tmpl.Spec.NodeSelector)
	}
	if tmpl.Spec.NodeSelector["disktype"] != "ssd" {
		t.Errorf("non-identity nodeSelector not preserved: %+v", tmpl.Spec.NodeSelector)
	}

	req := tmpl.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution
	if req == nil || len(req.NodeSelectorTerms) != 1 {
		t.Fatalf("expected one node selector term, got %+v", req)
	}
	term := req.NodeSelectorTerms[0]
	if len(term.MatchFields) != 0 {
		t.Errorf("metadata.name matchField not stripped: %+v", term.MatchFields)
	}
	if len(term.MatchExpressions) != 1 || term.MatchExpressions[0].Key != "zone" {
		t.Errorf("expected only the zone match expression to remain, got %+v", term.MatchExpressions)
	}
}

func TestSanitizePodTemplate_DoesNotMutateSource(t *testing.T) {
	pod := sourcePod()
	_ = SanitizePodTemplate(pod)

	if pod.Spec.NodeName != "node-1" {
		t.Errorf("source pod spec.NodeName was mutated: %q", pod.Spec.NodeName)
	}
	if pod.Spec.RestoreFrom == nil {
		t.Errorf("source pod spec.RestoreFrom was mutated (cleared)")
	} else {
		if pod.Spec.RestoreFrom.Name != "ckpt" {
			t.Errorf("source pod spec.RestoreFrom.Name was mutated: %q", pod.Spec.RestoreFrom.Name)
		}
		if pod.Spec.RestoreFrom.Options["example.runtime/target"] != "new-node" {
			t.Errorf("source pod spec.RestoreFrom.Options was mutated: %v", pod.Spec.RestoreFrom.Options)
		}
	}
	if pod.UID != types.UID("abc-123") {
		t.Errorf("source pod UID was mutated: %q", pod.UID)
	}
	if len(pod.ManagedFields) != 1 {
		t.Errorf("source pod ManagedFields was mutated: %+v", pod.ManagedFields)
	}
}

func TestSanitizePodTemplate_DropsAffinityCreatedOnlyForRestoreNode(t *testing.T) {
	pod := sourcePod()
	pod.Spec.Affinity = &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{{
					MatchFields: []v1.NodeSelectorRequirement{{
						Key:      "metadata.name",
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{"node-1"},
					}},
				}},
			},
		},
	}

	tmpl := SanitizePodTemplate(pod)
	if tmpl.Spec.Affinity != nil {
		t.Fatalf("expected empty affinity wrappers to be removed, got %+v", tmpl.Spec.Affinity)
	}
}

func TestSanitizePodTemplate_DropsEmptyAffinity(t *testing.T) {
	pod := sourcePod()
	pod.Spec.Affinity = &v1.Affinity{}

	tmpl := SanitizePodTemplate(pod)
	if tmpl.Spec.Affinity != nil {
		t.Fatalf("expected empty affinity to be removed, got %+v", tmpl.Spec.Affinity)
	}
}

func TestSanitizePodTemplate_NilPod(t *testing.T) {
	if got := SanitizePodTemplate(nil); got != nil {
		t.Errorf("expected nil for nil pod, got %+v", got)
	}
}
