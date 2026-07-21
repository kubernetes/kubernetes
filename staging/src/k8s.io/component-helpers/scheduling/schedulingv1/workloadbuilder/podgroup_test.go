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

package workloadbuilder

import (
	"testing"

	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func gangWorkload() *schedulingv1beta1.Workload {
	return &schedulingv1beta1.Workload{
		ObjectMeta: metav1.ObjectMeta{Name: "job-abc", Namespace: "training"},
		Spec: schedulingv1beta1.WorkloadSpec{
			PodGroupTemplates: []schedulingv1beta1.PodGroupTemplate{
				{
					Name:             "workers",
					SchedulingPolicy: schedulingv1beta1.PodGroupSchedulingPolicy{Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: 4}},
					SchedulingConstraints: &schedulingv1beta1.PodGroupSchedulingConstraints{
						Topology: []schedulingv1beta1.TopologyConstraint{{Key: "topology.kubernetes.io/zone"}},
					},
					DisruptionMode:    &schedulingv1beta1.DisruptionMode{All: &schedulingv1beta1.AllDisruptionMode{}},
					ResourceClaims:    []schedulingv1beta1.PodGroupResourceClaim{{Name: "gpu", ResourceClaimName: new("shared-gpu")}},
					PriorityClassName: "high",
					Priority:          ptr.To[int32](1000),
				},
			},
		},
	}
}

func TestNewPodGroup(t *testing.T) {
	wl := gangWorkload()
	owners := []metav1.OwnerReference{
		{APIVersion: "batch/v1", Kind: "Job", Name: "job", Controller: new(true)},
		{APIVersion: "scheduling.k8s.io/v1beta1", Kind: "Workload", Name: wl.Name},
	}

	pg, err := newPodGroup(wl, "workers", "job-abc-workers-xyz", owners)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if pg.Name != "job-abc-workers-xyz" {
		t.Errorf("unexpected name: %q", pg.Name)
	}
	if pg.Namespace != "training" {
		t.Errorf("expected namespace inherited from workload, got %q", pg.Namespace)
	}
	if len(pg.OwnerReferences) != 2 {
		t.Errorf("expected 2 owner references, got %d", len(pg.OwnerReferences))
	}

	ref := pg.Spec.WorkloadRef
	if ref == nil || ref.WorkloadName == "" {
		t.Fatal("expected podGroupTemplateRef.workload to be set")
	}
	if ref.WorkloadName != "job-abc" || ref.TemplateName != "workers" {
		t.Errorf("unexpected template ref: %+v", ref)
	}

	if pg.Spec.SchedulingPolicy.Gang == nil || pg.Spec.SchedulingPolicy.Gang.MinCount != 4 {
		t.Error("expected gang policy with MinCount=4 copied from template")
	}
	if pg.Spec.SchedulingConstraints == nil || len(pg.Spec.SchedulingConstraints.Topology) != 1 {
		t.Error("expected topology constraint copied from template")
	}
	if pg.Spec.DisruptionMode == nil || pg.Spec.DisruptionMode.All == nil {
		t.Error("expected All disruption mode copied from template")
	}
	if len(pg.Spec.ResourceClaims) != 1 || pg.Spec.ResourceClaims[0].Name != "gpu" {
		t.Error("expected resource claim copied from template")
	}
	if pg.Spec.PriorityClassName != "high" {
		t.Errorf("expected priorityClassName copied, got %q", pg.Spec.PriorityClassName)
	}
	if pg.Spec.Priority == nil || *pg.Spec.Priority != 1000 {
		t.Error("expected priority copied from template")
	}
}

func TestNewPodGroupDoesNotAliasWorkload(t *testing.T) {
	wl := gangWorkload()
	pg, err := newPodGroup(wl, "workers", "pg", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Mutating the PodGroup must not leak back into the Workload template.
	pg.Spec.SchedulingPolicy.Gang.MinCount = 99
	pg.Spec.SchedulingConstraints.Topology[0].Key = "changed"
	pg.Spec.ResourceClaims[0].Name = "changed"
	*pg.Spec.Priority = 1

	tmpl := wl.Spec.PodGroupTemplates[0]
	if tmpl.SchedulingPolicy.Gang.MinCount != 4 {
		t.Error("template MinCount mutated through the PodGroup")
	}
	if tmpl.SchedulingConstraints.Topology[0].Key != "topology.kubernetes.io/zone" {
		t.Error("template topology mutated through the PodGroup")
	}
	if tmpl.ResourceClaims[0].Name != "gpu" {
		t.Error("template resource claim mutated through the PodGroup")
	}
	if *tmpl.Priority != 1000 {
		t.Error("template priority mutated through the PodGroup")
	}
}

func TestNewPodGroupErrors(t *testing.T) {
	if _, err := newPodGroup(nil, "workers", "pg", nil); err == nil {
		t.Error("expected error for nil workload")
	}
	if _, err := newPodGroup(gangWorkload(), "missing", "pg", nil); err == nil {
		t.Error("expected error for unknown template name")
	}
}
