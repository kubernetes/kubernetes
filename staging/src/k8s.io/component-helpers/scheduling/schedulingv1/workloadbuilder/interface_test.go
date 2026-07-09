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
	"strings"
	"testing"

	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestNewBuilder(t *testing.T) {
	item := createGangWorkloadItem()
	opts := BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()}

	b := NewBuilder(item, opts)
	if b.root != item {
		t.Errorf("root not stored: got %p want %p", b.root, item)
	}
	if b.opts.Name != "wl" || b.opts.Namespace != "ns" || b.opts.Owner == nil {
		t.Errorf("opts not stored: %+v", b.opts)
	}
	if b.workload != nil {
		t.Error("workload should be nil before BuildWorkload")
	}
}

func TestBuilderBuildWorkload(t *testing.T) {
	t.Run("compiles identity and controllerRef", func(t *testing.T) {
		b := NewBuilder(createGangWorkloadItem(), BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()})
		wl, err := b.BuildWorkload()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if wl.Name != "wl" || wl.Namespace != "ns" {
			t.Errorf("unexpected identity: %s/%s", wl.Namespace, wl.Name)
		}
		if len(wl.OwnerReferences) != 1 || wl.OwnerReferences[0].Name != "job" {
			t.Errorf("expected single job ownerReference, got %+v", wl.OwnerReferences)
		}
		if wl.Spec.ControllerRef == nil ||
			wl.Spec.ControllerRef.APIGroup != "batch" ||
			wl.Spec.ControllerRef.Kind != "Job" ||
			wl.Spec.ControllerRef.Name != "job" {
			t.Errorf("unexpected controllerRef: %+v", wl.Spec.ControllerRef)
		}
		if len(wl.Spec.PodGroupTemplates) != 1 || wl.Spec.PodGroupTemplates[0].Name != "pgt-0" {
			t.Errorf("expected single template pgt-0, got %+v", wl.Spec.PodGroupTemplates)
		}
	})

	t.Run("caches the compiled workload", func(t *testing.T) {
		b := NewBuilder(createGangWorkloadItem(), BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()})
		first, err := b.BuildWorkload()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		second, err := b.BuildWorkload()
		if err != nil {
			t.Fatalf("unexpected error on second call: %v", err)
		}
		if first != second {
			t.Error("expected BuildWorkload to return the cached workload on repeat calls")
		}
	})

	t.Run("errors are returned and nothing is cached", func(t *testing.T) {
		tests := []struct {
			name string
			b    *Builder
		}{
			{
				name: "nil root",
				b:    NewBuilder(nil, BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()}),
			},
			{
				name: "missing owner",
				b:    NewBuilder(createGangWorkloadItem(), BuildOptions{Name: "wl", Namespace: "ns"}),
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				wl, err := tt.b.BuildWorkload()
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if wl != nil {
					t.Errorf("expected nil workload on error, got %+v", wl)
				}
				if tt.b.workload != nil {
					t.Error("failed build must not populate the cache")
				}
			})
		}
	})
}

func TestBuilderNewPodGroup(t *testing.T) {
	t.Run("requires a workload first", func(t *testing.T) {
		b := NewBuilder(createGangWorkloadItem(), BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()})
		_, err := b.NewPodGroup("pg", "pgt-0")
		if err == nil {
			t.Fatal("expected error when neither BuildWorkload nor SetExistingWorkload has run")
		}
		if !strings.Contains(err.Error(), "call BuildWorkload or SetExistingWorkload") {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("materializes from the compiled template owned by the configured owner", func(t *testing.T) {
		b := NewBuilder(createGangWorkloadItem(), BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()})
		if _, err := b.BuildWorkload(); err != nil {
			t.Fatalf("unexpected build error: %v", err)
		}
		pg, err := b.NewPodGroup("pg", "pgt-0")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if pg.Name != "pg" || pg.Namespace != "ns" {
			t.Errorf("unexpected identity: %s/%s", pg.Namespace, pg.Name)
		}
		if len(pg.OwnerReferences) != 1 || pg.OwnerReferences[0].Name != "job" {
			t.Errorf("expected single job ownerReference from opts.Owner, got %+v", pg.OwnerReferences)
		}
		if pg.Spec.WorkloadRef == nil ||
			pg.Spec.WorkloadRef.WorkloadName != "wl" ||
			pg.Spec.WorkloadRef.TemplateName != "pgt-0" {
			t.Errorf("unexpected workloadRef: %+v", pg.Spec.WorkloadRef)
		}
	})

	t.Run("omits ownerReferences when no owner is configured", func(t *testing.T) {
		// build() requires an owner, so compile with one, then clear it to
		// exercise the ownerless NewPodGroup path against the cached workload.
		b := NewBuilder(createGangWorkloadItem(), BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()})
		if _, err := b.BuildWorkload(); err != nil {
			t.Fatalf("unexpected build error: %v", err)
		}
		b.opts.Owner = nil
		pg, err := b.NewPodGroup("pg", "pgt-0")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(pg.OwnerReferences) != 0 {
			t.Errorf("expected no ownerReferences, got %+v", pg.OwnerReferences)
		}
	})

	t.Run("errors on unknown template", func(t *testing.T) {
		b := NewBuilder(createGangWorkloadItem(), BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()})
		if _, err := b.BuildWorkload(); err != nil {
			t.Fatalf("unexpected build error: %v", err)
		}
		if _, err := b.NewPodGroup("pg", "missing"); err == nil {
			t.Error("expected error for unknown template name")
		}
	})
}

func TestSetExistingWorkload(t *testing.T) {
	t.Run("materializes from the supplied workload without BuildWorkload", func(t *testing.T) {
		b := NewBuilder(createGangWorkloadItem(), BuildOptions{Owner: jobOwner()})
		b.SetExistingWorkload(existingWorkload())
		pg, err := b.NewPodGroup("pg", "pgt-0")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if pg.Name != "pg" || pg.Namespace != "ns" {
			t.Errorf("unexpected identity: %s/%s", pg.Namespace, pg.Name)
		}
		if len(pg.OwnerReferences) != 1 || pg.OwnerReferences[0].Name != "job" {
			t.Errorf("expected single job ownerReference, got %+v", pg.OwnerReferences)
		}
		// Fields the builder never sets must be copied from the supplied template.
		if pg.Spec.PriorityClassName != "high-priority" {
			t.Errorf("expected priorityClassName copied, got %q", pg.Spec.PriorityClassName)
		}
		if pg.Spec.Priority == nil || *pg.Spec.Priority != 1000 {
			t.Errorf("expected priority copied, got %v", pg.Spec.Priority)
		}
		if pg.Spec.WorkloadRef == nil || pg.Spec.WorkloadRef.WorkloadName != "parent-wl" {
			t.Errorf("unexpected workloadRef: %+v", pg.Spec.WorkloadRef)
		}
	})

	t.Run("BuildWorkload is refused in existing mode", func(t *testing.T) {
		b := NewBuilder(createGangWorkloadItem(), BuildOptions{Name: "wl", Namespace: "ns", Owner: jobOwner()})
		b.SetExistingWorkload(existingWorkload())
		_, err := b.BuildWorkload()
		if err == nil {
			t.Fatal("expected BuildWorkload to be refused after SetExistingWorkload")
		}
		if !strings.Contains(err.Error(), "cannot be used after SetExistingWorkload") {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("errors on unknown template", func(t *testing.T) {
		b := NewBuilder(createGangWorkloadItem(), BuildOptions{Owner: jobOwner()})
		b.SetExistingWorkload(existingWorkload())
		if _, err := b.NewPodGroup("pg", "missing"); err == nil {
			t.Error("expected error for unknown template name")
		}
	})
}

func jobOwner() *metav1.OwnerReference {
	return &metav1.OwnerReference{APIVersion: "batch/v1", Kind: "Job", Name: "job", UID: "job-uid"}
}

func createGangWorkloadItem() *WorkloadItem {
	return &WorkloadItem{
		Name:       "pgt-0",
		UserConfig: &SchedulingConfig{Policy: &SchedulingPolicy{Gang: &GangSchedulingPolicy{MinCount: ptr.To[int32](3)}}},
	}
}

func existingWorkload() *schedulingv1alpha3.Workload {
	return &schedulingv1alpha3.Workload{
		ObjectMeta: metav1.ObjectMeta{Name: "parent-wl", Namespace: "ns", UID: "wl-uid"},
		Spec: schedulingv1alpha3.WorkloadSpec{
			PodGroupTemplates: []schedulingv1alpha3.PodGroupTemplate{{
				Name:              "pgt-0",
				SchedulingPolicy:  schedulingv1alpha3.PodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.GangSchedulingPolicy{MinCount: 4}},
				PriorityClassName: "high-priority",
				Priority:          ptr.To[int32](1000),
			}},
		},
	}
}
