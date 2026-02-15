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

package job

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	schedulingv1alpha2 "k8s.io/api/scheduling/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func TestIsGangSchedulingEligible(t *testing.T) {
	indexed := batch.IndexedCompletion
	nonIndexed := batch.NonIndexedCompletion

	tests := map[string]struct {
		job  *batch.Job
		want bool
	}{
		"eligible: indexed, parallelism=completions>1, no schedulingGroup": {
			job: &batch.Job{
				Spec: batch.JobSpec{
					Parallelism:    ptr.To[int32](4),
					Completions:    ptr.To[int32](4),
					CompletionMode: &indexed,
				},
			},
			want: true,
		},
		"not eligible: parallelism is nil": {
			job: &batch.Job{
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](4),
					CompletionMode: &indexed,
				},
			},
			want: false,
		},
		"not eligible: parallelism=1": {
			job: &batch.Job{
				Spec: batch.JobSpec{
					Parallelism:    ptr.To[int32](1),
					Completions:    ptr.To[int32](1),
					CompletionMode: &indexed,
				},
			},
			want: false,
		},
		"not eligible: non-indexed completion mode": {
			job: &batch.Job{
				Spec: batch.JobSpec{
					Parallelism:    ptr.To[int32](4),
					Completions:    ptr.To[int32](4),
					CompletionMode: &nonIndexed,
				},
			},
			want: false,
		},
		"not eligible: completions is nil": {
			job: &batch.Job{
				Spec: batch.JobSpec{
					Parallelism:    ptr.To[int32](4),
					CompletionMode: &indexed,
				},
			},
			want: false,
		},
		"not eligible: completions != parallelism": {
			job: &batch.Job{
				Spec: batch.JobSpec{
					Parallelism:    ptr.To[int32](4),
					Completions:    ptr.To[int32](8),
					CompletionMode: &indexed,
				},
			},
			want: false,
		},
		"not eligible: schedulingGroup already set (opt-out)": {
			job: &batch.Job{
				Spec: batch.JobSpec{
					Parallelism:    ptr.To[int32](4),
					Completions:    ptr.To[int32](4),
					CompletionMode: &indexed,
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							SchedulingGroup: &v1.PodSchedulingGroup{
								PodGroupName: ptr.To("existing-pg"),
							},
						},
					},
				},
			},
			want: false,
		},
		"not eligible: completionMode is nil": {
			job: &batch.Job{
				Spec: batch.JobSpec{
					Parallelism: ptr.To[int32](4),
					Completions: ptr.To[int32](4),
				},
			},
			want: false,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			got := isGangSchedulingEligible(tc.job)
			if got != tc.want {
				t.Errorf("isGangSchedulingEligible() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestBuildPodGroupTemplates(t *testing.T) {
	tests := map[string]struct {
		jobName      string
		parallelism  int32
		completions  int32
		wantName     string
		wantMinCount int32
	}{
		"basic job": {
			jobName:      "my-training-job",
			parallelism:  8,
			completions:  8,
			wantName:     "my-training-job-worker-0",
			wantMinCount: 8,
		},
		"single template with small parallelism": {
			jobName:      "small-job",
			parallelism:  2,
			completions:  2,
			wantName:     "small-job-worker-0",
			wantMinCount: 2,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			job := &batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: tc.jobName},
				Spec: batch.JobSpec{
					Parallelism: ptr.To(tc.parallelism),
					Completions: ptr.To(tc.completions),
				},
			}
			templates := buildPodGroupTemplates(job)

			if len(templates) != 1 {
				t.Fatalf("expected 1 template, got %d", len(templates))
			}
			tpl := templates[0]
			if tpl.Name != tc.wantName {
				t.Errorf("template name = %q, want %q", tpl.Name, tc.wantName)
			}
			if tpl.SchedulingPolicy.Gang == nil {
				t.Fatal("expected gang scheduling policy, got nil")
			}
			if tpl.SchedulingPolicy.Gang.MinCount != tc.wantMinCount {
				t.Errorf("minCount = %d, want %d", tpl.SchedulingPolicy.Gang.MinCount, tc.wantMinCount)
			}
		})
	}
}

func TestComputeWorkloadName(t *testing.T) {
	tests := map[string]struct {
		jobName string
		jobUID  types.UID
	}{
		"short name": {
			jobName: "my-job",
			jobUID:  "uid-123",
		},
		"very long name exceeding DNS limit": {
			jobName: strings.Repeat("a", 300),
			jobUID:  "uid-long",
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			job := &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name: tc.jobName,
					UID:  tc.jobUID,
				},
			}
			result := computeWorkloadName(job)

			// Must fit within DNS subdomain length.
			if len(result) > validation.DNS1123SubdomainMaxLength {
				t.Errorf("name length %d exceeds DNS1123SubdomainMaxLength %d", len(result), validation.DNS1123SubdomainMaxLength)
			}

			// Must contain a hash suffix (separated by "-").
			parts := strings.Split(result, "-")
			if len(parts) < 2 {
				t.Errorf("expected at least 2 parts (prefix-hash), got %q", result)
			}

			// Must be deterministic.
			result2 := computeWorkloadName(job)
			if result != result2 {
				t.Errorf("non-deterministic: %q != %q", result, result2)
			}

			// Different UIDs must produce different names.
			job2 := job.DeepCopy()
			job2.UID = "different-uid"
			result3 := computeWorkloadName(job2)
			if result == result3 {
				t.Error("different UIDs should produce different names")
			}
		})
	}
}

func TestComputePodGroupName(t *testing.T) {
	tests := map[string]struct {
		workloadName string
		templateName string
	}{
		"short names": {
			workloadName: "my-workload",
			templateName: "my-template",
		},
		"long workload name": {
			workloadName: strings.Repeat("w", 250),
			templateName: "tpl",
		},
		"long template name": {
			workloadName: "wl",
			templateName: strings.Repeat("t", 250),
		},
		"both long names": {
			workloadName: strings.Repeat("w", 200),
			templateName: strings.Repeat("t", 200),
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			result := computePodGroupName(tc.workloadName, tc.templateName)

			// Must fit within DNS subdomain length.
			if len(result) > validation.DNS1123SubdomainMaxLength {
				t.Errorf("name length %d exceeds DNS1123SubdomainMaxLength %d", len(result), validation.DNS1123SubdomainMaxLength)
			}

			// Must contain the hash (at least 3 parts: wl-tpl-hash).
			parts := strings.Split(result, "-")
			if len(parts) < 3 {
				t.Errorf("expected at least 3 parts (wl-tpl-hash), got %q", result)
			}

			// Must be deterministic.
			result2 := computePodGroupName(tc.workloadName, tc.templateName)
			if result != result2 {
				t.Errorf("non-deterministic: %q != %q", result, result2)
			}

			// Different inputs must produce different names.
			result3 := computePodGroupName(tc.workloadName, "other-template")
			if result == result3 {
				t.Error("different template names should produce different PodGroup names")
			}
		})
	}
}

func TestComputePodGroupName_Truncation(t *testing.T) {
	// When both names are long, each gets roughly half the available space.
	longWL := strings.Repeat("w", 200)
	longTPL := strings.Repeat("t", 200)
	result := computePodGroupName(longWL, longTPL)

	if len(result) > validation.DNS1123SubdomainMaxLength {
		t.Errorf("name length %d exceeds max %d", len(result), validation.DNS1123SubdomainMaxLength)
	}

	// The result should contain truncated versions of both parts.
	// Split to check: the last segment is the hash.
	lastDash := strings.LastIndex(result, "-")
	if lastDash < 0 {
		t.Fatalf("expected at least one dash in %q", result)
	}
	prefix := result[:lastDash]
	// prefix should contain both truncated wl and tpl parts.
	if !strings.Contains(prefix, "www") {
		t.Errorf("expected truncated workload name in prefix %q", prefix)
	}
	if !strings.Contains(prefix, "ttt") {
		t.Errorf("expected truncated template name in prefix %q", prefix)
	}

	// When one name is short, it should not be truncated.
	shortWL := "wl"
	result2 := computePodGroupName(shortWL, longTPL)
	if !strings.HasPrefix(result2, "wl-") {
		t.Errorf("short workload name should not be truncated, got %q", result2)
	}
	if len(result2) > validation.DNS1123SubdomainMaxLength {
		t.Errorf("name length %d exceeds max %d", len(result2), validation.DNS1123SubdomainMaxLength)
	}
}

// newGangSchedulingJob creates a Job that meets gang scheduling eligibility criteria.
func newGangSchedulingJob(name string, parallelism int32) *batch.Job {
	indexed := batch.IndexedCompletion
	return &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceDefault,
			UID:       types.UID(name + "-uid"),
		},
		Spec: batch.JobSpec{
			Parallelism:    ptr.To(parallelism),
			Completions:    ptr.To(parallelism),
			CompletionMode: &indexed,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"job-name": name},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"job-name": name},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Image: "test"}},
				},
			},
		},
	}
}

// newControllerWithSchedulingInformers creates a Job controller with Workload and PodGroup
// informers wired up, suitable for testing scheduling manager functionality.
func newControllerWithSchedulingInformers(ctx context.Context, t *testing.T, kubeClient *fake.Clientset) (*Controller, informers.SharedInformerFactory) {
	t.Helper()
	sharedInformers := informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())
	jm, err := newControllerWithClock(ctx,
		sharedInformers.Core().V1().Pods(),
		sharedInformers.Batch().V1().Jobs(),
		kubeClient,
		realClock,
		sharedInformers.Scheduling().V1alpha2().Workloads(),
		sharedInformers.Scheduling().V1alpha2().PodGroups(),
	)
	if err != nil {
		t.Fatalf("Error creating Job controller: %v", err)
	}
	jm.podControl = &controller.FakePodControl{}
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady
	jm.workloadStoreSynced = alwaysReady
	jm.podGroupStoreSynced = alwaysReady
	return jm, sharedInformers
}

func TestEnsureWorkloadAndPodGroup_NotEligible(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	clientSet := fake.NewClientset()
	jm, _ := newControllerWithSchedulingInformers(ctx, t, clientSet)

	// Job with parallelism=1 is not eligible.
	job := newJob(1, 1, 0, batch.IndexedCompletion)
	pg, err := jm.ensureWorkloadAndPodGroup(ctx, job, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pg != nil {
		t.Errorf("expected nil PodGroup for non-eligible Job, got %v", pg)
	}
}

func TestEnsureWorkloadAndPodGroup_CreatesObjects(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	clientSet := fake.NewClientset()
	jm, _ := newControllerWithSchedulingInformers(ctx, t, clientSet)

	job := newGangSchedulingJob("test-job", 4)

	// With no pods and no existing objects, it should create both Workload and PodGroup.
	pg, err := jm.ensureWorkloadAndPodGroup(ctx, job, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pg == nil {
		t.Fatal("expected non-nil PodGroup")
	}

	// Verify the Workload was created.
	workloadName := computeWorkloadName(job)
	wl, err := clientSet.SchedulingV1alpha2().Workloads(job.Namespace).Get(ctx, workloadName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("expected Workload to be created, got error: %v", err)
	}

	// Verify Workload spec.
	if wl.Spec.ControllerRef == nil {
		t.Fatal("Workload should have controllerRef")
	}
	if wl.Spec.ControllerRef.Name != job.Name {
		t.Errorf("Workload controllerRef.Name = %q, want %q", wl.Spec.ControllerRef.Name, job.Name)
	}
	if len(wl.Spec.PodGroupTemplates) != 1 {
		t.Fatalf("expected 1 PodGroupTemplate, got %d", len(wl.Spec.PodGroupTemplates))
	}
	if wl.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang == nil {
		t.Fatal("expected gang scheduling policy")
	}
	if wl.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount != 4 {
		t.Errorf("minCount = %d, want 4", wl.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount)
	}

	// Verify Workload ownerRef.
	if len(wl.OwnerReferences) != 1 {
		t.Fatalf("expected 1 ownerReference, got %d", len(wl.OwnerReferences))
	}
	if wl.OwnerReferences[0].Kind != "Job" {
		t.Errorf("ownerRef kind = %q, want Job", wl.OwnerReferences[0].Kind)
	}
	if wl.OwnerReferences[0].Name != job.Name {
		t.Errorf("ownerRef name = %q, want %q", wl.OwnerReferences[0].Name, job.Name)
	}

	// Verify the PodGroup was created.
	pgName := computePodGroupName(wl.Name, wl.Spec.PodGroupTemplates[0].Name)
	createdPG, err := clientSet.SchedulingV1alpha2().PodGroups(job.Namespace).Get(ctx, pgName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("expected PodGroup to be created, got error: %v", err)
	}

	// Verify PodGroup spec.
	if createdPG.Spec.PodGroupTemplateRef == nil {
		t.Fatal("PodGroup should have podGroupTemplateRef")
	}
	if createdPG.Spec.PodGroupTemplateRef.WorkloadName != wl.Name {
		t.Errorf("PodGroup templateRef.workloadName = %q, want %q", createdPG.Spec.PodGroupTemplateRef.WorkloadName, wl.Name)
	}

	// Verify PodGroup has two ownerRefs (Job + Workload).
	if len(createdPG.OwnerReferences) != 2 {
		t.Fatalf("expected 2 ownerReferences, got %d", len(createdPG.OwnerReferences))
	}
	var hasJobOwner, hasWorkloadOwner bool
	for _, ref := range createdPG.OwnerReferences {
		if ref.Kind == "Job" && ref.Name == job.Name {
			hasJobOwner = true
		}
		if ref.Kind == "Workload" && ref.Name == wl.Name {
			hasWorkloadOwner = true
		}
	}
	if !hasJobOwner {
		t.Error("PodGroup missing Job ownerReference")
	}
	if !hasWorkloadOwner {
		t.Error("PodGroup missing Workload ownerReference")
	}
}

func TestEnsureWorkloadAndPodGroup_SkipsWhenHasPods(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	clientSet := fake.NewClientset()
	jm, _ := newControllerWithSchedulingInformers(ctx, t, clientSet)

	job := newGangSchedulingJob("test-job", 4)
	existingPods := []*v1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "pod-0", Namespace: job.Namespace}}}

	// With existing pods but no Workload, should return nil (don't create late).
	pg, err := jm.ensureWorkloadAndPodGroup(ctx, job, existingPods)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pg != nil {
		t.Errorf("expected nil PodGroup when Job has pods but no Workload, got %v", pg)
	}
}

func TestEnsureWorkloadAndPodGroup_DiscoversExistingObjects(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	job := newGangSchedulingJob("test-job", 4)

	// Pre-create Workload and PodGroup.
	workloadName := computeWorkloadName(job)
	templateName := fmt.Sprintf("%s-worker-0", job.Name)
	podGroupName := computePodGroupName(workloadName, templateName)

	existingWorkload := &schedulingv1alpha2.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      workloadName,
			Namespace: job.Namespace,
			UID:       "workload-uid",
		},
		Spec: schedulingv1alpha2.WorkloadSpec{
			ControllerRef: &schedulingv1alpha2.TypedLocalObjectReference{
				APIGroup: "batch",
				Kind:     "Job",
				Name:     job.Name,
			},
			PodGroupTemplates: []schedulingv1alpha2.PodGroupTemplate{
				{
					Name: templateName,
					SchedulingPolicy: schedulingv1alpha2.PodGroupSchedulingPolicy{
						Gang: &schedulingv1alpha2.GangSchedulingPolicy{
							MinCount: 4,
						},
					},
				},
			},
		},
	}
	existingPG := &schedulingv1alpha2.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podGroupName,
			Namespace: job.Namespace,
			UID:       "podgroup-uid",
		},
		Spec: schedulingv1alpha2.PodGroupSpec{
			PodGroupTemplateRef: &schedulingv1alpha2.PodGroupTemplateReference{
				WorkloadName:         workloadName,
				PodGroupTemplateName: templateName,
			},
		},
	}

	clientSet := fake.NewClientset(existingWorkload, existingPG)
	jm, sharedInformers := newControllerWithSchedulingInformers(ctx, t, clientSet)

	// Populate the informer caches.
	sharedInformers.Start(ctx.Done())
	sharedInformers.WaitForCacheSync(ctx.Done())

	// Should discover the existing objects instead of creating new ones.
	pg, err := jm.ensureWorkloadAndPodGroup(ctx, job, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pg == nil {
		t.Fatal("expected non-nil PodGroup")
	}
	if pg.Name != podGroupName {
		t.Errorf("PodGroup name = %q, want %q", pg.Name, podGroupName)
	}
}

func TestEnsureWorkload_UnsupportedTemplateCount(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	job := newGangSchedulingJob("test-job", 4)
	workloadName := computeWorkloadName(job)

	// Pre-create a Workload with 2 PodGroupTemplates (unsupported in alpha).
	existingWorkload := &schedulingv1alpha2.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      workloadName,
			Namespace: job.Namespace,
		},
		Spec: schedulingv1alpha2.WorkloadSpec{
			ControllerRef: &schedulingv1alpha2.TypedLocalObjectReference{
				APIGroup: "batch",
				Kind:     "Job",
				Name:     job.Name,
			},
			PodGroupTemplates: []schedulingv1alpha2.PodGroupTemplate{
				{Name: "group-a"},
				{Name: "group-b"},
			},
		},
	}

	clientSet := fake.NewClientset(existingWorkload)
	jm, sharedInformers := newControllerWithSchedulingInformers(ctx, t, clientSet)
	sharedInformers.Start(ctx.Done())
	sharedInformers.WaitForCacheSync(ctx.Done())

	// Should return nil without error (falls back to normal scheduling).
	wl, err := jm.ensureWorkload(ctx, job, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if wl != nil {
		t.Errorf("expected nil Workload for unsupported template count, got %v", wl)
	}
}

func TestCreateWorkloadForJob(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	clientSet := fake.NewClientset()
	jm, _ := newControllerWithSchedulingInformers(ctx, t, clientSet)

	job := newGangSchedulingJob("my-job", 8)

	wl, err := jm.createWorkloadForJob(ctx, job)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedName := computeWorkloadName(job)
	if wl.Name != expectedName {
		t.Errorf("workload name = %q, want %q", wl.Name, expectedName)
	}
	if wl.Namespace != job.Namespace {
		t.Errorf("workload namespace = %q, want %q", wl.Namespace, job.Namespace)
	}

	// Verify controllerRef.
	if wl.Spec.ControllerRef == nil {
		t.Fatal("expected controllerRef")
	}
	if diff := cmp.Diff(wl.Spec.ControllerRef, &schedulingv1alpha2.TypedLocalObjectReference{
		APIGroup: "batch",
		Kind:     "Job",
		Name:     job.Name,
	}); diff != "" {
		t.Errorf("unexpected controllerRef (-got +want):\n%s", diff)
	}

	// Verify templates.
	if len(wl.Spec.PodGroupTemplates) != 1 {
		t.Fatalf("expected 1 template, got %d", len(wl.Spec.PodGroupTemplates))
	}
	tpl := wl.Spec.PodGroupTemplates[0]
	if tpl.SchedulingPolicy.Gang == nil || tpl.SchedulingPolicy.Gang.MinCount != 8 {
		t.Errorf("expected gang minCount=8, got %+v", tpl.SchedulingPolicy)
	}

	// Verify ownerRef.
	if len(wl.OwnerReferences) != 1 || wl.OwnerReferences[0].Name != job.Name {
		t.Errorf("unexpected ownerReferences: %v", wl.OwnerReferences)
	}
}

func TestCreatePodGroupForWorkload(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	clientSet := fake.NewClientset()
	jm, _ := newControllerWithSchedulingInformers(ctx, t, clientSet)

	job := newGangSchedulingJob("my-job", 4)
	templateName := fmt.Sprintf("%s-worker-0", job.Name)

	workload := &schedulingv1alpha2.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-workload",
			Namespace: job.Namespace,
			UID:       "wl-uid",
		},
		Spec: schedulingv1alpha2.WorkloadSpec{
			PodGroupTemplates: []schedulingv1alpha2.PodGroupTemplate{
				{
					Name: templateName,
					SchedulingPolicy: schedulingv1alpha2.PodGroupSchedulingPolicy{
						Gang: &schedulingv1alpha2.GangSchedulingPolicy{
							MinCount: 4,
						},
					},
				},
			},
		},
	}

	pg, err := jm.createPodGroupForWorkload(ctx, job, workload)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedName := computePodGroupName(workload.Name, templateName)
	if pg.Name != expectedName {
		t.Errorf("PodGroup name = %q, want %q", pg.Name, expectedName)
	}

	// Verify templateRef.
	if pg.Spec.PodGroupTemplateRef == nil {
		t.Fatal("expected podGroupTemplateRef")
	}
	if pg.Spec.PodGroupTemplateRef.WorkloadName != workload.Name {
		t.Errorf("templateRef.workloadName = %q, want %q", pg.Spec.PodGroupTemplateRef.WorkloadName, workload.Name)
	}
	if pg.Spec.PodGroupTemplateRef.PodGroupTemplateName != templateName {
		t.Errorf("templateRef.podGroupTemplateName = %q, want %q", pg.Spec.PodGroupTemplateRef.PodGroupTemplateName, templateName)
	}

	// Verify scheduling policy was copied.
	if pg.Spec.SchedulingPolicy.Gang == nil || pg.Spec.SchedulingPolicy.Gang.MinCount != 4 {
		t.Errorf("expected copied gang policy with minCount=4, got %+v", pg.Spec.SchedulingPolicy)
	}

	// Verify two ownerRefs.
	if len(pg.OwnerReferences) != 2 {
		t.Fatalf("expected 2 ownerReferences, got %d", len(pg.OwnerReferences))
	}
}

func TestCreatePodGroupForWorkload_NoTemplates(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	clientSet := fake.NewClientset()
	jm, _ := newControllerWithSchedulingInformers(ctx, t, clientSet)

	job := newGangSchedulingJob("my-job", 4)
	workload := &schedulingv1alpha2.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-workload",
			Namespace: job.Namespace,
		},
		Spec: schedulingv1alpha2.WorkloadSpec{},
	}

	_, err := jm.createPodGroupForWorkload(ctx, job, workload)
	if err == nil {
		t.Fatal("expected error for Workload with no templates")
	}
}

func TestDiscoverWorkloadForJob(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	job := newGangSchedulingJob("test-job", 4)

	t.Run("not found returns nil", func(t *testing.T) {
		clientSet := fake.NewClientset()
		jm, sharedInformers := newControllerWithSchedulingInformers(ctx, t, clientSet)
		sharedInformers.Start(ctx.Done())
		sharedInformers.WaitForCacheSync(ctx.Done())

		wl, err := jm.discoverWorkloadForJob(ctx, job)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if wl != nil {
			t.Errorf("expected nil, got %v", wl)
		}
	})

	t.Run("found returns workload", func(t *testing.T) {
		workloadName := computeWorkloadName(job)
		existing := &schedulingv1alpha2.Workload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      workloadName,
				Namespace: job.Namespace,
			},
		}
		clientSet := fake.NewClientset(existing)
		jm, sharedInformers := newControllerWithSchedulingInformers(ctx, t, clientSet)
		sharedInformers.Start(ctx.Done())
		sharedInformers.WaitForCacheSync(ctx.Done())

		wl, err := jm.discoverWorkloadForJob(ctx, job)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if wl == nil {
			t.Fatal("expected Workload, got nil")
		}
		if wl.Name != workloadName {
			t.Errorf("name = %q, want %q", wl.Name, workloadName)
		}
	})
}

func TestDiscoverPodGroupForWorkload(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	templateName := "my-template"
	workload := &schedulingv1alpha2.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-workload",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: schedulingv1alpha2.WorkloadSpec{
			PodGroupTemplates: []schedulingv1alpha2.PodGroupTemplate{
				{Name: templateName},
			},
		},
	}

	t.Run("not found returns nil", func(t *testing.T) {
		clientSet := fake.NewClientset()
		jm, sharedInformers := newControllerWithSchedulingInformers(ctx, t, clientSet)
		sharedInformers.Start(ctx.Done())
		sharedInformers.WaitForCacheSync(ctx.Done())

		pg, err := jm.discoverPodGroupForWorkload(ctx, workload)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if pg != nil {
			t.Errorf("expected nil, got %v", pg)
		}
	})

	t.Run("found returns podgroup", func(t *testing.T) {
		pgName := computePodGroupName(workload.Name, templateName)
		existing := &schedulingv1alpha2.PodGroup{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pgName,
				Namespace: workload.Namespace,
			},
		}
		clientSet := fake.NewClientset(existing)
		jm, sharedInformers := newControllerWithSchedulingInformers(ctx, t, clientSet)
		sharedInformers.Start(ctx.Done())
		sharedInformers.WaitForCacheSync(ctx.Done())

		pg, err := jm.discoverPodGroupForWorkload(ctx, workload)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if pg == nil {
			t.Fatal("expected PodGroup, got nil")
		}
		if pg.Name != pgName {
			t.Errorf("name = %q, want %q", pg.Name, pgName)
		}
	})

	t.Run("no templates returns nil", func(t *testing.T) {
		noTemplatesWL := &schedulingv1alpha2.Workload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "empty-workload",
				Namespace: metav1.NamespaceDefault,
			},
			Spec: schedulingv1alpha2.WorkloadSpec{},
		}
		clientSet := fake.NewClientset()
		jm, sharedInformers := newControllerWithSchedulingInformers(ctx, t, clientSet)
		sharedInformers.Start(ctx.Done())
		sharedInformers.WaitForCacheSync(ctx.Done())

		pg, err := jm.discoverPodGroupForWorkload(ctx, noTemplatesWL)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if pg != nil {
			t.Errorf("expected nil for workload with no templates, got %v", pg)
		}
	})
}

func TestEnqueueJobFromOwnerRef(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	clientSet := fake.NewClientset()
	jm, _ := newControllerWithSchedulingInformers(ctx, t, clientSet)
	logger := ktesting.NewLogger(t, ktesting.NewConfig())

	tests := map[string]struct {
		obj      interface{}
		wantKey  string
		wantSize int
	}{
		"workload with Job ownerRef enqueues job": {
			obj: &schedulingv1alpha2.Workload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "my-workload",
					Namespace: "ns",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "batch/v1",
							Kind:       "Job",
							Name:       "my-job",
						},
					},
				},
			},
			wantKey:  "ns/my-job",
			wantSize: 1,
		},
		"podgroup with Job ownerRef enqueues job": {
			obj: &schedulingv1alpha2.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "my-pg",
					Namespace: "ns",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "batch/v1",
							Kind:       "Job",
							Name:       "another-job",
						},
					},
				},
			},
			wantKey:  "ns/another-job",
			wantSize: 1,
		},
		"object without Job ownerRef does not enqueue": {
			obj: &schedulingv1alpha2.Workload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "orphan-workload",
					Namespace: "ns",
				},
			},
			wantSize: 0,
		},
		"tombstone-wrapped object enqueues job": {
			obj: cache.DeletedFinalStateUnknown{
				Key: "ns/my-workload",
				Obj: &schedulingv1alpha2.Workload{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "my-workload",
						Namespace: "ns",
						OwnerReferences: []metav1.OwnerReference{
							{
								APIVersion: "batch/v1",
								Kind:       "Job",
								Name:       "tombstone-job",
							},
						},
					},
				},
			},
			wantKey:  "ns/tombstone-job",
			wantSize: 1,
		},
		"non-batch ownerRef is ignored": {
			obj: &schedulingv1alpha2.Workload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "wl-owned-by-other",
					Namespace: "ns",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "Deployment",
							Name:       "my-deploy",
						},
					},
				},
			},
			wantSize: 0,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			// Drain the queue from any previous test case.
			for jm.queue.Len() > 0 {
				item, _ := jm.queue.Get()
				jm.queue.Done(item)
			}

			jm.enqueueJobFromOwnerRef(logger, tc.obj)

			if jm.queue.Len() != tc.wantSize {
				t.Errorf("queue size = %d, want %d", jm.queue.Len(), tc.wantSize)
			}
			if tc.wantSize > 0 {
				item, _ := jm.queue.Get()
				jm.queue.Done(item)
				if item != tc.wantKey {
					t.Errorf("enqueued key = %q, want %q", item, tc.wantKey)
				}
			}
		})
	}
}

func TestAddSchedulingInformers(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	clientSet := fake.NewClientset()
	sharedInformers := informers.NewSharedInformerFactory(clientSet, controller.NoResyncPeriodFunc())

	// Create controller without scheduling informers.
	jm, err := newControllerWithClock(ctx,
		sharedInformers.Core().V1().Pods(),
		sharedInformers.Batch().V1().Jobs(),
		clientSet,
		realClock,
		nil, nil,
	)
	if err != nil {
		t.Fatalf("Error creating controller: %v", err)
	}

	// Verify listers are nil before adding informers.
	if jm.workloadLister != nil {
		t.Error("workloadLister should be nil before addSchedulingInformers")
	}
	if jm.podGroupLister != nil {
		t.Error("podGroupLister should be nil before addSchedulingInformers")
	}

	logger := ktesting.NewLogger(t, ktesting.NewConfig())
	err = jm.addSchedulingInformers(logger,
		sharedInformers.Scheduling().V1alpha2().Workloads(),
		sharedInformers.Scheduling().V1alpha2().PodGroups(),
	)
	if err != nil {
		t.Fatalf("addSchedulingInformers failed: %v", err)
	}

	// Verify listers are set after adding informers.
	if jm.workloadLister == nil {
		t.Error("workloadLister should be set after addSchedulingInformers")
	}
	if jm.podGroupLister == nil {
		t.Error("podGroupLister should be set after addSchedulingInformers")
	}
	if jm.workloadStoreSynced == nil {
		t.Error("workloadStoreSynced should be set after addSchedulingInformers")
	}
	if jm.podGroupStoreSynced == nil {
		t.Error("podGroupStoreSynced should be set after addSchedulingInformers")
	}
}

func TestEnsureWorkloadAndPodGroup_FeatureGateDisabled(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// Ensure the feature gate is disabled (it's alpha/default-off, but be explicit).
	featuregatetesting.SetFeatureGatesDuringTest(t, feature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.EnableWorkloadWithJob: false,
	})

	clientSet := fake.NewClientset()
	// Create controller without scheduling informers (simulating feature gate off).
	sharedInformers := informers.NewSharedInformerFactory(clientSet, controller.NoResyncPeriodFunc())
	jm, err := newControllerWithClock(ctx,
		sharedInformers.Core().V1().Pods(),
		sharedInformers.Batch().V1().Jobs(),
		clientSet,
		realClock,
		nil, nil,
	)
	if err != nil {
		t.Fatalf("Error creating controller: %v", err)
	}
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady

	// The controller's workloadLister is nil, so ensureWorkloadAndPodGroup
	// would panic if called. The feature gate check in syncJob prevents that.
	// Verify the feature gate is off.
	if feature.DefaultFeatureGate.Enabled(features.EnableWorkloadWithJob) {
		t.Fatal("expected EnableWorkloadWithJob to be disabled")
	}
	if jm.workloadLister != nil {
		t.Error("workloadLister should be nil when feature gate is disabled")
	}
}
