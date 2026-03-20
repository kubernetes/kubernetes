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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

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

func newWorkloadJobControllerRef(jobName string) *schedulingv1alpha2.TypedLocalObjectReference {
	return &schedulingv1alpha2.TypedLocalObjectReference{
		APIGroup: batch.SchemeGroupVersion.Group,
		Kind:     "Job",
		Name:     jobName,
	}
}

// newControllerWithSchedulingInformers creates a Job controller with Workload and PodGroup
// informers wired up, suitable for testing scheduling manager functionality.
func newControllerWithSchedulingInformers(ctx context.Context, t *testing.T, kubeClient *fake.Clientset) (*Controller, informers.SharedInformerFactory) {
	t.Helper()
	featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, feature.DefaultFeatureGate, utilversion.MustParse("1.36"))
	featuregatetesting.SetFeatureGatesDuringTest(t, feature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:       true,
		features.EnableWorkloadWithJob: true,
	})
	sharedInformers := informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())
	jm, err := newControllerWithClock(ctx,
		kubeClient,
		realClock,
		sharedInformers.Core().V1().Pods(),
		sharedInformers.Batch().V1().Jobs(),
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

func TestShouldManageWorkloadForJob(t *testing.T) {
	indexed := batch.IndexedCompletion
	nonIndexed := batch.NonIndexedCompletion

	tests := map[string]struct {
		enableWorkloadWithJob bool
		job                   *batch.Job
		want                  bool
	}{
		"eligible: indexed, parallelism=completions>1, no schedulingGroup": {
			enableWorkloadWithJob: true,
			job: &batch.Job{
				Spec: batch.JobSpec{
					Parallelism:    ptr.To[int32](4),
					Completions:    ptr.To[int32](4),
					CompletionMode: &indexed,
				},
			},
			want: true,
		},
		"feature gate disabled": {
			enableWorkloadWithJob: false,
			job: &batch.Job{
				Spec: batch.JobSpec{
					Parallelism:    ptr.To[int32](4),
					Completions:    ptr.To[int32](4),
					CompletionMode: &indexed,
				},
			},
			want: false,
		},
		"parallelism is nil": {
			enableWorkloadWithJob: true,
			job: &batch.Job{
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](4),
					CompletionMode: &indexed,
				},
			},
			want: false,
		},
		"parallelism=1": {
			enableWorkloadWithJob: true,
			job: &batch.Job{
				Spec: batch.JobSpec{
					Parallelism:    ptr.To[int32](1),
					Completions:    ptr.To[int32](1),
					CompletionMode: &indexed,
				},
			},
			want: false,
		},
		"non-indexed completion mode": {
			enableWorkloadWithJob: true,
			job: &batch.Job{
				Spec: batch.JobSpec{
					Parallelism:    ptr.To[int32](4),
					Completions:    ptr.To[int32](4),
					CompletionMode: &nonIndexed,
				},
			},
			want: false,
		},
		"completions is nil": {
			enableWorkloadWithJob: true,
			job: &batch.Job{
				Spec: batch.JobSpec{
					Parallelism:    ptr.To[int32](4),
					CompletionMode: &indexed,
				},
			},
			want: false,
		},
		"completions != parallelism": {
			enableWorkloadWithJob: true,
			job: &batch.Job{
				Spec: batch.JobSpec{
					Parallelism:    ptr.To[int32](4),
					Completions:    ptr.To[int32](8),
					CompletionMode: &indexed,
				},
			},
			want: false,
		},
		"schedulingGroup already set (opt-out)": {
			enableWorkloadWithJob: true,
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
		"completionMode is nil": {
			enableWorkloadWithJob: true,
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
			featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, feature.DefaultFeatureGate, utilversion.MustParse("1.36"))
			featuregatetesting.SetFeatureGatesDuringTest(t, feature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:       tc.enableWorkloadWithJob,
				features.EnableWorkloadWithJob: tc.enableWorkloadWithJob,
			})
			got := shouldManageWorkloadForJob(tc.job)
			if got != tc.want {
				t.Errorf("shouldManageWorkloadForJob() = %v, want %v", got, tc.want)
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
			wantName:     "my-training-job-pgt-0",
			wantMinCount: 8,
		},
		"single template with small parallelism": {
			jobName:      "small-job",
			parallelism:  2,
			completions:  2,
			wantName:     "small-job-pgt-0",
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

			if len(result) > validation.DNS1123SubdomainMaxLength {
				t.Errorf("name length %d exceeds DNS1123SubdomainMaxLength %d", len(result), validation.DNS1123SubdomainMaxLength)
			}

			parts := strings.Split(result, "-")
			if len(parts) < 3 {
				t.Errorf("expected at least 3 parts (wl-tpl-hash), got %q", result)
			}

			result2 := computePodGroupName(tc.workloadName, tc.templateName)
			if result != result2 {
				t.Errorf("non-deterministic: %q != %q", result, result2)
			}

			result3 := computePodGroupName(tc.workloadName, "other-template")
			if result == result3 {
				t.Error("different template names should produce different PodGroup names")
			}
		})
	}

	// Verify truncation preserves both parts.
	t.Run("truncation preserves both parts", func(t *testing.T) {
		result := computePodGroupName(strings.Repeat("w", 200), strings.Repeat("t", 200))
		lastDash := strings.LastIndex(result, "-")
		if lastDash < 0 {
			t.Fatalf("expected at least one dash in %q", result)
		}
		prefix := result[:lastDash]
		if !strings.Contains(prefix, "www") {
			t.Errorf("expected truncated workload name in prefix %q", prefix)
		}
		if !strings.Contains(prefix, "ttt") {
			t.Errorf("expected truncated template name in prefix %q", prefix)
		}
	})

	// Verify short name is not truncated when the other is long.
	t.Run("short workload name is not truncated", func(t *testing.T) {
		result := computePodGroupName("wl", strings.Repeat("t", 250))
		if !strings.HasPrefix(result, "wl-") {
			t.Errorf("short workload name should not be truncated, got %q", result)
		}
	})
}

func TestEnsureWorkloadAndPodGroup(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	baseJob := newGangSchedulingJob("test-job", 4)
	workloadName := computeWorkloadName(baseJob)
	templateName := fmt.Sprintf("%s-pgt-%d", baseJob.Name, 0)
	podGroupName := computePodGroupName(workloadName, templateName)

	makeWorkload := func(name, jobName string) *schedulingv1alpha2.Workload {
		return &schedulingv1alpha2.Workload{
			ObjectMeta: metav1.ObjectMeta{
				Name:            name,
				Namespace:       metav1.NamespaceDefault,
				UID:             types.UID(name + "-uid"),
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(baseJob, controllerKind)},
			},
			Spec: schedulingv1alpha2.WorkloadSpec{
				ControllerRef: newWorkloadJobControllerRef(jobName),
				PodGroupTemplates: []schedulingv1alpha2.PodGroupTemplate{
					{
						Name: templateName,
						SchedulingPolicy: schedulingv1alpha2.PodGroupSchedulingPolicy{
							Gang: &schedulingv1alpha2.GangSchedulingPolicy{MinCount: 4},
						},
					},
				},
			},
		}
	}

	makePodGroup := func(name, wlName string) *schedulingv1alpha2.PodGroup {
		return &schedulingv1alpha2.PodGroup{
			ObjectMeta: metav1.ObjectMeta{
				Name:            name,
				Namespace:       metav1.NamespaceDefault,
				UID:             types.UID(name + "-uid"),
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(baseJob, controllerKind)},
			},
			Spec: schedulingv1alpha2.PodGroupSpec{
				PodGroupTemplateRef: &schedulingv1alpha2.PodGroupTemplateReference{
					Workload: &schedulingv1alpha2.WorkloadPodGroupTemplateReference{
						WorkloadName:         wlName,
						PodGroupTemplateName: templateName,
					},
				},
			},
		}
	}

	testCases := map[string]struct {
		job               *batch.Job
		pods              []*v1.Pod
		existingWorkloads []*schedulingv1alpha2.Workload
		existingPodGroups []*schedulingv1alpha2.PodGroup

		// When true, the Workload create reactor returns a generic error.
		simulateCreateError bool
		// When true, PodGroup create returns a generic error.
		simulatePGCreateError bool

		wantPodGroup     bool
		wantPodGroupName string
		wantErr          bool
	}{
		"creates both Workload and PodGroup when no pods exist": {
			job:              baseJob,
			wantPodGroup:     true,
			wantPodGroupName: podGroupName,
		},
		"skips creation when Job has pods but no Workload": {
			job:  baseJob,
			pods: []*v1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "pod-0", Namespace: metav1.NamespaceDefault}}},
		},
		"skips creation when Job has StartTime set": {
			job: func() *batch.Job {
				j := baseJob.DeepCopy()
				now := metav1.Now()
				j.Status.StartTime = &now
				return j
			}(),
		},
		"skips creation when Job has succeeded pods": {
			job: func() *batch.Job {
				j := baseJob.DeepCopy()
				j.Status.Succeeded = 1
				return j
			}(),
		},
		"skips creation when Job has failed pods": {
			job: func() *batch.Job {
				j := baseJob.DeepCopy()
				j.Status.Failed = 1
				return j
			}(),
		},
		"skips creating Workload/PodGroup when Job was previously suspended": {
			job: func() *batch.Job {
				j := baseJob.DeepCopy()
				j.Status.Conditions = []batch.JobCondition{
					{
						Type:   batch.JobSuspended,
						Status: v1.ConditionTrue,
					},
				}
				return j
			}(),
		},
		"discovers existing Workload and PodGroup": {
			job:               baseJob,
			existingWorkloads: []*schedulingv1alpha2.Workload{makeWorkload(workloadName, baseJob.Name)},
			existingPodGroups: []*schedulingv1alpha2.PodGroup{makePodGroup(podGroupName, workloadName)},
			wantPodGroup:      true,
			wantPodGroupName:  podGroupName,
		},
		"discovers existing objects even when pods exist": {
			job:               baseJob,
			pods:              []*v1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "pod-0", Namespace: metav1.NamespaceDefault}}},
			existingWorkloads: []*schedulingv1alpha2.Workload{makeWorkload(workloadName, baseJob.Name)},
			existingPodGroups: []*schedulingv1alpha2.PodGroup{makePodGroup(podGroupName, workloadName)},
			wantPodGroup:      true,
			wantPodGroupName:  podGroupName,
		},
		"workload found but no PodGroup creates one": {
			job:               baseJob,
			existingWorkloads: []*schedulingv1alpha2.Workload{makeWorkload("user-workload", baseJob.Name)},
			wantPodGroup:      true,
		},

		"BYO Workload without controller ownerRef falls back": {
			job: baseJob,
			existingWorkloads: []*schedulingv1alpha2.Workload{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "user-created-workload",
						Namespace: metav1.NamespaceDefault,
						UID:       types.UID("user-wl-uid"),
					},
					Spec: schedulingv1alpha2.WorkloadSpec{
						ControllerRef: newWorkloadJobControllerRef(baseJob.Name),
						PodGroupTemplates: []schedulingv1alpha2.PodGroupTemplate{
							{
								Name: templateName,
								SchedulingPolicy: schedulingv1alpha2.PodGroupSchedulingPolicy{
									Gang: &schedulingv1alpha2.GangSchedulingPolicy{MinCount: 4},
								},
							},
						},
					},
				},
			},
		},
		"ambiguous Workloads fall back": {
			job: baseJob,
			existingWorkloads: []*schedulingv1alpha2.Workload{
				makeWorkload("workload-1", baseJob.Name),
				makeWorkload("workload-2", baseJob.Name),
			},
		},
		"BYO PodGroup without controller ownerRef falls back": {
			job:               baseJob,
			existingWorkloads: []*schedulingv1alpha2.Workload{makeWorkload(workloadName, baseJob.Name)},
			existingPodGroups: []*schedulingv1alpha2.PodGroup{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "external-pg",
						Namespace: metav1.NamespaceDefault,
						UID:       types.UID("ext-pg-uid"),
					},
					Spec: schedulingv1alpha2.PodGroupSpec{
						PodGroupTemplateRef: &schedulingv1alpha2.PodGroupTemplateReference{
							Workload: &schedulingv1alpha2.WorkloadPodGroupTemplateReference{
								WorkloadName: workloadName,
							},
						},
					},
				},
			},
		},
		"ambiguous PodGroups fall back": {
			job:               baseJob,
			existingWorkloads: []*schedulingv1alpha2.Workload{makeWorkload(workloadName, baseJob.Name)},
			existingPodGroups: []*schedulingv1alpha2.PodGroup{
				makePodGroup("pg-1", workloadName),
				makePodGroup("pg-2", workloadName),
			},
		},
		"unsupported PodGroupTemplate count falls back": {
			job: baseJob,
			existingWorkloads: []*schedulingv1alpha2.Workload{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:            workloadName,
						Namespace:       metav1.NamespaceDefault,
						UID:             types.UID(workloadName + "-uid"),
						OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(baseJob, controllerKind)},
					},
					Spec: schedulingv1alpha2.WorkloadSpec{
						ControllerRef: newWorkloadJobControllerRef(baseJob.Name),
						PodGroupTemplates: []schedulingv1alpha2.PodGroupTemplate{
							{Name: "group-a"},
							{Name: "group-b"},
						},
					},
				},
			},
		},

		"Workload create error propagates": {
			job:                 baseJob,
			simulateCreateError: true,
			wantErr:             true,
		},

		"PodGroup create error propagates": {
			job:                   baseJob,
			existingWorkloads:     []*schedulingv1alpha2.Workload{makeWorkload(workloadName, baseJob.Name)},
			simulatePGCreateError: true,
			wantErr:               true,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			var objs []runtime.Object
			for _, wl := range tc.existingWorkloads {
				objs = append(objs, wl)
			}
			for _, pg := range tc.existingPodGroups {
				objs = append(objs, pg)
			}
			clientSet := fake.NewClientset(objs...)
			jm, sharedInformers := newControllerWithSchedulingInformers(ctx, t, clientSet)

			if tc.simulateCreateError {
				clientSet.PrependReactor("create", "workloads", func(action k8stesting.Action) (bool, runtime.Object, error) {
					return true, nil, fmt.Errorf("synthetic API error")
				})
			}

			if tc.simulatePGCreateError {
				clientSet.PrependReactor("create", "podgroups", func(action k8stesting.Action) (bool, runtime.Object, error) {
					return true, nil, fmt.Errorf("synthetic PG API error")
				})
			}

			sharedInformers.Start(ctx.Done())
			sharedInformers.WaitForCacheSync(ctx.Done())

			_, pg, err := jm.ensureWorkloadAndPodGroup(ctx, tc.job, tc.pods)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.wantPodGroup {
				if pg == nil {
					t.Fatal("expected non-nil PodGroup")
				}
				if tc.wantPodGroupName != "" && pg.Name != tc.wantPodGroupName {
					t.Errorf("PodGroup name = %q, want %q", pg.Name, tc.wantPodGroupName)
				}
			} else if pg != nil {
				t.Errorf("expected nil PodGroup, got %v", pg)
			}
		})
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

	if wl.Spec.ControllerRef == nil {
		t.Fatal("expected controllerRef")
	}
	if diff := cmp.Diff(wl.Spec.ControllerRef, newWorkloadJobControllerRef(job.Name)); diff != "" {
		t.Errorf("unexpected controllerRef (-got +want):\n%s", diff)
	}

	if len(wl.Spec.PodGroupTemplates) != 1 {
		t.Fatalf("expected 1 template, got %d", len(wl.Spec.PodGroupTemplates))
	}
	tpl := wl.Spec.PodGroupTemplates[0]
	if tpl.SchedulingPolicy.Gang == nil || tpl.SchedulingPolicy.Gang.MinCount != 8 {
		t.Errorf("expected gang minCount=8, got %+v", tpl.SchedulingPolicy)
	}

	if len(wl.OwnerReferences) != 1 || wl.OwnerReferences[0].Name != job.Name {
		t.Errorf("unexpected ownerReferences: %v", wl.OwnerReferences)
	}
}

func TestCreatePodGroupForWorkload(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	clientSet := fake.NewClientset()
	jm, _ := newControllerWithSchedulingInformers(ctx, t, clientSet)
	job := newGangSchedulingJob("my-job", 4)
	templateName := fmt.Sprintf("%s-pgt-%d", job.Name, 0)

	t.Run("creates PodGroup with correct name and ownerReferences", func(t *testing.T) {
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
							Gang: &schedulingv1alpha2.GangSchedulingPolicy{MinCount: 4},
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
		if len(pg.OwnerReferences) != 2 {
			t.Fatalf("expected 2 ownerReferences, got %d", len(pg.OwnerReferences))
		}
	})

	t.Run("errors when Workload has no templates", func(t *testing.T) {
		workload := &schedulingv1alpha2.Workload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "empty-workload",
				Namespace: job.Namespace,
			},
			Spec: schedulingv1alpha2.WorkloadSpec{},
		}

		_, err := jm.createPodGroupForWorkload(ctx, job, workload)
		if err == nil {
			t.Fatal("expected error for Workload with no templates")
		}
	})
}
