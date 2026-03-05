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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	schedulingv1alpha2 "k8s.io/api/scheduling/v1alpha2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
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
			wantName:     "worker-0",
			wantMinCount: 8,
		},
		"single template with small parallelism": {
			jobName:      "small-job",
			parallelism:  2,
			completions:  2,
			wantName:     "worker-0",
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

func TestComputePodGroupNameTruncation(t *testing.T) {
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

func TestEnsureWorkloadAndPodGroup(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	baseJob := newGangSchedulingJob("test-job", 4)
	workloadName := computeWorkloadName(baseJob)
	templateName := "worker-0"
	podGroupName := computePodGroupName(workloadName, templateName)

	makeWorkload := func(name, jobName string) *schedulingv1alpha2.Workload {
		return &schedulingv1alpha2.Workload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID(name + "-uid"),
			},
			Spec: schedulingv1alpha2.WorkloadSpec{
				ControllerRef: &schedulingv1alpha2.TypedLocalObjectReference{
					APIGroup: "batch",
					Kind:     "Job",
					Name:     jobName,
				},
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
				Name:      name,
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID(name + "-uid"),
			},
			Spec: schedulingv1alpha2.PodGroupSpec{
				PodGroupTemplateRef: &schedulingv1alpha2.PodGroupTemplateReference{
					WorkloadName:         wlName,
					PodGroupTemplateName: templateName,
				},
			},
		}
	}

	testCases := map[string]struct {
		job               *batch.Job
		pods              []*v1.Pod
		existingWorkloads []*schedulingv1alpha2.Workload
		existingPodGroups []*schedulingv1alpha2.PodGroup

		// When true, create reactors return AlreadyExists and inject objects
		// into the informer cache (simulating another controller winning the race).
		simulateAlreadyExists bool

		wantPodGroup     bool
		wantPodGroupName string
	}{
		"not eligible: parallelism=1": {
			job: newJob(1, 1, 0, batch.IndexedCompletion),
		},
		"creates both Workload and PodGroup when no pods exist": {
			job:              baseJob,
			wantPodGroup:     true,
			wantPodGroupName: podGroupName,
		},
		"skips creation when Job has pods but no Workload": {
			job:  baseJob,
			pods: []*v1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "pod-0", Namespace: metav1.NamespaceDefault}}},
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
		"workload found but no PodGroup returns nil": {
			job:               baseJob,
			existingWorkloads: []*schedulingv1alpha2.Workload{makeWorkload("user-workload", baseJob.Name)},
		},
		"retries on AlreadyExists": {
			job:                   baseJob,
			simulateAlreadyExists: true,
			wantPodGroup:          true,
			wantPodGroupName:      podGroupName,
		},
		"discovers user-created (BYO) Workload without ownerRef": {
			job: baseJob,
			existingWorkloads: []*schedulingv1alpha2.Workload{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "user-created-workload",
						Namespace: metav1.NamespaceDefault,
						UID:       types.UID("user-wl-uid"),
					},
					Spec: schedulingv1alpha2.WorkloadSpec{
						ControllerRef: &schedulingv1alpha2.TypedLocalObjectReference{
							APIGroup: "batch",
							Kind:     "Job",
							Name:     baseJob.Name,
						},
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
			existingPodGroups: []*schedulingv1alpha2.PodGroup{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "user-created-pg",
						Namespace: metav1.NamespaceDefault,
						UID:       types.UID("user-pg-uid"),
					},
					Spec: schedulingv1alpha2.PodGroupSpec{
						PodGroupTemplateRef: &schedulingv1alpha2.PodGroupTemplateReference{
							WorkloadName:         "user-created-workload",
							PodGroupTemplateName: templateName,
						},
					},
				},
			},
			wantPodGroup:     true,
			wantPodGroupName: "user-created-pg",
		},
		"schedulingGroup already set (opt-out)": {
			job: func() *batch.Job {
				j := newGangSchedulingJob("opt-out-job", 4)
				j.Spec.Template.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
					PodGroupName: ptr.To("external-pg"),
				}
				return j
			}(),
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

			if tc.simulateAlreadyExists {
				wlIndexer := sharedInformers.Scheduling().V1alpha2().Workloads().Informer().GetIndexer()
				pgIndexer := sharedInformers.Scheduling().V1alpha2().PodGroups().Informer().GetIndexer()
				wlToInject := makeWorkload(workloadName, tc.job.Name)
				pgToInject := makePodGroup(podGroupName, workloadName)

				clientSet.PrependReactor("create", "workloads", func(action k8stesting.Action) (bool, runtime.Object, error) {
					wlIndexer.Add(wlToInject)
					return true, nil, apierrors.NewAlreadyExists(schedulingv1alpha2.Resource("workloads"), workloadName)
				})
				clientSet.PrependReactor("create", "podgroups", func(action k8stesting.Action) (bool, runtime.Object, error) {
					pgIndexer.Add(pgToInject)
					return true, nil, apierrors.NewAlreadyExists(schedulingv1alpha2.Resource("podgroups"), podGroupName)
				})
			}

			sharedInformers.Start(ctx.Done())
			sharedInformers.WaitForCacheSync(ctx.Done())

			pg, err := jm.ensureWorkloadAndPodGroup(ctx, tc.job, tc.pods)
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

func TestEnsureWorkloadUnsupportedTemplateCount(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	job := newGangSchedulingJob("test-job", 4)
	workloadName := computeWorkloadName(job)

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

	wl, err := jm.ensureWorkload(ctx, job)
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
	templateName := "worker-0"

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

	if pg.Spec.PodGroupTemplateRef == nil {
		t.Fatal("expected podGroupTemplateRef")
	}
	if pg.Spec.PodGroupTemplateRef.WorkloadName != workload.Name {
		t.Errorf("templateRef.workloadName = %q, want %q", pg.Spec.PodGroupTemplateRef.WorkloadName, workload.Name)
	}
	if pg.Spec.PodGroupTemplateRef.PodGroupTemplateName != templateName {
		t.Errorf("templateRef.podGroupTemplateName = %q, want %q", pg.Spec.PodGroupTemplateRef.PodGroupTemplateName, templateName)
	}

	if pg.Spec.SchedulingPolicy.Gang == nil || pg.Spec.SchedulingPolicy.Gang.MinCount != 4 {
		t.Errorf("expected copied gang policy with minCount=4, got %+v", pg.Spec.SchedulingPolicy)
	}

	if len(pg.OwnerReferences) != 2 {
		t.Fatalf("expected 2 ownerReferences, got %d", len(pg.OwnerReferences))
	}
}

func TestCreatePodGroupForWorkloadNoTemplates(t *testing.T) {
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

	testCases := map[string]struct {
		workloads        []*schedulingv1alpha2.Workload
		wantWorkloadName string
	}{
		"not found returns nil": {},
		"found by controllerRef": {
			workloads: []*schedulingv1alpha2.Workload{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "user-chosen-name",
						Namespace: job.Namespace,
					},
					Spec: schedulingv1alpha2.WorkloadSpec{
						ControllerRef: &schedulingv1alpha2.TypedLocalObjectReference{
							APIGroup: "batch",
							Kind:     "Job",
							Name:     job.Name,
						},
					},
				},
			},
			wantWorkloadName: "user-chosen-name",
		},
		"ignores workload without controllerRef": {
			workloads: []*schedulingv1alpha2.Workload{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "orphan-workload",
						Namespace: job.Namespace,
					},
				},
			},
		},
		"ignores workload referencing different job": {
			workloads: []*schedulingv1alpha2.Workload{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "other-workload",
						Namespace: job.Namespace,
					},
					Spec: schedulingv1alpha2.WorkloadSpec{
						ControllerRef: &schedulingv1alpha2.TypedLocalObjectReference{
							APIGroup: "batch",
							Kind:     "Job",
							Name:     "different-job",
						},
					},
				},
			},
		},
		"ambiguous returns nil with event": {
			workloads: []*schedulingv1alpha2.Workload{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "workload-1",
						Namespace: job.Namespace,
					},
					Spec: schedulingv1alpha2.WorkloadSpec{
						ControllerRef: &schedulingv1alpha2.TypedLocalObjectReference{
							APIGroup: "batch",
							Kind:     "Job",
							Name:     job.Name,
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "workload-2",
						Namespace: job.Namespace,
					},
					Spec: schedulingv1alpha2.WorkloadSpec{
						ControllerRef: &schedulingv1alpha2.TypedLocalObjectReference{
							APIGroup: "batch",
							Kind:     "Job",
							Name:     job.Name,
						},
					},
				},
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			var objs []runtime.Object
			for _, wl := range tc.workloads {
				objs = append(objs, wl)
			}
			clientSet := fake.NewClientset(objs...)
			jm, sharedInformers := newControllerWithSchedulingInformers(ctx, t, clientSet)
			sharedInformers.Start(ctx.Done())
			sharedInformers.WaitForCacheSync(ctx.Done())

			wl, err := jm.discoverWorkloadForJob(ctx, job)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.wantWorkloadName != "" {
				if wl == nil {
					t.Fatal("expected Workload, got nil")
				}
				if wl.Name != tc.wantWorkloadName {
					t.Errorf("name = %q, want %q", wl.Name, tc.wantWorkloadName)
				}
			} else if wl != nil {
				t.Errorf("expected nil, got %v", wl)
			}
		})
	}
}

func TestDiscoverPodGroupForWorkload(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	job := newGangSchedulingJob("test-job", 4)
	workload := &schedulingv1alpha2.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "my-workload",
			Namespace: metav1.NamespaceDefault,
		},
	}

	testCases := map[string]struct {
		podGroups        []*schedulingv1alpha2.PodGroup
		wantPodGroupName string
	}{
		"not found returns nil": {},
		"found by podGroupTemplateRef": {
			podGroups: []*schedulingv1alpha2.PodGroup{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "user-chosen-pg",
						Namespace: workload.Namespace,
					},
					Spec: schedulingv1alpha2.PodGroupSpec{
						PodGroupTemplateRef: &schedulingv1alpha2.PodGroupTemplateReference{
							WorkloadName: workload.Name,
						},
					},
				},
			},
			wantPodGroupName: "user-chosen-pg",
		},
		"ignores podgroup without ref": {
			podGroups: []*schedulingv1alpha2.PodGroup{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "orphan-pg",
						Namespace: workload.Namespace,
					},
				},
			},
		},
		"ignores podgroup referencing different workload": {
			podGroups: []*schedulingv1alpha2.PodGroup{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "other-pg",
						Namespace: workload.Namespace,
					},
					Spec: schedulingv1alpha2.PodGroupSpec{
						PodGroupTemplateRef: &schedulingv1alpha2.PodGroupTemplateReference{
							WorkloadName: "different-workload",
						},
					},
				},
			},
		},
		"unsupported structure returns nil": {
			podGroups: []*schedulingv1alpha2.PodGroup{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pg-1",
						Namespace: workload.Namespace,
					},
					Spec: schedulingv1alpha2.PodGroupSpec{
						PodGroupTemplateRef: &schedulingv1alpha2.PodGroupTemplateReference{
							WorkloadName: workload.Name,
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "pg-2",
						Namespace: workload.Namespace,
					},
					Spec: schedulingv1alpha2.PodGroupSpec{
						PodGroupTemplateRef: &schedulingv1alpha2.PodGroupTemplateReference{
							WorkloadName: workload.Name,
						},
					},
				},
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			var objs []runtime.Object
			for _, pg := range tc.podGroups {
				objs = append(objs, pg)
			}
			clientSet := fake.NewClientset(objs...)
			jm, sharedInformers := newControllerWithSchedulingInformers(ctx, t, clientSet)
			sharedInformers.Start(ctx.Done())
			sharedInformers.WaitForCacheSync(ctx.Done())

			pg, err := jm.discoverPodGroupForWorkload(ctx, job, workload)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.wantPodGroupName != "" {
				if pg == nil {
					t.Fatal("expected PodGroup, got nil")
				}
				if pg.Name != tc.wantPodGroupName {
					t.Errorf("name = %q, want %q", pg.Name, tc.wantPodGroupName)
				}
			} else if pg != nil {
				t.Errorf("expected nil, got %v", pg)
			}
		})
	}
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

func TestEnsureWorkloadAndPodGroupFeatureGateDisabled(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	featuregatetesting.SetFeatureGatesDuringTest(t, feature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.EnableWorkloadWithJob: false,
	})

	clientSet := fake.NewClientset()
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

	if feature.DefaultFeatureGate.Enabled(features.EnableWorkloadWithJob) {
		t.Fatal("expected EnableWorkloadWithJob to be disabled")
	}
	if jm.workloadLister != nil {
		t.Error("workloadLister should be nil when feature gate is disabled")
	}
}
