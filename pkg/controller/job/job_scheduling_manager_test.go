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
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
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
	apischeduling "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
)

// newGangSchedulingJob creates a Job that opts into gang scheduling via
// spec.scheduling with minCount equal to parallelism.
func newGangSchedulingJob(name string, parallelism int32) *batch.Job {
	indexed := batch.IndexedCompletion
	return &batch.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceDefault,
			UID:       types.UID(name + "-uid"),
		},
		Spec: batch.JobSpec{
			Parallelism:    new(parallelism),
			Completions:    new(parallelism),
			CompletionMode: &indexed,
			Scheduling: &batch.JobSchedulingConfiguration{
				Policy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{
					Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: new(parallelism)},
				},
			},
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

func newWorkloadJobControllerRef(jobName string) *schedulingv1alpha3.TypedLocalObjectReference {
	return &schedulingv1alpha3.TypedLocalObjectReference{
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
		features.GenericWorkload: true,
		features.WorkloadWithJob: true,
	})
	sharedInformers := informers.NewSharedInformerFactory(kubeClient, controller.NoResyncPeriodFunc())
	jm, err := newControllerWithClock(ctx,
		kubeClient,
		realClock,
		sharedInformers.Core().V1().Pods(),
		sharedInformers.Batch().V1().Jobs(),
		sharedInformers.Scheduling().V1alpha3().Workloads(),
		sharedInformers.Scheduling().V1alpha3().PodGroups(),
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
	parentWorkloadOwner := []metav1.OwnerReference{{
		APIVersion: "jobset.x-k8s.io/v1alpha2",
		Kind:       "JobSet",
		Name:       "js",
		Controller: new(true),
	}}
	cronJobOwner := []metav1.OwnerReference{{
		APIVersion: "batch/v1",
		Kind:       "CronJob",
		Name:       "cj",
		Controller: new(true),
	}}

	tests := map[string]struct {
		workloadWithJob bool
		job             *batch.Job
		want            bool
	}{
		"gate enabled, standalone Job -> managed (Universal Representation)": {
			workloadWithJob: true,
			job:             &batch.Job{Spec: batch.JobSpec{Parallelism: new(int32(4))}},
			want:            true,
		},
		"gate enabled, basic single-pod Job -> still managed": {
			workloadWithJob: true,
			job:             &batch.Job{Spec: batch.JobSpec{Parallelism: new(int32(1))}},
			want:            true,
		},
		"feature gate disabled -> not managed": {
			workloadWithJob: false,
			job:             &batch.Job{Spec: batch.JobSpec{Parallelism: new(int32(4))}},
			want:            false,
		},
		"BYO PodGroup via pod template schedulingGroup -> not managed": {
			workloadWithJob: true,
			job: &batch.Job{
				Spec: batch.JobSpec{
					Parallelism: new(int32(4)),
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							SchedulingGroup: &v1.PodSchedulingGroup{PodGroupName: new("existing-pg")},
						},
					},
				},
			},
			want: false,
		},
		"parent-owned Workload without delegation annotation -> not managed": {
			workloadWithJob: true,
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{OwnerReferences: parentWorkloadOwner},
				Spec:       batch.JobSpec{Parallelism: new(int32(4))},
			},
			want: false,
		},
		"parent-owned Workload with delegation annotation -> managed (delegated PodGroup)": {
			workloadWithJob: true,
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					OwnerReferences: parentWorkloadOwner,
					Annotations:     map[string]string{apischeduling.GroupTemplateNameAnnotation: "workers"},
				},
				Spec: batch.JobSpec{Parallelism: new(int32(4))},
			},
			want: true,
		},
		"CronJob-owned Job is treated as parent-owned (not managed)": {
			workloadWithJob: true,
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{OwnerReferences: cronJobOwner},
				Spec:       batch.JobSpec{Parallelism: new(int32(4))},
			},
			want: false,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, feature.DefaultFeatureGate, utilversion.MustParse("1.36"))
			featuregatetesting.SetFeatureGatesDuringTest(t, feature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload: tc.workloadWithJob,
				features.WorkloadWithJob: tc.workloadWithJob,
			})
			got := shouldManageWorkloadForJob(tc.job)
			if got != tc.want {
				t.Errorf("shouldManageWorkloadForJob() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestGenerateWorkload(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	clientSet := fake.NewClientset()
	jm, _ := newControllerWithSchedulingInformers(ctx, t, clientSet)

	newJob := func(name string, parallelism int32, scheduling *batch.JobSchedulingConfiguration) *batch.Job {
		return &batch.Job{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: metav1.NamespaceDefault, UID: types.UID(name + "-uid")},
			Spec:       batch.JobSpec{Parallelism: new(parallelism), Scheduling: scheduling},
		}
	}

	testCases := map[string]struct {
		job              *batch.Job
		wantErr          bool
		wantBasic        bool
		wantGangMinCount int32
		wantTopologyKey  string
		wantAllDisrupt   bool
		wantClaimNames   []string
	}{
		"nil scheduling defaults to Basic": {
			job:       newJob("basic-default", 4, nil),
			wantBasic: true,
		},
		"scheduling set but policy nil defaults to Basic": {
			job:       newJob("basic-nil-policy", 4, &batch.JobSchedulingConfiguration{}),
			wantBasic: true,
		},
		"explicit basic policy": {
			job: newJob("basic-explicit", 4, &batch.JobSchedulingConfiguration{
				Policy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Basic: &schedulingv1alpha3.WorkloadPodGroupBasicSchedulingPolicy{}},
			}),
			wantBasic: true,
		},
		"gang with explicit minCount": {
			job: newJob("gang", 4, &batch.JobSchedulingConfiguration{
				Policy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: new(int32(3))}},
			}),
			wantGangMinCount: 3,
		},
		"gang without minCount defaults to parallelism": {
			job: newJob("gang-zero", 4, &batch.JobSchedulingConfiguration{
				Policy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{}},
			}),
			wantGangMinCount: 4,
		},
		"gang, topology, disruption, and claims passthrough": {
			job: newJob("full", 4, &batch.JobSchedulingConfiguration{
				Policy: &schedulingv1alpha3.WorkloadPodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.WorkloadPodGroupGangSchedulingPolicy{MinCount: new(int32(4))}},
				Constraints: &schedulingv1alpha3.WorkloadPodGroupSchedulingConstraints{
					Topology: []schedulingv1alpha3.TopologyConstraint{{Key: "topology.kubernetes.io/zone"}},
				},
				DisruptionMode: &schedulingv1alpha3.WorkloadPodGroupDisruptionMode{All: &schedulingv1alpha3.WorkloadPodGroupAllDisruptionMode{}},
				ResourceClaims: []schedulingv1alpha3.WorkloadPodGroupResourceClaim{{Name: "gpu", ResourceClaimName: new("my-claim")}},
			}),
			wantGangMinCount: 4,
			wantTopologyKey:  "topology.kubernetes.io/zone",
			wantAllDisrupt:   true,
			wantClaimNames:   []string{"gpu"},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			wl, err := jm.generateWorkload(tc.job)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got none")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(wl.Spec.PodGroupTemplates) != 1 {
				t.Fatalf("expected 1 template, got %d", len(wl.Spec.PodGroupTemplates))
			}
			tpl := wl.Spec.PodGroupTemplates[0]
			if tpl.Name != podGroupTemplateName(tc.job) {
				t.Errorf("template name = %q, want %q", tpl.Name, podGroupTemplateName(tc.job))
			}
			if tc.wantBasic {
				if tpl.SchedulingPolicy.Basic == nil {
					t.Errorf("expected Basic policy, got %+v", tpl.SchedulingPolicy)
				}
			} else {
				if tpl.SchedulingPolicy.Gang == nil {
					t.Fatalf("expected Gang policy, got %+v", tpl.SchedulingPolicy)
				}
				if tpl.SchedulingPolicy.Gang.MinCount != tc.wantGangMinCount {
					t.Errorf("gang minCount = %d, want %d", tpl.SchedulingPolicy.Gang.MinCount, tc.wantGangMinCount)
				}
			}
			if tc.wantTopologyKey != "" {
				if tpl.SchedulingConstraints == nil || len(tpl.SchedulingConstraints.Topology) != 1 ||
					tpl.SchedulingConstraints.Topology[0].Key != tc.wantTopologyKey {
					t.Errorf("topology = %+v, want key %q", tpl.SchedulingConstraints, tc.wantTopologyKey)
				}
			}
			if tc.wantAllDisrupt && (tpl.DisruptionMode == nil || tpl.DisruptionMode.All == nil) {
				t.Errorf("expected All disruption mode, got %+v", tpl.DisruptionMode)
			}
			if len(tc.wantClaimNames) > 0 {
				if len(tpl.ResourceClaims) != len(tc.wantClaimNames) {
					t.Fatalf("resourceClaims = %d, want %d", len(tpl.ResourceClaims), len(tc.wantClaimNames))
				}
				for i, want := range tc.wantClaimNames {
					if tpl.ResourceClaims[i].Name != want {
						t.Errorf("claim[%d].Name = %q, want %q", i, tpl.ResourceClaims[i].Name, want)
					}
				}
			}
			if wl.Spec.ControllerRef == nil || wl.Spec.ControllerRef.Name != tc.job.Name {
				t.Errorf("unexpected controllerRef: %+v", wl.Spec.ControllerRef)
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

	makeWorkload := func(name, jobName string) *schedulingv1alpha3.Workload {
		return &schedulingv1alpha3.Workload{
			ObjectMeta: metav1.ObjectMeta{
				Name:            name,
				Namespace:       metav1.NamespaceDefault,
				UID:             types.UID(name + "-uid"),
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(baseJob, controllerKind)},
			},
			Spec: schedulingv1alpha3.WorkloadSpec{
				ControllerRef: newWorkloadJobControllerRef(jobName),
				PodGroupTemplates: []schedulingv1alpha3.PodGroupTemplate{
					{
						Name: templateName,
						SchedulingPolicy: schedulingv1alpha3.PodGroupSchedulingPolicy{
							Gang: &schedulingv1alpha3.GangSchedulingPolicy{MinCount: 4},
						},
					},
				},
			},
		}
	}

	makePodGroup := func(name, wlName string) *schedulingv1alpha3.PodGroup {
		return &schedulingv1alpha3.PodGroup{
			ObjectMeta: metav1.ObjectMeta{
				Name:            name,
				Namespace:       metav1.NamespaceDefault,
				UID:             types.UID(name + "-uid"),
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(baseJob, controllerKind)},
			},
			Spec: schedulingv1alpha3.PodGroupSpec{
				WorkloadRef: &schedulingv1alpha3.WorkloadReference{
					WorkloadName: wlName,
					TemplateName: templateName,
				},
			},
		}
	}

	testCases := map[string]struct {
		job               *batch.Job
		pods              []*v1.Pod
		existingWorkloads []*schedulingv1alpha3.Workload
		existingPodGroups []*schedulingv1alpha3.PodGroup

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
			existingWorkloads: []*schedulingv1alpha3.Workload{makeWorkload(workloadName, baseJob.Name)},
			existingPodGroups: []*schedulingv1alpha3.PodGroup{makePodGroup(podGroupName, workloadName)},
			wantPodGroup:      true,
			wantPodGroupName:  podGroupName,
		},
		"discovers existing objects even when pods exist": {
			job:               baseJob,
			pods:              []*v1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "pod-0", Namespace: metav1.NamespaceDefault}}},
			existingWorkloads: []*schedulingv1alpha3.Workload{makeWorkload(workloadName, baseJob.Name)},
			existingPodGroups: []*schedulingv1alpha3.PodGroup{makePodGroup(podGroupName, workloadName)},
			wantPodGroup:      true,
			wantPodGroupName:  podGroupName,
		},
		"workload found but no PodGroup creates one": {
			job:               baseJob,
			existingWorkloads: []*schedulingv1alpha3.Workload{makeWorkload("user-workload", baseJob.Name)},
			wantPodGroup:      true,
		},

		"BYO Workload (no controller ownerRef) is ignored; controller creates its own": {
			job: baseJob,
			existingWorkloads: []*schedulingv1alpha3.Workload{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "user-created-workload",
						Namespace: metav1.NamespaceDefault,
						UID:       types.UID("user-wl-uid"),
					},
					Spec: schedulingv1alpha3.WorkloadSpec{
						ControllerRef: newWorkloadJobControllerRef(baseJob.Name),
						PodGroupTemplates: []schedulingv1alpha3.PodGroupTemplate{
							{
								Name: templateName,
								SchedulingPolicy: schedulingv1alpha3.PodGroupSchedulingPolicy{
									Gang: &schedulingv1alpha3.GangSchedulingPolicy{MinCount: 4},
								},
							},
						},
					},
				},
			},
			wantPodGroup:     true,
			wantPodGroupName: podGroupName,
		},
		"ambiguous Workloads fall back": {
			job: baseJob,
			existingWorkloads: []*schedulingv1alpha3.Workload{
				makeWorkload("workload-1", baseJob.Name),
				makeWorkload("workload-2", baseJob.Name),
			},
		},
		"BYO PodGroup (no controller ownerRef) is ignored; controller creates its own": {
			job:               baseJob,
			existingWorkloads: []*schedulingv1alpha3.Workload{makeWorkload(workloadName, baseJob.Name)},
			existingPodGroups: []*schedulingv1alpha3.PodGroup{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "external-pg",
						Namespace: metav1.NamespaceDefault,
						UID:       types.UID("ext-pg-uid"),
					},
					Spec: schedulingv1alpha3.PodGroupSpec{
						WorkloadRef: &schedulingv1alpha3.WorkloadReference{
							WorkloadName: workloadName,
						},
					},
				},
			},
			wantPodGroup:     true,
			wantPodGroupName: podGroupName,
		},
		"ambiguous PodGroups fall back": {
			job:               baseJob,
			existingWorkloads: []*schedulingv1alpha3.Workload{makeWorkload(workloadName, baseJob.Name)},
			existingPodGroups: []*schedulingv1alpha3.PodGroup{
				makePodGroup("pg-1", workloadName),
				makePodGroup("pg-2", workloadName),
			},
		},
		"unsupported PodGroupTemplate count falls back": {
			job: baseJob,
			existingWorkloads: []*schedulingv1alpha3.Workload{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:            workloadName,
						Namespace:       metav1.NamespaceDefault,
						UID:             types.UID(workloadName + "-uid"),
						OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(baseJob, controllerKind)},
					},
					Spec: schedulingv1alpha3.WorkloadSpec{
						ControllerRef: newWorkloadJobControllerRef(baseJob.Name),
						PodGroupTemplates: []schedulingv1alpha3.PodGroupTemplate{
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
			existingWorkloads:     []*schedulingv1alpha3.Workload{makeWorkload(workloadName, baseJob.Name)},
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

	if got := wl.Annotations[batch.JobWorkloadManagedByAnnotation]; got != batch.JobControllerName {
		t.Errorf("managed-by annotation = %q, want %q", got, batch.JobControllerName)
	}
}

func TestCreatePodGroup(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	clientSet := fake.NewClientset()
	jm, _ := newControllerWithSchedulingInformers(ctx, t, clientSet)
	job := newGangSchedulingJob("my-job", 4)
	templateName := fmt.Sprintf("%s-pgt-%d", job.Name, 0)

	testCases := map[string]struct {
		workloadName      string
		linkWorkloadOwner bool
		wantOwnerRefs     int
	}{
		"root Job links the Workload as a non-controller owner": {
			workloadName:      "root-workload",
			linkWorkloadOwner: true,
			wantOwnerRefs:     2,
		},
		"delegated Job does not link the parent-owned Workload": {
			workloadName:      "delegated-workload",
			linkWorkloadOwner: false,
			wantOwnerRefs:     1,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			workload := &schedulingv1alpha3.Workload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      tc.workloadName,
					Namespace: job.Namespace,
					UID:       "wl-uid",
				},
				Spec: schedulingv1alpha3.WorkloadSpec{
					PodGroupTemplates: []schedulingv1alpha3.PodGroupTemplate{
						{
							Name: templateName,
							SchedulingPolicy: schedulingv1alpha3.PodGroupSchedulingPolicy{
								Gang: &schedulingv1alpha3.GangSchedulingPolicy{MinCount: 4},
							},
						},
					},
				},
			}

			pg, err := jm.createPodGroup(ctx, job, workload, "", tc.linkWorkloadOwner)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			expectedName := computePodGroupName(workload.Name, templateName)
			if pg.Name != expectedName {
				t.Errorf("PodGroup name = %q, want %q", pg.Name, expectedName)
			}
			if len(pg.OwnerReferences) != tc.wantOwnerRefs {
				t.Fatalf("expected %d ownerReferences, got %d", tc.wantOwnerRefs, len(pg.OwnerReferences))
			}
		})
	}
}

func TestSyncGangMinCount(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// The Job opts into gang with minCount=5, so the desired runtime size is 5.
	job := newGangSchedulingJob("scale-job", 5)
	wlName := computeWorkloadName(job)
	tmplName := podGroupTemplateName(job)
	pgName := computePodGroupName(wlName, tmplName)

	// Controller-created objects always carry a controller ownerRef; the
	// managed-by annotation (managed) is the additional signal the controller
	// requires before mutating them. BYO objects lack the annotation.
	makeWL := func(minCount int32, managed bool) *schedulingv1alpha3.Workload {
		wl := &schedulingv1alpha3.Workload{
			ObjectMeta: metav1.ObjectMeta{
				Name:            wlName,
				Namespace:       job.Namespace,
				UID:             "wl-uid",
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(job, controllerKind)},
			},
			Spec: schedulingv1alpha3.WorkloadSpec{
				ControllerRef: newWorkloadJobControllerRef(job.Name),
				PodGroupTemplates: []schedulingv1alpha3.PodGroupTemplate{{
					Name:             tmplName,
					SchedulingPolicy: schedulingv1alpha3.PodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.GangSchedulingPolicy{MinCount: minCount}},
				}},
			},
		}
		if managed {
			wl.Annotations = map[string]string{batch.JobWorkloadManagedByAnnotation: batch.JobControllerName}
		}
		return wl
	}
	makePG := func(minCount int32, managed bool) *schedulingv1alpha3.PodGroup {
		pg := &schedulingv1alpha3.PodGroup{
			ObjectMeta: metav1.ObjectMeta{
				Name:            pgName,
				Namespace:       job.Namespace,
				UID:             "pg-uid",
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(job, controllerKind)},
			},
			Spec: schedulingv1alpha3.PodGroupSpec{
				WorkloadRef:      &schedulingv1alpha3.WorkloadReference{WorkloadName: wlName, TemplateName: tmplName},
				SchedulingPolicy: schedulingv1alpha3.PodGroupSchedulingPolicy{Gang: &schedulingv1alpha3.GangSchedulingPolicy{MinCount: minCount}},
			},
		}
		if managed {
			pg.Annotations = map[string]string{batch.JobWorkloadManagedByAnnotation: batch.JobControllerName}
		}
		return pg
	}

	testCases := map[string]struct {
		workload     *schedulingv1alpha3.Workload
		podGroup     *schedulingv1alpha3.PodGroup
		wantMinCount int32
	}{
		"managed objects are resized to the Job's minCount": {
			workload: makeWL(2, true), podGroup: makePG(2, true), wantMinCount: 5,
		},
		"already at desired minCount is left unchanged": {
			workload: makeWL(5, true), podGroup: makePG(5, true), wantMinCount: 5,
		},
		"BYO objects (no managed-by marker) are never resized": {
			workload: makeWL(2, false), podGroup: makePG(2, false), wantMinCount: 2,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			clientSet := fake.NewClientset(tc.workload, tc.podGroup)
			jm, _ := newControllerWithSchedulingInformers(ctx, t, clientSet)

			if err := jm.syncGangMinCount(ctx, job, tc.workload, tc.podGroup); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			gotWL, err := clientSet.SchedulingV1alpha3().Workloads(job.Namespace).Get(ctx, wlName, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("get workload: %v", err)
			}
			if got := gotWL.Spec.PodGroupTemplates[0].SchedulingPolicy.Gang.MinCount; got != tc.wantMinCount {
				t.Errorf("workload minCount = %d, want %d", got, tc.wantMinCount)
			}
			gotPG, err := clientSet.SchedulingV1alpha3().PodGroups(job.Namespace).Get(ctx, pgName, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("get podGroup: %v", err)
			}
			if got := gotPG.Spec.SchedulingPolicy.Gang.MinCount; got != tc.wantMinCount {
				t.Errorf("podGroup minCount = %d, want %d", got, tc.wantMinCount)
			}
		})
	}
}
