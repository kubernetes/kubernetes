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
	"testing"

	schedulingv1alpha2 "k8s.io/api/scheduling/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func TestGangSchedulingParallelism(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	gangPG := &schedulingv1alpha2.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "gang-pg",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: schedulingv1alpha2.PodGroupSpec{
			SchedulingPolicy: schedulingv1alpha2.PodGroupSchedulingPolicy{
				Gang: &schedulingv1alpha2.GangSchedulingPolicy{MinCount: 4},
			},
		},
	}

	basicPG := &schedulingv1alpha2.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "basic-pg",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: schedulingv1alpha2.PodGroupSpec{
			SchedulingPolicy: schedulingv1alpha2.PodGroupSchedulingPolicy{
				Basic: &schedulingv1alpha2.BasicSchedulingPolicy{},
			},
		},
	}

	indexedMode := batch.IndexedCompletion

	baseJob := func(sgName *string) *batch.Job {
		j := &batch.Job{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-job",
				Namespace: metav1.NamespaceDefault,
				UID:       types.UID("test-job-uid"),
			},
			Spec: batch.JobSpec{
				CompletionMode: &indexedMode,
				Completions:    ptr.To[int32](4),
				Parallelism:    ptr.To[int32](4),
			},
		}
		if sgName != nil {
			j.Spec.Template.Spec.SchedulingGroup = &api.PodSchedulingGroup{
				PodGroupName: sgName,
			}
		}
		return j
	}

	cases := map[string]struct {
		enableFeatureGate bool
		oldJob            *batch.Job
		newJob            *batch.Job
		podGroups         []*schedulingv1alpha2.PodGroup
		wantErr           bool
	}{
		"feature gate disabled: allows parallelism change": {
			oldJob: baseJob(ptr.To("gang-pg")),
			newJob: func() *batch.Job {
				j := baseJob(ptr.To("gang-pg"))
				j.Spec.Parallelism = ptr.To[int32](2)
				return j
			}(),
			podGroups: []*schedulingv1alpha2.PodGroup{gangPG},
		},
		"no schedulingGroup: skips check (handled by validation)": {
			enableFeatureGate: true,
			oldJob:            baseJob(nil),
			newJob: func() *batch.Job {
				j := baseJob(nil)
				j.Spec.Parallelism = ptr.To[int32](2)
				return j
			}(),
		},
		"parallelism unchanged: allows": {
			enableFeatureGate: true,
			oldJob:            baseJob(ptr.To("gang-pg")),
			newJob:            baseJob(ptr.To("gang-pg")),
			podGroups:         []*schedulingv1alpha2.PodGroup{gangPG},
		},
		"gang PodGroup: rejects parallelism change": {
			enableFeatureGate: true,
			oldJob:            baseJob(ptr.To("gang-pg")),
			newJob: func() *batch.Job {
				j := baseJob(ptr.To("gang-pg"))
				j.Spec.Parallelism = ptr.To[int32](2)
				return j
			}(),
			podGroups: []*schedulingv1alpha2.PodGroup{gangPG},
			wantErr:   true,
		},
		"basic PodGroup: allows parallelism change": {
			enableFeatureGate: true,
			oldJob:            baseJob(ptr.To("basic-pg")),
			newJob: func() *batch.Job {
				j := baseJob(ptr.To("basic-pg"))
				j.Spec.Parallelism = ptr.To[int32](2)
				return j
			}(),
			podGroups: []*schedulingv1alpha2.PodGroup{basicPG},
		},
		"PodGroup not found: allows parallelism change": {
			enableFeatureGate: true,
			oldJob:            baseJob(ptr.To("missing-pg")),
			newJob: func() *batch.Job {
				j := baseJob(ptr.To("missing-pg"))
				j.Spec.Parallelism = ptr.To[int32](2)
				return j
			}(),
		},
		"controller-created gang PodGroup - rejects parallelism change": {
			enableFeatureGate: true,
			oldJob:            baseJob(nil),
			newJob: func() *batch.Job {
				j := baseJob(nil)
				j.Spec.Parallelism = ptr.To[int32](2)
				return j
			}(),
			podGroups: []*schedulingv1alpha2.PodGroup{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "controller-created-pg",
						Namespace: metav1.NamespaceDefault,
						OwnerReferences: []metav1.OwnerReference{
							{
								APIVersion: "batch/v1",
								Kind:       "Job",
								Name:       "test-job",
								UID:        types.UID("test-job-uid"),
								Controller: ptr.To(true),
							},
						},
					},
					Spec: schedulingv1alpha2.PodGroupSpec{
						SchedulingPolicy: schedulingv1alpha2.PodGroupSchedulingPolicy{
							Gang: &schedulingv1alpha2.GangSchedulingPolicy{MinCount: 4},
						},
					},
				},
			},
			wantErr: true,
		},
		"controller-created basic PodGroup - allows parallelism change": {
			enableFeatureGate: true,
			oldJob:            baseJob(nil),
			newJob: func() *batch.Job {
				j := baseJob(nil)
				j.Spec.Parallelism = ptr.To[int32](2)
				return j
			}(),
			podGroups: []*schedulingv1alpha2.PodGroup{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "controller-created-pg",
						Namespace: metav1.NamespaceDefault,
						OwnerReferences: []metav1.OwnerReference{
							{
								APIVersion: "batch/v1",
								Kind:       "Job",
								Name:       "test-job",
								UID:        types.UID("test-job-uid"),
								Controller: ptr.To(true),
							},
						},
					},
					Spec: schedulingv1alpha2.PodGroupSpec{
						SchedulingPolicy: schedulingv1alpha2.PodGroupSchedulingPolicy{
							Basic: &schedulingv1alpha2.BasicSchedulingPolicy{},
						},
					},
				},
			},
		},
		"gang PodGroup owned by different Job - allows parallelism change": {
			enableFeatureGate: true,
			oldJob:            baseJob(nil),
			newJob: func() *batch.Job {
				j := baseJob(nil)
				j.Spec.Parallelism = ptr.To[int32](2)
				return j
			}(),
			podGroups: []*schedulingv1alpha2.PodGroup{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "other-gang-pg",
						Namespace: metav1.NamespaceDefault,
						OwnerReferences: []metav1.OwnerReference{
							{
								APIVersion: "batch/v1",
								Kind:       "Job",
								Name:       "other-job",
								UID:        types.UID("other-job-uid"),
								Controller: ptr.To(true),
							},
						},
					},
					Spec: schedulingv1alpha2.PodGroupSpec{
						SchedulingPolicy: schedulingv1alpha2.PodGroupSchedulingPolicy{
							Gang: &schedulingv1alpha2.GangSchedulingPolicy{MinCount: 4},
						},
					},
				},
			},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t,
				utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.GenericWorkload:       tc.enableFeatureGate,
					features.EnableWorkloadWithJob: tc.enableFeatureGate,
				})

			p := NewPlugin()
			p.InspectFeatureGates(utilfeature.DefaultFeatureGate)

			informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
			p.SetExternalKubeInformerFactory(informerFactory)

			for _, pg := range tc.podGroups {
				if err := informerFactory.Scheduling().V1alpha2().PodGroups().Informer().GetStore().Add(pg); err != nil {
					t.Fatalf("failed to add PodGroup: %v", err)
				}
			}

			attrs := admission.NewAttributesRecord(
				tc.newJob, tc.oldJob,
				schema.GroupVersionKind{Group: "batch", Version: "v1", Kind: "Job"},
				metav1.NamespaceDefault, "test-job",
				batch.Resource("jobs").WithVersion("v1"),
				"", admission.Update, &metav1.UpdateOptions{}, false, nil,
			)

			err := p.Validate(ctx, attrs, nil)
			if tc.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
