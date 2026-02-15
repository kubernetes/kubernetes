/*
Copyright 2025 The Kubernetes Authors.

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

package scheduling

import (
	"context"

	schedulingv1alpha2 "k8s.io/api/scheduling/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2econformance "k8s.io/kubernetes/test/e2e/framework/conformance"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

var _ = SIGDescribe("Workload", framework.WithFeatureGate(features.GenericWorkload), func() {
	f := framework.NewDefaultFramework("workload-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	f.Context("CRUD Tests", func() {
		/*
			Release: v1.36
			Testname: CRUD operations for Workloads
			Description: kube-apiserver must support create/get/list/update/patch/delete operations for scheduling.k8s.io/v1alpha2 Workload.
		*/
		framework.It("Workload API availability", func(ctx context.Context) {
			e2econformance.TestResource(ctx, f,
				&e2econformance.ResourceTestcase[*schedulingv1alpha2.Workload]{
					GVR:        schedulingv1alpha2.SchemeGroupVersion.WithResource("workloads"),
					Namespaced: ptr.To(true),
					InitialSpec: &schedulingv1alpha2.Workload{
						Spec: schedulingv1alpha2.WorkloadSpec{
							PodGroupTemplates: []schedulingv1alpha2.PodGroupTemplate{
								{
									Name: "pg1",
									SchedulingPolicy: schedulingv1alpha2.PodGroupSchedulingPolicy{
										Gang: &schedulingv1alpha2.GangSchedulingPolicy{
											MinCount: 5,
										},
									},
								},
							},
						},
					},
					UpdateSpec: func(obj *schedulingv1alpha2.Workload) *schedulingv1alpha2.Workload {
						obj.Spec.ControllerRef = &schedulingv1alpha2.TypedLocalObjectReference{
							Kind: "foo",
							Name: "bar",
						}
						return obj
					},
					StrategicMergePatchSpec: `{"metadata": {"labels": {"foo": "bar"}}}`,
				},
			)
		})

		/*
			Release: v1.36
			Testname: CRUD operations for PodGroups
			Description: kube-apiserver must support create/get/list/update/patch/delete operations for scheduling.k8s.io/v1alpha2 PodGroup.
		*/
		framework.It("PodGroup API availability", func(ctx context.Context) {
			e2econformance.TestResource(ctx, f,
				&e2econformance.ResourceTestcase[*schedulingv1alpha2.PodGroup]{
					GVR:        schedulingv1alpha2.SchemeGroupVersion.WithResource("podgroups"),
					Namespaced: ptr.To(true),
					InitialSpec: &schedulingv1alpha2.PodGroup{
						Spec: schedulingv1alpha2.PodGroupSpec{
							PodGroupTemplateRef: &schedulingv1alpha2.PodGroupTemplateReference{
								WorkloadName:         "w1",
								PodGroupTemplateName: "pg1",
							},
							SchedulingPolicy: schedulingv1alpha2.PodGroupSchedulingPolicy{
								Gang: &schedulingv1alpha2.GangSchedulingPolicy{
									MinCount: 5,
								},
							},
						},
					},
					UpdateSpec: func(obj *schedulingv1alpha2.PodGroup) *schedulingv1alpha2.PodGroup {
						obj.Labels["foo"] = "bar"
						return obj
					},
					UpdateStatus: func(obj *schedulingv1alpha2.PodGroup) *schedulingv1alpha2.PodGroup {
						obj.Status.Conditions = append(obj.Status.Conditions, metav1.Condition{
							Type:   "PodGroupScheduled",
							Status: metav1.ConditionTrue,
						})
						return obj
					},
					StrategicMergePatchSpec: `{"metadata": {"labels": {"foo": "bar"}}}`,
				},
			)
		})
	})
})
