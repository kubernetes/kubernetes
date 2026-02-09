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

	schedulingv1alpha1 "k8s.io/api/scheduling/v1alpha1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2econformance "k8s.io/kubernetes/test/e2e/framework/conformance"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

var _ = SIGDescribe("Workload", framework.WithFeatureGate(features.GenericWorkload), func() {
	f := framework.NewDefaultFramework("workload-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	/*
	   Release: v1.?
	   Testname: CRUD operations for Workloads
	   Description: kube-apiserver must support create/get/list/update/patch/delete operations for scheduling.k8s.io/v1alpha1 Workload.
	*/
	framework.It("Workload API availability", func(ctx context.Context) {
		e2econformance.TestResource(ctx, f,
			&e2econformance.ResourceTestcase[*schedulingv1alpha1.Workload]{
				GVR:        schedulingv1alpha1.SchemeGroupVersion.WithResource("workloads"),
				Namespaced: ptr.To(true),
				InitialSpec: &schedulingv1alpha1.Workload{
					Spec: schedulingv1alpha1.WorkloadSpec{
						PodGroups: []schedulingv1alpha1.PodGroup{
							{
								Name: "pg1",
								Policy: schedulingv1alpha1.PodGroupPolicy{
									Gang: &schedulingv1alpha1.GangSchedulingPolicy{
										MinCount: 5,
									},
								},
							},
						},
					},
				},
				UpdateSpec: func(obj *schedulingv1alpha1.Workload) *schedulingv1alpha1.Workload {
					obj.Spec.ControllerRef = &schedulingv1alpha1.TypedLocalObjectReference{
						Kind: "foo",
						Name: "bar",
					}
					return obj
				},
				StrategicMergePatchSpec: `{"metadata": {"labels": {"foo": "bar"}}}`,
			},
		)
	})
})
