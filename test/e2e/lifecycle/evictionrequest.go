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

package lifecycle

import (
	"context"

	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2econformance "k8s.io/kubernetes/test/e2e/framework/conformance"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = framework.SIGDescribe("node")("EvictionRequest", framework.WithFeatureGate(features.EvictionRequestAPI), func() {
	f := framework.NewDefaultFramework("evictionrequest-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	f.Context("CRUD Tests", func() {
		/*
			Testname: CRUD operations for EvictionRequests
			Description: kube-apiserver must support create/get/list/update/patch/delete
			operations for lifecycle.k8s.io/v1alpha1 EvictionRequest.
		*/
		framework.It("EvictionRequest API availability", func(ctx context.Context) {
			e2econformance.TestResource(ctx, f,
				&e2econformance.ResourceTestcase[*lifecyclev1alpha1.EvictionRequest]{
					GVR:        lifecyclev1alpha1.SchemeGroupVersion.WithResource("evictionrequests"),
					Namespaced: new(true),
					InitialSpec: &lifecyclev1alpha1.EvictionRequest{
						Spec: lifecyclev1alpha1.EvictionRequestSpec{
							Target: lifecyclev1alpha1.EvictionRequestTarget{
								Pod: &lifecyclev1alpha1.EvictionRequestPodReference{
									Name: "foo",
									UID:  "b37d123b-8637-4b23-a8f3-41523c344fc2",
								}},
							Requester: "foo.example.com/bar",
							Intent:    lifecyclev1alpha1.EvictionRequestIntentEviction,
						},
					},
					UpdateSpec: func(obj *lifecyclev1alpha1.EvictionRequest) *lifecyclev1alpha1.EvictionRequest {
						obj.Spec.Intent = lifecyclev1alpha1.EvictionRequestIntentWithdrawn
						return obj
					},
					UpdateStatus: func(obj *lifecyclev1alpha1.EvictionRequest) *lifecyclev1alpha1.EvictionRequest {
						obj.Status.Conditions = append(obj.Status.Conditions, metav1.Condition{
							Type:               "FooCondition",
							Status:             metav1.ConditionFalse,
							Reason:             "FooReason",
							Message:            "Test status condition message",
							LastTransitionTime: metav1.Now(),
						})
						return obj
					},
					StrategicMergePatchSpec: `{"spec": {"intent": "Withdrawn"}}`,
				},
			)
		})

		/*
			Testname: CRUD operations for Evictions
			Description: kube-apiserver must support create/get/list/update/patch/delete
			operations for lifecycle.k8s.io/v1alpha1 Eviction.
		*/
		framework.It("Eviction API availability", func(ctx context.Context) {
			e2econformance.TestResource(ctx, f,
				&e2econformance.ResourceTestcase[*lifecyclev1alpha1.Eviction]{
					GVR:        lifecyclev1alpha1.SchemeGroupVersion.WithResource("evictions"),
					Namespaced: new(true),
					InitialSpec: &lifecyclev1alpha1.Eviction{
						Spec: lifecyclev1alpha1.EvictionSpec{
							Target: lifecyclev1alpha1.EvictionTarget{
								Pod: &lifecyclev1alpha1.EvictionPodReference{
									Name: "foo",
									UID:  "b37d123b-8637-4b23-a8f3-41523c344fc2",
								}},
						},
					},
					UpdateSpec: func(obj *lifecyclev1alpha1.Eviction) *lifecyclev1alpha1.Eviction {
						obj.Labels["foo"] = "bar"
						return obj
					},
					UpdateStatus: func(obj *lifecyclev1alpha1.Eviction) *lifecyclev1alpha1.Eviction {
						obj.Status.Conditions = append(obj.Status.Conditions, metav1.Condition{
							Type:               "FooCondition",
							Status:             metav1.ConditionFalse,
							Reason:             "FooReason",
							Message:            "Test status condition message",
							LastTransitionTime: metav1.Now(),
						})
						return obj
					},
					StrategicMergePatchSpec: `{"metadata": {"labels": {"foo": "bar"}}}`,
				},
			)
		})
	})
})
