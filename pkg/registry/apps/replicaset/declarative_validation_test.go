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

package replicaset

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"
)

var apiVersions = []string{"v1"}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "apps",
				APIVersion:        apiVersion,
				Resource:          "replicasets",
				IsResourceRequest: true,
				Verb:              "create",
			})

			testCases := map[string]struct {
				input        apps.ReplicaSet
				expectedErrs field.ErrorList
			}{
				"valid": {
					input: mkValidReplicaSet(),
				},
			}
			for name, tc := range testCases {
				t.Run(name, func(t *testing.T) {
					apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
				})
			}
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testCases := map[string]struct {
				oldObj       apps.ReplicaSet
				updateObj    apps.ReplicaSet
				expectedErrs field.ErrorList
			}{
				"valid update": {
					oldObj:    mkValidReplicaSet(),
					updateObj: mkValidReplicaSet(),
				},
				"immutable selector": {
					oldObj: mkValidReplicaSet(),
					updateObj: mkValidReplicaSet(func(obj *apps.ReplicaSet) {
						obj.Spec.Selector = &metav1.LabelSelector{
							MatchLabels: map[string]string{"app": "changed"},
						}
						obj.Spec.Template.Labels = map[string]string{"app": "changed"}
					}),
					expectedErrs: field.ErrorList{
						field.Invalid(field.NewPath("spec", "selector"), &metav1.LabelSelector{
							MatchLabels: map[string]string{"app": "changed"},
						}, "field is immutable").WithOrigin("immutable").MarkAlpha(),
					},
				},
			}
			for name, tc := range testCases {
				t.Run(name, func(t *testing.T) {
					ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
						APIPrefix:         "apis",
						APIGroup:          "apps",
						APIVersion:        apiVersion,
						Resource:          "replicasets",
						Name:              "valid-replicaset",
						IsResourceRequest: true,
						Verb:              "update",
					})
					tc.oldObj.ResourceVersion = "1"
					tc.updateObj.ResourceVersion = "2"
					apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs)
				})
			}
		})
	}
}

func mkValidReplicaSet(tweaks ...func(*apps.ReplicaSet)) apps.ReplicaSet {
	obj := apps.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-replicaset",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: apps.ReplicaSetSpec{
			Replicas: 1,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "test"},
			},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": "test"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:                     "test",
							Image:                    "test",
							ImagePullPolicy:          api.PullAlways,
							TerminationMessagePolicy: api.TerminationMessageReadFile,
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU: resource.MustParse("100m"),
								},
							},
						},
					},
					RestartPolicy:                 api.RestartPolicyAlways,
					DNSPolicy:                     api.DNSClusterFirst,
					TerminationGracePeriodSeconds: ptr.To[int64](30),
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}
