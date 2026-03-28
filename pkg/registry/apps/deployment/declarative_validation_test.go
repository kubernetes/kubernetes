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

package deployment

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
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
				Resource:          "deployments",
				IsResourceRequest: true,
				Verb:              "create",
			})

			testCases := map[string]struct {
				input        apps.Deployment
				expectedErrs field.ErrorList
			}{
				"valid": {
					input: mkValidDeployment(),
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
				oldObj       apps.Deployment
				updateObj    apps.Deployment
				expectedErrs field.ErrorList
			}{
				"valid update": {
					oldObj:    mkValidDeployment(),
					updateObj: mkValidDeployment(),
				},
				"immutable selector": {
					oldObj: mkValidDeployment(),
					updateObj: mkValidDeployment(func(obj *apps.Deployment) {
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
						Resource:          "deployments",
						Name:              "valid-deployment",
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

func mkValidDeployment(tweaks ...func(*apps.Deployment)) apps.Deployment {
	maxSurge := intstr.FromInt32(1)
	maxUnavailable := intstr.FromInt32(0)
	obj := apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-deployment",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: apps.DeploymentSpec{
			Replicas: 1,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "test"},
			},
			Strategy: apps.DeploymentStrategy{
				Type: apps.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &apps.RollingUpdateDeployment{
					MaxSurge:       maxSurge,
					MaxUnavailable: maxUnavailable,
				},
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
