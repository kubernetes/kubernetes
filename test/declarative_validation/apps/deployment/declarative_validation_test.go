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

package deployment

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	apps "k8s.io/kubernetes/pkg/apis/apps"
	api "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/apps/deployment"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func tweakPodSpec(podTweaks ...podtest.Tweak) func(*apps.Deployment) {
	return func(deploy *apps.Deployment) {
		pod := &api.Pod{
			Spec: deploy.Spec.Template.Spec,
		}
		for _, tweak := range podTweaks {
			tweak(pod)
		}
		deploy.Spec.Template.Spec = pod.Spec
	}
}

func tweakSelector(selector *metav1.LabelSelector) func(*apps.Deployment) {
	return func(deploy *apps.Deployment) {
		deploy.Spec.Selector = selector
	}
}

func mkDeployment(tweaks ...func(*apps.Deployment)) *apps.Deployment {
	deploy := &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: apps.DeploymentSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"name": "abc"}},
			Strategy: apps.DeploymentStrategy{
				Type: apps.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &apps.RollingUpdateDeployment{
					MaxSurge:       intstr.FromInt32(1),
					MaxUnavailable: intstr.FromInt32(1),
				},
			},
			Template: api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"name": "abc"}},
				Spec:       podtest.MakePodSpec(),
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(deploy)
	}
	return deploy
}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIGroup:          "apps",
			APIVersion:        apiVersion,
			IsResourceRequest: true,
			Verb:              "create",
		})
		testCases := map[string]struct {
			input        *apps.Deployment
			expectedErrs field.ErrorList
		}{
			"valid": {
				input: mkDeployment(),
			},
			"valid toleration key": {
				input: mkDeployment(tweakPodSpec(podtest.SetTolerations(api.Toleration{Key: "example.com/valid-key", Operator: api.TolerationOpExists}))),
			},
			"valid toleration key without prefix": {
				input: mkDeployment(tweakPodSpec(podtest.SetTolerations(api.Toleration{Key: "simple-key", Operator: api.TolerationOpExists}))),
			},
			"invalid toleration key format": {
				input: mkDeployment(tweakPodSpec(podtest.SetTolerations(api.Toleration{Key: "invalid key", Operator: api.TolerationOpExists}))),
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("spec", "template", "spec", "tolerations").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key").MarkAlpha(),
				},
			},
			"missing selector": {
				input: mkDeployment(tweakSelector(nil)),
				expectedErrs: field.ErrorList{
					field.Required(field.NewPath("spec", "selector"), "").MarkAlpha(),
					field.Invalid(field.NewPath("spec", "template", "metadata", "labels"), nil, "").MarkFromImperative(),
				},
			},
		}
		for k, tc := range testCases {
			t.Run(k, func(t *testing.T) {
				apitesting.VerifyValidationEquivalence(t, ctx, tc.input, registry.Strategy, tc.expectedErrs)
			})
		}
		meta.RunObjectMetaTestCases(t, ctx, mkDeployment(), registry.Strategy,
			meta.WithStringentFinalizerValidation(),
		)
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "apps",
		APIVersion:        apiVersion,
		IsResourceRequest: true,
		Verb:              "update",
	})
	meta.RunObjectMetaUpdateTestCases(t, ctx, mkDeployment(), registry.Strategy,
		meta.WithStringentFinalizerValidation(),
	)

	testCases := map[string]struct {
		old          *apps.Deployment
		update       *apps.Deployment
		expectedErrs field.ErrorList
	}{
		"valid update": {
			old:    mkDeployment(),
			update: mkDeployment(),
		},
		"selector changed": {
			old:    mkDeployment(),
			update: mkDeployment(tweakSelector(&metav1.LabelSelector{MatchLabels: map[string]string{"name": "different"}})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "template", "metadata", "labels"), nil, "").MarkFromImperative(),
				field.Invalid(field.NewPath("spec", "selector"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
	}

	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, tc.update, tc.old, registry.Strategy, tc.expectedErrs)
		})
	}
}
