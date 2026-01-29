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

package horizontalpodautoscaler

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
)

func TestDeclarativeValidate(t *testing.T) {
	apiVersions := []string{"v1", "v2"}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	apiVersions := []string{"v1", "v2"}
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "autoscaling",
		APIVersion:        apiVersion,
		Resource:          "horizontalpodautoscalers",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input        autoscaling.HorizontalPodAutoscaler
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidHPA(),
		},
		"invalid maxReplicas (zero)": {
			input: mkValidHPA(TweakMaxReplicas(0)),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "maxReplicas"), ""),
			},
		},
		"invalid maxReplicas (negative)": {
			input: mkValidHPA(TweakMaxReplicas(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "maxReplicas"), nil, "").WithOrigin("minimum"),
			},
		},
		"invalid scaleTargetRef.kind (empty)": {
			input: mkValidHPA(TweakScaleTargetRefKind("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "scaleTargetRef", "kind"), ""),
			},
		},
		"invalid scaleTargetRef.name (empty)": {
			input: mkValidHPA(TweakScaleTargetRefName("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "scaleTargetRef", "name"), ""),
			},
		},
		"invalid scaleTargetRef.kind (special characters)": {
			input: mkValidHPA(TweakScaleTargetRefKind("asd/asd")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "scaleTargetRef", "kind"), nil, "").WithOrigin("format=k8s-path-segment-name"),
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "autoscaling",
		APIVersion:        apiVersion,
		Resource:          "horizontalpodautoscalers",
		IsResourceRequest: true,
		Verb:              "update",
	})

	testCases := map[string]struct {
		oldInput     autoscaling.HorizontalPodAutoscaler
		newInput     autoscaling.HorizontalPodAutoscaler
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldInput: mkValidHPA(func(obj *autoscaling.HorizontalPodAutoscaler) {
				obj.ResourceVersion = "1"
			}),
			newInput: mkValidHPA(func(obj *autoscaling.HorizontalPodAutoscaler) {
				obj.ResourceVersion = "1"
			}),
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.newInput, &tc.oldInput, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func TweakMaxReplicas(max int32) func(obj *autoscaling.HorizontalPodAutoscaler) {
	return func(obj *autoscaling.HorizontalPodAutoscaler) {
		obj.Spec.MaxReplicas = max
	}
}

func TweakScaleTargetRefKind(kind string) func(obj *autoscaling.HorizontalPodAutoscaler) {
	return func(obj *autoscaling.HorizontalPodAutoscaler) {
		obj.Spec.ScaleTargetRef.Kind = kind
	}
}

func TweakScaleTargetRefName(name string) func(obj *autoscaling.HorizontalPodAutoscaler) {
	return func(obj *autoscaling.HorizontalPodAutoscaler) {
		obj.Spec.ScaleTargetRef.Name = name
	}
}

func mkValidHPA(tweaks ...func(obj *autoscaling.HorizontalPodAutoscaler)) autoscaling.HorizontalPodAutoscaler {
	obj := autoscaling.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-hpa",
			Namespace: "default",
		},
		Spec: autoscaling.HorizontalPodAutoscalerSpec{
			MaxReplicas: 10,
			ScaleTargetRef: autoscaling.CrossVersionObjectReference{
				Kind: "Deployment",
				Name: "foo",
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}