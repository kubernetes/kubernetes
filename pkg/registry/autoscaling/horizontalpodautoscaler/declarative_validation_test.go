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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
)

var apiVersions = []string{"v1", "v2"}

func TestDeclarativeValidate(t *testing.T) {

	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
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
			input: mkValidHorizontalPodAutoscaler(),
		},
		"invalid scaleTargetRef - missing name": {
			input: mkValidHorizontalPodAutoscaler(func(obj *autoscaling.HorizontalPodAutoscaler) {
				obj.Spec.ScaleTargetRef.Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "scaleTargetRef", "name"), ""),
			},
		},
		"invalid scaleTargetRef - missing kind": {
			input: mkValidHorizontalPodAutoscaler(func(obj *autoscaling.HorizontalPodAutoscaler) {
				obj.Spec.ScaleTargetRef.Kind = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "scaleTargetRef", "kind"), ""),
			},
		},

		// TODO: Add more test cases
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
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
	testCases := map[string]struct {
		oldObj       autoscaling.HorizontalPodAutoscaler
		updateObj    autoscaling.HorizontalPodAutoscaler
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidHorizontalPodAutoscaler(func(obj *autoscaling.HorizontalPodAutoscaler) { obj.ResourceVersion = "1" }),
			updateObj: mkValidHorizontalPodAutoscaler(func(obj *autoscaling.HorizontalPodAutoscaler) { obj.ResourceVersion = "1" }),
		},
		"invalid update name": {
			oldObj: mkValidHorizontalPodAutoscaler(func(obj *autoscaling.HorizontalPodAutoscaler) { obj.ResourceVersion = "1" }),
			updateObj: mkValidHorizontalPodAutoscaler(func(obj *autoscaling.HorizontalPodAutoscaler) {
				obj.ResourceVersion = "1"
				obj.Spec.ScaleTargetRef.Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "scaleTargetRef", "name"), ""),
			},
		},
		"invalid update kind": {
			oldObj: mkValidHorizontalPodAutoscaler(func(obj *autoscaling.HorizontalPodAutoscaler) { obj.ResourceVersion = "1" }),
			updateObj: mkValidHorizontalPodAutoscaler(func(obj *autoscaling.HorizontalPodAutoscaler) {
				obj.ResourceVersion = "1"
				obj.Spec.ScaleTargetRef.Kind = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "scaleTargetRef", "kind"), ""),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "autoscaling",
				APIVersion:        apiVersion,
				Resource:          "horizontalpodautoscalers",
				IsResourceRequest: true,
				Verb:              "update",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

func mkValidHorizontalPodAutoscaler(tweaks ...func(obj *autoscaling.HorizontalPodAutoscaler)) autoscaling.HorizontalPodAutoscaler {
	obj := autoscaling.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-hpa",
			Namespace: "default",
		},
		Spec: autoscaling.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscaling.CrossVersionObjectReference{
				Name:       "hpaSpec",
				Kind:       "Deployment",
				APIVersion: "apps/v1",
			},
			MaxReplicas: 10,
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}
