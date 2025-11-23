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

package horizontalpodautoscaler

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/autoscaling"
)

var apiVersions = []string{"v2beta1", "v2beta2", "v1", "v2"}

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	for _, apiVersion := range apiVersions {
		testDeclarativeValidateForDeclarative(t, apiVersion)
	}
}

func testDeclarativeValidateForDeclarative(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "autoscaling",
		APIVersion: apiVersion,
		Resource:   "horizontalpodautoscalers",
	})
	testCases := map[string]struct {
		input        api.HorizontalPodAutoscaler
		expectedErrs field.ErrorList
	}{
		"valid: minReplicas = 5": {
			input: makeValidHPA(tweakMinReplicas(5)),
		},
		"valid: minReplicas not set (nil)": {
			input: makeValidHPA(), // Default, no minReplicas set
		},
		"invalid: minReplicas negative": {
			input: makeValidHPA(tweakMinReplicas(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "minReplicas"), int32(-1), "must be greater than or equal to 1").WithOrigin("minimum"),
			},
		},
		"invalid: maxReplicas = 0": {
			input: makeValidHPA(tweakMaxReplicas(0)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "maxReplicas"), int32(0), "must be greater than 0").WithOrigin("minimum"),
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func makeValidHPA(mutators ...func(*api.HorizontalPodAutoscaler)) api.HorizontalPodAutoscaler {
	hpa := api.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-hpa",
			Namespace: "default",
		},
		Spec: api.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: api.CrossVersionObjectReference{
				Kind:       "Deployment",
				Name:       "test-deployment",
				APIVersion: "apps/v1",
			},
			MaxReplicas: 10,
		},
	}
	for _, mutate := range mutators {
		mutate(&hpa)
	}
	return hpa
}

func tweakMinReplicas(replicas int32) func(*api.HorizontalPodAutoscaler) {
	return func(hpa *api.HorizontalPodAutoscaler) {
		hpa.Spec.MinReplicas = &replicas
	}
}

func tweakMaxReplicas(replicas int32) func(*api.HorizontalPodAutoscaler) {
	return func(hpa *api.HorizontalPodAutoscaler) {
		hpa.Spec.MaxReplicas = replicas
	}
}
