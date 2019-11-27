/*
Copyright 2019 The Kubernetes Authors.

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

package apiapproval

import (
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestCalculateCondition(t *testing.T) {
	noConditionFn := func(t *testing.T, condition *apiextensions.CustomResourceDefinitionCondition) {
		t.Helper()
		if condition != nil {
			t.Fatal(condition)
		}
	}

	verifyCondition := func(status apiextensions.ConditionStatus, message string) func(t *testing.T, condition *apiextensions.CustomResourceDefinitionCondition) {
		return func(t *testing.T, condition *apiextensions.CustomResourceDefinitionCondition) {
			t.Helper()
			if condition == nil {
				t.Fatal("missing condition")
			}
			if e, a := status, condition.Status; e != a {
				t.Errorf("expected %v, got %v", e, a)
			}
			if e, a := message, condition.Message; e != a {
				t.Errorf("expected %v, got %v", e, a)
			}
		}
	}

	tests := []struct {
		name string

		group             string
		annotationValue   string
		validateCondition func(t *testing.T, condition *apiextensions.CustomResourceDefinitionCondition)
	}{
		{
			name:              "for other group",
			group:             "other.io",
			annotationValue:   "",
			validateCondition: noConditionFn,
		},
		{
			name:              "missing annotation",
			group:             "sigs.k8s.io",
			annotationValue:   "",
			validateCondition: verifyCondition(apiextensions.ConditionFalse, `protected groups must have approval annotation "api-approved.kubernetes.io", see https://github.com/kubernetes/enhancements/pull/1111`),
		},
		{
			name:              "invalid annotation",
			group:             "sigs.k8s.io",
			annotationValue:   "bad value",
			validateCondition: verifyCondition(apiextensions.ConditionFalse, `protected groups must have approval annotation "api-approved.kubernetes.io" with either a URL or a reason starting with "unapproved", see https://github.com/kubernetes/enhancements/pull/1111`),
		},
		{
			name:              "approved",
			group:             "sigs.k8s.io",
			annotationValue:   "https://github.com/kubernetes/kubernetes/pull/79724",
			validateCondition: verifyCondition(apiextensions.ConditionTrue, `approved in https://github.com/kubernetes/kubernetes/pull/79724`),
		},
		{
			name:              "unapproved",
			group:             "sigs.k8s.io",
			annotationValue:   "unapproved for reasons",
			validateCondition: verifyCondition(apiextensions.ConditionFalse, `not approved: "unapproved for reasons"`),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			crd := &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Annotations: map[string]string{v1beta1.KubeAPIApprovedAnnotation: test.annotationValue}},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: test.group,
				},
			}

			actual := calculateCondition(crd)
			test.validateCondition(t, actual)

		})
	}

}
