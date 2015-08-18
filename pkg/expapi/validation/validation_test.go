/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package validation

import (
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/expapi"
	errors "k8s.io/kubernetes/pkg/util/fielderrors"
)

func TestValidateHorizontalPodAutoscaler(t *testing.T) {
	successCases := []expapi.HorizontalPodAutoscaler{
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.HorizontalPodAutoscalerSpec{
				ScaleRef: &expapi.SubresourceReference{
					Subresource: "scale",
				},
				MinCount: 1,
				MaxCount: 5,
				Target:   expapi.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse("0.8")},
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateHorizontalPodAutoscaler(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]expapi.HorizontalPodAutoscaler{
		"must be non-negative": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.HorizontalPodAutoscalerSpec{
				ScaleRef: &expapi.SubresourceReference{
					Subresource: "scale",
				},
				MinCount: -1,
				MaxCount: 5,
				Target:   expapi.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse("0.8")},
			},
		},
		"must be bigger or equal to minCount": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.HorizontalPodAutoscalerSpec{
				ScaleRef: &expapi.SubresourceReference{
					Subresource: "scale",
				},
				MinCount: 7,
				MaxCount: 5,
				Target:   expapi.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse("0.8")},
			},
		},
		"invalid value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.HorizontalPodAutoscalerSpec{
				ScaleRef: &expapi.SubresourceReference{
					Subresource: "scale",
				},
				MinCount: 1,
				MaxCount: 5,
				Target:   expapi.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse("-0.8")},
			},
		},
		"resource not supported": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.HorizontalPodAutoscalerSpec{
				ScaleRef: &expapi.SubresourceReference{
					Subresource: "scale",
				},
				MinCount: 1,
				MaxCount: 5,
				Target:   expapi.ResourceConsumption{Resource: api.ResourceName("NotSupportedResource"), Quantity: resource.MustParse("0.8")},
			},
		},
		"required value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.HorizontalPodAutoscalerSpec{
				MinCount: 1,
				MaxCount: 5,
				Target:   expapi.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse("0.8")},
			},
		},
	}

	for k, v := range errorCases {
		errs := ValidateHorizontalPodAutoscaler(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		} else if !strings.Contains(errs[0].Error(), k) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], k)
		}
	}
}

func TestValidateJob(t *testing.T) {
	validSelector := map[string]string{"a": "b"}
	validPodTemplateSpec := api.PodTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Labels: validSelector,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
		},
	}
	successCases := []expapi.Job{
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.JobSpec{
				Selector: validSelector,
				Template: &validPodTemplateSpec,
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateJob(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	negative := -1
	errorCases := map[string]expapi.Job{
		"spec.parallelism:must be non-negative": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.JobSpec{
				Parallelism: &negative,
				Selector:    validSelector,
				Template:    &validPodTemplateSpec,
			},
		},
		"spec.completions:must be non-negative": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.JobSpec{
				Completions: &negative,
				Selector:    validSelector,
				Template:    &validPodTemplateSpec,
			},
		},
		"spec.selector:required value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.JobSpec{
				Selector: map[string]string{},
				Template: &validPodTemplateSpec,
			},
		},
		"spec.template:required value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.JobSpec{
				Selector: validSelector,
			},
		},
		"spec.template.labels:selector does not match template": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.JobSpec{
				Selector: validSelector,
				Template: &api.PodTemplateSpec{
					ObjectMeta: api.ObjectMeta{
						Labels: map[string]string{"y": "z"},
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyOnFailure,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
					},
				},
			},
		},
		"spec.template.spec.restartPolicy:unsupported value": {
			ObjectMeta: api.ObjectMeta{
				Name:      "myjob",
				Namespace: api.NamespaceDefault,
			},
			Spec: expapi.JobSpec{
				Selector: validSelector,
				Template: &api.PodTemplateSpec{
					ObjectMeta: api.ObjectMeta{
						Labels: validSelector,
					},
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicyAlways,
						DNSPolicy:     api.DNSClusterFirst,
						Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent"}},
					},
				},
			},
		},
	}

	for k, v := range errorCases {
		errs := ValidateJob(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		} else {
			s := strings.Split(k, ":")
			err := errs[0].(*errors.ValidationError)
			if err.Field != s[0] || !strings.Contains(err.Error(), s[1]) {
				t.Errorf("unexpected error: %v, expected: %s", errs[0], k)
			}
		}
	}
}
