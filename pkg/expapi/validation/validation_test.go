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
				Target:   expapi.TargetConsumption{api.ResourceCPU, resource.MustParse("0.8")},
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
				Target:   expapi.TargetConsumption{api.ResourceCPU, resource.MustParse("0.8")},
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
				Target:   expapi.TargetConsumption{api.ResourceCPU, resource.MustParse("0.8")},
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
				Target:   expapi.TargetConsumption{api.ResourceCPU, resource.MustParse("-0.8")},
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
				Target:   expapi.TargetConsumption{api.ResourceName("NotSupportedResource"), resource.MustParse("0.8")},
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
				Target:   expapi.TargetConsumption{api.ResourceCPU, resource.MustParse("0.8")},
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
