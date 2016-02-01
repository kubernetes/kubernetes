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
	"fmt"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func TestValidateHorizontalPodAutoscaler(t *testing.T) {
	successCases := []autoscaling.HorizontalPodAutoscaler{
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: autoscaling.HorizontalPodAutoscalerSpec{
				ScaleRef: autoscaling.SubresourceReference{
					Kind:        "ReplicationController",
					Name:        "myrc",
					Subresource: "scale",
				},
				MinReplicas:    newInt(1),
				MaxReplicas:    5,
				CPUUtilization: &autoscaling.CPUTargetUtilization{TargetPercentage: 70},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: autoscaling.HorizontalPodAutoscalerSpec{
				ScaleRef: autoscaling.SubresourceReference{
					Kind:        "ReplicationController",
					Name:        "myrc",
					Subresource: "scale",
				},
				MinReplicas: newInt(1),
				MaxReplicas: 5,
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateHorizontalPodAutoscaler(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		horizontalPodAutoscaler autoscaling.HorizontalPodAutoscaler
		msg                     string
	}{
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleRef:       autoscaling.SubresourceReference{Name: "myrc", Subresource: "scale"},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &autoscaling.CPUTargetUtilization{TargetPercentage: 70},
				},
			},
			msg: "scaleRef.kind: Required",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleRef:       autoscaling.SubresourceReference{Kind: "..", Name: "myrc", Subresource: "scale"},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &autoscaling.CPUTargetUtilization{TargetPercentage: 70},
				},
			},
			msg: "scaleRef.kind: Invalid",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleRef:       autoscaling.SubresourceReference{Kind: "ReplicationController", Subresource: "scale"},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &autoscaling.CPUTargetUtilization{TargetPercentage: 70},
				},
			},
			msg: "scaleRef.name: Required",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleRef:       autoscaling.SubresourceReference{Kind: "ReplicationController", Name: "..", Subresource: "scale"},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &autoscaling.CPUTargetUtilization{TargetPercentage: 70},
				},
			},
			msg: "scaleRef.name: Invalid",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleRef:       autoscaling.SubresourceReference{Kind: "ReplicationController", Name: "myrc", Subresource: ""},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &autoscaling.CPUTargetUtilization{TargetPercentage: 70},
				},
			},
			msg: "scaleRef.subresource: Required",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleRef:       autoscaling.SubresourceReference{Kind: "ReplicationController", Name: "myrc", Subresource: ".."},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &autoscaling.CPUTargetUtilization{TargetPercentage: 70},
				},
			},
			msg: "scaleRef.subresource: Invalid",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleRef:       autoscaling.SubresourceReference{Kind: "ReplicationController", Name: "myrc", Subresource: "randomsubresource"},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &autoscaling.CPUTargetUtilization{TargetPercentage: 70},
				},
			},
			msg: "scaleRef.subresource: Unsupported",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name:      "myautoscaler",
					Namespace: api.NamespaceDefault,
				},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleRef: autoscaling.SubresourceReference{
						Subresource: "scale",
					},
					MinReplicas: newInt(-1),
					MaxReplicas: 5,
				},
			},
			msg: "must be greater than 0",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name:      "myautoscaler",
					Namespace: api.NamespaceDefault,
				},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleRef: autoscaling.SubresourceReference{
						Subresource: "scale",
					},
					MinReplicas: newInt(7),
					MaxReplicas: 5,
				},
			},
			msg: "must be greater than or equal to `minReplicas`",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name:      "myautoscaler",
					Namespace: api.NamespaceDefault,
				},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleRef: autoscaling.SubresourceReference{
						Subresource: "scale",
					},
					MinReplicas:    newInt(1),
					MaxReplicas:    5,
					CPUUtilization: &autoscaling.CPUTargetUtilization{TargetPercentage: -70},
				},
			},
			msg: "must be greater than 0",
		},
	}

	for _, c := range errorCases {
		errs := ValidateHorizontalPodAutoscaler(&c.horizontalPodAutoscaler)
		if len(errs) == 0 {
			t.Errorf("expected failure for %q", c.msg)
		} else if !strings.Contains(errs[0].Error(), c.msg) {
			t.Errorf("unexpected error: %q, expected: %q", errs[0], c.msg)
		}
	}
}
