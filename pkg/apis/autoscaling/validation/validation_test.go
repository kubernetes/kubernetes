/*
Copyright 2016 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/controller/podautoscaler"
)

func TestValidateScale(t *testing.T) {
	successCases := []autoscaling.Scale{
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "frontend",
				Namespace: api.NamespaceDefault,
			},
			Spec: autoscaling.ScaleSpec{
				Replicas: 1,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "frontend",
				Namespace: api.NamespaceDefault,
			},
			Spec: autoscaling.ScaleSpec{
				Replicas: 10,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "frontend",
				Namespace: api.NamespaceDefault,
			},
			Spec: autoscaling.ScaleSpec{
				Replicas: 0,
			},
		},
	}

	for _, successCase := range successCases {
		if errs := ValidateScale(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		scale autoscaling.Scale
		msg   string
	}{
		{
			scale: autoscaling.Scale{
				ObjectMeta: api.ObjectMeta{
					Name:      "frontend",
					Namespace: api.NamespaceDefault,
				},
				Spec: autoscaling.ScaleSpec{
					Replicas: -1,
				},
			},
			msg: "must be greater than or equal to 0",
		},
	}

	for _, c := range errorCases {
		if errs := ValidateScale(&c.scale); len(errs) == 0 {
			t.Errorf("expected failure for %s", c.msg)
		} else if !strings.Contains(errs[0].Error(), c.msg) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], c.msg)
		}
	}
}

func TestValidateHorizontalPodAutoscaler(t *testing.T) {
	successCases := []autoscaling.HorizontalPodAutoscaler{
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: autoscaling.HorizontalPodAutoscalerSpec{
				ScaleTargetRef: autoscaling.CrossVersionObjectReference{
					Kind: "ReplicationController",
					Name: "myrc",
				},
				MinReplicas:                    newInt32(1),
				MaxReplicas:                    5,
				TargetCPUUtilizationPercentage: newInt32(70),
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
			},
			Spec: autoscaling.HorizontalPodAutoscalerSpec{
				ScaleTargetRef: autoscaling.CrossVersionObjectReference{
					Kind: "ReplicationController",
					Name: "myrc",
				},
				MinReplicas: newInt32(1),
				MaxReplicas: 5,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Name:      "myautoscaler",
				Namespace: api.NamespaceDefault,
				Annotations: map[string]string{
					podautoscaler.HpaCustomMetricsTargetAnnotationName: "{\"items\":[{\"name\":\"qps\",\"value\":\"20\"}]}",
				},
			},
			Spec: autoscaling.HorizontalPodAutoscalerSpec{
				ScaleTargetRef: autoscaling.CrossVersionObjectReference{
					Kind: "ReplicationController",
					Name: "myrc",
				},
				MinReplicas: newInt32(1),
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
					ScaleTargetRef:                 autoscaling.CrossVersionObjectReference{Name: "myrc"},
					MinReplicas:                    newInt32(1),
					MaxReplicas:                    5,
					TargetCPUUtilizationPercentage: newInt32(70),
				},
			},
			msg: "scaleTargetRef.kind: Required",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef:                 autoscaling.CrossVersionObjectReference{Kind: "..", Name: "myrc"},
					MinReplicas:                    newInt32(1),
					MaxReplicas:                    5,
					TargetCPUUtilizationPercentage: newInt32(70),
				},
			},
			msg: "scaleTargetRef.kind: Invalid",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef:                 autoscaling.CrossVersionObjectReference{Kind: "ReplicationController"},
					MinReplicas:                    newInt32(1),
					MaxReplicas:                    5,
					TargetCPUUtilizationPercentage: newInt32(70),
				},
			},
			msg: "scaleTargetRef.name: Required",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{Name: "myautoscaler", Namespace: api.NamespaceDefault},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef:                 autoscaling.CrossVersionObjectReference{Kind: "ReplicationController", Name: ".."},
					MinReplicas:                    newInt32(1),
					MaxReplicas:                    5,
					TargetCPUUtilizationPercentage: newInt32(70),
				},
			},
			msg: "scaleTargetRef.name: Invalid",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name:      "myautoscaler",
					Namespace: api.NamespaceDefault,
				},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{},
					MinReplicas:    newInt32(-1),
					MaxReplicas:    5,
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
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{},
					MinReplicas:    newInt32(7),
					MaxReplicas:    5,
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
					ScaleTargetRef:                 autoscaling.CrossVersionObjectReference{},
					MinReplicas:                    newInt32(1),
					MaxReplicas:                    5,
					TargetCPUUtilizationPercentage: newInt32(-70),
				},
			},
			msg: "must be greater than 0",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name:      "myautoscaler",
					Namespace: api.NamespaceDefault,
					Annotations: map[string]string{
						podautoscaler.HpaCustomMetricsTargetAnnotationName: "broken",
					},
				},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Kind: "ReplicationController",
						Name: "myrc",
					},
					MinReplicas: newInt32(1),
					MaxReplicas: 5,
				},
			},
			msg: "failed to parse custom metrics target annotation",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name:      "myautoscaler",
					Namespace: api.NamespaceDefault,
					Annotations: map[string]string{
						podautoscaler.HpaCustomMetricsTargetAnnotationName: "{}",
					},
				},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Kind: "ReplicationController",
						Name: "myrc",
					},
					MinReplicas: newInt32(1),
					MaxReplicas: 5,
				},
			},
			msg: "custom metrics target must not be empty",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name:      "myautoscaler",
					Namespace: api.NamespaceDefault,
					Annotations: map[string]string{
						podautoscaler.HpaCustomMetricsTargetAnnotationName: "{\"items\":[{\"value\":\"20\"}]}",
					},
				},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Kind: "ReplicationController",
						Name: "myrc",
					},
					MinReplicas: newInt32(1),
					MaxReplicas: 5,
				},
			},
			msg: "missing custom metric target name",
		},
		{
			horizontalPodAutoscaler: autoscaling.HorizontalPodAutoscaler{
				ObjectMeta: api.ObjectMeta{
					Name:      "myautoscaler",
					Namespace: api.NamespaceDefault,
					Annotations: map[string]string{
						podautoscaler.HpaCustomMetricsTargetAnnotationName: "{\"items\":[{\"name\":\"qps\",\"value\":\"0\"}]}",
					},
				},
				Spec: autoscaling.HorizontalPodAutoscalerSpec{
					ScaleTargetRef: autoscaling.CrossVersionObjectReference{
						Kind: "ReplicationController",
						Name: "myrc",
					},
					MinReplicas: newInt32(1),
					MaxReplicas: 5,
				},
			},
			msg: "custom metric target value must be greater than 0",
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

func newInt32(val int32) *int32 {
	p := new(int32)
	*p = val
	return p
}
