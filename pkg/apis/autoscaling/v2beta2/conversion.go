/*
Copyright 2020 The Kubernetes Authors.

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

package v2beta2

import (
	"fmt"

	autoscalingv2beta2 "k8s.io/api/autoscaling/v2beta2"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
)

func Convert_autoscaling_HorizontalPodAutoscaler_To_v2beta2_HorizontalPodAutoscaler(in *autoscaling.HorizontalPodAutoscaler, out *autoscalingv2beta2.HorizontalPodAutoscaler, s conversion.Scope) error {
	if err := autoConvert_autoscaling_HorizontalPodAutoscaler_To_v2beta2_HorizontalPodAutoscaler(in, out, s); err != nil {
		return err
	}
	// Ensure old round-trips annotations are discarded
	annotations, copiedAnnotations := autoscaling.DropRoundTripHorizontalPodAutoscalerAnnotations(out.Annotations)
	out.Annotations = annotations

	behavior := in.Spec.Behavior
	if behavior == nil {
		return nil
	}
	// Save the tolerance fields in annotations for round-trip
	if behavior.ScaleDown != nil && behavior.ScaleDown.Tolerance != nil {
		if !copiedAnnotations {
			copiedAnnotations = true
			out.Annotations = autoscaling.DeepCopyStringMap(out.Annotations)
		}
		out.Annotations[autoscaling.ToleranceScaleDownAnnotation] = behavior.ScaleDown.Tolerance.String()
	}
	if behavior.ScaleUp != nil && behavior.ScaleUp.Tolerance != nil {
		if !copiedAnnotations {
			copiedAnnotations = true
			out.Annotations = autoscaling.DeepCopyStringMap(out.Annotations)
		}
		out.Annotations[autoscaling.ToleranceScaleUpAnnotation] = behavior.ScaleUp.Tolerance.String()
	}
	return nil
}

func Convert_v2beta2_HorizontalPodAutoscaler_To_autoscaling_HorizontalPodAutoscaler(in *autoscalingv2beta2.HorizontalPodAutoscaler, out *autoscaling.HorizontalPodAutoscaler, s conversion.Scope) error {
	if err := autoConvert_v2beta2_HorizontalPodAutoscaler_To_autoscaling_HorizontalPodAutoscaler(in, out, s); err != nil {
		return err
	}
	// Restore the tolerance fields from annotations for round-trip
	if tolerance, ok := out.Annotations[autoscaling.ToleranceScaleDownAnnotation]; ok {
		if out.Spec.Behavior == nil {
			out.Spec.Behavior = &autoscaling.HorizontalPodAutoscalerBehavior{}
		}
		if out.Spec.Behavior.ScaleDown == nil {
			out.Spec.Behavior.ScaleDown = &autoscaling.HPAScalingRules{}
		}
		q, err := resource.ParseQuantity(tolerance)
		if err != nil {
			return fmt.Errorf("failed to parse annotation %q: %w", autoscaling.ToleranceScaleDownAnnotation, err)
		}
		out.Spec.Behavior.ScaleDown.Tolerance = &q
	}
	if tolerance, ok := out.Annotations[autoscaling.ToleranceScaleUpAnnotation]; ok {
		if out.Spec.Behavior == nil {
			out.Spec.Behavior = &autoscaling.HorizontalPodAutoscalerBehavior{}
		}
		if out.Spec.Behavior.ScaleUp == nil {
			out.Spec.Behavior.ScaleUp = &autoscaling.HPAScalingRules{}
		}
		q, err := resource.ParseQuantity(tolerance)
		if err != nil {
			return fmt.Errorf("failed to parse annotation %q: %w", autoscaling.ToleranceScaleUpAnnotation, err)
		}
		out.Spec.Behavior.ScaleUp.Tolerance = &q
	}
	// Do not save round-trip annotations in internal resource
	out.Annotations, _ = autoscaling.DropRoundTripHorizontalPodAutoscalerAnnotations(out.Annotations)
	return nil
}

func Convert_v2beta2_HPAScalingRules_To_autoscaling_HPAScalingRules(in *autoscalingv2beta2.HPAScalingRules, out *autoscaling.HPAScalingRules, s conversion.Scope) error {
	// Tolerance field is handled in the HorizontalPodAutoscaler conversion function.
	return autoConvert_v2beta2_HPAScalingRules_To_autoscaling_HPAScalingRules(in, out, s)
}

func Convert_autoscaling_HPAScalingRules_To_v2beta2_HPAScalingRules(in *autoscaling.HPAScalingRules, out *autoscalingv2beta2.HPAScalingRules, s conversion.Scope) error {
	// Tolerance field is handled in the HorizontalPodAutoscaler conversion function.
	return autoConvert_autoscaling_HPAScalingRules_To_v2beta2_HPAScalingRules(in, out, s)
}
