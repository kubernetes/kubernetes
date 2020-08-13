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
	autoscalingv2beta2 "k8s.io/api/autoscaling/v2beta2"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
)

func Convert_autoscaling_HorizontalPodAutoscaler_To_v2beta2_HorizontalPodAutoscaler(in *autoscaling.HorizontalPodAutoscaler, out *autoscalingv2beta2.HorizontalPodAutoscaler, s conversion.Scope) error {
	if err := autoConvert_autoscaling_HorizontalPodAutoscaler_To_v2beta2_HorizontalPodAutoscaler(in, out, s); err != nil {
		return err
	}
	// v2beta2 round-trips to internal without any serialized annotations, make sure any from other versions don't get serialized
	out.Annotations, _ = autoscaling.DropRoundTripHorizontalPodAutoscalerAnnotations(out.Annotations)
	return nil
}

func Convert_v2beta2_HorizontalPodAutoscaler_To_autoscaling_HorizontalPodAutoscaler(in *autoscalingv2beta2.HorizontalPodAutoscaler, out *autoscaling.HorizontalPodAutoscaler, s conversion.Scope) error {
	if err := autoConvert_v2beta2_HorizontalPodAutoscaler_To_autoscaling_HorizontalPodAutoscaler(in, out, s); err != nil {
		return err
	}
	// v2beta2 round-trips to internal without any serialized annotations, make sure any from other versions don't get serialized
	out.Annotations, _ = autoscaling.DropRoundTripHorizontalPodAutoscalerAnnotations(out.Annotations)
	return nil
}
