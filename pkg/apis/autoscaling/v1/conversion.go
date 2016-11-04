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

package v1

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	// Add non-generated conversion functions
	err := scheme.AddConversionFuncs(
		Convert_autoscaling_HorizontalPodAutoscalerSpec_To_v1_HorizontalPodAutoscalerSpec,
		Convert_v1_HorizontalPodAutoscalerSpec_To_autoscaling_HorizontalPodAutoscalerSpec,
		Convert_autoscaling_HorizontalPodAutoscalerStatus_To_v1_HorizontalPodAutoscalerStatus,
		Convert_v1_HorizontalPodAutoscalerStatus_To_autoscaling_HorizontalPodAutoscalerStatus,
	)
	if err != nil {
		return err
	}

	return nil
}

func Convert_autoscaling_HorizontalPodAutoscalerSpec_To_v1_HorizontalPodAutoscalerSpec(in *autoscaling.HorizontalPodAutoscalerSpec, out *HorizontalPodAutoscalerSpec, s conversion.Scope) error {
	if err := Convert_autoscaling_CrossVersionObjectReference_To_v1_CrossVersionObjectReference(&in.ScaleTargetRef, &out.ScaleTargetRef, s); err != nil {
		return err
	}

	out.MinReplicas = in.MinReplicas
	out.MaxReplicas = in.MaxReplicas

	for _, metric := range in.Metrics {
		if metric.Type == autoscaling.ResourceSourceType && metric.Resource != nil && metric.Resource.Name == api.ResourceCPU {
			if metric.Resource.TargetPercentageOfRequest != nil {

				out.TargetCPUUtilizationPercentage = new(int32)
				*out.TargetCPUUtilizationPercentage = *metric.Resource.TargetPercentageOfRequest
			}
			break
		}
	}

	return nil
}

func Convert_v1_HorizontalPodAutoscalerSpec_To_autoscaling_HorizontalPodAutoscalerSpec(in *HorizontalPodAutoscalerSpec, out *autoscaling.HorizontalPodAutoscalerSpec, s conversion.Scope) error {
	if err := Convert_v1_CrossVersionObjectReference_To_autoscaling_CrossVersionObjectReference(&in.ScaleTargetRef, &out.ScaleTargetRef, s); err != nil {
		return err
	}

	out.MinReplicas = in.MinReplicas
	out.MaxReplicas = in.MaxReplicas

	out.Metrics = []autoscaling.MetricSpec{
		{
			Type: autoscaling.ResourceSourceType,
			Resource: &autoscaling.ResourceMetricSource{
				Name: api.ResourceCPU,
			},
		},
	}
	if in.TargetCPUUtilizationPercentage != nil {
		out.Metrics[0].Resource.TargetPercentageOfRequest = new(int32)
		*out.Metrics[0].Resource.TargetPercentageOfRequest = *in.TargetCPUUtilizationPercentage
	}

	return nil
}

func Convert_autoscaling_HorizontalPodAutoscalerStatus_To_v1_HorizontalPodAutoscalerStatus(in *autoscaling.HorizontalPodAutoscalerStatus, out *HorizontalPodAutoscalerStatus, s conversion.Scope) error {
	out.ObservedGeneration = in.ObservedGeneration
	out.LastScaleTime = in.LastScaleTime

	out.CurrentReplicas = in.CurrentReplicas
	out.DesiredReplicas = in.DesiredReplicas

	for _, metric := range in.CurrentMetrics {
		if metric.Type == autoscaling.ResourceSourceType && metric.Resource != nil && metric.Resource.Name == api.ResourceCPU {
			if metric.Resource.CurrentPercentageOfRequest != nil {

				out.CurrentCPUUtilizationPercentage = new(int32)
				*out.CurrentCPUUtilizationPercentage = *metric.Resource.CurrentPercentageOfRequest
			}
		}
	}
	return nil
}

func Convert_v1_HorizontalPodAutoscalerStatus_To_autoscaling_HorizontalPodAutoscalerStatus(in *HorizontalPodAutoscalerStatus, out *autoscaling.HorizontalPodAutoscalerStatus, s conversion.Scope) error {
	out.ObservedGeneration = in.ObservedGeneration
	out.LastScaleTime = in.LastScaleTime

	out.CurrentReplicas = in.CurrentReplicas
	out.DesiredReplicas = in.DesiredReplicas

	out.CurrentMetrics = []autoscaling.MetricStatus{
		{
			Type: autoscaling.ResourceSourceType,
			Resource: &autoscaling.ResourceMetricStatus{
				Name: api.ResourceCPU,
			},
		},
	}
	if in.CurrentCPUUtilizationPercentage != nil {
		out.CurrentMetrics[0].Resource.CurrentPercentageOfRequest = new(int32)
		*out.CurrentMetrics[0].Resource.CurrentPercentageOfRequest = *in.CurrentCPUUtilizationPercentage
	}
	return nil
}
