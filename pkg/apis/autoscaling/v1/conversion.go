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
	"encoding/json"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	// Add non-generated conversion functions
	err := scheme.AddConversionFuncs(
		Convert_autoscaling_HorizontalPodAutoscaler_To_v1_HorizontalPodAutoscaler,
		Convert_v1_HorizontalPodAutoscaler_To_autoscaling_HorizontalPodAutoscaler,
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

func Convert_autoscaling_HorizontalPodAutoscaler_To_v1_HorizontalPodAutoscaler(in *autoscaling.HorizontalPodAutoscaler, out *HorizontalPodAutoscaler, s conversion.Scope) error {
	if err := autoConvert_autoscaling_HorizontalPodAutoscaler_To_v1_HorizontalPodAutoscaler(in, out, s); err != nil {
		return err
	}

	otherMetrics := make([]MetricSpec, 0, len(in.Spec.Metrics))
	for _, metric := range in.Spec.Metrics {
		if metric.Type == autoscaling.ResourceMetricSourceType && metric.Resource != nil && metric.Resource.Name == api.ResourceCPU && metric.Resource.TargetAverageUtilization != nil {
			continue
		}

		convMetric := MetricSpec{}
		if err := Convert_autoscaling_MetricSpec_To_v1_MetricSpec(&metric, &convMetric, s); err != nil {
			return err
		}
		otherMetrics = append(otherMetrics, convMetric)
	}

	// NB: we need to save the status even if it maps to a CPU utilization status in order to save the raw value as well
	currentMetrics := make([]MetricStatus, len(in.Status.CurrentMetrics))
	for i, currentMetric := range in.Status.CurrentMetrics {
		if err := Convert_autoscaling_MetricStatus_To_v1_MetricStatus(&currentMetric, &currentMetrics[i], s); err != nil {
			return err
		}
	}

	if len(otherMetrics) > 0 || len(in.Status.CurrentMetrics) > 0 {
		old := out.Annotations
		out.Annotations = make(map[string]string, len(old)+2)
		if old != nil {
			for k, v := range old {
				out.Annotations[k] = v
			}
		}
	}

	if len(otherMetrics) > 0 {
		otherMetricsEnc, err := json.Marshal(otherMetrics)
		if err != nil {
			return err
		}
		out.Annotations[autoscaling.MetricSpecsAnnotation] = string(otherMetricsEnc)
	}

	if len(in.Status.CurrentMetrics) > 0 {
		currentMetricsEnc, err := json.Marshal(currentMetrics)
		if err != nil {
			return err
		}
		out.Annotations[autoscaling.MetricStatusesAnnotation] = string(currentMetricsEnc)
	}

	return nil
}

func Convert_v1_HorizontalPodAutoscaler_To_autoscaling_HorizontalPodAutoscaler(in *HorizontalPodAutoscaler, out *autoscaling.HorizontalPodAutoscaler, s conversion.Scope) error {
	if err := autoConvert_v1_HorizontalPodAutoscaler_To_autoscaling_HorizontalPodAutoscaler(in, out, s); err != nil {
		return err
	}

	if otherMetricsEnc, hasOtherMetrics := out.Annotations[autoscaling.MetricSpecsAnnotation]; hasOtherMetrics {
		var otherMetrics []MetricSpec
		if err := json.Unmarshal([]byte(otherMetricsEnc), &otherMetrics); err != nil {
			return err
		}

		// the normal Spec conversion could have populated out.Spec.Metrics with a single element, so deal with that
		outMetrics := make([]autoscaling.MetricSpec, len(otherMetrics)+len(out.Spec.Metrics))
		for i, metric := range otherMetrics {
			if err := Convert_v1_MetricSpec_To_autoscaling_MetricSpec(&metric, &outMetrics[i], s); err != nil {
				return err
			}
		}
		if out.Spec.Metrics != nil {
			outMetrics[len(otherMetrics)] = out.Spec.Metrics[0]
		}
		out.Spec.Metrics = outMetrics
		delete(out.Annotations, autoscaling.MetricSpecsAnnotation)
	}

	if currentMetricsEnc, hasCurrentMetrics := out.Annotations[autoscaling.MetricStatusesAnnotation]; hasCurrentMetrics {
		// ignore any existing status values -- the ones here have more information
		var currentMetrics []MetricStatus
		if err := json.Unmarshal([]byte(currentMetricsEnc), &currentMetrics); err != nil {
			return err
		}

		out.Status.CurrentMetrics = make([]autoscaling.MetricStatus, len(currentMetrics))
		for i, currentMetric := range currentMetrics {
			if err := Convert_v1_MetricStatus_To_autoscaling_MetricStatus(&currentMetric, &out.Status.CurrentMetrics[i], s); err != nil {
				return err
			}
		}
		delete(out.Annotations, autoscaling.MetricStatusesAnnotation)
	}

	// autoscaling/v1 formerly had an implicit default applied in the controller.  In v2alpha1, we apply it explicitly.
	// We apply it here, explicitly, since we have access to the full set of metrics from the annotation.
	if len(out.Spec.Metrics) == 0 {
		// no other metrics, no explicit CPU value set
		out.Spec.Metrics = []autoscaling.MetricSpec{
			{
				Type: autoscaling.ResourceMetricSourceType,
				Resource: &autoscaling.ResourceMetricSource{
					Name: api.ResourceCPU,
				},
			},
		}
		out.Spec.Metrics[0].Resource.TargetAverageUtilization = new(int32)
		*out.Spec.Metrics[0].Resource.TargetAverageUtilization = autoscaling.DefaultCPUUtilization
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
		if metric.Type == autoscaling.ResourceMetricSourceType && metric.Resource != nil && metric.Resource.Name == api.ResourceCPU {
			if metric.Resource.TargetAverageUtilization != nil {
				out.TargetCPUUtilizationPercentage = new(int32)
				*out.TargetCPUUtilizationPercentage = *metric.Resource.TargetAverageUtilization
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

	if in.TargetCPUUtilizationPercentage != nil {
		out.Metrics = []autoscaling.MetricSpec{
			{
				Type: autoscaling.ResourceMetricSourceType,
				Resource: &autoscaling.ResourceMetricSource{
					Name: api.ResourceCPU,
				},
			},
		}
		out.Metrics[0].Resource.TargetAverageUtilization = new(int32)
		*out.Metrics[0].Resource.TargetAverageUtilization = *in.TargetCPUUtilizationPercentage
	}

	return nil
}

func Convert_autoscaling_HorizontalPodAutoscalerStatus_To_v1_HorizontalPodAutoscalerStatus(in *autoscaling.HorizontalPodAutoscalerStatus, out *HorizontalPodAutoscalerStatus, s conversion.Scope) error {
	out.ObservedGeneration = in.ObservedGeneration
	out.LastScaleTime = in.LastScaleTime

	out.CurrentReplicas = in.CurrentReplicas
	out.DesiredReplicas = in.DesiredReplicas

	for _, metric := range in.CurrentMetrics {
		if metric.Type == autoscaling.ResourceMetricSourceType && metric.Resource != nil && metric.Resource.Name == api.ResourceCPU {
			if metric.Resource.CurrentAverageUtilization != nil {

				out.CurrentCPUUtilizationPercentage = new(int32)
				*out.CurrentCPUUtilizationPercentage = *metric.Resource.CurrentAverageUtilization
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

	if in.CurrentCPUUtilizationPercentage != nil {
		out.CurrentMetrics = []autoscaling.MetricStatus{
			{
				Type: autoscaling.ResourceMetricSourceType,
				Resource: &autoscaling.ResourceMetricStatus{
					Name: api.ResourceCPU,
				},
			},
		}
		out.CurrentMetrics[0].Resource.CurrentAverageUtilization = new(int32)
		*out.CurrentMetrics[0].Resource.CurrentAverageUtilization = *in.CurrentCPUUtilizationPercentage
	}
	return nil
}
