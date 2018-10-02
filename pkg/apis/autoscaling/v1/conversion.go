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

	autoscalingv1 "k8s.io/api/autoscaling/v1"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	api "k8s.io/kubernetes/pkg/apis/core"
	core "k8s.io/kubernetes/pkg/apis/core"
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
		Convert_autoscaling_ExternalMetricSource_To_v1_ExternalMetricSource,
		Convert_v1_ExternalMetricSource_To_autoscaling_ExternalMetricSource,
		Convert_autoscaling_ObjectMetricSource_To_v1_ObjectMetricSource,
		Convert_v1_ObjectMetricSource_To_autoscaling_ObjectMetricSource,
		Convert_autoscaling_PodsMetricSource_To_v1_PodsMetricSource,
		Convert_v1_PodsMetricSource_To_autoscaling_PodsMetricSource,
		Convert_autoscaling_ExternalMetricStatus_To_v1_ExternalMetricStatus,
		Convert_v1_ExternalMetricStatus_To_autoscaling_ExternalMetricStatus,
		Convert_autoscaling_ObjectMetricStatus_To_v1_ObjectMetricStatus,
		Convert_v1_ObjectMetricStatus_To_autoscaling_ObjectMetricStatus,
		Convert_autoscaling_PodsMetricStatus_To_v1_PodsMetricStatus,
		Convert_v1_PodsMetricStatus_To_autoscaling_PodsMetricStatus,
		Convert_autoscaling_MetricTarget_To_v1_CrossVersionObjectReference,
		Convert_v1_CrossVersionObjectReference_To_autoscaling_MetricTarget,
		Convert_autoscaling_ResourceMetricStatus_To_v1_ResourceMetricStatus,
		Convert_v1_ResourceMetricStatus_To_autoscaling_ResourceMetricStatus,
	)
	if err != nil {
		return err
	}

	return nil
}

func Convert_autoscaling_MetricTarget_To_v1_CrossVersionObjectReference(in *autoscaling.MetricTarget, out *autoscalingv1.CrossVersionObjectReference, s conversion.Scope) error {
	return nil
}

func Convert_v1_CrossVersionObjectReference_To_autoscaling_MetricTarget(in *autoscalingv1.CrossVersionObjectReference, out *autoscaling.MetricTarget, s conversion.Scope) error {
	return nil
}

func Convert_autoscaling_ExternalMetricSource_To_v1_ExternalMetricSource(in *autoscaling.ExternalMetricSource, out *autoscalingv1.ExternalMetricSource, s conversion.Scope) error {
	out.MetricName = in.Metric.Name
	out.TargetValue = in.Target.Value
	out.TargetAverageValue = in.Target.AverageValue
	out.MetricSelector = in.Metric.Selector
	return nil
}

func Convert_v1_ExternalMetricSource_To_autoscaling_ExternalMetricSource(in *autoscalingv1.ExternalMetricSource, out *autoscaling.ExternalMetricSource, s conversion.Scope) error {
	value := in.TargetValue
	averageValue := in.TargetAverageValue
	var metricType autoscaling.MetricTargetType
	if value == nil {
		metricType = autoscaling.AverageValueMetricType
	} else {
		metricType = autoscaling.ValueMetricType
	}
	out.Target = autoscaling.MetricTarget{
		Type:         metricType,
		Value:        value,
		AverageValue: averageValue,
	}

	out.Metric = autoscaling.MetricIdentifier{
		Name:     in.MetricName,
		Selector: in.MetricSelector,
	}
	return nil
}

func Convert_autoscaling_ObjectMetricSource_To_v1_ObjectMetricSource(in *autoscaling.ObjectMetricSource, out *autoscalingv1.ObjectMetricSource, s conversion.Scope) error {
	if in.Target.Value != nil {
		out.TargetValue = *in.Target.Value
	}
	out.AverageValue = in.Target.AverageValue
	out.Target = autoscalingv1.CrossVersionObjectReference{
		Kind:       in.DescribedObject.Kind,
		Name:       in.DescribedObject.Name,
		APIVersion: in.DescribedObject.APIVersion,
	}
	out.MetricName = in.Metric.Name
	out.Selector = in.Metric.Selector
	return nil
}

func Convert_v1_ObjectMetricSource_To_autoscaling_ObjectMetricSource(in *autoscalingv1.ObjectMetricSource, out *autoscaling.ObjectMetricSource, s conversion.Scope) error {
	var metricType autoscaling.MetricTargetType
	if in.AverageValue == nil {
		metricType = autoscaling.ValueMetricType
	} else {
		metricType = autoscaling.AverageValueMetricType
	}

	out.Target = autoscaling.MetricTarget{
		Type:         metricType,
		Value:        &in.TargetValue,
		AverageValue: in.AverageValue,
	}
	out.DescribedObject = autoscaling.CrossVersionObjectReference{
		Kind:       in.Target.Kind,
		Name:       in.Target.Name,
		APIVersion: in.Target.APIVersion,
	}
	out.Metric = autoscaling.MetricIdentifier{
		Name:     in.MetricName,
		Selector: in.Selector,
	}
	return nil
}

func Convert_autoscaling_PodsMetricSource_To_v1_PodsMetricSource(in *autoscaling.PodsMetricSource, out *autoscalingv1.PodsMetricSource, s conversion.Scope) error {
	if in.Target.AverageValue != nil {
		out.TargetAverageValue = *in.Target.AverageValue
	}

	out.MetricName = in.Metric.Name
	out.Selector = in.Metric.Selector
	return nil
}

func Convert_v1_PodsMetricSource_To_autoscaling_PodsMetricSource(in *autoscalingv1.PodsMetricSource, out *autoscaling.PodsMetricSource, s conversion.Scope) error {
	var metricType autoscaling.MetricTargetType
	metricType = autoscaling.AverageValueMetricType

	out.Target = autoscaling.MetricTarget{
		Type:         metricType,
		AverageValue: &in.TargetAverageValue,
	}
	out.Metric = autoscaling.MetricIdentifier{
		Name:     in.MetricName,
		Selector: in.Selector,
	}
	return nil
}

func Convert_autoscaling_ExternalMetricStatus_To_v1_ExternalMetricStatus(in *autoscaling.ExternalMetricStatus, out *autoscalingv1.ExternalMetricStatus, s conversion.Scope) error {
	out.MetricName = in.Metric.Name
	if in.Current.Value != nil {
		out.CurrentValue = *in.Current.Value
	}
	if in.Current.AverageValue != nil {
		out.CurrentAverageValue = in.Current.AverageValue
	}
	out.MetricSelector = in.Metric.Selector
	return nil
}

func Convert_v1_ExternalMetricStatus_To_autoscaling_ExternalMetricStatus(in *autoscalingv1.ExternalMetricStatus, out *autoscaling.ExternalMetricStatus, s conversion.Scope) error {
	value := in.CurrentValue
	averageValue := in.CurrentAverageValue
	out.Current = autoscaling.MetricValueStatus{
		Value:        &value,
		AverageValue: averageValue,
	}
	out.Metric = autoscaling.MetricIdentifier{
		Name:     in.MetricName,
		Selector: in.MetricSelector,
	}
	return nil
}

func Convert_autoscaling_ObjectMetricStatus_To_v1_ObjectMetricStatus(in *autoscaling.ObjectMetricStatus, out *autoscalingv1.ObjectMetricStatus, s conversion.Scope) error {
	if in.Current.Value != nil {
		out.CurrentValue = *in.Current.Value
	}
	if in.Current.AverageValue != nil {
		currentAverageValue := *in.Current.AverageValue
		out.AverageValue = &currentAverageValue
	}
	out.Target = autoscalingv1.CrossVersionObjectReference{
		Kind:       in.DescribedObject.Kind,
		Name:       in.DescribedObject.Name,
		APIVersion: in.DescribedObject.APIVersion,
	}
	out.MetricName = in.Metric.Name
	out.Selector = in.Metric.Selector
	return nil
}

func Convert_v1_ObjectMetricStatus_To_autoscaling_ObjectMetricStatus(in *autoscalingv1.ObjectMetricStatus, out *autoscaling.ObjectMetricStatus, s conversion.Scope) error {
	out.Current = autoscaling.MetricValueStatus{
		Value:        &in.CurrentValue,
		AverageValue: in.AverageValue,
	}
	out.DescribedObject = autoscaling.CrossVersionObjectReference{
		Kind:       in.Target.Kind,
		Name:       in.Target.Name,
		APIVersion: in.Target.APIVersion,
	}
	out.Metric = autoscaling.MetricIdentifier{
		Name:     in.MetricName,
		Selector: in.Selector,
	}
	return nil
}

func Convert_autoscaling_PodsMetricStatus_To_v1_PodsMetricStatus(in *autoscaling.PodsMetricStatus, out *autoscalingv1.PodsMetricStatus, s conversion.Scope) error {
	if in.Current.AverageValue != nil {
		out.CurrentAverageValue = *in.Current.AverageValue
	}
	out.MetricName = in.Metric.Name
	out.Selector = in.Metric.Selector
	return nil
}

func Convert_v1_PodsMetricStatus_To_autoscaling_PodsMetricStatus(in *autoscalingv1.PodsMetricStatus, out *autoscaling.PodsMetricStatus, s conversion.Scope) error {
	out.Current = autoscaling.MetricValueStatus{
		AverageValue: &in.CurrentAverageValue,
	}
	out.Metric = autoscaling.MetricIdentifier{
		Name:     in.MetricName,
		Selector: in.Selector,
	}
	return nil
}

func Convert_v1_ResourceMetricSource_To_autoscaling_ResourceMetricSource(in *autoscalingv1.ResourceMetricSource, out *autoscaling.ResourceMetricSource, s conversion.Scope) error {
	out.Name = core.ResourceName(in.Name)
	utilization := in.TargetAverageUtilization
	averageValue := in.TargetAverageValue
	var metricType autoscaling.MetricTargetType
	if utilization == nil {
		metricType = autoscaling.AverageValueMetricType
	} else {
		metricType = autoscaling.UtilizationMetricType
	}
	out.Target = autoscaling.MetricTarget{
		Type:               metricType,
		AverageValue:       averageValue,
		AverageUtilization: utilization,
	}
	return nil
}

func Convert_autoscaling_ResourceMetricSource_To_v1_ResourceMetricSource(in *autoscaling.ResourceMetricSource, out *autoscalingv1.ResourceMetricSource, s conversion.Scope) error {
	out.Name = v1.ResourceName(in.Name)
	out.TargetAverageUtilization = in.Target.AverageUtilization
	out.TargetAverageValue = in.Target.AverageValue
	return nil
}

func Convert_v1_ResourceMetricStatus_To_autoscaling_ResourceMetricStatus(in *autoscalingv1.ResourceMetricStatus, out *autoscaling.ResourceMetricStatus, s conversion.Scope) error {
	out.Name = core.ResourceName(in.Name)
	utilization := in.CurrentAverageUtilization
	averageValue := &in.CurrentAverageValue
	out.Current = autoscaling.MetricValueStatus{
		AverageValue:       averageValue,
		AverageUtilization: utilization,
	}
	return nil
}

func Convert_autoscaling_ResourceMetricStatus_To_v1_ResourceMetricStatus(in *autoscaling.ResourceMetricStatus, out *autoscalingv1.ResourceMetricStatus, s conversion.Scope) error {
	out.Name = v1.ResourceName(in.Name)
	out.CurrentAverageUtilization = in.Current.AverageUtilization
	if in.Current.AverageValue != nil {
		out.CurrentAverageValue = *in.Current.AverageValue
	}
	return nil
}

func Convert_autoscaling_HorizontalPodAutoscaler_To_v1_HorizontalPodAutoscaler(in *autoscaling.HorizontalPodAutoscaler, out *autoscalingv1.HorizontalPodAutoscaler, s conversion.Scope) error {
	if err := autoConvert_autoscaling_HorizontalPodAutoscaler_To_v1_HorizontalPodAutoscaler(in, out, s); err != nil {
		return err
	}

	otherMetrics := make([]autoscalingv1.MetricSpec, 0, len(in.Spec.Metrics))
	for _, metric := range in.Spec.Metrics {
		if metric.Type == autoscaling.ResourceMetricSourceType && metric.Resource != nil && metric.Resource.Name == api.ResourceCPU && metric.Resource.Target.AverageUtilization != nil {
			continue
		}

		convMetric := autoscalingv1.MetricSpec{}
		if err := Convert_autoscaling_MetricSpec_To_v1_MetricSpec(&metric, &convMetric, s); err != nil {
			return err
		}
		otherMetrics = append(otherMetrics, convMetric)
	}

	// NB: we need to save the status even if it maps to a CPU utilization status in order to save the raw value as well
	currentMetrics := make([]autoscalingv1.MetricStatus, len(in.Status.CurrentMetrics))
	for i, currentMetric := range in.Status.CurrentMetrics {
		if err := Convert_autoscaling_MetricStatus_To_v1_MetricStatus(&currentMetric, &currentMetrics[i], s); err != nil {
			return err
		}
	}

	// store HPA conditions in an annotation
	currentConditions := make([]autoscalingv1.HorizontalPodAutoscalerCondition, len(in.Status.Conditions))
	for i, currentCondition := range in.Status.Conditions {
		if err := Convert_autoscaling_HorizontalPodAutoscalerCondition_To_v1_HorizontalPodAutoscalerCondition(&currentCondition, &currentConditions[i], s); err != nil {
			return err
		}
	}

	if len(otherMetrics) > 0 || len(in.Status.CurrentMetrics) > 0 || len(currentConditions) > 0 {
		old := out.Annotations
		out.Annotations = make(map[string]string, len(old)+3)
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

	if len(in.Status.Conditions) > 0 {
		currentConditionsEnc, err := json.Marshal(currentConditions)
		if err != nil {
			return err
		}
		out.Annotations[autoscaling.HorizontalPodAutoscalerConditionsAnnotation] = string(currentConditionsEnc)
	}

	return nil
}

func Convert_v1_HorizontalPodAutoscaler_To_autoscaling_HorizontalPodAutoscaler(in *autoscalingv1.HorizontalPodAutoscaler, out *autoscaling.HorizontalPodAutoscaler, s conversion.Scope) error {
	if err := autoConvert_v1_HorizontalPodAutoscaler_To_autoscaling_HorizontalPodAutoscaler(in, out, s); err != nil {
		return err
	}

	if otherMetricsEnc, hasOtherMetrics := out.Annotations[autoscaling.MetricSpecsAnnotation]; hasOtherMetrics {
		var otherMetrics []autoscalingv1.MetricSpec
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
		var currentMetrics []autoscalingv1.MetricStatus
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

	// autoscaling/v1 formerly had an implicit default applied in the controller.  In v2beta1, we apply it explicitly.
	// We apply it here, explicitly, since we have access to the full set of metrics from the annotation.
	if len(out.Spec.Metrics) == 0 {
		// no other metrics, no explicit CPU value set
		out.Spec.Metrics = []autoscaling.MetricSpec{
			{
				Type: autoscaling.ResourceMetricSourceType,
				Resource: &autoscaling.ResourceMetricSource{
					Name: api.ResourceCPU,
					Target: autoscaling.MetricTarget{
						Type: autoscaling.UtilizationMetricType,
					},
				},
			},
		}
		out.Spec.Metrics[0].Resource.Target.AverageUtilization = new(int32)
		*out.Spec.Metrics[0].Resource.Target.AverageUtilization = autoscaling.DefaultCPUUtilization
	}

	if currentConditionsEnc, hasCurrentConditions := out.Annotations[autoscaling.HorizontalPodAutoscalerConditionsAnnotation]; hasCurrentConditions {
		var currentConditions []autoscalingv1.HorizontalPodAutoscalerCondition
		if err := json.Unmarshal([]byte(currentConditionsEnc), &currentConditions); err != nil {
			return err
		}

		out.Status.Conditions = make([]autoscaling.HorizontalPodAutoscalerCondition, len(currentConditions))
		for i, currentCondition := range currentConditions {
			if err := Convert_v1_HorizontalPodAutoscalerCondition_To_autoscaling_HorizontalPodAutoscalerCondition(&currentCondition, &out.Status.Conditions[i], s); err != nil {
				return err
			}
		}
		delete(out.Annotations, autoscaling.HorizontalPodAutoscalerConditionsAnnotation)
	}

	return nil
}

func Convert_autoscaling_HorizontalPodAutoscalerSpec_To_v1_HorizontalPodAutoscalerSpec(in *autoscaling.HorizontalPodAutoscalerSpec, out *autoscalingv1.HorizontalPodAutoscalerSpec, s conversion.Scope) error {
	if err := Convert_autoscaling_CrossVersionObjectReference_To_v1_CrossVersionObjectReference(&in.ScaleTargetRef, &out.ScaleTargetRef, s); err != nil {
		return err
	}

	out.MinReplicas = in.MinReplicas
	out.MaxReplicas = in.MaxReplicas

	for _, metric := range in.Metrics {
		if metric.Type == autoscaling.ResourceMetricSourceType && metric.Resource != nil && metric.Resource.Name == api.ResourceCPU {
			if metric.Resource.Target.AverageUtilization != nil {
				out.TargetCPUUtilizationPercentage = new(int32)
				*out.TargetCPUUtilizationPercentage = *metric.Resource.Target.AverageUtilization
			}
			break
		}
	}

	return nil
}

func Convert_v1_HorizontalPodAutoscalerSpec_To_autoscaling_HorizontalPodAutoscalerSpec(in *autoscalingv1.HorizontalPodAutoscalerSpec, out *autoscaling.HorizontalPodAutoscalerSpec, s conversion.Scope) error {
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
					Target: autoscaling.MetricTarget{
						Type: autoscaling.UtilizationMetricType,
					},
				},
			},
		}
		out.Metrics[0].Resource.Target.AverageUtilization = new(int32)
		*out.Metrics[0].Resource.Target.AverageUtilization = *in.TargetCPUUtilizationPercentage
	}

	return nil
}

func Convert_autoscaling_HorizontalPodAutoscalerStatus_To_v1_HorizontalPodAutoscalerStatus(in *autoscaling.HorizontalPodAutoscalerStatus, out *autoscalingv1.HorizontalPodAutoscalerStatus, s conversion.Scope) error {
	out.ObservedGeneration = in.ObservedGeneration
	out.LastScaleTime = in.LastScaleTime

	out.CurrentReplicas = in.CurrentReplicas
	out.DesiredReplicas = in.DesiredReplicas

	for _, metric := range in.CurrentMetrics {
		if metric.Type == autoscaling.ResourceMetricSourceType && metric.Resource != nil && metric.Resource.Name == api.ResourceCPU {
			if metric.Resource.Current.AverageUtilization != nil {

				out.CurrentCPUUtilizationPercentage = new(int32)
				*out.CurrentCPUUtilizationPercentage = *metric.Resource.Current.AverageUtilization
			}
		}
	}
	return nil
}

func Convert_v1_HorizontalPodAutoscalerStatus_To_autoscaling_HorizontalPodAutoscalerStatus(in *autoscalingv1.HorizontalPodAutoscalerStatus, out *autoscaling.HorizontalPodAutoscalerStatus, s conversion.Scope) error {
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
		out.CurrentMetrics[0].Resource.Current.AverageUtilization = new(int32)
		*out.CurrentMetrics[0].Resource.Current.AverageUtilization = *in.CurrentCPUUtilizationPercentage
	}
	return nil
}
