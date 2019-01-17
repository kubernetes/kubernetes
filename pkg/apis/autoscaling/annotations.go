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

package autoscaling

const (
	// MetricSpecsAnnotation is the annotation which holds non-CPU-utilization HPA metric
	// specs when converting the `Metrics` field from autoscaling/v2beta1
	MetricSpecsAnnotation = "autoscaling.alpha.kubernetes.io/metrics"

	// MetricStatusesAnnotation is the annotation which holds non-CPU-utilization HPA metric
	// statuses when converting the `CurrentMetrics` field from autoscaling/v2beta1
	MetricStatusesAnnotation = "autoscaling.alpha.kubernetes.io/current-metrics"

	// HorizontalPodAutoscalerConditionsAnnotation is the annotation which holds the conditions
	// of an HPA when converting the `Conditions` field from autoscaling/v2beta1
	HorizontalPodAutoscalerConditionsAnnotation = "autoscaling.alpha.kubernetes.io/conditions"

	// DefaultCPUUtilization is the default value for CPU utilization, provided no other
	// metrics are present.  This is here because it's used by both the v2beta1 defaulting
	// logic, and the pseudo-defaulting done in v1 conversion.
	DefaultCPUUtilization = 80

	// HorizontalPodAutoscalerDownscaleStabilizationWindow is the annotation which holds the
	// downscaleStabilizationWindow of an HPA ... TODO
	HorizontalPodAutoscalerDownscaleStabilizationWindow = "autoscaling.alpha.kubernetes.io/downscaleStabilizationWindow"

	// HorizontalPodAutoscalerUpscaleForbiddenWindow is the annotation which holds the
	// upscaleForbiddenWindow of an HPA ... TODO
	HorizontalPodAutoscalerUpscaleForbiddenWindow = "autoscaling.alpha.kubernetes.io/upscaleForbiddenWindow"

	// HorizontalPodAutoscalerDownscaleForbiddenWindow is the annotation which holds the
	// downscaleForbiddenWindow of an HPA ... TODO
	HorizontalPodAutoscalerDownscaleForbiddenWindow = "autoscaling.alpha.kubernetes.io/downscaleForbiddenWindow"

	// HorizontalPodAutoscalerInitialReadinessDelay is the annotation which holds the
	// initialReadinessDelay of an HPA ... TODO
	HorizontalPodAutoscalerInitialReadinessDelay = "autoscaling.alpha.kubernetes.io/initialReadinessDelay"

	// HorizontalPodAutoscalerTolerance is the annotation which holds the tolerance value
	// of an HPA ... TODO
	HorizontalPodAutoscalerTolerance = "autoscaling.alpha.kubernetes.io/tolerance"
)
