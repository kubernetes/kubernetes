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

// MetricSpecsAnnotation is the annotation which holds non-CPU-utilization HPA metric
// specs when converting the `Metrics` field from autoscaling/v2beta1
const MetricSpecsAnnotation = "autoscaling.alpha.kubernetes.io/metrics"

// MetricStatusesAnnotation is the annotation which holds non-CPU-utilization HPA metric
// statuses when converting the `CurrentMetrics` field from autoscaling/v2beta1
const MetricStatusesAnnotation = "autoscaling.alpha.kubernetes.io/current-metrics"

// HorizontalPodAutoscalerConditionsAnnotation is the annotation which holds the conditions
// of an HPA when converting the `Conditions` field from autoscaling/v2beta1
const HorizontalPodAutoscalerConditionsAnnotation = "autoscaling.alpha.kubernetes.io/conditions"

// DefaultCPUUtilization is the default value for CPU utilization, provided no other
// metrics are present.  This is here because it's used by both the v2beta1 defaulting
// logic, and the pseudo-defaulting done in v1 conversion.
const DefaultCPUUtilization = 80

// BehaviorSpecsAnnotation is the annotation which holds the HPA constraints specs
// when converting the `Behavior` field from autoscaling/v2beta2
const BehaviorSpecsAnnotation = "autoscaling.alpha.kubernetes.io/behavior"

// ToleranceScaleDownAnnotation is the annotation which holds the HPA tolerance specs
// when converting the `ScaleDown.Tolerance` field from autoscaling/v2
const ToleranceScaleDownAnnotation = "autoscaling.alpha.kubernetes.io/scale-down-tolerance"

// ToleranceScaleUpAnnotation is the annotation which holds the HPA tolerance specs
// when converting the `ScaleUp.Tolerance` field from autoscaling/v2
const ToleranceScaleUpAnnotation = "autoscaling.alpha.kubernetes.io/scale-up-tolerance"
