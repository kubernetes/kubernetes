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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
)

// CrossVersionObjectReference contains enough information to let you identify the referred resource.
type CrossVersionObjectReference struct {
	// Kind of the referent; More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds"
	Kind string `json:"kind" protobuf:"bytes,1,opt,name=kind"`
	// Name of the referent; More info: http://kubernetes.io/docs/user-guide/identifiers#names
	Name string `json:"name" protobuf:"bytes,2,opt,name=name"`
	// API version of the referent
	// +optional
	APIVersion string `json:"apiVersion,omitempty" protobuf:"bytes,3,opt,name=apiVersion"`
}

// specification of a horizontal pod autoscaler.
type HorizontalPodAutoscalerSpec struct {
	// reference to scaled resource; horizontal pod autoscaler will learn the current resource consumption
	// and will set the desired number of pods by using its Scale subresource.
	ScaleTargetRef CrossVersionObjectReference `json:"scaleTargetRef" protobuf:"bytes,1,opt,name=scaleTargetRef"`
	// lower limit for the number of pods that can be set by the autoscaler, default 1.
	// +optional
	MinReplicas *int32 `json:"minReplicas,omitempty" protobuf:"varint,2,opt,name=minReplicas"`
	// upper limit for the number of pods that can be set by the autoscaler; cannot be smaller than MinReplicas.
	MaxReplicas int32 `json:"maxReplicas" protobuf:"varint,3,opt,name=maxReplicas"`
	// target average CPU utilization (represented as a percentage of requested CPU) over all the pods;
	// if not specified the default autoscaling policy will be used.
	// +optional
	TargetCPUUtilizationPercentage *int32 `json:"targetCPUUtilizationPercentage,omitempty" protobuf:"varint,4,opt,name=targetCPUUtilizationPercentage"`
}

// current status of a horizontal pod autoscaler
type HorizontalPodAutoscalerStatus struct {
	// most recent generation observed by this autoscaler.
	// +optional
	ObservedGeneration *int64 `json:"observedGeneration,omitempty" protobuf:"varint,1,opt,name=observedGeneration"`

	// last time the HorizontalPodAutoscaler scaled the number of pods;
	// used by the autoscaler to control how often the number of pods is changed.
	// +optional
	LastScaleTime *metav1.Time `json:"lastScaleTime,omitempty" protobuf:"bytes,2,opt,name=lastScaleTime"`

	// current number of replicas of pods managed by this autoscaler.
	CurrentReplicas int32 `json:"currentReplicas" protobuf:"varint,3,opt,name=currentReplicas"`

	// desired number of replicas of pods managed by this autoscaler.
	DesiredReplicas int32 `json:"desiredReplicas" protobuf:"varint,4,opt,name=desiredReplicas"`

	// current average CPU utilization over all pods, represented as a percentage of requested CPU,
	// e.g. 70 means that an average pod is using now 70% of its requested CPU.
	// +optional
	CurrentCPUUtilizationPercentage *int32 `json:"currentCPUUtilizationPercentage,omitempty" protobuf:"varint,5,opt,name=currentCPUUtilizationPercentage"`
}

// +genclient=true

// configuration of a horizontal pod autoscaler.
type HorizontalPodAutoscaler struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// behaviour of autoscaler. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status.
	// +optional
	Spec HorizontalPodAutoscalerSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// current information about the autoscaler.
	// +optional
	Status HorizontalPodAutoscalerStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// list of horizontal pod autoscaler objects.
type HorizontalPodAutoscalerList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// list of horizontal pod autoscaler objects.
	Items []HorizontalPodAutoscaler `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// Scale represents a scaling request for a resource.
type Scale struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata; More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// defines the behavior of the scale. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status.
	// +optional
	Spec ScaleSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// current status of the scale. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status. Read-only.
	// +optional
	Status ScaleStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// ScaleSpec describes the attributes of a scale subresource.
type ScaleSpec struct {
	// desired number of instances for the scaled object.
	// +optional
	Replicas int32 `json:"replicas,omitempty" protobuf:"varint,1,opt,name=replicas"`
}

// ScaleStatus represents the current status of a scale subresource.
type ScaleStatus struct {
	// actual number of observed instances of the scaled object.
	Replicas int32 `json:"replicas" protobuf:"varint,1,opt,name=replicas"`

	// label query over pods that should match the replicas count. This is same
	// as the label selector but in the string format to avoid introspection
	// by clients. The string will be in the same format as the query-param syntax.
	// More info about label selectors: http://kubernetes.io/docs/user-guide/labels#label-selectors
	// +optional
	Selector string `json:"selector,omitempty" protobuf:"bytes,2,opt,name=selector"`
}

// the types below are used in the alpha metrics annotation

// a type of metric source
type MetricSourceType string

var (
	// a metric describing a kubernetes object (for example, hits-per-second on an Ingress object)
	ObjectSourceType MetricSourceType = "Object"
	// a metric describing each pod in the current scale target (for example, transactions-processed-per-second).
	// The values will be averaged together before being compared to the target value
	PodsSourceType MetricSourceType = "Pods"
	// a resource metric known to Kubernetes, as specified in requests and limits, describing each pod
	// in the current scale target (e.g. CPU or memory).  Such metrics are built in to Kubernetes,
	// and have special scaling options on top of those available to normal per-pod metrics (the "pods" source)
	ResourceSourceType MetricSourceType = "Resource"
)

// a specification for how to scale based on a single metric
// (only `type` and one other matching field should be set at once)
type MetricSpec struct {
	// the type of metric source (should match one of the fields below)
	Type MetricSourceType `json:"type" protobuf:"bytes,1,name=type"`

	// a metric describing a single kubernetes object (for example, hits-per-second on an Ingress object)
	// +optional
	Object *ObjectMetricSource `json:"object,omitempty" protobuf:"bytes,2,opt,name=object"`
	// a metric describing each pod in the current scale target (for example, transactions-processed-per-second).
	// The values will be averaged together before being compared to the target value
	// +optional
	Pods *PodsMetricSource `json:"pods,omitempty" protobuf:"bytes,3,opt,name=pods"`
	// a resource metric (such as those specified in requests and limits) known to Kubernetes
	// describing each pod in the current scale target (e.g. CPU or memory). Such metrics are
	// built in to Kubernetes, and have special scaling options on top of those available to
	// normal per-pod metrics using the "pods" source.
	// +optional
	Resource *ResourceMetricSource `json:"resource,omitempty" protobuf:"bytes,4,opt,name=resource"`
}

// a metric describing a single kubernetes object (for example, hits-per-second on an Ingress object)
type ObjectMetricSource struct {
	// the described Kubernetes object
	Target CrossVersionObjectReference `json:"target" protobuf:"bytes,1,name=target"`

	// the name of the metric in question
	MetricName string `json:"metricName" protobuf:"bytes,2,name=metricName"`
	// the target value of the metric (as a quantity)
	TargetValue resource.Quantity `json:"targetValue" protobuf:"bytes,3,name=targetValue"`
}

// a metric describing each pod in the current scale target (for example, transactions-processed-per-second).
// The values will be averaged together before being compared to the target value
type PodsMetricSource struct {
	// the name of the metric in question
	MetricName string `json:"metricName" protobuf:"bytes,1,name=metricName"`
	// the target value of the metric (as a quantity)
	TargetAverageValue resource.Quantity `json:"targetAverageValue" protobuf:"bytes,2,name=targetAverageValue"`
}

// a resource metric known to Kubernetes, as specified in requests and limits, describing each pod
// in the current scale target (e.g. CPU or memory).  The values will be averaged together before
// being compared to the target.  Such metrics are built in to Kubernetes, and have special
// scaling options on top of those available to normal per-pod metrics using the "pods" source.
// Only one "target" type should be set.
type ResourceMetricSource struct {
	// the name of the resource in question
	Name v1.ResourceName `json:"name" protobuf:"bytes,1,name=name"`
	// the target value of the resource metric, represented as
	// a percentage of the requested value of the resource on the pods.
	// +optional
	TargetAverageUtilization *int32 `json:"targetAverageUtilization,omitempty" protobuf:"varint,2,opt,name=targetAverageUtilization"`
	// the target value of the resource metric as a raw value, similarly
	// to the "pods" metric source type.
	// +optional
	TargetAverageValue *resource.Quantity `json:"targetAverageValue,omitempty" protobuf:"bytes,3,opt,name=targetAverageValue"`
}

// the status of a single metric
type MetricStatus struct {
	// the type of metric source
	Type MetricSourceType `json:"type" protobuf:"bytes,1,name=type"`

	// a metric describing a single kubernetes object (for example, hits-per-second on an Ingress object)
	// +optional
	Object *ObjectMetricStatus `json:"object,omitempty" protobuf:"bytes,2,opt,name=object"`
	// a metric describing each pod in the current scale target (for example, transactions-processed-per-second).
	// The values will be averaged together before being compared to the target value
	// +optional
	Pods *PodsMetricStatus `json:"pods,omitempty" protobuf:"bytes,3,opt,name=pods"`
	// a resource metric known to Kubernetes, as specified in requests and limits, describing each pod
	// in the current scale target (e.g. CPU or memory).  Such metrics are built in to Kubernetes,
	// and have special scaling options on top of those available to normal per-pod metrics using the "pods" source.
	// +optional
	Resource *ResourceMetricStatus `json:"resource,omitempty" protobuf:"bytes,4,opt,name=resource"`
}

// a metric describing a single kubernetes object (for example, hits-per-second on an Ingress object)
type ObjectMetricStatus struct {
	// the described Kubernetes object
	Target CrossVersionObjectReference `json:"target" protobuf:"bytes,1,name=target"`

	// the name of the metric in question
	MetricName string `json:"metricName" protobuf:"bytes,2,name=metricName"`
	// the current value of the metric (as a quantity)
	CurrentValue resource.Quantity `json:"currentValue" protobuf:"bytes,3,name=currentValue"`
}

// a metric describing each pod in the current scale target (for example, transactions-processed-per-second).
// The values will be averaged together before being compared to the target value
type PodsMetricStatus struct {
	// the name of the metric in question
	MetricName string `json:"metricName" protobuf:"bytes,1,name=metricName"`
	// the current value of the metric (as a quantity)
	CurrentAverageValue resource.Quantity `json:"currentAverageValue" protobuf:"bytes,2,name=currentAverageValue"`
}

// a resource metric known to Kubernetes, as specified in requests and limits, describing each pod
// in the current scale target (e.g. CPU or memory).  The values will be averaged together before
// being compared to the target.  Such metrics are built in to Kubernetes, and have special
// scaling options on top of those available to normal per-pod metrics using the "pods" source.
// Only one "target" type should be set.  Note that the current raw value is always displayed
// (even when the current values as request utilization is also displayed).
type ResourceMetricStatus struct {
	// the name of the resource in question
	Name v1.ResourceName `json:"name" protobuf:"bytes,1,name=name"`
	// the target value of the resource metric, represented as
	// a percentage of the requested value of the resource on the pods
	// (only populated if the corresponding request target was set)
	// +optional
	CurrentAverageUtilization *int32 `json:"currentAverageUtilization,omitempty" protobuf:"bytes,2,opt,name=currentAverageUtilization"`
	// the current value of the resource metric as a raw value
	CurrentAverageValue resource.Quantity `json:"currentAverageValue" protobuf:"bytes,3,name=currentAverageValue"`
}
