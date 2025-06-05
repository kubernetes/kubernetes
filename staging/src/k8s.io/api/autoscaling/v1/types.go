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
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// CrossVersionObjectReference contains enough information to let you identify the referred resource.
// +structType=atomic
type CrossVersionObjectReference struct {
	// kind is the kind of the referent; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	Kind string `json:"kind" protobuf:"bytes,1,opt,name=kind"`

	// name is the name of the referent; More info: https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#names
	Name string `json:"name" protobuf:"bytes,2,opt,name=name"`

	// apiVersion is the API version of the referent
	// +optional
	APIVersion string `json:"apiVersion,omitempty" protobuf:"bytes,3,opt,name=apiVersion"`
}

// specification of a horizontal pod autoscaler.
type HorizontalPodAutoscalerSpec struct {
	// reference to scaled resource; horizontal pod autoscaler will learn the current resource consumption
	// and will set the desired number of pods by using its Scale subresource.
	ScaleTargetRef CrossVersionObjectReference `json:"scaleTargetRef" protobuf:"bytes,1,opt,name=scaleTargetRef"`
	// minReplicas is the lower limit for the number of replicas to which the autoscaler
	// can scale down.  It defaults to 1 pod.  minReplicas is allowed to be 0 if the
	// alpha feature gate HPAScaleToZero is enabled and at least one Object or External
	// metric is configured.  Scaling is active as long as at least one metric value is
	// available.
	// +optional
	MinReplicas *int32 `json:"minReplicas,omitempty" protobuf:"varint,2,opt,name=minReplicas"`

	// maxReplicas is the upper limit for the number of pods that can be set by the autoscaler; cannot be smaller than MinReplicas.
	MaxReplicas int32 `json:"maxReplicas" protobuf:"varint,3,opt,name=maxReplicas"`

	// targetCPUUtilizationPercentage is the target average CPU utilization (represented as a percentage of requested CPU) over all the pods;
	// if not specified the default autoscaling policy will be used.
	// +optional
	TargetCPUUtilizationPercentage *int32 `json:"targetCPUUtilizationPercentage,omitempty" protobuf:"varint,4,opt,name=targetCPUUtilizationPercentage"`
}

// current status of a horizontal pod autoscaler
type HorizontalPodAutoscalerStatus struct {
	// observedGeneration is the most recent generation observed by this autoscaler.
	// +optional
	ObservedGeneration *int64 `json:"observedGeneration,omitempty" protobuf:"varint,1,opt,name=observedGeneration"`

	// lastScaleTime is the last time the HorizontalPodAutoscaler scaled the number of pods;
	// used by the autoscaler to control how often the number of pods is changed.
	// +optional
	LastScaleTime *metav1.Time `json:"lastScaleTime,omitempty" protobuf:"bytes,2,opt,name=lastScaleTime"`

	// currentReplicas is the current number of replicas of pods managed by this autoscaler.
	CurrentReplicas int32 `json:"currentReplicas" protobuf:"varint,3,opt,name=currentReplicas"`

	// desiredReplicas is the  desired number of replicas of pods managed by this autoscaler.
	DesiredReplicas int32 `json:"desiredReplicas" protobuf:"varint,4,opt,name=desiredReplicas"`

	// currentCPUUtilizationPercentage is the current average CPU utilization over all pods, represented as a percentage of requested CPU,
	// e.g. 70 means that an average pod is using now 70% of its requested CPU.
	// +optional
	CurrentCPUUtilizationPercentage *int32 `json:"currentCPUUtilizationPercentage,omitempty" protobuf:"varint,5,opt,name=currentCPUUtilizationPercentage"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.2

// configuration of a horizontal pod autoscaler.
type HorizontalPodAutoscaler struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata. More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec defines the behaviour of autoscaler. More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status.
	// +optional
	Spec HorizontalPodAutoscalerSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// status is the current information about the autoscaler.
	// +optional
	Status HorizontalPodAutoscalerStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.2

// list of horizontal pod autoscaler objects.
type HorizontalPodAutoscalerList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of horizontal pod autoscaler objects.
	Items []HorizontalPodAutoscaler `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.2
// +k8s:isSubresource=/scale

// Scale represents a scaling request for a resource.
type Scale struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata; More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec defines the behavior of the scale. More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status.
	// +optional
	Spec ScaleSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// status is the current status of the scale. More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status. Read-only.
	// +optional
	Status ScaleStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// ScaleSpec describes the attributes of a scale subresource.
type ScaleSpec struct {
	// replicas is the desired number of instances for the scaled object.
	// +optional
	// +k8s:optional
	// +default=0
	// +k8s:minimum=0
	Replicas int32 `json:"replicas,omitempty" protobuf:"varint,1,opt,name=replicas"`
}

// ScaleStatus represents the current status of a scale subresource.
type ScaleStatus struct {
	// replicas is the actual number of observed instances of the scaled object.
	Replicas int32 `json:"replicas" protobuf:"varint,1,opt,name=replicas"`

	// selector is the label query over pods that should match the replicas count. This is same
	// as the label selector but in the string format to avoid introspection
	// by clients. The string will be in the same format as the query-param syntax.
	// More info about label selectors: https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/
	// +optional
	Selector string `json:"selector,omitempty" protobuf:"bytes,2,opt,name=selector"`
}

// the types below are used in the alpha metrics annotation

// MetricSourceType indicates the type of metric.
// +enum
type MetricSourceType string

const (
	// ObjectMetricSourceType is a metric describing a kubernetes object
	// (for example, hits-per-second on an Ingress object).
	ObjectMetricSourceType MetricSourceType = "Object"
	// PodsMetricSourceType is a metric describing each pod in the current scale
	// target (for example, transactions-processed-per-second).  The values
	// will be averaged together before being compared to the target value.
	PodsMetricSourceType MetricSourceType = "Pods"
	// ResourceMetricSourceType is a resource metric known to Kubernetes, as
	// specified in requests and limits, describing each pod in the current
	// scale target (e.g. CPU or memory).  Such metrics are built in to
	// Kubernetes, and have special scaling options on top of those available
	// to normal per-pod metrics (the "pods" source).
	ResourceMetricSourceType MetricSourceType = "Resource"
	// ContainerResourceMetricSourceType is a resource metric known to Kubernetes, as
	// specified in requests and limits, describing a single container in each pod in the current
	// scale target (e.g. CPU or memory).  Such metrics are built in to
	// Kubernetes, and have special scaling options on top of those available
	// to normal per-pod metrics (the "pods" source).
	ContainerResourceMetricSourceType MetricSourceType = "ContainerResource"
	// ExternalMetricSourceType is a global metric that is not associated
	// with any Kubernetes object. It allows autoscaling based on information
	// coming from components running outside of cluster
	// (for example length of queue in cloud messaging service, or
	// QPS from loadbalancer running outside of cluster).
	ExternalMetricSourceType MetricSourceType = "External"
)

// MetricSpec specifies how to scale based on a single metric
// (only `type` and one other matching field should be set at once).
type MetricSpec struct {
	// type is the type of metric source.  It should be one of "ContainerResource",
	// "External", "Object", "Pods" or "Resource", each mapping to a matching field in the object.
	Type MetricSourceType `json:"type" protobuf:"bytes,1,name=type"`

	// object refers to a metric describing a single kubernetes object
	// (for example, hits-per-second on an Ingress object).
	// +optional
	Object *ObjectMetricSource `json:"object,omitempty" protobuf:"bytes,2,opt,name=object"`

	// pods refers to a metric describing each pod in the current scale target
	// (for example, transactions-processed-per-second).  The values will be
	// averaged together before being compared to the target value.
	// +optional
	Pods *PodsMetricSource `json:"pods,omitempty" protobuf:"bytes,3,opt,name=pods"`

	// resource refers to a resource metric (such as those specified in
	// requests and limits) known to Kubernetes describing each pod in the
	// current scale target (e.g. CPU or memory). Such metrics are built in to
	// Kubernetes, and have special scaling options on top of those available
	// to normal per-pod metrics using the "pods" source.
	// +optional
	Resource *ResourceMetricSource `json:"resource,omitempty" protobuf:"bytes,4,opt,name=resource"`

	// containerResource refers to a resource metric (such as those specified in
	// requests and limits) known to Kubernetes describing a single container in each pod of the
	// current scale target (e.g. CPU or memory). Such metrics are built in to
	// Kubernetes, and have special scaling options on top of those available
	// to normal per-pod metrics using the "pods" source.
	// +optional
	ContainerResource *ContainerResourceMetricSource `json:"containerResource,omitempty" protobuf:"bytes,7,opt,name=containerResource"`

	// external refers to a global metric that is not associated
	// with any Kubernetes object. It allows autoscaling based on information
	// coming from components running outside of cluster
	// (for example length of queue in cloud messaging service, or
	// QPS from loadbalancer running outside of cluster).
	// +optional
	External *ExternalMetricSource `json:"external,omitempty" protobuf:"bytes,5,opt,name=external"`
}

// ObjectMetricSource indicates how to scale on a metric describing a
// kubernetes object (for example, hits-per-second on an Ingress object).
type ObjectMetricSource struct {
	// target is the described Kubernetes object.
	Target CrossVersionObjectReference `json:"target" protobuf:"bytes,1,name=target"`

	// metricName is the name of the metric in question.
	MetricName string `json:"metricName" protobuf:"bytes,2,name=metricName"`

	// targetValue is the target value of the metric (as a quantity).
	TargetValue resource.Quantity `json:"targetValue" protobuf:"bytes,3,name=targetValue"`

	// selector is the string-encoded form of a standard kubernetes label selector for the given metric.
	// When set, it is passed as an additional parameter to the metrics server for more specific metrics scoping
	// When unset, just the metricName will be used to gather metrics.
	// +optional
	Selector *metav1.LabelSelector `json:"selector,omitempty" protobuf:"bytes,4,name=selector"`

	// averageValue is the target value of the average of the
	// metric across all relevant pods (as a quantity)
	// +optional
	AverageValue *resource.Quantity `json:"averageValue,omitempty" protobuf:"bytes,5,name=averageValue"`
}

// PodsMetricSource indicates how to scale on a metric describing each pod in
// the current scale target (for example, transactions-processed-per-second).
// The values will be averaged together before being compared to the target
// value.
type PodsMetricSource struct {
	// metricName is the name of the metric in question
	MetricName string `json:"metricName" protobuf:"bytes,1,name=metricName"`

	// targetAverageValue is the target value of the average of the
	// metric across all relevant pods (as a quantity)
	TargetAverageValue resource.Quantity `json:"targetAverageValue" protobuf:"bytes,2,name=targetAverageValue"`

	// selector is the string-encoded form of a standard kubernetes label selector for the given metric
	// When set, it is passed as an additional parameter to the metrics server for more specific metrics scoping
	// When unset, just the metricName will be used to gather metrics.
	// +optional
	Selector *metav1.LabelSelector `json:"selector,omitempty" protobuf:"bytes,3,name=selector"`
}

// ResourceMetricSource indicates how to scale on a resource metric known to
// Kubernetes, as specified in requests and limits, describing each pod in the
// current scale target (e.g. CPU or memory).  The values will be averaged
// together before being compared to the target.  Such metrics are built in to
// Kubernetes, and have special scaling options on top of those available to
// normal per-pod metrics using the "pods" source.  Only one "target" type
// should be set.
type ResourceMetricSource struct {
	// name is the name of the resource in question.
	Name v1.ResourceName `json:"name" protobuf:"bytes,1,name=name"`

	// targetAverageUtilization is the target value of the average of the
	// resource metric across all relevant pods, represented as a percentage of
	// the requested value of the resource for the pods.
	// +optional
	TargetAverageUtilization *int32 `json:"targetAverageUtilization,omitempty" protobuf:"varint,2,opt,name=targetAverageUtilization"`

	// targetAverageValue is the target value of the average of the
	// resource metric across all relevant pods, as a raw value (instead of as
	// a percentage of the request), similar to the "pods" metric source type.
	// +optional
	TargetAverageValue *resource.Quantity `json:"targetAverageValue,omitempty" protobuf:"bytes,3,opt,name=targetAverageValue"`
}

// ContainerResourceMetricSource indicates how to scale on a resource metric known to
// Kubernetes, as specified in the requests and limits, describing a single container in
// each of the pods of the current scale target(e.g. CPU or memory). The values will be
// averaged together before being compared to the target. Such metrics are built into
// Kubernetes, and have special scaling options on top of those available to
// normal per-pod metrics using the "pods" source. Only one "target" type
// should be set.
type ContainerResourceMetricSource struct {
	// name is the name of the resource in question.
	Name v1.ResourceName `json:"name" protobuf:"bytes,1,name=name"`

	// targetAverageUtilization is the target value of the average of the
	// resource metric across all relevant pods, represented as a percentage of
	// the requested value of the resource for the pods.
	// +optional
	TargetAverageUtilization *int32 `json:"targetAverageUtilization,omitempty" protobuf:"varint,2,opt,name=targetAverageUtilization"`

	// targetAverageValue is the target value of the average of the
	// resource metric across all relevant pods, as a raw value (instead of as
	// a percentage of the request), similar to the "pods" metric source type.
	// +optional
	TargetAverageValue *resource.Quantity `json:"targetAverageValue,omitempty" protobuf:"bytes,3,opt,name=targetAverageValue"`

	// container is the name of the container in the pods of the scaling target.
	Container string `json:"container" protobuf:"bytes,5,opt,name=container"`
}

// ExternalMetricSource indicates how to scale on a metric not associated with
// any Kubernetes object (for example length of queue in cloud
// messaging service, or QPS from loadbalancer running outside of cluster).
type ExternalMetricSource struct {
	// metricName is the name of the metric in question.
	MetricName string `json:"metricName" protobuf:"bytes,1,name=metricName"`

	// metricSelector is used to identify a specific time series
	// within a given metric.
	// +optional
	MetricSelector *metav1.LabelSelector `json:"metricSelector,omitempty" protobuf:"bytes,2,opt,name=metricSelector"`

	// targetValue is the target value of the metric (as a quantity).
	// Mutually exclusive with TargetAverageValue.
	// +optional
	TargetValue *resource.Quantity `json:"targetValue,omitempty" protobuf:"bytes,3,opt,name=targetValue"`

	// targetAverageValue is the target per-pod value of global metric (as a quantity).
	// Mutually exclusive with TargetValue.
	// +optional
	TargetAverageValue *resource.Quantity `json:"targetAverageValue,omitempty" protobuf:"bytes,4,opt,name=targetAverageValue"`
}

// MetricStatus describes the last-read state of a single metric.
type MetricStatus struct {
	// type is the type of metric source.  It will be one of "ContainerResource",
	// "External", "Object", "Pods" or "Resource", each corresponds to a matching field in the object.
	Type MetricSourceType `json:"type" protobuf:"bytes,1,name=type"`

	// object refers to a metric describing a single kubernetes object
	// (for example, hits-per-second on an Ingress object).
	// +optional
	Object *ObjectMetricStatus `json:"object,omitempty" protobuf:"bytes,2,opt,name=object"`

	// pods refers to a metric describing each pod in the current scale target
	// (for example, transactions-processed-per-second).  The values will be
	// averaged together before being compared to the target value.
	// +optional
	Pods *PodsMetricStatus `json:"pods,omitempty" protobuf:"bytes,3,opt,name=pods"`

	// resource refers to a resource metric (such as those specified in
	// requests and limits) known to Kubernetes describing each pod in the
	// current scale target (e.g. CPU or memory). Such metrics are built in to
	// Kubernetes, and have special scaling options on top of those available
	// to normal per-pod metrics using the "pods" source.
	// +optional
	Resource *ResourceMetricStatus `json:"resource,omitempty" protobuf:"bytes,4,opt,name=resource"`

	// containerResource refers to a resource metric (such as those specified in
	// requests and limits) known to Kubernetes describing a single container in each pod in the
	// current scale target (e.g. CPU or memory). Such metrics are built in to
	// Kubernetes, and have special scaling options on top of those available
	// to normal per-pod metrics using the "pods" source.
	// +optional
	ContainerResource *ContainerResourceMetricStatus `json:"containerResource,omitempty" protobuf:"bytes,7,opt,name=containerResource"`

	// external refers to a global metric that is not associated
	// with any Kubernetes object. It allows autoscaling based on information
	// coming from components running outside of cluster
	// (for example length of queue in cloud messaging service, or
	// QPS from loadbalancer running outside of cluster).
	// +optional
	External *ExternalMetricStatus `json:"external,omitempty" protobuf:"bytes,5,opt,name=external"`
}

// HorizontalPodAutoscalerConditionType are the valid conditions of
// a HorizontalPodAutoscaler.
type HorizontalPodAutoscalerConditionType string

const (
	// ScalingActive indicates that the HPA controller is able to scale if necessary:
	// it's correctly configured, can fetch the desired metrics, and isn't disabled.
	ScalingActive HorizontalPodAutoscalerConditionType = "ScalingActive"
	// AbleToScale indicates a lack of transient issues which prevent scaling from occurring,
	// such as being in a backoff window, or being unable to access/update the target scale.
	AbleToScale HorizontalPodAutoscalerConditionType = "AbleToScale"
	// ScalingLimited indicates that the calculated scale based on metrics would be above or
	// below the range for the HPA, and has thus been capped.
	ScalingLimited HorizontalPodAutoscalerConditionType = "ScalingLimited"
)

// HorizontalPodAutoscalerCondition describes the state of
// a HorizontalPodAutoscaler at a certain point.
type HorizontalPodAutoscalerCondition struct {
	// type describes the current condition
	Type HorizontalPodAutoscalerConditionType `json:"type" protobuf:"bytes,1,name=type"`

	// status is the status of the condition (True, False, Unknown)
	Status v1.ConditionStatus `json:"status" protobuf:"bytes,2,name=status"`

	// lastTransitionTime is the last time the condition transitioned from
	// one status to another
	// +optional
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty" protobuf:"bytes,3,opt,name=lastTransitionTime"`

	// reason is the reason for the condition's last transition.
	// +optional
	Reason string `json:"reason,omitempty" protobuf:"bytes,4,opt,name=reason"`

	// message is a human-readable explanation containing details about
	// the transition
	// +optional
	Message string `json:"message,omitempty" protobuf:"bytes,5,opt,name=message"`
}

// ObjectMetricStatus indicates the current value of a metric describing a
// kubernetes object (for example, hits-per-second on an Ingress object).
type ObjectMetricStatus struct {
	// target is the described Kubernetes object.
	Target CrossVersionObjectReference `json:"target" protobuf:"bytes,1,name=target"`

	// metricName is the name of the metric in question.
	MetricName string `json:"metricName" protobuf:"bytes,2,name=metricName"`

	// currentValue is the current value of the metric (as a quantity).
	CurrentValue resource.Quantity `json:"currentValue" protobuf:"bytes,3,name=currentValue"`

	// selector is the string-encoded form of a standard kubernetes label selector for the given metric
	// When set in the ObjectMetricSource, it is passed as an additional parameter to the metrics server for more specific metrics scoping.
	// When unset, just the metricName will be used to gather metrics.
	// +optional
	Selector *metav1.LabelSelector `json:"selector,omitempty" protobuf:"bytes,4,name=selector"`

	// averageValue is the current value of the average of the
	// metric across all relevant pods (as a quantity)
	// +optional
	AverageValue *resource.Quantity `json:"averageValue,omitempty" protobuf:"bytes,5,name=averageValue"`
}

// PodsMetricStatus indicates the current value of a metric describing each pod in
// the current scale target (for example, transactions-processed-per-second).
type PodsMetricStatus struct {
	// metricName is the name of the metric in question
	MetricName string `json:"metricName" protobuf:"bytes,1,name=metricName"`

	// currentAverageValue is the current value of the average of the
	// metric across all relevant pods (as a quantity)
	CurrentAverageValue resource.Quantity `json:"currentAverageValue" protobuf:"bytes,2,name=currentAverageValue"`

	// selector is the string-encoded form of a standard kubernetes label selector for the given metric
	// When set in the PodsMetricSource, it is passed as an additional parameter to the metrics server for more specific metrics scoping.
	// When unset, just the metricName will be used to gather metrics.
	// +optional
	Selector *metav1.LabelSelector `json:"selector,omitempty" protobuf:"bytes,3,name=selector"`
}

// ResourceMetricStatus indicates the current value of a resource metric known to
// Kubernetes, as specified in requests and limits, describing each pod in the
// current scale target (e.g. CPU or memory).  Such metrics are built in to
// Kubernetes, and have special scaling options on top of those available to
// normal per-pod metrics using the "pods" source.
type ResourceMetricStatus struct {
	// name is the name of the resource in question.
	Name v1.ResourceName `json:"name" protobuf:"bytes,1,name=name"`

	// currentAverageUtilization is the current value of the average of the
	// resource metric across all relevant pods, represented as a percentage of
	// the requested value of the resource for the pods.  It will only be
	// present if `targetAverageValue` was set in the corresponding metric
	// specification.
	// +optional
	CurrentAverageUtilization *int32 `json:"currentAverageUtilization,omitempty" protobuf:"bytes,2,opt,name=currentAverageUtilization"`

	// currentAverageValue is the current value of the average of the
	// resource metric across all relevant pods, as a raw value (instead of as
	// a percentage of the request), similar to the "pods" metric source type.
	// It will always be set, regardless of the corresponding metric specification.
	CurrentAverageValue resource.Quantity `json:"currentAverageValue" protobuf:"bytes,3,name=currentAverageValue"`
}

// ContainerResourceMetricStatus indicates the current value of a resource metric known to
// Kubernetes, as specified in requests and limits, describing a single container in each pod in the
// current scale target (e.g. CPU or memory).  Such metrics are built in to
// Kubernetes, and have special scaling options on top of those available to
// normal per-pod metrics using the "pods" source.
type ContainerResourceMetricStatus struct {
	// name is the name of the resource in question.
	Name v1.ResourceName `json:"name" protobuf:"bytes,1,name=name"`

	// currentAverageUtilization is the current value of the average of the
	// resource metric across all relevant pods, represented as a percentage of
	// the requested value of the resource for the pods.  It will only be
	// present if `targetAverageValue` was set in the corresponding metric
	// specification.
	// +optional
	CurrentAverageUtilization *int32 `json:"currentAverageUtilization,omitempty" protobuf:"bytes,2,opt,name=currentAverageUtilization"`

	// currentAverageValue is the current value of the average of the
	// resource metric across all relevant pods, as a raw value (instead of as
	// a percentage of the request), similar to the "pods" metric source type.
	// It will always be set, regardless of the corresponding metric specification.
	CurrentAverageValue resource.Quantity `json:"currentAverageValue" protobuf:"bytes,3,name=currentAverageValue"`

	// container is the name of the container in the pods of the scaling taget
	Container string `json:"container" protobuf:"bytes,4,opt,name=container"`
}

// ExternalMetricStatus indicates the current value of a global metric
// not associated with any Kubernetes object.
type ExternalMetricStatus struct {
	// metricName is the name of a metric used for autoscaling in
	// metric system.
	MetricName string `json:"metricName" protobuf:"bytes,1,name=metricName"`

	// metricSelector is used to identify a specific time series
	// within a given metric.
	// +optional
	MetricSelector *metav1.LabelSelector `json:"metricSelector,omitempty" protobuf:"bytes,2,opt,name=metricSelector"`
	// currentValue is the current value of the metric (as a quantity)
	CurrentValue resource.Quantity `json:"currentValue" protobuf:"bytes,3,name=currentValue"`

	// currentAverageValue is the current value of metric averaged over autoscaled pods.
	// +optional
	CurrentAverageValue *resource.Quantity `json:"currentAverageValue,omitempty" protobuf:"bytes,4,opt,name=currentAverageValue"`
}
