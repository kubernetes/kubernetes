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

import (
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Scale represents a scaling request for a resource.
type Scale struct {
	metav1.TypeMeta
	// Standard object metadata; More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta

	// defines the behavior of the scale. More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#spec-and-status.
	// +optional
	Spec ScaleSpec

	// current status of the scale. More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#spec-and-status. Read-only.
	// +optional
	Status ScaleStatus
}

// ScaleSpec describes the attributes of a scale subresource.
type ScaleSpec struct {
	// desired number of instances for the scaled object.
	// +optional
	Replicas int32
}

// ScaleStatus represents the current status of a scale subresource.
type ScaleStatus struct {
	// actual number of observed instances of the scaled object.
	Replicas int32

	// label query over pods that should match the replicas count. This is same
	// as the label selector but in the string format to avoid introspection
	// by clients. The string will be in the same format as the query-param syntax.
	// More info: https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#label-selectors
	// +optional
	Selector string
}

// CrossVersionObjectReference contains enough information to let you identify the referred resource.
type CrossVersionObjectReference struct {
	// Kind of the referent; More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#types-kinds"
	Kind string
	// Name of the referent; More info: http://kubernetes.io/docs/user-guide/identifiers#names
	Name string
	// API version of the referent
	// +optional
	APIVersion string
}

// HorizontalPodAutoscalerSpec describes the desired functionality of the HorizontalPodAutoscaler.
type HorizontalPodAutoscalerSpec struct {
	// ScaleTargetRef points to the target resource to scale, and is used to the pods for which metrics
	// should be collected, as well as to actually change the replica count.
	ScaleTargetRef CrossVersionObjectReference
	// MinReplicas is the lower limit for the number of replicas to which the autoscaler can scale down.
	// It defaults to 1 pod.
	// +optional
	MinReplicas *int32
	// MaxReplicas is the upper limit for the number of replicas to which the autoscaler can scale up.
	// It cannot be less that minReplicas.
	MaxReplicas int32
	// Metrics contains the specifications for which to use to calculate the
	// desired replica count (the maximum replica count across all metrics will
	// be used).  The desired replica count is calculated multiplying the
	// ratio between the target value and the current value by the current
	// number of pods.  Ergo, metrics used must decrease as the pod count is
	// increased, and vice-versa.  See the individual metric source types for
	// more information about how each type of metric must respond.
	// +optional
	Metrics []MetricSpec
}

// MetricSourceType indicates the type of metric.
type MetricSourceType string

var (
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
	// Type is the type of metric source.  It should be one of "Object",
	// "Pods" or "Resource", each mapping to a matching field in the object.
	Type MetricSourceType

	// Object refers to a metric describing a single kubernetes object
	// (for example, hits-per-second on an Ingress object).
	// +optional
	Object *ObjectMetricSource
	// Pods refers to a metric describing each pod in the current scale target
	// (for example, transactions-processed-per-second).  The values will be
	// averaged together before being compared to the target value.
	// +optional
	Pods *PodsMetricSource
	// Resource refers to a resource metric (such as those specified in
	// requests and limits) known to Kubernetes describing each pod in the
	// current scale target (e.g. CPU or memory). Such metrics are built in to
	// Kubernetes, and have special scaling options on top of those available
	// to normal per-pod metrics using the "pods" source.
	// +optional
	Resource *ResourceMetricSource
	// External refers to a global metric that is not associated
	// with any Kubernetes object. It allows autoscaling based on information
	// coming from components running outside of cluster
	// (for example length of queue in cloud messaging service, or
	// QPS from loadbalancer running outside of cluster).
	// +optional
	External *ExternalMetricSource
}

// ObjectMetricSource indicates how to scale on a metric describing a
// kubernetes object (for example, hits-per-second on an Ingress object).
type ObjectMetricSource struct {
	DescribedObject CrossVersionObjectReference
	Target          MetricTarget
	Metric          MetricIdentifier
}

// PodsMetricSource indicates how to scale on a metric describing each pod in
// the current scale target (for example, transactions-processed-per-second).
// The values will be averaged together before being compared to the target
// value.
type PodsMetricSource struct {
	// metric identifies the target metric by name and selector
	Metric MetricIdentifier
	// target specifies the target value for the given metric
	Target MetricTarget
}

// ResourceMetricSource indicates how to scale on a resource metric known to
// Kubernetes, as specified in requests and limits, describing each pod in the
// current scale target (e.g. CPU or memory).  The values will be averaged
// together before being compared to the target.  Such metrics are built in to
// Kubernetes, and have special scaling options on top of those available to
// normal per-pod metrics using the "pods" source.  Only one "target" type
// should be set.
type ResourceMetricSource struct {
	// Name is the name of the resource in question.
	Name api.ResourceName
	// Target specifies the target value for the given metric
	Target MetricTarget
}

// ExternalMetricSource indicates how to scale on a metric not associated with
// any Kubernetes object (for example length of queue in cloud
// messaging service, or QPS from loadbalancer running outside of cluster).
type ExternalMetricSource struct {
	// Metric identifies the target metric by name and selector
	Metric MetricIdentifier
	// Target specifies the target value for the given metric
	Target MetricTarget
}

// MetricIdentifier defines the name and optionally selector for a metric
type MetricIdentifier struct {
	// Name is the name of the given metric
	Name string
	// Selector is the selector for the given metric
	// it is the string-encoded form of a standard kubernetes label selector
	// +optional
	Selector *metav1.LabelSelector
}

// MetricTarget defines the target value, average value, or average utilization of a specific metric
type MetricTarget struct {
	// Type represents whether the metric type is Utilization, Value, or AverageValue
	Type MetricTargetType
	// Value is the target value of the metric (as a quantity).
	Value *resource.Quantity
	// TargetAverageValue is the target value of the average of the
	// metric across all relevant pods (as a quantity)
	AverageValue *resource.Quantity

	// AverageUtilization is the target value of the average of the
	// resource metric across all relevant pods, represented as a percentage of
	// the requested value of the resource for the pods.
	// Currently only valid for Resource metric source type
	AverageUtilization *int32
}

// MetricTargetType specifies the type of metric being targeted, and should be either
// "Value", "AverageValue", or "Utilization"
type MetricTargetType string

var (
	// UtilizationMetricType is a possible value for MetricTarget.Type.
	UtilizationMetricType MetricTargetType = "Utilization"
	// ValueMetricType is a possible value for MetricTarget.Type.
	ValueMetricType MetricTargetType = "Value"
	// AverageValueMetricType is a possible value for MetricTarget.Type.
	AverageValueMetricType MetricTargetType = "AverageValue"
)

// HorizontalPodAutoscalerStatus describes the current status of a horizontal pod autoscaler.
type HorizontalPodAutoscalerStatus struct {
	// ObservedGeneration is the most recent generation observed by this autoscaler.
	// +optional
	ObservedGeneration *int64

	// LastScaleTime is the last time the HorizontalPodAutoscaler scaled the number of pods,
	// used by the autoscaler to control how often the number of pods is changed.
	// +optional
	LastScaleTime *metav1.Time

	// CurrentReplicas is current number of replicas of pods managed by this autoscaler,
	// as last seen by the autoscaler.
	CurrentReplicas int32

	// DesiredReplicas is the desired number of replicas of pods managed by this autoscaler,
	// as last calculated by the autoscaler.
	DesiredReplicas int32

	// CurrentMetrics is the last read state of the metrics used by this autoscaler.
	// +optional
	CurrentMetrics []MetricStatus

	// Conditions is the set of conditions required for this autoscaler to scale its target,
	// and indicates whether or not those conditions are met.
	Conditions []HorizontalPodAutoscalerCondition
}

// ConditionStatus indicates the status of a condition (true, false, or unknown).
type ConditionStatus string

// These are valid condition statuses. "ConditionTrue" means a resource is in the condition;
// "ConditionFalse" means a resource is not in the condition; "ConditionUnknown" means kubernetes
// can't decide if a resource is in the condition or not. In the future, we could add other
// intermediate conditions, e.g. ConditionDegraded.
const (
	ConditionTrue    ConditionStatus = "True"
	ConditionFalse   ConditionStatus = "False"
	ConditionUnknown ConditionStatus = "Unknown"
)

// HorizontalPodAutoscalerConditionType are the valid conditions of
// a HorizontalPodAutoscaler.
type HorizontalPodAutoscalerConditionType string

var (
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
	// Type describes the current condition
	Type HorizontalPodAutoscalerConditionType
	// Status is the status of the condition (True, False, Unknown)
	Status ConditionStatus
	// LastTransitionTime is the last time the condition transitioned from
	// one status to another
	// +optional
	LastTransitionTime metav1.Time
	// Reason is the reason for the condition's last transition.
	// +optional
	Reason string
	// Message is a human-readable explanation containing details about
	// the transition
	// +optional
	Message string
}

// MetricStatus describes the last-read state of a single metric.
type MetricStatus struct {
	// Type is the type of metric source.  It will be one of "Object",
	// "Pods" or "Resource", each corresponds to a matching field in the object.
	Type MetricSourceType

	// Object refers to a metric describing a single kubernetes object
	// (for example, hits-per-second on an Ingress object).
	// +optional
	Object *ObjectMetricStatus
	// Pods refers to a metric describing each pod in the current scale target
	// (for example, transactions-processed-per-second).  The values will be
	// averaged together before being compared to the target value.
	// +optional
	Pods *PodsMetricStatus
	// Resource refers to a resource metric (such as those specified in
	// requests and limits) known to Kubernetes describing each pod in the
	// current scale target (e.g. CPU or memory). Such metrics are built in to
	// Kubernetes, and have special scaling options on top of those available
	// to normal per-pod metrics using the "pods" source.
	// +optional
	Resource *ResourceMetricStatus
	// External refers to a global metric that is not associated
	// with any Kubernetes object. It allows autoscaling based on information
	// coming from components running outside of cluster
	// (for example length of queue in cloud messaging service, or
	// QPS from loadbalancer running outside of cluster).
	// +optional
	External *ExternalMetricStatus
}

// ObjectMetricStatus indicates the current value of a metric describing a
// kubernetes object (for example, hits-per-second on an Ingress object).
type ObjectMetricStatus struct {
	Metric  MetricIdentifier
	Current MetricValueStatus

	DescribedObject CrossVersionObjectReference
}

// PodsMetricStatus indicates the current value of a metric describing each pod in
// the current scale target (for example, transactions-processed-per-second).
type PodsMetricStatus struct {
	Metric  MetricIdentifier
	Current MetricValueStatus
}

// ResourceMetricStatus indicates the current value of a resource metric known to
// Kubernetes, as specified in requests and limits, describing each pod in the
// current scale target (e.g. CPU or memory).  Such metrics are built in to
// Kubernetes, and have special scaling options on top of those available to
// normal per-pod metrics using the "pods" source.
type ResourceMetricStatus struct {
	// Name is the name of the resource in question.
	Name    api.ResourceName
	Current MetricValueStatus
}

// ExternalMetricStatus indicates the current value of a global metric
// not associated with any Kubernetes object.
type ExternalMetricStatus struct {
	Metric  MetricIdentifier
	Current MetricValueStatus
}

// MetricValueStatus indicates the current value of a metric.
type MetricValueStatus struct {
	Value              *resource.Quantity
	AverageValue       *resource.Quantity
	AverageUtilization *int32
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// HorizontalPodAutoscaler is the configuration for a horizontal pod
// autoscaler, which automatically manages the replica count of any resource
// implementing the scale subresource based on the metrics specified.
type HorizontalPodAutoscaler struct {
	metav1.TypeMeta
	// Metadata is the standard object metadata.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta

	// Spec is the specification for the behaviour of the autoscaler.
	// More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#spec-and-status.
	// +optional
	Spec HorizontalPodAutoscalerSpec

	// Status is the current information about the autoscaler.
	// +optional
	Status HorizontalPodAutoscalerStatus
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// HorizontalPodAutoscalerList is a list of horizontal pod autoscaler objects.
type HorizontalPodAutoscalerList struct {
	metav1.TypeMeta
	// Metadata is the standard list metadata.
	// +optional
	metav1.ListMeta

	// Items is the list of horizontal pod autoscaler objects.
	Items []HorizontalPodAutoscaler
}
