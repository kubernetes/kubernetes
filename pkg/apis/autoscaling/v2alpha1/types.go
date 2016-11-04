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

package v2alpha1

import (
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
)

// TODO: do we want to change this to just refer to an API group, and not a version?
// contains enough information to let you identify a resource in the current namespace
type CrossVersionObjectReference struct {
	// Kind of the referent; More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds"
	Kind string `json:"kind" protobuf:"bytes,1,opt,name=kind"`
	// Name of the referent; More info: http://kubernetes.io/docs/user-guide/identifiers#names
	Name string `json:"name" protobuf:"bytes,2,opt,name=name"`
	// API version of the referent
	// +optional
	APIVersion string `json:"apiVersion,omitempty" protobuf:"bytes,3,opt,name=apiVersion"`
}

// specification of a horizontal pod autoscaler
type HorizontalPodAutoscalerSpec struct {
	// the target scalable object to autoscale
	ScaleTargetRef CrossVersionObjectReference `json:"scaleTargetRef" protobuf:"bytes,1,opt,name=scaleTargetRef"`

	// the minimum number of replicas to which the autoscaler may scale
	// +optional
	MinReplicas *int32 `json:"minReplicas,omitempty" protobuf:"varint,2,opt,name=minReplicas"`
	// the maximum number of replicas to which the autoscaler may scale
	MaxReplicas int32 `json:"maxReplicas" protobuf:"varint,3,opt,name=maxReplicas"`

	// the metrics to use to calculate the desired replica count (the
	// maximum replica count across all metrics will be used).  It is
	// expected that any metrics used will decrease as the replica count
	// increases, and will eventually increase if we decrease the replica
	// count.
	Metrics []MetricSpec `json:"metrics,omitempty" protobuf:"bytes,4,rep,name=metrics"`
}

// a type of metric source
type MetricSourceType string

var (
	// a metric describing a kubernetes object
	ObjectSourceType MetricSourceType = "object"
	// a metric describing pods in the scale target
	PodsSourceType MetricSourceType = "pods"
	// a resource metric known to Kubernetes
	ResourceSourceType MetricSourceType = "resource"
)

// a specification for how to scale based on a single metric
// (only `type` and one other matching field should be set at once)
type MetricSpec struct {
	// the type of metric source (should match one of the fields below)
	Type MetricSourceType `json:"type" protobuf:"bytes,1,opt,name=type"`

	// metric describing a single Kubernetes object
	Object *ObjectMetricSource `json:"object,omitempty" protobuf:"bytes,2,opt,name=object"`
	// metric describing pods in the scale target
	Pods *PodsMetricSource `json:"pods,omitempty" protobuf:"bytes,3,opt,name=pods"`
	// resource metric describing pods in the scale target
	// (guaranteed to be available and have the same names across clusters)
	Resource *ResourceMetricSource `json:"resource,omitempty" protobuf:"bytes,4,opt,name=resource"`
}

// a metric describing a Kubernetes object
type ObjectMetricSource struct {
	// the described Kubernetes object
	Target CrossVersionObjectReference `json:"target" protobuf:"bytes,1,opt,name=target"`

	// the name of the metric in question
	MetricName string `json:"metricName" protobuf:"bytes,2,opt,name=metricName"`
	// the target value of the metric (as a quantity)
	TargetValue resource.Quantity `json:"targetValue" protobuf:"bytes,3,opt,name=targetValue"`
}

// metric describing pods in the scale target
type PodsMetricSource struct {
	// the name of the metric in question
	MetricName string `json:"metricName" protobuf:"bytes,1,opt,name=metricName"`
	// the target value of the metric (as a quantity)
	TargetValue resource.Quantity `json:"targetValue" protobuf:"bytes,2,opt,name=targetValue"`
}

// resource metric describing pods in the scale target
// (guaranteed to be available and have the same names across clusters)
type ResourceMetricSource struct {
	// the name of the resource in question
	Name v1.ResourceName `json:"name" protobuf:"bytes,1,opt,name=name"`
	// the target value of the resource metric as a percentage of the
	// request on the pods
	// +optional
	TargetPercentageOfRequest *int32 `json:"targetPercentageOfRequest,omitempty" protobuf:"bytes,2,opt,name=targetPercentageOfRequest"`
	// the target value of the resource metric as a raw value
	// +optional
	TargetRawValue *resource.Quantity `json:"targetRawValue,omitempty" protobuf:"bytes,3,opt,name=targetRawValue"`
}

// the status of a horizontal pod autoscaler
type HorizontalPodAutoscalerStatus struct {
	// most recent generation observed by this autoscaler.
	// +optional
	ObservedGeneration *int64 `json:"observedGeneration,omitempty" protobuf:"varint,1,opt,name=observedGeneration"`
	// last time the HorizontalPodAutoscaler scaled the number of pods;
	// used by the autoscaler to control how often the number of pods is changed.
	// +optional
	LastScaleTime *unversioned.Time `json:"lastScaleTime,omitempty" protobuf:"bytes,2,opt,name=lastScaleTime"`

	// the last observed number of replicas from the target object.
	CurrentReplicas int32 `json:"currentReplicas" protobuf:"varint,3,opt,name=currentReplicas"`
	// the desired number of replicas as last computed by the autoscaler
	DesiredReplicas int32 `json:"desiredReplicas" protobuf:"varint,4,opt,name=desiredReplicas"`

	// the last read state of the metrics used by this autoscaler
	CurrentMetrics []MetricStatus `json:"currentMetrics" protobuf:"bytes,5,rep,name=currentMetrics"`
}

// the status of a single metric
type MetricStatus struct {
	// the type of metric source (should match one of the fields below)
	Type MetricSourceType `json:"type" protobuf:"bytes,1,opt,name=type"`

	// metric describing a single Kubernetes object
	Object *ObjectMetricStatus `json:"object,omitempty" protobuf:"bytes,2,opt,name=object"`
	// metric describing pods in the scale target
	Pods *PodsMetricStatus `json:"pods,omitempty" protobuf:"bytes,3,opt,name=pods"`
	// resource metric describing pods in the scale target
	// (guaranteed to be available and have the same names across clusters)
	Resource *ResourceMetricStatus `json:"resource,omitempty" protobuf:"bytes,4,opt,name=resource"`
}

// a metric describing a Kubernetes object
type ObjectMetricStatus struct {
	// the described Kubernetes object
	Target CrossVersionObjectReference `json:"target" protobuf:"bytes,1,opt,name=target"`

	// the name of the metric in question
	MetricName string `json:"metricName" protobuf:"bytes,2,opt,name=metricName"`
	// the target value of the metric (as a quantity)
	CurrentValue resource.Quantity `json:"currentValue" protobuf:"bytes,3,opt,name=currentValue"`
}

// metric describing pods in the scale target
type PodsMetricStatus struct {
	// the name of the metric in question
	MetricName string `json:"metricName" protobuf:"bytes,1,opt,name=metricName"`
	// the current value of the metric (as a quantity)
	CurrentValue resource.Quantity `json:"currentValue" protobuf:"bytes,2,opt,name=currentValue"`
}

// resource metric describing pods in the scale target
type ResourceMetricStatus struct {
	// the name of the resource in question
	Name v1.ResourceName `json:"name" protobuf:"bytes,1,opt,name=name"`
	// the current value of the resource metric as a percentage of the
	// request on the pods (only populated if request is available)
	// +optional
	CurrentPercentageOfRequest *int32 `json:"currentPercentageOfRequest,omitempty" protobuf:"bytes,2,opt,name=currentPercentageOfRequest"`
	// the target value of the resource metric as a raw value
	// +optional
	CurrentRawValue *resource.Quantity `json:"currentRawValue,omitempty" protobuf:"bytes,3,opt,name=currentRawValue"`
}

// +genclient=true

// configuration of a horizontal pod autoscaler.
type HorizontalPodAutoscaler struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object metadata. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	// +optional
	v1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// behaviour of autoscaler. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status.
	// +optional
	Spec HorizontalPodAutoscalerSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// current information about the autoscaler.
	// +optional
	Status HorizontalPodAutoscalerStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// list of horizontal pod autoscaler objects.
type HorizontalPodAutoscalerList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata.
	// +optional
	unversioned.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// list of horizontal pod autoscaler objects.
	Items []HorizontalPodAutoscaler `json:"items" protobuf:"bytes,2,rep,name=items"`
}
