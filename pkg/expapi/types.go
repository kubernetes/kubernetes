/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

/*
This file (together with pkg/expapi/v1/types.go) contain the experimental
types in kubernetes. These API objects are experimental, meaning that the
APIs may be broken at any time by the kubernetes team.

DISCLAIMER: The implementation of the experimental API group itself is
a temporary one meant as a stopgap solution until kubernetes has proper
support for multiple API groups. The transition may require changes
beyond registration differences. In other words, experimental API group
support is experimental.
*/

package expapi

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
)

// ScaleSpec describes the attributes a Scale subresource
type ScaleSpec struct {
	// Replicas is the number of desired replicas.
	Replicas int `json:"replicas,omitempty" description:"number of replicas desired;  http://releases.k8s.io/HEAD/docs/user-guide/replication-controller.md#what-is-a-replication-controller"`
}

// ScaleStatus represents the current status of a Scale subresource.
type ScaleStatus struct {
	// Replicas is the number of actual replicas.
	Replicas int `json:"replicas" description:"most recently oberved number of replicas; see http://releases.k8s.io/HEAD/docs/user-guide/replication-controller.md#what-is-a-replication-controller"`

	// Selector is a label query over pods that should match the replicas count.
	Selector map[string]string `json:"selector,omitempty" description:"label keys and values that must match in order to be controlled by this replication controller, if empty defaulted to labels on Pod template; see http://releases.k8s.io/HEAD/docs/user-guide/labels.md#label-selectors"`
}

// Scale subresource, applicable to ReplicationControllers and (in future) Deployment.
type Scale struct {
	api.TypeMeta   `json:",inline"`
	api.ObjectMeta `json:"metadata,omitempty" description:"standard object metadata; see http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata"`

	// Spec defines the behavior of the scale.
	Spec ScaleSpec `json:"spec,omitempty" description:"specification of the desired behavior of the scale; http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status"`

	// Status represents the current status of the scale.
	Status ScaleStatus `json:"status,omitempty" description:"most recently observed status of the service; populated by the system, read-only; http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status"`
}

// Dummy definition
type ReplicationControllerDummy struct {
	api.TypeMeta `json:",inline"`
}

// SubresourceReference contains enough information to let you inspect or modify the referred subresource.
type SubresourceReference struct {
	Kind        string `json:"kind,omitempty"`
	Namespace   string `json:"namespace,omitempty"`
	Name        string `json:"name,omitempty"`
	APIVersion  string `json:"apiVersion,omitempty"`
	Subresource string `json:"subresource,omitempty"`
}

// TargetConsumption is an object for specifying target average resource consumption of a particular resource.
type TargetConsumption struct {
	Resource api.ResourceName  `json:"resource,omitempty"`
	Quantity resource.Quantity `json:"quantity,omitempty"`
}

// HorizontalPodAutoscalerSpec is the specification of a horizontal pod autoscaler.
type HorizontalPodAutoscalerSpec struct {
	// ScaleRef is a reference to Scale subresource. HorizontalPodAutoscaler will learn the current resource consumption from its status,
	// and will set the desired number of pods by modyfying its spec.
	ScaleRef *SubresourceReference `json:"scaleRef"`
	// MinCount is the lower limit for the number of pods that can be set by the autoscaler.
	MinCount int `json:"minCount"`
	// MaxCount is the upper limit for the number of pods that can be set by the autoscaler. It cannot be smaller than MinCount.
	MaxCount int `json:"maxCount"`
	// Target is the target average consumption of the given resource that the autoscaler will try to maintain by adjusting the desired number of pods.
	// Currently two types of resources are supported: "cpu" and "memory".
	Target TargetConsumption `json:"target"`
}

// HorizontalPodAutoscaler represents the configuration of a horizontal pod autoscaler.
type HorizontalPodAutoscaler struct {
	api.TypeMeta   `json:",inline"`
	api.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the behaviour of autoscaler.
	Spec HorizontalPodAutoscalerSpec `json:"spec,omitempty"`
}

// HorizontalPodAutoscaler is a collection of pod autoscalers.
type HorizontalPodAutoscalerList struct {
	api.TypeMeta `json:",inline"`
	api.ListMeta `json:"metadata,omitempty"`

	Items []HorizontalPodAutoscaler `json:"items"`
}
