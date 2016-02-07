/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
)

// SubresourceReference contains enough information to let you inspect or modify the referred subresource.
type SubresourceReference struct {
	// Kind of the referent; More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds"
	Kind string `json:"kind,omitempty"`
	// Name of the referent; More info: http://releases.k8s.io/HEAD/docs/user-guide/identifiers.md#names
	Name string `json:"name,omitempty"`
	// API version of the referent
	APIVersion string `json:"apiVersion,omitempty"`
	// Subresource name of the referent
	Subresource string `json:"subresource,omitempty"`
}

type CPUTargetUtilization struct {
	// fraction of the requested CPU that should be utilized/used,
	// e.g. 70 means that 70% of the requested CPU should be in use.
	TargetPercentage int32 `json:"targetPercentage"`
}

// specification of a horizontal pod autoscaler.
type HorizontalPodAutoscalerSpec struct {
	// reference to Scale subresource; horizontal pod autoscaler will learn the current resource consumption from its status,
	// and will set the desired number of pods by modifying its spec.
	ScaleRef SubresourceReference `json:"scaleRef"`
	// lower limit for the number of pods that can be set by the autoscaler, default 1.
	MinReplicas *int32 `json:"minReplicas,omitempty"`
	// upper limit for the number of pods that can be set by the autoscaler; cannot be smaller than MinReplicas.
	MaxReplicas int32 `json:"maxReplicas"`
	// target average CPU utilization (represented as a percentage of requested CPU) over all the pods;
	// if not specified it defaults to the target CPU utilization at 80% of the requested resources.
	CPUUtilization *CPUTargetUtilization `json:"cpuUtilization,omitempty"`
}

// current status of a horizontal pod autoscaler
type HorizontalPodAutoscalerStatus struct {
	// most recent generation observed by this autoscaler.
	ObservedGeneration *int64 `json:"observedGeneration,omitempty"`

	// last time the HorizontalPodAutoscaler scaled the number of pods;
	// used by the autoscaler to control how often the number of pods is changed.
	LastScaleTime *unversioned.Time `json:"lastScaleTime,omitempty"`

	// current number of replicas of pods managed by this autoscaler.
	CurrentReplicas int32 `json:"currentReplicas"`

	// desired number of replicas of pods managed by this autoscaler.
	DesiredReplicas int32 `json:"desiredReplicas"`

	// current average CPU utilization over all pods, represented as a percentage of requested CPU,
	// e.g. 70 means that an average pod is using now 70% of its requested CPU.
	CurrentCPUUtilizationPercentage *int32 `json:"currentCPUUtilizationPercentage,omitempty"`
}

// configuration of a horizontal pod autoscaler.
type HorizontalPodAutoscaler struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object metadata. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata
	v1.ObjectMeta `json:"metadata,omitempty"`

	// behaviour of autoscaler. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status.
	Spec HorizontalPodAutoscalerSpec `json:"spec,omitempty"`

	// current information about the autoscaler.
	Status HorizontalPodAutoscalerStatus `json:"status,omitempty"`
}

// list of horizontal pod autoscaler objects.
type HorizontalPodAutoscalerList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata.
	unversioned.ListMeta `json:"metadata,omitempty"`

	// list of horizontal pod autoscaler objects.
	Items []HorizontalPodAutoscaler `json:"items"`
}
