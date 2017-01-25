/*
Copyright 2017 The Kubernetes Authors.

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

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +genclient=true

// Status and (in future) Configuration of ClusterAutoscaler.
type ClusterAutoscaler struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object metadata.
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Specification of ClusterAutoscaler. Currently empty. Protobuf index placeholder.
	// +optional
	Spec ClusterAutoscalerSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// Current information about ClusterAutoscaler.
	// +optional
	Status ClusterAutoscalerStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// Specification of ClusterAutoscaler. Empty for for now.
type ClusterAutoscalerSpec struct {
}

// Type of ClusterAutoscalerCondition
type ClusterAutoscalerConditionType string

const (
	// Condition that explains what is the current health of ClusterAutoscaler or its node groups.
	ClusterAutoscalerHealth ClusterAutoscalerConditionType = "Health"
	// Condition that explains what is the current status of a node group with regard to
	// scale down activities.
	ClusterAutoscalerScaleDown ClusterAutoscalerConditionType = "ScaleDown"
	// Condition that explains what is the current status of a node group with regard to
	// scale down activities.
	ClusterAutoscalerScaleUp ClusterAutoscalerConditionType = "ScaleUp"
)

// Status of ClusterAutoscalerCondition.
type ClusterAutoscalerConditionStatus string

const (
	// Statuses for Health condition type.
	ClusterAutoscalerHealthy   ClusterAutoscalerConditionStatus = "Healthy"
	ClusterAutoscalerUnhealthy ClusterAutoscalerConditionStatus = "Unhealthy"

	// Statuses for ScaleDown condition type.
	ClusterAutoscalerCandidatesPresent ClusterAutoscalerConditionStatus = "CandidatesPresent"
	ClusterAutoscalerNoCandidates      ClusterAutoscalerConditionStatus = "NoCandidates"

	// Statuses for ScaleUp condition type.
	ClusterAutoscalerCandidatesNeeded ClusterAutoscalerConditionStatus = "Needed"
	ClusterAutoscalerNotNeeded        ClusterAutoscalerConditionStatus = "NotNeeded"
	ClusterAutoscalerInProgress       ClusterAutoscalerConditionStatus = "InProgress"
	ClusterAutoscalerNoActivity       ClusterAutoscalerConditionStatus = "NoActivity"
)

// ClusterAutoscalerCondition describes some aspect of ClusterAutoscaler work.
type ClusterAutoscalerCondition struct {
	// Defines the aspect that the condition describes. For example, it can be Health or ScaleUp/Down activity.
	Type ClusterAutoscalerConditionType `json:"type,omitempty" protobuf:"bytes,1,opt,name=type"`
	// Status of the condition. Tells how given aspect
	Status ClusterAutoscalerConditionStatus `json:"status,omitempty" protobuf:"bytes,2,opt,name=status"`
	// Free text extra information about the condition. It may contain some
	// extra debugging data, like why the cluster is unhealthy.
	Message string `json:"message,omitempty" protobuf:"bytes,3,opt,name=message"`
	// Since when the condition was in the given state.
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty" protobuf:"bytes,4,opt,name=lastTransitionTime"`
}

// Status of ClusterAutoscaler
type ClusterAutoscalerStatus struct {
	// Status information of individual node groups on which CA works.
	NodeGroupStatuses []NodeGroupsStatus
	// Conditions that apply to the whole autoscaler.
	ClusterwideConditions []ClusterAutoscalerCondition
}

// Status of a group of nodes controlled by ClusterAutoscaler.
type NodeGroupStatus struct {
	// Name of the node group. On GCE it will be equal to MIG url, on AWS it will be ASG name, etc.
	Name string `json:"name,omitempty" protobuf:"bytes,1,opt,name=name"`
	// List of conditions that describe the state of the node group.
	Conditions []ClusterAutoscalerCondition
}
