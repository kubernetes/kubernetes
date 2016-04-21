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

package nodelifecycle

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

type NodeDeployment struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty"`

	// Specification of the desired behavior of the NodeDeployment.
	Spec NodeDeploymentSpec `json:"spec,omitempty"`

	// Most recently observed status of the NodeDeployment.
	Status NodeDeploymentStatus `json:"status,omitempty"`
}

type NodeDeploymentSpec struct {
	// Number of desired pods. This is a pointer to distinguish between explicit
	// zero and not specified. Defaults to 1.
	Replicas int `json:"replicas,omitempty"`

	// Label selector for pods. Existing ReplicaSets whose pods are
	// selected by this will be the ones affected by this deployment.
	Selector *unversioned.LabelSelector `json:"selector,omitempty"`

	// Template describes the pods that will be created.
	Template api.PodTemplateSpec `json:"template"`

	// The deployment strategy to use to replace existing pods with new ones.
	Strategy DeploymentStrategy `json:"strategy,omitempty"`

	// Minimum number of seconds for which a newly created pod should be ready
	// without any of its container crashing, for it to be considered available.
	// Defaults to 0 (pod will be considered available as soon as it is ready)
	MinReadySeconds int `json:"minReadySeconds,omitempty"`

	// The number of old ReplicaSets to retain to allow rollback.
	// This is a pointer to distinguish between explicit zero and not specified.
	RevisionHistoryLimit *int `json:"revisionHistoryLimit,omitempty"`

	// Indicates that the deployment is paused and will not be processed by the
	// deployment controller.
	Paused bool `json:"paused,omitempty"`
	// The config this deployment is rolling back to. Will be cleared after rollback is done.
	RollbackTo *RollbackConfig `json:"rollbackTo,omitempty"`
}

type NodeDeploymentStatus struct {
	// The generation observed by the node deployment controller.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Total number of non-terminated pods targeted by this deployment (their labels match the
	// selector).
	Replicas int `json:"replicas,omitempty"`

	// Total number of non-terminated pods targeted by this deployment that have the desired
	// template spec.
	UpdatedReplicas int `json:"updatedReplicas,omitempty"`

	// Total number of available pods (ready for at least minReadySeconds) targeted by this
	// deployment.
	AvailableReplicas int `json:"availableReplicas,omitempty"`

	// Total number of unavailable pods targeted by this deployment.
	UnavailableReplicas int `json:"unavailableReplicas,omitempty"`
}
