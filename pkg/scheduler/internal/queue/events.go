/*
Copyright 2019 The Kubernetes Authors.

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

package queue

import (
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// Events that trigger scheduler queue to change.
const (
	// PodAdd is the event when a new pod is added to API server.
	PodAdd = "PodAdd"
	// NodeAdd is the event when a new node is added to the cluster.
	NodeAdd = "NodeAdd"
	// ScheduleAttemptFailure is the event when a schedule attempt fails.
	ScheduleAttemptFailure = "ScheduleAttemptFailure"
	// BackoffComplete is the event when a pod finishes backoff.
	BackoffComplete = "BackoffComplete"
	// UnschedulableTimeout is the event when a pod stays in unschedulable for longer than timeout.
	UnschedulableTimeout = "UnschedulableTimeout"
	// AssignedPodAdd is the event when a pod is added that causes pods with matching affinity terms
	// to be more schedulable.
	AssignedPodAdd = "AssignedPodAdd"
	// AssignedPodUpdate is the event when a pod is updated that causes pods with matching affinity
	// terms to be more schedulable.
	AssignedPodUpdate = "AssignedPodUpdate"
	// AssignedPodDelete is the event when a pod is deleted that causes pods with matching affinity
	// terms to be more schedulable.
	AssignedPodDelete = "AssignedPodDelete"
	// PvAdd is the event when a persistent volume is added in the cluster.
	PvAdd = "PvAdd"
	// PvUpdate is the event when a persistent volume is updated in the cluster.
	PvUpdate = "PvUpdate"
	// PvcAdd is the event when a persistent volume claim is added in the cluster.
	PvcAdd = "PvcAdd"
	// PvcUpdate is the event when a persistent volume claim is updated in the cluster.
	PvcUpdate = "PvcUpdate"
	// StorageClassAdd is the event when a StorageClass is added in the cluster.
	StorageClassAdd = "StorageClassAdd"
	// ServiceAdd is the event when a service is added in the cluster.
	ServiceAdd = "ServiceAdd"
	// ServiceUpdate is the event when a service is updated in the cluster.
	ServiceUpdate = "ServiceUpdate"
	// ServiceDelete is the event when a service is deleted in the cluster.
	ServiceDelete = "ServiceDelete"
	// CSINodeAdd is the event when a CSI node is added in the cluster.
	CSINodeAdd = "CSINodeAdd"
	// CSINodeUpdate is the event when a CSI node is updated in the cluster.
	CSINodeUpdate = "CSINodeUpdate"
	// NodeSpecUnschedulableChange is the event when unschedulable node spec is changed.
	NodeSpecUnschedulableChange = "NodeSpecUnschedulableChange"
	// NodeAllocatableChange is the event when node allocatable is changed.
	NodeAllocatableChange = "NodeAllocatableChange"
	// NodeLabelChange is the event when node label is changed.
	NodeLabelChange = "NodeLabelChange"
	// NodeTaintChange is the event when node taint is changed.
	NodeTaintChange = "NodeTaintChange"
	// NodeConditionChange is the event when node condition is changed.
	NodeConditionChange = "NodeConditionChange"
)

// TODO: benchmark the perf gain if making the keys as enums (int), and then
// making clusterEventReg a []framework.ClusterEvent.
var clusterEventReg = map[string]framework.ClusterEvent{
	AssignedPodAdd:              {Resource: framework.Pod, ActionType: framework.Add},
	AssignedPodUpdate:           {Resource: framework.Pod, ActionType: framework.Update},
	AssignedPodDelete:           {Resource: framework.Pod, ActionType: framework.Delete},
	NodeAdd:                     {Resource: framework.Node, ActionType: framework.Add},
	NodeSpecUnschedulableChange: {Resource: framework.Node, ActionType: framework.UpdateNodeTaint},
	NodeAllocatableChange:       {Resource: framework.Node, ActionType: framework.UpdateNodeAllocatable},
	NodeLabelChange:             {Resource: framework.Node, ActionType: framework.UpdateNodeLabel},
	NodeTaintChange:             {Resource: framework.Node, ActionType: framework.UpdateNodeTaint},
	NodeConditionChange:         {Resource: framework.Node, ActionType: framework.UpdateNodeCondition},
	PvAdd:                       {Resource: framework.PersistentVolume, ActionType: framework.Add},
	PvUpdate:                    {Resource: framework.PersistentVolume, ActionType: framework.Update},
	PvcAdd:                      {Resource: framework.PersistentVolumeClaim, ActionType: framework.Add},
	PvcUpdate:                   {Resource: framework.PersistentVolumeClaim, ActionType: framework.Update},
	StorageClassAdd:             {Resource: framework.StorageClass, ActionType: framework.Add},
	CSINodeAdd:                  {Resource: framework.CSINode, ActionType: framework.Add},
	CSINodeUpdate:               {Resource: framework.CSINode, ActionType: framework.Update},
	ServiceAdd:                  {Resource: framework.Service, ActionType: framework.Add},
	ServiceUpdate:               {Resource: framework.Service, ActionType: framework.Update},
	ServiceDelete:               {Resource: framework.Service, ActionType: framework.Delete},
	UnschedulableTimeout:        framework.WildCardEvent,
}
