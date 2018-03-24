/*
Copyright 2018 The Kubernetes Authors.

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

package kubelet

import (
	"fmt"
	"strings"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/events"
)

// Set Ready condition for the node.
func (kl *Kubelet) setNodeReadyCondition(node *v1.Node) {
	// NOTE(aaronlevy): NodeReady condition needs to be the last in the list of node conditions.
	// This is due to an issue with version skewed kubelet and master components.
	// ref: https://github.com/kubernetes/kubernetes/issues/16961
	currentTime := metav1.NewTime(kl.clock.Now())
	newNodeReadyCondition := v1.NodeCondition{
		Type:              v1.NodeReady,
		Status:            v1.ConditionTrue,
		Reason:            "KubeletReady",
		Message:           "kubelet is posting ready status",
		LastHeartbeatTime: currentTime,
	}
	rs := append(kl.runtimeState.runtimeErrors(), kl.runtimeState.networkErrors()...)
	requiredCapacities := []v1.ResourceName{v1.ResourceCPU, v1.ResourceMemory, v1.ResourcePods}
	if utilfeature.DefaultFeatureGate.Enabled(features.LocalStorageCapacityIsolation) {
		requiredCapacities = append(requiredCapacities, v1.ResourceEphemeralStorage)
	}
	missingCapacities := []string{}
	for _, resource := range requiredCapacities {
		if _, found := node.Status.Capacity[resource]; !found {
			missingCapacities = append(missingCapacities, string(resource))
		}
	}
	if len(missingCapacities) > 0 {
		rs = append(rs, fmt.Sprintf("Missing node capacity for resources: %s", strings.Join(missingCapacities, ", ")))
	}
	if len(rs) > 0 {
		newNodeReadyCondition = v1.NodeCondition{
			Type:              v1.NodeReady,
			Status:            v1.ConditionFalse,
			Reason:            "KubeletNotReady",
			Message:           strings.Join(rs, ","),
			LastHeartbeatTime: currentTime,
		}
	}
	// Append AppArmor status if it's enabled.
	if newNodeReadyCondition.Status == v1.ConditionTrue &&
		kl.appArmorValidator != nil && kl.appArmorValidator.ValidateHost() == nil {
		newNodeReadyCondition.Message = fmt.Sprintf("%s. AppArmor enabled", newNodeReadyCondition.Message)
	}

	// Record any soft requirements that were not met in the container manager.
	status := kl.containerManager.Status()
	if status.SoftRequirements != nil {
		newNodeReadyCondition.Message = fmt.Sprintf("%s. WARNING: %s", newNodeReadyCondition.Message, status.SoftRequirements.Error())
	}

	readyConditionUpdated := false
	needToRecordEvent := false
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == v1.NodeReady {
			if node.Status.Conditions[i].Status == newNodeReadyCondition.Status {
				newNodeReadyCondition.LastTransitionTime = node.Status.Conditions[i].LastTransitionTime
			} else {
				newNodeReadyCondition.LastTransitionTime = currentTime
				needToRecordEvent = true
			}
			node.Status.Conditions[i] = newNodeReadyCondition
			readyConditionUpdated = true
			break
		}
	}
	if !readyConditionUpdated {
		newNodeReadyCondition.LastTransitionTime = currentTime
		node.Status.Conditions = append(node.Status.Conditions, newNodeReadyCondition)
	}
	if needToRecordEvent {
		if newNodeReadyCondition.Status == v1.ConditionTrue {
			kl.recordNodeStatusEvent(v1.EventTypeNormal, events.NodeReady)
		} else {
			kl.recordNodeStatusEvent(v1.EventTypeNormal, events.NodeNotReady)
			glog.Infof("Node became not ready: %+v", newNodeReadyCondition)
		}
	}
}

// setNodeMemoryPressureCondition for the node.
func (kl *Kubelet) setNodeMemoryPressureCondition(node *v1.Node) {
	currentTime := metav1.NewTime(kl.clock.Now())
	var condition *v1.NodeCondition

	// Check if NodeMemoryPressure condition already exists and if it does, just pick it up for update.
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == v1.NodeMemoryPressure {
			condition = &node.Status.Conditions[i]
		}
	}

	newCondition := false
	// If the NodeMemoryPressure condition doesn't exist, create one
	if condition == nil {
		condition = &v1.NodeCondition{
			Type:   v1.NodeMemoryPressure,
			Status: v1.ConditionUnknown,
		}
		// cannot be appended to node.Status.Conditions here because it gets
		// copied to the slice. So if we append to the slice here none of the
		// updates we make below are reflected in the slice.
		newCondition = true
	}

	// Update the heartbeat time
	condition.LastHeartbeatTime = currentTime

	// Note: The conditions below take care of the case when a new NodeMemoryPressure condition is
	// created and as well as the case when the condition already exists. When a new condition
	// is created its status is set to v1.ConditionUnknown which matches either
	// condition.Status != v1.ConditionTrue or
	// condition.Status != v1.ConditionFalse in the conditions below depending on whether
	// the kubelet is under memory pressure or not.
	if kl.evictionManager.IsUnderMemoryPressure() {
		if condition.Status != v1.ConditionTrue {
			condition.Status = v1.ConditionTrue
			condition.Reason = "KubeletHasInsufficientMemory"
			condition.Message = "kubelet has insufficient memory available"
			condition.LastTransitionTime = currentTime
			kl.recordNodeStatusEvent(v1.EventTypeNormal, "NodeHasInsufficientMemory")
		}
	} else if condition.Status != v1.ConditionFalse {
		condition.Status = v1.ConditionFalse
		condition.Reason = "KubeletHasSufficientMemory"
		condition.Message = "kubelet has sufficient memory available"
		condition.LastTransitionTime = currentTime
		kl.recordNodeStatusEvent(v1.EventTypeNormal, "NodeHasSufficientMemory")
	}

	if newCondition {
		node.Status.Conditions = append(node.Status.Conditions, *condition)
	}
}

// setNodePIDPressureCondition for the node.
func (kl *Kubelet) setNodePIDPressureCondition(node *v1.Node) {
	currentTime := metav1.NewTime(kl.clock.Now())
	var condition *v1.NodeCondition

	// Check if NodePIDPressure condition already exists and if it does, just pick it up for update.
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == v1.NodePIDPressure {
			condition = &node.Status.Conditions[i]
		}
	}

	newCondition := false
	// If the NodePIDPressure condition doesn't exist, create one
	if condition == nil {
		condition = &v1.NodeCondition{
			Type:   v1.NodePIDPressure,
			Status: v1.ConditionUnknown,
		}
		// cannot be appended to node.Status.Conditions here because it gets
		// copied to the slice. So if we append to the slice here none of the
		// updates we make below are reflected in the slice.
		newCondition = true
	}

	// Update the heartbeat time
	condition.LastHeartbeatTime = currentTime

	// Note: The conditions below take care of the case when a new NodePIDPressure condition is
	// created and as well as the case when the condition already exists. When a new condition
	// is created its status is set to v1.ConditionUnknown which matches either
	// condition.Status != v1.ConditionTrue or
	// condition.Status != v1.ConditionFalse in the conditions below depending on whether
	// the kubelet is under PID pressure or not.
	if kl.evictionManager.IsUnderPIDPressure() {
		if condition.Status != v1.ConditionTrue {
			condition.Status = v1.ConditionTrue
			condition.Reason = "KubeletHasInsufficientPID"
			condition.Message = "kubelet has insufficient PID available"
			condition.LastTransitionTime = currentTime
			kl.recordNodeStatusEvent(v1.EventTypeNormal, "NodeHasInsufficientPID")
		}
	} else if condition.Status != v1.ConditionFalse {
		condition.Status = v1.ConditionFalse
		condition.Reason = "KubeletHasSufficientPID"
		condition.Message = "kubelet has sufficient PID available"
		condition.LastTransitionTime = currentTime
		kl.recordNodeStatusEvent(v1.EventTypeNormal, "NodeHasSufficientPID")
	}

	if newCondition {
		node.Status.Conditions = append(node.Status.Conditions, *condition)
	}
}

// setNodeDiskPressureCondition for the node.
func (kl *Kubelet) setNodeDiskPressureCondition(node *v1.Node) {
	currentTime := metav1.NewTime(kl.clock.Now())
	var condition *v1.NodeCondition

	// Check if NodeDiskPressure condition already exists and if it does, just pick it up for update.
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == v1.NodeDiskPressure {
			condition = &node.Status.Conditions[i]
		}
	}

	newCondition := false
	// If the NodeDiskPressure condition doesn't exist, create one
	if condition == nil {
		condition = &v1.NodeCondition{
			Type:   v1.NodeDiskPressure,
			Status: v1.ConditionUnknown,
		}
		// cannot be appended to node.Status.Conditions here because it gets
		// copied to the slice. So if we append to the slice here none of the
		// updates we make below are reflected in the slice.
		newCondition = true
	}

	// Update the heartbeat time
	condition.LastHeartbeatTime = currentTime

	// Note: The conditions below take care of the case when a new NodeDiskPressure condition is
	// created and as well as the case when the condition already exists. When a new condition
	// is created its status is set to v1.ConditionUnknown which matches either
	// condition.Status != v1.ConditionTrue or
	// condition.Status != v1.ConditionFalse in the conditions below depending on whether
	// the kubelet is under disk pressure or not.
	if kl.evictionManager.IsUnderDiskPressure() {
		if condition.Status != v1.ConditionTrue {
			condition.Status = v1.ConditionTrue
			condition.Reason = "KubeletHasDiskPressure"
			condition.Message = "kubelet has disk pressure"
			condition.LastTransitionTime = currentTime
			kl.recordNodeStatusEvent(v1.EventTypeNormal, "NodeHasDiskPressure")
		}
	} else if condition.Status != v1.ConditionFalse {
		condition.Status = v1.ConditionFalse
		condition.Reason = "KubeletHasNoDiskPressure"
		condition.Message = "kubelet has no disk pressure"
		condition.LastTransitionTime = currentTime
		kl.recordNodeStatusEvent(v1.EventTypeNormal, "NodeHasNoDiskPressure")
	}

	if newCondition {
		node.Status.Conditions = append(node.Status.Conditions, *condition)
	}
}

// Set OODCondition for the node.
func (kl *Kubelet) setNodeOODCondition(node *v1.Node) {
	currentTime := metav1.NewTime(kl.clock.Now())
	var nodeOODCondition *v1.NodeCondition

	// Check if NodeOutOfDisk condition already exists and if it does, just pick it up for update.
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == v1.NodeOutOfDisk {
			nodeOODCondition = &node.Status.Conditions[i]
		}
	}

	newOODCondition := nodeOODCondition == nil
	if newOODCondition {
		nodeOODCondition = &v1.NodeCondition{}
	}
	if nodeOODCondition.Status != v1.ConditionFalse {
		nodeOODCondition.Type = v1.NodeOutOfDisk
		nodeOODCondition.Status = v1.ConditionFalse
		nodeOODCondition.Reason = "KubeletHasSufficientDisk"
		nodeOODCondition.Message = "kubelet has sufficient disk space available"
		nodeOODCondition.LastTransitionTime = currentTime
		kl.recordNodeStatusEvent(v1.EventTypeNormal, "NodeHasSufficientDisk")
	}

	// Update the heartbeat time irrespective of all the conditions.
	nodeOODCondition.LastHeartbeatTime = currentTime

	if newOODCondition {
		node.Status.Conditions = append(node.Status.Conditions, *nodeOODCondition)
	}
}
