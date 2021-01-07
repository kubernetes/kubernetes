/*
Copyright 2021 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"time"

	"k8s.io/api/core/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog/v2"
)

// Returns 0 for resyncPeriod in case resyncing is not needed.
func NoResyncPeriodFunc() time.Duration {
	return 0
}

// CreateAddNodeHandler creates an add node handler.
func CreateAddNodeHandler(f func(node *v1.Node) error) func(obj interface{}) {
	return func(originalObj interface{}) {
		node := originalObj.(*v1.Node).DeepCopy()
		if err := f(node); err != nil {
			utilruntime.HandleError(fmt.Errorf("Error while processing Node Add: %v", err))
		}
	}
}

// CreateDeleteNodeHandler creates a delete node handler. (Common to lifecycle and ipam)
func CreateDeleteNodeHandler(f func(node *v1.Node) error) func(obj interface{}) {
	return func(originalObj interface{}) {
		originalNode, isNode := originalObj.(*v1.Node)
		// We can get DeletedFinalStateUnknown instead of *v1.Node here and
		// we need to handle that correctly. #34692
		if !isNode {
			deletedState, ok := originalObj.(cache.DeletedFinalStateUnknown)
			if !ok {
				klog.Errorf("Received unexpected object: %v", originalObj)
				return
			}
			originalNode, ok = deletedState.Obj.(*v1.Node)
			if !ok {
				klog.Errorf("DeletedFinalStateUnknown contained non-Node object: %v", deletedState.Obj)
				return
			}
		}
		node := originalNode.DeepCopy()
		if err := f(node); err != nil {
			utilruntime.HandleError(fmt.Errorf("Error while processing Node Add/Delete: %v", err))
		}
	}
}

// CreateUpdateNodeHandler creates a node update handler. (Common to lifecycle and ipam)
func CreateUpdateNodeHandler(f func(oldNode, newNode *v1.Node) error) func(oldObj, newObj interface{}) {
	return func(origOldObj, origNewObj interface{}) {
		node := origNewObj.(*v1.Node).DeepCopy()
		prevNode := origOldObj.(*v1.Node).DeepCopy()

		if err := f(prevNode, node); err != nil {
			utilruntime.HandleError(fmt.Errorf("Error while processing Node Add/Delete: %v", err))
		}
	}
}

// GetNodeCondition extracts the provided condition from the given status and returns that.
// Returns nil and -1 if the condition is not present, and the index of the located condition.
func GetNodeCondition(status *v1.NodeStatus, conditionType v1.NodeConditionType) (int, *v1.NodeCondition) {
	if status == nil {
		return -1, nil
	}
	for i := range status.Conditions {
		if status.Conditions[i].Type == conditionType {
			return i, &status.Conditions[i]
		}
	}
	return -1, nil
}

// RecordNodeStatusChange records a event related to a node status change. (Common to lifecycle and ipam)
func RecordNodeStatusChange(recorder record.EventRecorder, node *v1.Node, newStatus string) {
	ref := &v1.ObjectReference{
		APIVersion: "v1",
		Kind:       "Node",
		Name:       node.Name,
		UID:        node.UID,
		Namespace:  "",
	}
	klog.V(2).Infof("Recording status change %s event message for node %s", newStatus, node.Name)
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	recorder.Eventf(ref, v1.EventTypeNormal, newStatus, "Node %s status is now: %s", node.Name, newStatus)
}
