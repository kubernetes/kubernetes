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

// Package statusupdater implements interfaces that enable updating the status
// of API objects.
package statusupdater

import (
	"encoding/json"
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api/v1"
	kcache "k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/util/strategicpatch"
)

// NodeStatusUpdater defines a set of operations for updating the
// VolumesAttached field in the Node Status.
type NodeStatusUpdater interface {
	// Gets a list of node statuses that should be updated from the actual state
	// of the world and updates them.
	UpdateNodeStatuses() error
}

// NewNodeStatusUpdater returns a new instance of NodeStatusUpdater.
func NewNodeStatusUpdater(
	kubeClient clientset.Interface,
	nodeInformer kcache.SharedInformer,
	actualStateOfWorld cache.ActualStateOfWorld) NodeStatusUpdater {
	return &nodeStatusUpdater{
		actualStateOfWorld: actualStateOfWorld,
		nodeInformer:       nodeInformer,
		kubeClient:         kubeClient,
	}
}

type nodeStatusUpdater struct {
	kubeClient         clientset.Interface
	nodeInformer       kcache.SharedInformer
	actualStateOfWorld cache.ActualStateOfWorld
}

func (nsu *nodeStatusUpdater) UpdateNodeStatuses() error {
	// TODO: investigate right behavior if nodeName is empty
	// kubernetes/kubernetes/issues/37777
	nodesToUpdate := nsu.actualStateOfWorld.GetVolumesToReportAttached()
	for nodeName, attachedVolumes := range nodesToUpdate {
		nodeObj, exists, err := nsu.nodeInformer.GetStore().GetByKey(string(nodeName))
		if nodeObj == nil || !exists || err != nil {
			// If node does not exist, its status cannot be updated, log error and
			// reset flag statusUpdateNeeded back to true to indicate this node status
			// needs to be updated again
			glog.V(2).Infof(
				"Could not update node status. Failed to find node %q in NodeInformer cache. %v",
				nodeName,
				err)
			nsu.actualStateOfWorld.SetNodeStatusUpdateNeeded(nodeName)
			continue
		}

		clonedNode, err := conversion.NewCloner().DeepCopy(nodeObj)
		if err != nil {
			return fmt.Errorf("error cloning node %q: %v",
				nodeName,
				err)
		}

		node, ok := clonedNode.(*v1.Node)
		if !ok || node == nil {
			return fmt.Errorf(
				"failed to cast %q object %#v to Node",
				nodeName,
				clonedNode)
		}

		oldData, err := json.Marshal(node)
		if err != nil {
			return fmt.Errorf(
				"failed to Marshal oldData for node %q. %v",
				nodeName,
				err)
		}

		node.Status.VolumesAttached = attachedVolumes

		newData, err := json.Marshal(node)
		if err != nil {
			return fmt.Errorf(
				"failed to Marshal newData for node %q. %v",
				nodeName,
				err)
		}

		patchBytes, err :=
			strategicpatch.CreateStrategicMergePatch(oldData, newData, node)
		if err != nil {
			return fmt.Errorf(
				"failed to CreateStrategicMergePatch for node %q. %v",
				nodeName,
				err)
		}

		_, err = nsu.kubeClient.Core().Nodes().PatchStatus(string(nodeName), patchBytes)
		if err != nil {
			// If update node status fails, reset flag statusUpdateNeeded back to true
			// to indicate this node status needs to be updated again
			nsu.actualStateOfWorld.SetNodeStatusUpdateNeeded(nodeName)
			return fmt.Errorf(
				"failed to kubeClient.Core().Nodes().Patch for node %q. %v",
				nodeName,
				err)
		}
		glog.V(2).Infof(
			"Updating status for node %q succeeded. patchBytes: %q VolumesAttached: %v",
			nodeName,
			string(patchBytes),
			node.Status.VolumesAttached)

	}
	return nil
}
