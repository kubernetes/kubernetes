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
	"k8s.io/klog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
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
	nodeLister corelisters.NodeLister,
	actualStateOfWorld cache.ActualStateOfWorld) NodeStatusUpdater {
	return &nodeStatusUpdater{
		actualStateOfWorld: actualStateOfWorld,
		nodeLister:         nodeLister,
		kubeClient:         kubeClient,
	}
}

type nodeStatusUpdater struct {
	kubeClient         clientset.Interface
	nodeLister         corelisters.NodeLister
	actualStateOfWorld cache.ActualStateOfWorld
}

func (nsu *nodeStatusUpdater) UpdateNodeStatuses() error {
	// TODO: investigate right behavior if nodeName is empty
	// kubernetes/kubernetes/issues/37777
	nodesToUpdate := nsu.actualStateOfWorld.GetVolumesToReportAttached()
	for nodeName, attachedVolumes := range nodesToUpdate {
		nodeObj, err := nsu.nodeLister.Get(string(nodeName))
		if errors.IsNotFound(err) {
			// If node does not exist, its status cannot be updated.
			// Do nothing so that there is no retry until node is created.
			klog.V(2).Infof(
				"Could not update node status. Failed to find node %q in NodeInformer cache. Error: '%v'",
				nodeName,
				err)
			continue
		} else if err != nil {
			// For all other errors, log error and reset flag statusUpdateNeeded
			// back to true to indicate this node status needs to be updated again.
			klog.V(2).Infof("Error retrieving nodes from node lister. Error: %v", err)
			nsu.actualStateOfWorld.SetNodeStatusUpdateNeeded(nodeName)
			continue
		}

		if err := nsu.updateNodeStatus(nodeName, nodeObj, attachedVolumes); err != nil {
			// If update node status fails, reset flag statusUpdateNeeded back to true
			// to indicate this node status needs to be updated again
			nsu.actualStateOfWorld.SetNodeStatusUpdateNeeded(nodeName)

			klog.V(2).Infof(
				"Could not update node status for %q; re-marking for update. %v",
				nodeName,
				err)

			// We currently always return immediately on error
			return err
		}
	}
	return nil
}

func (nsu *nodeStatusUpdater) updateNodeStatus(nodeName types.NodeName, nodeObj *v1.Node, attachedVolumes []v1.AttachedVolume) error {
	node := nodeObj.DeepCopy()
	node.Status.VolumesAttached = attachedVolumes
	_, patchBytes, err := nodeutil.PatchNodeStatus(nsu.kubeClient.CoreV1(), nodeName, nodeObj, node)
	if err != nil {
		return err
	}

	klog.V(4).Infof("Updating status %q for node %q succeeded. VolumesAttached: %v", patchBytes, nodeName, attachedVolumes)
	return nil
}
