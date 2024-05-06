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
	"fmt"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	nodeutil "k8s.io/component-helpers/node/util"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
)

// NodeStatusUpdater defines a set of operations for updating the
// VolumesAttached field in the Node Status.
type NodeStatusUpdater interface {
	// Gets a list of node statuses that should be updated from the actual state
	// of the world and updates them.
	UpdateNodeStatuses(logger klog.Logger) error
	// Update any pending status change for the given node
	UpdateNodeStatusForNode(logger klog.Logger, nodeName types.NodeName) error
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

func (nsu *nodeStatusUpdater) UpdateNodeStatuses(logger klog.Logger) error {
	var nodeIssues int
	// TODO: investigate right behavior if nodeName is empty
	// kubernetes/kubernetes/issues/37777
	nodesToUpdate := nsu.actualStateOfWorld.GetVolumesToReportAttached(logger)
	for nodeName, attachedVolumes := range nodesToUpdate {
		err := nsu.processNodeVolumes(logger, nodeName, attachedVolumes)
		if err != nil {
			nodeIssues += 1
		}
	}
	if nodeIssues > 0 {
		return fmt.Errorf("unable to update %d nodes", nodeIssues)
	}
	return nil
}

func (nsu *nodeStatusUpdater) UpdateNodeStatusForNode(logger klog.Logger, nodeName types.NodeName) error {
	needsUpdate, attachedVolumes := nsu.actualStateOfWorld.GetVolumesToReportAttachedForNode(logger, nodeName)
	if !needsUpdate {
		return nil
	}
	return nsu.processNodeVolumes(logger, nodeName, attachedVolumes)
}

func (nsu *nodeStatusUpdater) processNodeVolumes(logger klog.Logger, nodeName types.NodeName, attachedVolumes []v1.AttachedVolume) error {
	nodeObj, err := nsu.nodeLister.Get(string(nodeName))
	if errors.IsNotFound(err) {
		// If node does not exist, its status cannot be updated.
		// Do nothing so that there is no retry until node is created.
		logger.V(2).Info(
			"Could not update node status. Failed to find node in NodeInformer cache", "node", klog.KRef("", string(nodeName)), "err", err)
		return nil
	} else if err != nil {
		// For all other errors, log error and reset flag statusUpdateNeeded
		// back to true to indicate this node status needs to be updated again.
		logger.V(2).Info("Error retrieving nodes from node lister", "err", err)
		nsu.actualStateOfWorld.SetNodeStatusUpdateNeeded(logger, nodeName)
		return err
	}

	err = nsu.updateNodeStatus(logger, nodeName, nodeObj, attachedVolumes)
	if errors.IsNotFound(err) {
		// If node does not exist, its status cannot be updated.
		// Do nothing so that there is no retry until node is created.
		logger.V(2).Info(
			"Could not update node status, node does not exist - skipping", "node", klog.KObj(nodeObj))
		return nil
	} else if err != nil {
		// If update node status fails, reset flag statusUpdateNeeded back to true
		// to indicate this node status needs to be updated again
		nsu.actualStateOfWorld.SetNodeStatusUpdateNeeded(logger, nodeName)

		logger.V(2).Info("Could not update node status; re-marking for update", "node", klog.KObj(nodeObj), "err", err)

		return err
	}
	return nil
}

func (nsu *nodeStatusUpdater) updateNodeStatus(logger klog.Logger, nodeName types.NodeName, nodeObj *v1.Node, attachedVolumes []v1.AttachedVolume) error {
	node := nodeObj.DeepCopy()
	node.Status.VolumesAttached = attachedVolumes
	_, patchBytes, err := nodeutil.PatchNodeStatus(nsu.kubeClient.CoreV1(), nodeName, nodeObj, node)
	if err != nil {
		return err
	}

	logger.V(4).Info("Updating status for node succeeded", "node", klog.KObj(node), "patchBytes", patchBytes, "attachedVolumes", attachedVolumes)
	return nil
}
