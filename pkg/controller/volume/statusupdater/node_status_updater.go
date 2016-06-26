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

// Package statusupdater implements interfaces that enable updating the status
// of API objects.
package statusupdater

import (
	"encoding/json"
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/controller/volume/cache"
	"k8s.io/kubernetes/pkg/util/strategicpatch"
)

// NodeStatusUpdater defines a set of operations for updating the
// VolumesAttached field in the Node Status.
type NodeStatusUpdater interface {
	// Gets a list of node statuses that should be updated from the actual state
	// of the world and updates them.
	UpdateNodeStatuses() error

	// IsVolumeMounted returns true if the given volume is still being
	// used by the node.
	// TODO: find a better home for this? it really isn't about node status update.
	IsVolumeMounted(volume cache.AttachedVolume) (bool, error)
}

// NewNodeStatusUpdater returns a new instance of NodeStatusUpdater.
func NewNodeStatusUpdater(
	kubeClient internalclientset.Interface,
	nodeInformer framework.SharedInformer,
	actualStateOfWorld cache.ActualStateOfWorld) NodeStatusUpdater {
	return &nodeStatusUpdater{
		actualStateOfWorld: actualStateOfWorld,
		nodeInformer:       nodeInformer,
		kubeClient:         kubeClient,
	}
}

type nodeStatusUpdater struct {
	kubeClient         internalclientset.Interface
	nodeInformer       framework.SharedInformer
	actualStateOfWorld cache.ActualStateOfWorld
}

// IsVolumeMountedByNode returns true if a volume is mounted by the node.
// This is a safety barrier. If there's a bug that leaks volumes attached
// to a node, we will not be able to detach it till the GC timeout. This is
// preferable to most other scenarios which would lead to data corruption.
// TODO: Verify GC actually cleans up the detacher keeps ignoring it because
// it's in use.
func (nsu *nodeStatusUpdater) IsVolumeMounted(attachedVolume cache.AttachedVolume) (bool, error) {
	node, err := nsu.kubeClient.Core().Nodes().Get(attachedVolume.NodeName)
	if err != nil {
		return true, err
	}
	for _, v := range node.Status.VolumesInUse {
		if v == attachedVolume.VolumeName {
			// TODO: Make sure this isn't too spammy.
			glog.Infof("Volume %v in use by node %v", v, node.Name)
			return true, nil
		}
	}
	return false, nil
}

func (nsu *nodeStatusUpdater) UpdateNodeStatuses() error {
	nodesToUpdate := nsu.actualStateOfWorld.GetVolumesToReportAttached()
	for nodeName, attachedVolumes := range nodesToUpdate {
		nodeObj, exists, err := nsu.nodeInformer.GetStore().GetByKey(nodeName)
		if nodeObj == nil || !exists || err != nil {
			return fmt.Errorf(
				"failed to find node %q in NodeInformer cache. %v",
				nodeName,
				err)
		}

		node, ok := nodeObj.(*api.Node)
		if !ok || node == nil {
			return fmt.Errorf(
				"failed to cast %q object %#v to Node",
				nodeName,
				nodeObj)
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

		_, err = nsu.kubeClient.Core().Nodes().PatchStatus(nodeName, patchBytes)
		if err != nil {
			return fmt.Errorf(
				"failed to kubeClient.Core().Nodes().Patch for node %q. %v",
				nodeName,
				err)
		}

		err = nsu.actualStateOfWorld.ResetNodeStatusUpdateNeeded(nodeName)
		if err != nil {
			return fmt.Errorf(
				"failed to ResetNodeStatusUpdateNeeded for node %q. %v",
				nodeName,
				err)
		}

		glog.V(3).Infof(
			"Updating status for node %q succeeded. patchBytes: %q",
			nodeName,
			string(patchBytes))
	}
	return nil
}
