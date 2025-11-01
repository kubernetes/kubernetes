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
	"context"
	"encoding/json"
	"slices"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
)

// NodeStatusUpdater defines a set of operations for updating the
// VolumesAttached field in the Node Status.
type NodeStatusUpdater interface {
	// Run starts asynchronous updates queued by [QueueUpdate].
	Run(ctx context.Context, workers int)
	// Queue updating any pending status change for the given node
	QueueUpdate(nodeName types.NodeName)
}

// NewNodeStatusUpdater returns a new instance of NodeStatusUpdater.
func NewNodeStatusUpdater(
	kubeClient clientset.Interface,
	nodeLister corelisters.NodeLister,
	actualStateOfWorld cache.ActualStateOfWorld) NodeStatusUpdater {
	u := &nodeStatusUpdater{
		actualStateOfWorld: actualStateOfWorld,
		nodeLister:         nodeLister,
		kubeClient:         kubeClient,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[types.NodeName](),
			workqueue.TypedRateLimitingQueueConfig[types.NodeName]{Name: "node-attachedVolumes"},
		),
	}
	actualStateOfWorld.SetNodeUpdateHook(u.QueueUpdate)
	return u
}

type nodeStatusUpdater struct {
	kubeClient         clientset.Interface
	nodeLister         corelisters.NodeLister
	actualStateOfWorld cache.ActualStateOfWorld
	queue              workqueue.TypedRateLimitingInterface[types.NodeName]
}

func (nsu *nodeStatusUpdater) Run(ctx context.Context, worker int) {
	defer nsu.queue.ShutDown()
	for range worker {
		go wait.UntilWithContext(ctx, nsu.worker, time.Second)
	}
	<-ctx.Done()
}

func (nsu *nodeStatusUpdater) QueueUpdate(nodeName types.NodeName) {
	nsu.queue.Add(nodeName)
}

func (nsu *nodeStatusUpdater) worker(ctx context.Context) {
	logger := klog.FromContext(ctx)
	for nsu.syncNode(logger) {
	}
	logger.Info("NodeStatusUpdater worker shutting down")
}

func (nsu *nodeStatusUpdater) syncNode(logger klog.Logger) bool {
	nodeName, quit := nsu.queue.Get()
	if quit {
		return false
	}
	defer nsu.queue.Done(nodeName)

	err := nsu.processNodeVolumes(logger, nodeName)
	if err != nil {
		nsu.queue.AddRateLimited(nodeName)
	} else {
		nsu.queue.Forget(nodeName)
	}
	return true
}

type nodeStatusPatch struct {
	VolumesAttached []v1.AttachedVolume `json:"volumesAttached"`
}

type objectMetaPatch struct {
	ResourceVersion string `json:"resourceVersion,omitempty"`
}

type nodePatch struct {
	ObjectMeta objectMetaPatch `json:"metadata,omitzero"`
	Status     nodeStatusPatch `json:"status"`
}

func (nsu *nodeStatusUpdater) processNodeVolumes(logger klog.Logger, nodeName types.NodeName) error {
	attachedVolumes, removingVolumes := nsu.actualStateOfWorld.GetVolumesToReportAttachedForNode(logger, nodeName)
	if attachedVolumes == nil && removingVolumes == nil {
		return nil // We know nothing about this node
	}

	nodeObj, err := nsu.nodeLister.Get(string(nodeName))
	if errors.IsNotFound(err) {
		// If node does not exist, all detaching volumes are already removed.
		nsu.actualStateOfWorld.ConfirmNodeStatusRemoved(logger, nodeName, removingVolumes)
		// Should queue again when the node is created to populate attached volumes.
		logger.V(2).Info(
			"Could not update node status. Failed to find node in NodeInformer cache", "node", klog.KRef("", string(nodeName)), "err", err)
		return nil
	} else if err != nil {
		logger.V(2).Info("Error retrieving nodes from node lister", "err", err)
		return err
	}

	var removedVolumes []v1.UniqueVolumeName
	var resourceVersion string // only set if we remove some volumes, to ensure kubelet doesn't add inUse volumes back after we check.
	if len(removingVolumes) != 0 {
		// Before remove, we check VolumesInUse to avoid removing in-use volumes
		inUseVolumes := sets.New(nodeObj.Status.VolumesInUse...)
		var oldVolumes map[v1.UniqueVolumeName]*v1.AttachedVolume
		removedVolumes = make([]v1.UniqueVolumeName, 0, len(removingVolumes))
		for _, vol := range removingVolumes {
			if !inUseVolumes.Has(vol) {
				resourceVersion = nodeObj.ResourceVersion
				removedVolumes = append(removedVolumes, vol)
				continue
			}
			logger.V(4).Info("Volume is still in use on node; will not remove from status",
				"node", klog.KObj(nodeObj), "volumeName", vol)
			if oldVolumes == nil {
				oldVolumes = make(map[v1.UniqueVolumeName]*v1.AttachedVolume, len(nodeObj.Status.VolumesAttached))
				for i := range nodeObj.Status.VolumesAttached {
					v := &nodeObj.Status.VolumesAttached[i]
					oldVolumes[v.Name] = v
				}
			}
			attachedVolumes = append(attachedVolumes, *oldVolumes[vol])
		}
	}
	if resourceVersion == "" && slices.Equal(attachedVolumes, nodeObj.Status.VolumesAttached) {
		return nil // No update required
	}

	nodePatch := nodePatch{
		ObjectMeta: objectMetaPatch{
			ResourceVersion: resourceVersion,
		},
		Status: nodeStatusPatch{
			VolumesAttached: attachedVolumes,
		},
	}
	patchData, err := json.Marshal(&nodePatch)
	if err != nil {
		logger.Error(err, "Failed to marshal node patch", "node", klog.KObj(nodeObj))
		return err
	}
	_, err = nsu.kubeClient.CoreV1().Nodes().PatchStatus(context.TODO(), nodeObj.Name, patchData)
	if errors.IsNotFound(err) {
		// If node does not exist, all detaching volumes are already removed.
		nsu.actualStateOfWorld.ConfirmNodeStatusRemoved(logger, nodeName, removingVolumes)
		// Should queue again when the node is created to populate attached volumes.
		logger.V(2).Info(
			"Could not update node status, node does not exist - skipping", "node", klog.KObj(nodeObj))
		return nil
	} else if errors.IsConflict(err) {
		logger.V(2).Info("Conflict updating node status; waiting for next event", "node", klog.KObj(nodeObj), "err", err)
		return nil
	} else if err != nil {
		logger.V(2).Info("Could not update node status; re-marking for update", "node", klog.KObj(nodeObj), "err", err)
		return err
	}
	if len(removedVolumes) > 0 {
		nsu.actualStateOfWorld.ConfirmNodeStatusRemoved(logger, nodeName, removedVolumes)
	}
	return nil
}
