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
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	toolscache "k8s.io/client-go/tools/cache"
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
	logger klog.Logger,
	kubeClient clientset.Interface,
	nodeInformer coreinformers.NodeInformer,
	actualStateOfWorld cache.ActualStateOfWorld) (NodeStatusUpdater, error) {
	u := &nodeStatusUpdater{
		actualStateOfWorld: actualStateOfWorld,
		nodeLister:         nodeInformer.Lister(),
		kubeClient:         kubeClient,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[types.NodeName](),
			workqueue.TypedRateLimitingQueueConfig[types.NodeName]{Name: "node-attachedVolumes"},
		),
	}
	actualStateOfWorld.SetNodeUpdateHook(u.QueueUpdate)
	_, err := nodeInformer.Informer().AddEventHandlerWithOptions(toolscache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			u.queueNode(logger, obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			// in case others modified volumesAttached or our last sync see a stale state
			u.queueNode(logger, newObj)
		},
		DeleteFunc: func(obj interface{}) {
			// all volumes should be detached
			tombstone, ok := obj.(toolscache.DeletedFinalStateUnknown)
			if ok {
				obj = tombstone.Obj
			}
			u.queueNode(logger, obj)
		},
	}, toolscache.HandlerOptions{Logger: &logger})
	if err != nil {
		return nil, err
	}
	return u, nil
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

// kubernetes/kubernetes/issues/37586
// When a node add/update causes to wipe out the attached volumes field.
// This function ensures that we sync with the actual status.
func (nsu *nodeStatusUpdater) queueNode(logger klog.Logger, obj any) {
	node, ok := obj.(*v1.Node)
	if !ok {
		utilruntime.HandleErrorWithLogger(logger, nil, "object is not a node", "obj", obj)
		return
	}
	nsu.QueueUpdate(types.NodeName(node.Name))
}

func (nsu *nodeStatusUpdater) QueueUpdate(nodeName types.NodeName) {
	nsu.queue.Add(nodeName)
}

func (nsu *nodeStatusUpdater) worker(ctx context.Context) {
	for nsu.syncNode(ctx) {
	}
	klog.FromContext(ctx).Info("NodeStatusUpdater worker shutting down")
}

func (nsu *nodeStatusUpdater) syncNode(ctx context.Context) bool {
	nodeName, quit := nsu.queue.Get()
	if quit {
		return false
	}
	defer nsu.queue.Done(nodeName)

	err := nsu.processNodeVolumes(ctx, nodeName)
	if err != nil {
		nsu.queue.AddRateLimited(nodeName)
	} else {
		nsu.queue.Forget(nodeName)
	}
	return true
}

type nodeStatusPatch struct {
	VolumesAttached []v1.AttachedVolume `json:"volumesAttached"` // intentional not omitempty
}

type nodePatch struct {
	Status nodeStatusPatch `json:"status"`
}

func removingVolumes(volumes []cache.VolumeToReport) []v1.UniqueVolumeName {
	removed := make([]v1.UniqueVolumeName, 0)
	for _, vol := range volumes {
		if vol.Report == cache.NodeStatusReportForceRemoving {
			removed = append(removed, vol.Volume.Name)
		}
	}
	return removed
}

func (nsu *nodeStatusUpdater) processNodeVolumes(ctx context.Context, nodeName types.NodeName) error {
	logger := klog.FromContext(ctx)
	volumesToReport := nsu.actualStateOfWorld.GetVolumesToReportAttachedForNode(logger, nodeName)
	if volumesToReport == nil {
		return nil // We know nothing about this node
	}

	nodeObj, err := nsu.nodeLister.Get(string(nodeName))
	if errors.IsNotFound(err) {
		// If node does not exist, all detaching volumes are already removed.
		nsu.actualStateOfWorld.ConfirmNodeStatusRemoved(logger, nodeName, removingVolumes(volumesToReport))
		// Should queue again when the node is created to populate attached volumes.
		logger.V(2).Info(
			"Could not update node status. Failed to find node in NodeInformer cache", "node", klog.KRef("", string(nodeName)), "err", err)
		return nil
	} else if err != nil {
		logger.V(2).Info("Error retrieving nodes from node lister", "err", err)
		return err
	}

	var attachedVolumes []v1.AttachedVolume
	var removedVolumes []v1.UniqueVolumeName
	for _, vol := range volumesToReport {
		switch vol.Report {
		case cache.NodeStatusReportAdding:
			attachedVolumes = append(attachedVolumes, vol.Volume)
		case cache.NodeStatusReportForceRemoving:
			removedVolumes = append(removedVolumes, vol.Volume.Name)
		}
	}
	if len(removedVolumes) == 0 && slices.Equal(attachedVolumes, nodeObj.Status.VolumesAttached) {
		return nil // No update required
		// we always do a patch if removedVolumes is non-empty
		// in case the nodeObj from the lister is stale
	}

	nodePatch := nodePatch{
		Status: nodeStatusPatch{
			VolumesAttached: attachedVolumes,
		},
	}
	patchData, err := json.Marshal(&nodePatch)
	if err != nil {
		logger.Error(err, "Failed to marshal node patch", "node", klog.KObj(nodeObj))
		return err
	}
	_, err = nsu.kubeClient.CoreV1().Nodes().PatchStatus(ctx, nodeObj.Name, patchData)
	if errors.IsNotFound(err) {
		// If node does not exist, all detaching volumes are already removed.
		nsu.actualStateOfWorld.ConfirmNodeStatusRemoved(logger, nodeName, removingVolumes(volumesToReport))
		// Should queue again when the node is created to populate attached volumes.
		logger.V(2).Info(
			"Could not update node status, node does not exist - skipping", "node", klog.KObj(nodeObj))
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
