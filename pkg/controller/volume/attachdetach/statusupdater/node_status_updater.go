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
	"slices"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/util/workqueue"
	nodeutil "k8s.io/component-helpers/node/util"
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

func (nsu *nodeStatusUpdater) processNodeVolumes(logger klog.Logger, nodeName types.NodeName) error {
	nodeObj, err := nsu.nodeLister.Get(string(nodeName))
	if errors.IsNotFound(err) {
		// If node does not exist, its status cannot be updated.
		// Do nothing so that there is no retry until node is created.
		logger.V(2).Info(
			"Could not update node status. Failed to find node in NodeInformer cache", "node", klog.KRef("", string(nodeName)), "err", err)
		return nil
	} else if err != nil {
		logger.V(2).Info("Error retrieving nodes from node lister", "err", err)
		return err
	}

	attachedVolumes, generation := nsu.actualStateOfWorld.GetVolumesToReportAttachedForNode(logger, nodeName)
	if attachedVolumes == nil { // we know nothing about this node
		return nil
	}
	err = nsu.updateNodeStatus(nodeName, nodeObj, attachedVolumes)
	if errors.IsNotFound(err) {
		// If node does not exist, its status cannot be updated.
		// Do nothing so that there is no retry until node is created.
		logger.V(2).Info(
			"Could not update node status, node does not exist - skipping", "node", klog.KObj(nodeObj))
		return nil
	} else if err != nil {
		logger.V(2).Info("Could not update node status; re-marking for update", "node", klog.KObj(nodeObj), "err", err)
		return err
	}
	logger.V(4).Info("Updating status for node succeeded",
		"node", klog.KRef("", string(nodeName)), "generation", generation, "attachedVolumes", attachedVolumes)
	nsu.actualStateOfWorld.SetNodeStatusUpdateFinished(logger, nodeName, generation)
	return nil
}

func (nsu *nodeStatusUpdater) updateNodeStatus(nodeName types.NodeName, nodeObj *v1.Node, attachedVolumes []v1.AttachedVolume) error {
	if slices.Equal(attachedVolumes, nodeObj.Status.VolumesAttached) {
		return nil
	}

	node := nodeObj.DeepCopy()
	node.Status.VolumesAttached = attachedVolumes
	_, _, err := nodeutil.PatchNodeStatus(nsu.kubeClient.CoreV1(), nodeName, nodeObj, node)
	if err != nil {
		return err
	}
	return nil
}
