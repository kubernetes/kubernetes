/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package nodecidr

import (
	"net"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	// podCIDRUpdateRetry controls the number of retries of writing Node.Spec.PodCIDR update.
	podCIDRUpdateRetry = 5
	// controls how many NodeSpec updates NC can process in any moment.
	cidrUpdateWorkers   = 10
	cidrUpdateQueueSize = 5000
)

type nodeAndCIDR struct {
	nodeName string
	cidr     *net.IPNet
}

type NodeCIDRController struct {
	kubeClient    clientset.Interface
	clusterCIDR   *net.IPNet
	serviceCIDR   *net.IPNet
	recorder      record.EventRecorder
	cidrAllocator CIDRAllocator
	// Node framework and store
	nodeController *framework.Controller
	nodeStore      cache.StoreToNodeLister

	nodeCIDRUpdateChannel chan nodeAndCIDR
}

func NewNodeCIDRController(kubeClient clientset.Interface, clusterCIDR *net.IPNet, serviceCIDR *net.IPNet, nodeCIDRMaskSize int) *NodeCIDRController {
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(api.EventSource{Component: "controllermanager"})
	eventBroadcaster.StartLogging(glog.Infof)
	if kubeClient != nil {
		glog.V(0).Infof("Sending events to api server.")
		eventBroadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{Interface: kubeClient.Core().Events("")})
	} else {
		glog.V(0).Infof("No api server defined - no events will be sent to API server.")
	}

	if clusterCIDR == nil {
		glog.Fatal("NodeController: Must specify clusterCIDR if allocateNodeCIDRs == true.")
	}
	mask := clusterCIDR.Mask
	if maskSize, _ := mask.Size(); maskSize > nodeCIDRMaskSize {
		glog.Fatal("NodeController: Invalid clusterCIDR, mask size of clusterCIDR must be less than nodeCIDRMaskSize.")
	}

	ncc := &NodeCIDRController{
		kubeClient:    kubeClient,
		clusterCIDR:   clusterCIDR,
		serviceCIDR:   serviceCIDR,
		recorder:      recorder,
		cidrAllocator: NewCIDRRangeAllocator(clusterCIDR, nodeCIDRMaskSize),
	}

	nodeEventHandlerFuncs := framework.ResourceEventHandlerFuncs{
		AddFunc:    ncc.allocateOrOccupyCIDR,
		DeleteFunc: ncc.recycleCIDR,
	}

	ncc.nodeStore.Store, ncc.nodeController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return ncc.kubeClient.Core().Nodes().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return ncc.kubeClient.Core().Nodes().Watch(options)
			},
		},
		&api.Node{},
		controller.NoResyncPeriodFunc(),
		nodeEventHandlerFuncs,
	)

	return ncc
}

func (ncc *NodeCIDRController) Run(period time.Duration) {
	if ncc.serviceCIDR != nil {
		ncc.filterOutServiceRange()
	} else {
		glog.Info("No Service CIDR provided. Skipping filtering out service addresses.")
	}

	go ncc.nodeController.Run(wait.NeverStop)

	for i := 0; i < cidrUpdateWorkers; i++ {
		go func(stopChan <-chan struct{}) {
			for {
				select {
				case workItem, ok := <-ncc.nodeCIDRUpdateChannel:
					if !ok {
						glog.Warning("NodeCIDRUpdateChannel read returned false.")
						return
					}
					ncc.updateCIDRAllocation(workItem)
				case <-stopChan:
					glog.V(0).Info("StopChannel is closed.")
					return
				}
			}
		}(wait.NeverStop)
	}
}

func (ncc *NodeCIDRController) filterOutServiceRange() {
	if !ncc.clusterCIDR.Contains(ncc.serviceCIDR.IP.Mask(ncc.clusterCIDR.Mask)) && !ncc.serviceCIDR.Contains(ncc.clusterCIDR.IP.Mask(ncc.serviceCIDR.Mask)) {
		return
	}

	if err := ncc.cidrAllocator.Occupy(ncc.serviceCIDR); err != nil {
		glog.Errorf("Error filtering out service cidr: %v", err)
	}
}

// allocateOrOccupyCIDR looks at each new observed node, assigns it a valid CIDR
// if it doesn't currently have one or mark the CIDR as used if the node already have one.
func (ncc *NodeCIDRController) allocateOrOccupyCIDR(obj interface{}) {
	node := obj.(*api.Node)
	if node.Spec.PodCIDR != "" {
		_, podCIDR, err := net.ParseCIDR(node.Spec.PodCIDR)
		if err != nil {
			glog.Errorf("failed to parse node %s, CIDR %s", node.Name, node.Spec.PodCIDR)
			return
		}
		if err := ncc.cidrAllocator.Occupy(podCIDR); err != nil {
			glog.Errorf("failed to mark cidr as occupied :%v", err)
			return
		}
		return
	}
	podCIDR, err := ncc.cidrAllocator.AllocateNext()
	if err != nil {
		ncc.recordNodeStatusChange(node, "CIDRNotAvailable")
		return
	}

	glog.V(4).Infof("Putting node %s with CIDR %s into the work queue", node.Name, podCIDR)
	ncc.nodeCIDRUpdateChannel <- nodeAndCIDR{
		nodeName: node.Name,
		cidr:     podCIDR,
	}
}

// recycleCIDR recycles the CIDR of a removed node
func (ncc *NodeCIDRController) recycleCIDR(obj interface{}) {
	node := obj.(*api.Node)

	if node.Spec.PodCIDR == "" {
		return
	}

	_, podCIDR, err := net.ParseCIDR(node.Spec.PodCIDR)
	if err != nil {
		glog.Errorf("failed to parse node %s, CIDR %s", node.Name, node.Spec.PodCIDR)
		return
	}

	glog.V(4).Infof("recycle node %s CIDR %s", node.Name, podCIDR)
	if err := ncc.cidrAllocator.Release(podCIDR); err != nil {
		glog.Errorf("failed to release cidr: %v", err)
	}
}

func (ncc *NodeCIDRController) recordNodeStatusChange(node *api.Node, new_status string) {
	ref := &api.ObjectReference{
		Kind:      "Node",
		Name:      node.Name,
		UID:       types.UID(node.Name),
		Namespace: "",
	}
	glog.V(2).Infof("Recording status change %s event message for node %s", new_status, node.Name)
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	ncc.recorder.Eventf(ref, api.EventTypeNormal, new_status, "Node %s status is now: %s", node.Name, new_status)
}

func (ncc *NodeCIDRController) updateCIDRAllocation(data nodeAndCIDR) {
	var err error
	var node *api.Node
	for rep := 0; rep < podCIDRUpdateRetry; rep++ {
		node, err = ncc.kubeClient.Core().Nodes().Get(data.nodeName)
		if err != nil {
			glog.Errorf("Failed while getting node %v to retry updating Node.Spec.PodCIDR: %v", data.nodeName, err)
			continue
		}
		node.Spec.PodCIDR = data.cidr.String()
		if _, err := ncc.kubeClient.Core().Nodes().Update(node); err != nil {
			glog.Errorf("Failed while updating Node.Spec.PodCIDR (%d retries left): %v", podCIDRUpdateRetry-rep-1, err)
		} else {
			break
		}
	}
	if err != nil {
		ncc.recordNodeStatusChange(node, "CIDRAssignmentFailed")
		glog.Errorf("CIDR assignment for node %v failed: %v. Releasing allocated CIDR", data.nodeName, err)
		err := ncc.cidrAllocator.Release(data.cidr)
		glog.Errorf("Error releasing allocated CIDR for node %v: %v", data.nodeName, err)
	}
}
