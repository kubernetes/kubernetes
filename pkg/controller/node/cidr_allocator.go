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

package node

import (
	"errors"
	"fmt"
	"net"
	"sync"

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"

	"github.com/golang/glog"
)

// TODO: figure out the good setting for those constants.
const (
	// controls how many NodeSpec updates NC can process concurrently.
	cidrUpdateWorkers   = 10
	cidrUpdateQueueSize = 5000
	// podCIDRUpdateRetry controls the number of retries of writing Node.Spec.PodCIDR update.
	podCIDRUpdateRetry = 5
)

var errCIDRRangeNoCIDRsRemaining = errors.New("CIDR allocation failed; there are no remaining CIDRs left to allocate in the accepted range")

type nodeAndCIDR struct {
	cidr     *net.IPNet
	nodeName string
}

// CIDRAllocator is an interface implemented by things that know how to allocate/occupy/recycle CIDR for nodes.
type CIDRAllocator interface {
	AllocateOrOccupyCIDR(node *api.Node) error
	ReleaseCIDR(node *api.Node) error
}

type rangeAllocator struct {
	client      clientset.Interface
	cidrs       *cidrSet
	clusterCIDR *net.IPNet
	maxCIDRs    int
	// Channel that is used to pass updating Nodes with assigned CIDRs to the background
	// This increases a throughput of CIDR assignment by not blocking on long operations.
	nodeCIDRUpdateChannel chan nodeAndCIDR
	recorder              record.EventRecorder
	// Keep a set of nodes that are currectly being processed to avoid races in CIDR allocation
	sync.Mutex
	nodesInProcessing sets.String
}

// NewCIDRRangeAllocator returns a CIDRAllocator to allocate CIDR for node
// Caller must ensure subNetMaskSize is not less than cluster CIDR mask size.
// Caller must always pass in a list of existing nodes so the new allocator
// can initialize its CIDR map. NodeList is only nil in testing.
func NewCIDRRangeAllocator(client clientset.Interface, clusterCIDR *net.IPNet, serviceCIDR *net.IPNet, subNetMaskSize int, nodeList *api.NodeList) (CIDRAllocator, error) {
	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(api.EventSource{Component: "cidrAllocator"})
	eventBroadcaster.StartLogging(glog.Infof)

	ra := &rangeAllocator{
		client:                client,
		cidrs:                 newCIDRSet(clusterCIDR, subNetMaskSize),
		clusterCIDR:           clusterCIDR,
		nodeCIDRUpdateChannel: make(chan nodeAndCIDR, cidrUpdateQueueSize),
		recorder:              recorder,
		nodesInProcessing:     sets.NewString(),
	}

	if serviceCIDR != nil {
		ra.filterOutServiceRange(serviceCIDR)
	} else {
		glog.V(0).Info("No Service CIDR provided. Skipping filtering out service addresses.")
	}

	if nodeList != nil {
		for _, node := range nodeList.Items {
			if node.Spec.PodCIDR == "" {
				glog.Infof("Node %v has no CIDR, ignoring", node.Name)
				continue
			} else {
				glog.Infof("Node %v has CIDR %s, occupying it in CIDR map", node.Name, node.Spec.PodCIDR)
			}
			if err := ra.occupyCIDR(&node); err != nil {
				// This will happen if:
				// 1. We find garbage in the podCIDR field. Retrying is useless.
				// 2. CIDR out of range: This means a node CIDR has changed.
				// This error will keep crashing controller-manager.
				return nil, err
			}
		}
	}
	for i := 0; i < cidrUpdateWorkers; i++ {
		go func(stopChan <-chan struct{}) {
			for {
				select {
				case workItem, ok := <-ra.nodeCIDRUpdateChannel:
					if !ok {
						glog.Warning("NodeCIDRUpdateChannel read returned false.")
						return
					}
					ra.updateCIDRAllocation(workItem)
				case <-stopChan:
					return
				}
			}
		}(wait.NeverStop)
	}

	return ra, nil
}

func (r *rangeAllocator) insertNodeToProcessing(nodeName string) bool {
	r.Lock()
	defer r.Unlock()
	if r.nodesInProcessing.Has(nodeName) {
		return false
	}
	r.nodesInProcessing.Insert(nodeName)
	return true
}

func (r *rangeAllocator) removeNodeFromProcessing(nodeName string) {
	r.Lock()
	defer r.Unlock()
	r.nodesInProcessing.Delete(nodeName)
}

func (r *rangeAllocator) occupyCIDR(node *api.Node) error {
	defer r.removeNodeFromProcessing(node.Name)
	if node.Spec.PodCIDR == "" {
		return nil
	}
	_, podCIDR, err := net.ParseCIDR(node.Spec.PodCIDR)
	if err != nil {
		return fmt.Errorf("failed to parse node %s, CIDR %s", node.Name, node.Spec.PodCIDR)
	}
	if err := r.cidrs.occupy(podCIDR); err != nil {
		return fmt.Errorf("failed to mark cidr as occupied: %v", err)
	}
	return nil
}

// AllocateOrOccupyCIDR looks at the given node, assigns it a valid CIDR
// if it doesn't currently have one or mark the CIDR as used if the node already have one.
// WARNING: If you're adding any return calls or defer any more work from this function
// you have to handle correctly nodesInProcessing.
func (r *rangeAllocator) AllocateOrOccupyCIDR(node *api.Node) error {
	if node == nil {
		return nil
	}
	if !r.insertNodeToProcessing(node.Name) {
		glog.V(2).Infof("Node %v is already in a process of CIDR assignment.", node.Name)
		return nil
	}
	if node.Spec.PodCIDR != "" {
		return r.occupyCIDR(node)
	}
	podCIDR, err := r.cidrs.allocateNext()
	if err != nil {
		r.removeNodeFromProcessing(node.Name)
		recordNodeStatusChange(r.recorder, node, "CIDRNotAvailable")
		return fmt.Errorf("failed to allocate cidr: %v", err)
	}

	glog.V(10).Infof("Putting node %s with CIDR %s into the work queue", node.Name, podCIDR)
	r.nodeCIDRUpdateChannel <- nodeAndCIDR{
		nodeName: node.Name,
		cidr:     podCIDR,
	}
	return nil
}

// ReleaseCIDR releases the CIDR of the removed node
func (r *rangeAllocator) ReleaseCIDR(node *api.Node) error {
	if node == nil || node.Spec.PodCIDR == "" {
		return nil
	}
	_, podCIDR, err := net.ParseCIDR(node.Spec.PodCIDR)
	if err != nil {
		return fmt.Errorf("Failed to parse CIDR %s on Node %v: %v", node.Spec.PodCIDR, node.Name, err)
	}

	glog.V(4).Infof("release CIDR %s", node.Spec.PodCIDR)
	if err = r.cidrs.release(podCIDR); err != nil {
		return fmt.Errorf("Error when releasing CIDR %v: %v", node.Spec.PodCIDR, err)
	}
	return err
}

// Marks all CIDRs with subNetMaskSize that belongs to serviceCIDR as used,
// so that they won't be assignable.
func (r *rangeAllocator) filterOutServiceRange(serviceCIDR *net.IPNet) {
	// Checks if service CIDR has a nonempty intersection with cluster CIDR. It is the case if either
	// clusterCIDR contains serviceCIDR with clusterCIDR's Mask applied (this means that clusterCIDR contains serviceCIDR)
	// or vice versa (which means that serviceCIDR contains clusterCIDR).
	if !r.clusterCIDR.Contains(serviceCIDR.IP.Mask(r.clusterCIDR.Mask)) && !serviceCIDR.Contains(r.clusterCIDR.IP.Mask(serviceCIDR.Mask)) {
		return
	}

	if err := r.cidrs.occupy(serviceCIDR); err != nil {
		glog.Errorf("Error filtering out service cidr %v: %v", serviceCIDR, err)
	}
}

// Assigns CIDR to Node and sends an update to the API server.
func (r *rangeAllocator) updateCIDRAllocation(data nodeAndCIDR) error {
	var err error
	var node *api.Node
	defer r.removeNodeFromProcessing(data.nodeName)
	for rep := 0; rep < podCIDRUpdateRetry; rep++ {
		// TODO: change it to using PATCH instead of full Node updates.
		node, err = r.client.Core().Nodes().Get(data.nodeName)
		if err != nil {
			glog.Errorf("Failed while getting node %v to retry updating Node.Spec.PodCIDR: %v", data.nodeName, err)
			continue
		}
		if node.Spec.PodCIDR != "" {
			glog.Errorf("Node %v already has allocated CIDR %v. Releasing assigned one if different.", node.Name, node.Spec.PodCIDR)
			if node.Spec.PodCIDR != data.cidr.String() {
				if err := r.cidrs.release(data.cidr); err != nil {
					glog.Errorf("Error when releasing CIDR %v", data.cidr.String())
				}
			}
			return nil
		}
		node.Spec.PodCIDR = data.cidr.String()
		if _, err := r.client.Core().Nodes().Update(node); err != nil {
			glog.Errorf("Failed while updating Node.Spec.PodCIDR (%d retries left): %v", podCIDRUpdateRetry-rep-1, err)
		} else {
			break
		}
	}
	if err != nil {
		recordNodeStatusChange(r.recorder, node, "CIDRAssignmentFailed")
		// We accept the fact that we may leek CIDRs here. This is safer than releasing
		// them in case when we don't know if request went through.
		// NodeController restart will return all falsely allocated CIDRs to the pool.
		if !apierrors.IsServerTimeout(err) {
			glog.Errorf("CIDR assignment for node %v failed: %v. Releasing allocated CIDR", data.nodeName, err)
			if releaseErr := r.cidrs.release(data.cidr); releaseErr != nil {
				glog.Errorf("Error releasing allocated CIDR for node %v: %v", data.nodeName, releaseErr)
			}
		}
	}
	return err
}
