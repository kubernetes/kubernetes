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

package ipam

import (
	"context"
	"fmt"
	"net"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	informers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	nodeutil "k8s.io/component-helpers/node/util"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam/cidrset"
	controllerutil "k8s.io/kubernetes/pkg/controller/util/node"
)

type rangeAllocator struct {
	client clientset.Interface
	// cluster cidrs as passed in during controller creation
	clusterCIDRs []*net.IPNet
	// for each entry in clusterCIDRs we maintain a list of what is used and what is not
	cidrSets []*cidrset.CidrSet
	// nodeLister is able to list/get nodes and is populated by the shared informer passed to controller
	nodeLister corelisters.NodeLister
	// nodesSynced returns true if the node shared informer has been synced at least once.
	nodesSynced cache.InformerSynced
	// Channel that is used to pass updating Nodes and their reserved CIDRs to the background
	// This increases a throughput of CIDR assignment by not blocking on long operations.
	nodeCIDRUpdateChannel chan nodeReservedCIDRs
	broadcaster           record.EventBroadcaster
	recorder              record.EventRecorder
	// Keep a set of nodes that are currently being processed to avoid races in CIDR allocation
	lock              sync.Mutex
	nodesInProcessing sets.String
}

// NewCIDRRangeAllocator returns a CIDRAllocator to allocate CIDRs for node (one from each of clusterCIDRs)
// Caller must ensure subNetMaskSize is not less than cluster CIDR mask size.
// Caller must always pass in a list of existing nodes so the new allocator.
// Caller must ensure that ClusterCIDRs are semantically correct e.g (1 for non DualStack, 2 for DualStack etc..)
// can initialize its CIDR map. NodeList is only nil in testing.
func NewCIDRRangeAllocator(ctx context.Context, client clientset.Interface, nodeInformer informers.NodeInformer, allocatorParams CIDRAllocatorParams, nodeList *v1.NodeList) (CIDRAllocator, error) {
	logger := klog.FromContext(ctx)
	if client == nil {
		logger.Error(nil, "kubeClient is nil when starting CIDRRangeAllocator")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}

	eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cidrAllocator"})

	// create a cidrSet for each cidr we operate on
	// cidrSet are mapped to clusterCIDR by index
	cidrSets := make([]*cidrset.CidrSet, len(allocatorParams.ClusterCIDRs))
	for idx, cidr := range allocatorParams.ClusterCIDRs {
		cidrSet, err := cidrset.NewCIDRSet(cidr, allocatorParams.NodeCIDRMaskSizes[idx])
		if err != nil {
			return nil, err
		}
		cidrSets[idx] = cidrSet
	}

	ra := &rangeAllocator{
		client:                client,
		clusterCIDRs:          allocatorParams.ClusterCIDRs,
		cidrSets:              cidrSets,
		nodeLister:            nodeInformer.Lister(),
		nodesSynced:           nodeInformer.Informer().HasSynced,
		nodeCIDRUpdateChannel: make(chan nodeReservedCIDRs, cidrUpdateQueueSize),
		broadcaster:           eventBroadcaster,
		recorder:              recorder,
		nodesInProcessing:     sets.NewString(),
	}

	if allocatorParams.ServiceCIDR != nil {
		ra.filterOutServiceRange(logger, allocatorParams.ServiceCIDR)
	} else {
		logger.Info("No Service CIDR provided. Skipping filtering out service addresses")
	}

	if allocatorParams.SecondaryServiceCIDR != nil {
		ra.filterOutServiceRange(logger, allocatorParams.SecondaryServiceCIDR)
	} else {
		logger.Info("No Secondary Service CIDR provided. Skipping filtering out secondary service addresses")
	}

	if nodeList != nil {
		for _, node := range nodeList.Items {
			if len(node.Spec.PodCIDRs) == 0 {
				logger.V(4).Info("Node has no CIDR, ignoring", "node", klog.KObj(&node))
				continue
			}
			logger.V(4).Info("Node has CIDR, occupying it in CIDR map", "node", klog.KObj(&node), "podCIDR", node.Spec.PodCIDR)
			if err := ra.occupyCIDRs(&node); err != nil {
				// This will happen if:
				// 1. We find garbage in the podCIDRs field. Retrying is useless.
				// 2. CIDR out of range: This means a node CIDR has changed.
				// This error will keep crashing controller-manager.
				return nil, err
			}
		}
	}

	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: controllerutil.CreateAddNodeHandler(func(node *v1.Node) error {
			return ra.AllocateOrOccupyCIDR(logger, node)
		}),
		UpdateFunc: controllerutil.CreateUpdateNodeHandler(func(_, newNode *v1.Node) error {
			// If the PodCIDRs list is not empty we either:
			// - already processed a Node that already had CIDRs after NC restarted
			//   (cidr is marked as used),
			// - already processed a Node successfully and allocated CIDRs for it
			//   (cidr is marked as used),
			// - already processed a Node but we did saw a "timeout" response and
			//   request eventually got through in this case we haven't released
			//   the allocated CIDRs (cidr is still marked as used).
			// There's a possible error here:
			// - NC sees a new Node and assigns CIDRs X,Y.. to it,
			// - Update Node call fails with a timeout,
			// - Node is updated by some other component, NC sees an update and
			//   assigns CIDRs A,B.. to the Node,
			// - Both CIDR X,Y.. and CIDR A,B.. are marked as used in the local cache,
			//   even though Node sees only CIDR A,B..
			// The problem here is that in in-memory cache we see CIDR X,Y.. as marked,
			// which prevents it from being assigned to any new node. The cluster
			// state is correct.
			// Restart of NC fixes the issue.
			if len(newNode.Spec.PodCIDRs) == 0 {
				return ra.AllocateOrOccupyCIDR(logger, newNode)
			}
			return nil
		}),
		DeleteFunc: controllerutil.CreateDeleteNodeHandler(logger, func(node *v1.Node) error {
			return ra.ReleaseCIDR(logger, node)
		}),
	})

	return ra, nil
}

func (r *rangeAllocator) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()

	// Start event processing pipeline.
	r.broadcaster.StartStructuredLogging(3)
	logger := klog.FromContext(ctx)
	logger.Info("Sending events to api server")
	r.broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: r.client.CoreV1().Events("")})
	defer r.broadcaster.Shutdown()

	logger.Info("Starting range CIDR allocator")
	defer logger.Info("Shutting down range CIDR allocator")

	if !cache.WaitForNamedCacheSync("cidrallocator", ctx.Done(), r.nodesSynced) {
		return
	}

	for i := 0; i < cidrUpdateWorkers; i++ {
		go r.worker(ctx)
	}

	<-ctx.Done()
}

func (r *rangeAllocator) worker(ctx context.Context) {
	logger := klog.FromContext(ctx)
	for {
		select {
		case workItem, ok := <-r.nodeCIDRUpdateChannel:
			if !ok {
				logger.Info("Channel nodeCIDRUpdateChannel was unexpectedly closed")
				return
			}
			if err := r.updateCIDRsAllocation(logger, workItem); err != nil {
				// Requeue the failed node for update again.
				r.nodeCIDRUpdateChannel <- workItem
			}
		case <-ctx.Done():
			return
		}
	}
}

func (r *rangeAllocator) insertNodeToProcessing(nodeName string) bool {
	r.lock.Lock()
	defer r.lock.Unlock()
	if r.nodesInProcessing.Has(nodeName) {
		return false
	}
	r.nodesInProcessing.Insert(nodeName)
	return true
}

func (r *rangeAllocator) removeNodeFromProcessing(nodeName string) {
	r.lock.Lock()
	defer r.lock.Unlock()
	r.nodesInProcessing.Delete(nodeName)
}

// marks node.PodCIDRs[...] as used in allocator's tracked cidrSet
func (r *rangeAllocator) occupyCIDRs(node *v1.Node) error {
	defer r.removeNodeFromProcessing(node.Name)
	if len(node.Spec.PodCIDRs) == 0 {
		return nil
	}
	for idx, cidr := range node.Spec.PodCIDRs {
		_, podCIDR, err := netutils.ParseCIDRSloppy(cidr)
		if err != nil {
			return fmt.Errorf("failed to parse node %s, CIDR %s", node.Name, node.Spec.PodCIDR)
		}
		// If node has a pre allocate cidr that does not exist in our cidrs.
		// This will happen if cluster went from dualstack(multi cidrs) to non-dualstack
		// then we have now way of locking it
		if idx >= len(r.cidrSets) {
			return fmt.Errorf("node:%s has an allocated cidr: %v at index:%v that does not exist in cluster cidrs configuration", node.Name, cidr, idx)
		}

		if err := r.cidrSets[idx].Occupy(podCIDR); err != nil {
			return fmt.Errorf("failed to mark cidr[%v] at idx [%v] as occupied for node: %v: %v", podCIDR, idx, node.Name, err)
		}
	}
	return nil
}

// WARNING: If you're adding any return calls or defer any more work from this
// function you have to make sure to update nodesInProcessing properly with the
// disposition of the node when the work is done.
func (r *rangeAllocator) AllocateOrOccupyCIDR(logger klog.Logger, node *v1.Node) error {
	if node == nil {
		return nil
	}
	if !r.insertNodeToProcessing(node.Name) {
		logger.V(2).Info("Node is already in a process of CIDR assignment", "node", klog.KObj(node))
		return nil
	}

	if len(node.Spec.PodCIDRs) > 0 {
		return r.occupyCIDRs(node)
	}
	// allocate and queue the assignment
	allocated := nodeReservedCIDRs{
		nodeName:       node.Name,
		allocatedCIDRs: make([]*net.IPNet, len(r.cidrSets)),
	}

	for idx := range r.cidrSets {
		podCIDR, err := r.cidrSets[idx].AllocateNext()
		if err != nil {
			r.removeNodeFromProcessing(node.Name)
			controllerutil.RecordNodeStatusChange(logger, r.recorder, node, "CIDRNotAvailable")
			return fmt.Errorf("failed to allocate cidr from cluster cidr at idx:%v: %v", idx, err)
		}
		allocated.allocatedCIDRs[idx] = podCIDR
	}

	//queue the assignment
	logger.V(4).Info("Putting node with CIDR into the work queue", "node", klog.KObj(node), "CIDRs", allocated.allocatedCIDRs)
	r.nodeCIDRUpdateChannel <- allocated
	return nil
}

// ReleaseCIDR marks node.podCIDRs[...] as unused in our tracked cidrSets
func (r *rangeAllocator) ReleaseCIDR(logger klog.Logger, node *v1.Node) error {
	if node == nil || len(node.Spec.PodCIDRs) == 0 {
		return nil
	}

	for idx, cidr := range node.Spec.PodCIDRs {
		_, podCIDR, err := netutils.ParseCIDRSloppy(cidr)
		if err != nil {
			return fmt.Errorf("failed to parse CIDR %s on Node %v: %v", cidr, node.Name, err)
		}

		// If node has a pre allocate cidr that does not exist in our cidrs.
		// This will happen if cluster went from dualstack(multi cidrs) to non-dualstack
		// then we have now way of locking it
		if idx >= len(r.cidrSets) {
			return fmt.Errorf("node:%s has an allocated cidr: %v at index:%v that does not exist in cluster cidrs configuration", node.Name, cidr, idx)
		}

		logger.V(4).Info("Release CIDR for node", "CIDR", cidr, "node", klog.KObj(node))
		if err = r.cidrSets[idx].Release(podCIDR); err != nil {
			return fmt.Errorf("error when releasing CIDR %v: %v", cidr, err)
		}
	}
	return nil
}

// Marks all CIDRs with subNetMaskSize that belongs to serviceCIDR as used across all cidrs
// so that they won't be assignable.
func (r *rangeAllocator) filterOutServiceRange(logger klog.Logger, serviceCIDR *net.IPNet) {
	// Checks if service CIDR has a nonempty intersection with cluster
	// CIDR. It is the case if either clusterCIDR contains serviceCIDR with
	// clusterCIDR's Mask applied (this means that clusterCIDR contains
	// serviceCIDR) or vice versa (which means that serviceCIDR contains
	// clusterCIDR).
	for idx, cidr := range r.clusterCIDRs {
		// if they don't overlap then ignore the filtering
		if !cidr.Contains(serviceCIDR.IP.Mask(cidr.Mask)) && !serviceCIDR.Contains(cidr.IP.Mask(serviceCIDR.Mask)) {
			continue
		}

		// at this point, len(cidrSet) == len(clusterCidr)
		if err := r.cidrSets[idx].Occupy(serviceCIDR); err != nil {
			logger.Error(err, "Error filtering out service cidr out cluster cidr", "CIDR", cidr, "index", idx, "serviceCIDR", serviceCIDR)
		}
	}
}

// updateCIDRsAllocation assigns CIDR to Node and sends an update to the API server.
func (r *rangeAllocator) updateCIDRsAllocation(logger klog.Logger, data nodeReservedCIDRs) error {
	var err error
	var node *v1.Node
	defer r.removeNodeFromProcessing(data.nodeName)
	cidrsString := ipnetToStringList(data.allocatedCIDRs)
	node, err = r.nodeLister.Get(data.nodeName)
	if err != nil {
		logger.Error(err, "Failed while getting node for updating Node.Spec.PodCIDRs", "node", klog.KRef("", data.nodeName))
		return err
	}

	// if cidr list matches the proposed.
	// then we possibly updated this node
	// and just failed to ack the success.
	if len(node.Spec.PodCIDRs) == len(data.allocatedCIDRs) {
		match := true
		for idx, cidr := range cidrsString {
			if node.Spec.PodCIDRs[idx] != cidr {
				match = false
				break
			}
		}
		if match {
			logger.V(4).Info("Node already has allocated CIDR. It matches the proposed one", "node", klog.KObj(node), "CIDRs", data.allocatedCIDRs)
			return nil
		}
	}

	// node has cidrs, release the reserved
	if len(node.Spec.PodCIDRs) != 0 {
		logger.Error(nil, "Node already has a CIDR allocated. Releasing the new one", "node", klog.KObj(node), "podCIDRs", node.Spec.PodCIDRs)
		for idx, cidr := range data.allocatedCIDRs {
			if releaseErr := r.cidrSets[idx].Release(cidr); releaseErr != nil {
				logger.Error(releaseErr, "Error when releasing CIDR", "index", idx, "CIDR", cidr)
			}
		}
		return nil
	}

	// If we reached here, it means that the node has no CIDR currently assigned. So we set it.
	for i := 0; i < cidrUpdateRetries; i++ {
		if err = nodeutil.PatchNodeCIDRs(r.client, types.NodeName(node.Name), cidrsString); err == nil {
			logger.Info("Set node PodCIDR", "node", klog.KObj(node), "podCIDRs", cidrsString)
			return nil
		}
	}
	// failed release back to the pool
	logger.Error(err, "Failed to update node PodCIDR after multiple attempts", "node", klog.KObj(node), "podCIDRs", cidrsString)
	controllerutil.RecordNodeStatusChange(logger, r.recorder, node, "CIDRAssignmentFailed")
	// We accept the fact that we may leak CIDRs here. This is safer than releasing
	// them in case when we don't know if request went through.
	// NodeController restart will return all falsely allocated CIDRs to the pool.
	if !apierrors.IsServerTimeout(err) {
		logger.Error(err, "CIDR assignment for node failed. Releasing allocated CIDR", "node", klog.KObj(node))
		for idx, cidr := range data.allocatedCIDRs {
			if releaseErr := r.cidrSets[idx].Release(cidr); releaseErr != nil {
				logger.Error(releaseErr, "Error releasing allocated CIDR for node", "node", klog.KObj(node))
			}
		}
	}
	return err
}
