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
	"fmt"
	"net"
	"sync"

	"k8s.io/klog"

	"k8s.io/api/core/v1"
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
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam/cidrset"
	nodeutil "k8s.io/kubernetes/pkg/controller/util/node"
	utilnode "k8s.io/kubernetes/pkg/util/node"
)

type rangeAllocator struct {
	client      clientset.Interface
	cidrs       *cidrset.CidrSet
	clusterCIDR *net.IPNet
	maxCIDRs    int

	// nodeLister is able to list/get nodes and is populated by the shared informer passed to
	// NewCloudCIDRAllocator.
	nodeLister corelisters.NodeLister
	// nodesSynced returns true if the node shared informer has been synced at least once.
	nodesSynced cache.InformerSynced

	// Channel that is used to pass updating Nodes with assigned CIDRs to the background
	// This increases a throughput of CIDR assignment by not blocking on long operations.
	nodeCIDRUpdateChannel chan nodeAndCIDR
	recorder              record.EventRecorder

	// Keep a set of nodes that are currectly being processed to avoid races in CIDR allocation
	lock              sync.Mutex
	nodesInProcessing sets.String
}

// NewCIDRRangeAllocator returns a CIDRAllocator to allocate CIDR for node
// Caller must ensure subNetMaskSize is not less than cluster CIDR mask size.
// Caller must always pass in a list of existing nodes so the new allocator
// can initialize its CIDR map. NodeList is only nil in testing.
func NewCIDRRangeAllocator(client clientset.Interface, nodeInformer informers.NodeInformer, clusterCIDR *net.IPNet, serviceCIDR *net.IPNet, subNetMaskSize int, nodeList *v1.NodeList) (CIDRAllocator, error) {
	if client == nil {
		klog.Fatalf("kubeClient is nil when starting NodeController")
	}

	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cidrAllocator"})
	eventBroadcaster.StartLogging(klog.Infof)
	klog.V(0).Infof("Sending events to api server.")
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: client.CoreV1().Events("")})

	set, err := cidrset.NewCIDRSet(clusterCIDR, subNetMaskSize)
	if err != nil {
		return nil, err
	}
	ra := &rangeAllocator{
		client:                client,
		cidrs:                 set,
		clusterCIDR:           clusterCIDR,
		nodeLister:            nodeInformer.Lister(),
		nodesSynced:           nodeInformer.Informer().HasSynced,
		nodeCIDRUpdateChannel: make(chan nodeAndCIDR, cidrUpdateQueueSize),
		recorder:              recorder,
		nodesInProcessing:     sets.NewString(),
	}

	if serviceCIDR != nil {
		ra.filterOutServiceRange(serviceCIDR)
	} else {
		klog.V(0).Info("No Service CIDR provided. Skipping filtering out service addresses.")
	}

	if nodeList != nil {
		for _, node := range nodeList.Items {
			if node.Spec.PodCIDR == "" {
				klog.Infof("Node %v has no CIDR, ignoring", node.Name)
				continue
			} else {
				klog.Infof("Node %v has CIDR %s, occupying it in CIDR map",
					node.Name, node.Spec.PodCIDR)
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

	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: nodeutil.CreateAddNodeHandler(ra.AllocateOrOccupyCIDR),
		UpdateFunc: nodeutil.CreateUpdateNodeHandler(func(_, newNode *v1.Node) error {
			// If the PodCIDR is not empty we either:
			// - already processed a Node that already had a CIDR after NC restarted
			//   (cidr is marked as used),
			// - already processed a Node successfully and allocated a CIDR for it
			//   (cidr is marked as used),
			// - already processed a Node but we did saw a "timeout" response and
			//   request eventually got through in this case we haven't released
			//   the allocated CIDR (cidr is still marked as used).
			// There's a possible error here:
			// - NC sees a new Node and assigns a CIDR X to it,
			// - Update Node call fails with a timeout,
			// - Node is updated by some other component, NC sees an update and
			//   assigns CIDR Y to the Node,
			// - Both CIDR X and CIDR Y are marked as used in the local cache,
			//   even though Node sees only CIDR Y
			// The problem here is that in in-memory cache we see CIDR X as marked,
			// which prevents it from being assigned to any new node. The cluster
			// state is correct.
			// Restart of NC fixes the issue.
			if newNode.Spec.PodCIDR == "" {
				return ra.AllocateOrOccupyCIDR(newNode)
			}
			return nil
		}),
		DeleteFunc: nodeutil.CreateDeleteNodeHandler(ra.ReleaseCIDR),
	})

	return ra, nil
}

func (r *rangeAllocator) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	klog.Infof("Starting range CIDR allocator")
	defer klog.Infof("Shutting down range CIDR allocator")

	if !controller.WaitForCacheSync("cidrallocator", stopCh, r.nodesSynced) {
		return
	}

	for i := 0; i < cidrUpdateWorkers; i++ {
		go r.worker(stopCh)
	}

	<-stopCh
}

func (r *rangeAllocator) worker(stopChan <-chan struct{}) {
	for {
		select {
		case workItem, ok := <-r.nodeCIDRUpdateChannel:
			if !ok {
				klog.Warning("Channel nodeCIDRUpdateChannel was unexpectedly closed")
				return
			}
			if err := r.updateCIDRAllocation(workItem); err != nil {
				// Requeue the failed node for update again.
				r.nodeCIDRUpdateChannel <- workItem
			}
		case <-stopChan:
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

func (r *rangeAllocator) occupyCIDR(node *v1.Node) error {
	defer r.removeNodeFromProcessing(node.Name)
	if node.Spec.PodCIDR == "" {
		return nil
	}
	_, podCIDR, err := net.ParseCIDR(node.Spec.PodCIDR)
	if err != nil {
		return fmt.Errorf("failed to parse node %s, CIDR %s", node.Name, node.Spec.PodCIDR)
	}
	if err := r.cidrs.Occupy(podCIDR); err != nil {
		return fmt.Errorf("failed to mark cidr as occupied: %v", err)
	}
	return nil
}

// WARNING: If you're adding any return calls or defer any more work from this
// function you have to make sure to update nodesInProcessing properly with the
// disposition of the node when the work is done.
func (r *rangeAllocator) AllocateOrOccupyCIDR(node *v1.Node) error {
	if node == nil {
		return nil
	}
	if !r.insertNodeToProcessing(node.Name) {
		klog.V(2).Infof("Node %v is already in a process of CIDR assignment.", node.Name)
		return nil
	}
	if node.Spec.PodCIDR != "" {
		return r.occupyCIDR(node)
	}
	podCIDR, err := r.cidrs.AllocateNext()
	if err != nil {
		r.removeNodeFromProcessing(node.Name)
		nodeutil.RecordNodeStatusChange(r.recorder, node, "CIDRNotAvailable")
		return fmt.Errorf("failed to allocate cidr: %v", err)
	}

	klog.V(4).Infof("Putting node %s with CIDR %s into the work queue", node.Name, podCIDR)
	r.nodeCIDRUpdateChannel <- nodeAndCIDR{
		nodeName: node.Name,
		cidr:     podCIDR,
	}
	return nil
}

func (r *rangeAllocator) ReleaseCIDR(node *v1.Node) error {
	if node == nil || node.Spec.PodCIDR == "" {
		return nil
	}
	_, podCIDR, err := net.ParseCIDR(node.Spec.PodCIDR)
	if err != nil {
		return fmt.Errorf("Failed to parse CIDR %s on Node %v: %v", node.Spec.PodCIDR, node.Name, err)
	}

	klog.V(4).Infof("release CIDR %s", node.Spec.PodCIDR)
	if err = r.cidrs.Release(podCIDR); err != nil {
		return fmt.Errorf("Error when releasing CIDR %v: %v", node.Spec.PodCIDR, err)
	}
	return err
}

// Marks all CIDRs with subNetMaskSize that belongs to serviceCIDR as used,
// so that they won't be assignable.
func (r *rangeAllocator) filterOutServiceRange(serviceCIDR *net.IPNet) {
	// Checks if service CIDR has a nonempty intersection with cluster
	// CIDR. It is the case if either clusterCIDR contains serviceCIDR with
	// clusterCIDR's Mask applied (this means that clusterCIDR contains
	// serviceCIDR) or vice versa (which means that serviceCIDR contains
	// clusterCIDR).
	if !r.clusterCIDR.Contains(serviceCIDR.IP.Mask(r.clusterCIDR.Mask)) && !serviceCIDR.Contains(r.clusterCIDR.IP.Mask(serviceCIDR.Mask)) {
		return
	}

	if err := r.cidrs.Occupy(serviceCIDR); err != nil {
		klog.Errorf("Error filtering out service cidr %v: %v", serviceCIDR, err)
	}
}

// updateCIDRAllocation assigns CIDR to Node and sends an update to the API server.
func (r *rangeAllocator) updateCIDRAllocation(data nodeAndCIDR) error {
	var err error
	var node *v1.Node
	defer r.removeNodeFromProcessing(data.nodeName)

	podCIDR := data.cidr.String()

	node, err = r.nodeLister.Get(data.nodeName)
	if err != nil {
		klog.Errorf("Failed while getting node %v for updating Node.Spec.PodCIDR: %v", data.nodeName, err)
		return err
	}

	if node.Spec.PodCIDR == podCIDR {
		klog.V(4).Infof("Node %v already has allocated CIDR %v. It matches the proposed one.", node.Name, podCIDR)
		return nil
	}
	if node.Spec.PodCIDR != "" {
		klog.Errorf("Node %v already has a CIDR allocated %v. Releasing the new one %v.", node.Name, node.Spec.PodCIDR, podCIDR)
		if err := r.cidrs.Release(data.cidr); err != nil {
			klog.Errorf("Error when releasing CIDR %v", podCIDR)
		}
		return nil
	}
	// If we reached here, it means that the node has no CIDR currently assigned. So we set it.
	for i := 0; i < cidrUpdateRetries; i++ {
		if err = utilnode.PatchNodeCIDR(r.client, types.NodeName(node.Name), podCIDR); err == nil {
			klog.Infof("Set node %v PodCIDR to %v", node.Name, podCIDR)
			return nil
		}
	}
	klog.Errorf("Failed to update node %v PodCIDR to %v after multiple attempts: %v", node.Name, podCIDR, err)
	nodeutil.RecordNodeStatusChange(r.recorder, node, "CIDRAssignmentFailed")
	// We accept the fact that we may leak CIDRs here. This is safer than releasing
	// them in case when we don't know if request went through.
	// NodeController restart will return all falsely allocated CIDRs to the pool.
	if !apierrors.IsServerTimeout(err) {
		klog.Errorf("CIDR assignment for node %v failed: %v. Releasing allocated CIDR", node.Name, err)
		if releaseErr := r.cidrs.Release(data.cidr); releaseErr != nil {
			klog.Errorf("Error releasing allocated CIDR for node %v: %v", node.Name, releaseErr)
		}
	}
	return err
}
