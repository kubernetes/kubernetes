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

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	informers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"

	"k8s.io/kubernetes/pkg/controller/node/ipam/cidrset"
	"k8s.io/kubernetes/pkg/controller/node/util"
)

type rangeAllocator struct {
	client      clientset.Interface
	cidrs       *cidrset.CidrSet
	clusterCIDR *net.IPNet
	maxCIDRs    int

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
func NewCIDRRangeAllocator(client clientset.Interface, clusterCIDR *net.IPNet, serviceCIDR *net.IPNet, subNetMaskSize int, nodeList *v1.NodeList) (CIDRAllocator, error) {
	if client == nil {
		glog.Fatalf("kubeClient is nil when starting NodeController")
	}

	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cidrAllocator"})
	eventBroadcaster.StartLogging(glog.Infof)
	glog.V(0).Infof("Sending events to api server.")
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(client.CoreV1().RESTClient()).Events("")})

	ra := &rangeAllocator{
		client:                client,
		cidrs:                 cidrset.NewCIDRSet(clusterCIDR, subNetMaskSize),
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
				glog.Infof("Node %v has CIDR %s, occupying it in CIDR map",
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
	for i := 0; i < cidrUpdateWorkers; i++ {
		// TODO: Take stopChan as an argument to NewCIDRRangeAllocator and pass it to the worker.
		go ra.worker(wait.NeverStop)
	}

	return ra, nil
}

func (r *rangeAllocator) worker(stopChan <-chan struct{}) {
	for {
		select {
		case workItem, ok := <-r.nodeCIDRUpdateChannel:
			if !ok {
				glog.Warning("Channel nodeCIDRUpdateChannel was unexpectedly closed")
				return
			}
			r.updateCIDRAllocation(workItem)
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
		glog.V(2).Infof("Node %v is already in a process of CIDR assignment.", node.Name)
		return nil
	}
	if node.Spec.PodCIDR != "" {
		return r.occupyCIDR(node)
	}
	podCIDR, err := r.cidrs.AllocateNext()
	if err != nil {
		r.removeNodeFromProcessing(node.Name)
		util.RecordNodeStatusChange(r.recorder, node, "CIDRNotAvailable")
		return fmt.Errorf("failed to allocate cidr: %v", err)
	}

	glog.V(4).Infof("Putting node %s with CIDR %s into the work queue", node.Name, podCIDR)
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

	glog.V(4).Infof("release CIDR %s", node.Spec.PodCIDR)
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
		glog.Errorf("Error filtering out service cidr %v: %v", serviceCIDR, err)
	}
}

// updateCIDRAllocation assigns CIDR to Node and sends an update to the API server.
func (r *rangeAllocator) updateCIDRAllocation(data nodeAndCIDR) error {
	var err error
	var node *v1.Node
	defer r.removeNodeFromProcessing(data.nodeName)

	podCIDR := data.cidr.String()
	for rep := 0; rep < cidrUpdateRetries; rep++ {
		// TODO: change it to using PATCH instead of full Node updates.
		node, err = r.client.CoreV1().Nodes().Get(data.nodeName, metav1.GetOptions{})
		if err != nil {
			glog.Errorf("Failed while getting node %v to retry updating Node.Spec.PodCIDR: %v", data.nodeName, err)
			continue
		}
		if node.Spec.PodCIDR != "" {
			glog.V(4).Infof("Node %v already has allocated CIDR %v. Releasing assigned one if different.", node.Name, node.Spec.PodCIDR)
			if node.Spec.PodCIDR != podCIDR {
				glog.Errorf("Node %q PodCIDR seems to have changed (original=%v, current=%v), releasing original and occupying new CIDR",
					node.Name, node.Spec.PodCIDR, podCIDR)
				if err := r.cidrs.Release(data.cidr); err != nil {
					glog.Errorf("Error when releasing CIDR %v", podCIDR)
				}
			}
			return nil
		}
		node.Spec.PodCIDR = podCIDR
		if _, err = r.client.CoreV1().Nodes().Update(node); err == nil {
			glog.Infof("Set node %v PodCIDR to %v", node.Name, podCIDR)
			break
		}
		glog.Errorf("Failed to update node %v PodCIDR to %v (%d retries left): %v", node.Name, podCIDR, cidrUpdateRetries-rep-1, err)
	}
	if err != nil {
		util.RecordNodeStatusChange(r.recorder, node, "CIDRAssignmentFailed")
		// We accept the fact that we may leek CIDRs here. This is safer than releasing
		// them in case when we don't know if request went through.
		// NodeController restart will return all falsely allocated CIDRs to the pool.
		if !apierrors.IsServerTimeout(err) {
			glog.Errorf("CIDR assignment for node %v failed: %v. Releasing allocated CIDR", data.nodeName, err)
			if releaseErr := r.cidrs.Release(data.cidr); releaseErr != nil {
				glog.Errorf("Error releasing allocated CIDR for node %v: %v", data.nodeName, releaseErr)
			}
		}
	}
	return err
}

func (r *rangeAllocator) Register(nodeInformer informers.NodeInformer) {
	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: util.CreateAddNodeHandler(r.AllocateOrOccupyCIDR),
		UpdateFunc: util.CreateUpdateNodeHandler(func(_, newNode *v1.Node) error {
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
				return r.AllocateOrOccupyCIDR(newNode)
			}
			return nil
		}),
		DeleteFunc: util.CreateDeleteNodeHandler(r.ReleaseCIDR),
	})
}
