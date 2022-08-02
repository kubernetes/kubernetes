/*
Copyright 2022 The Kubernetes Authors.

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
	"container/heap"
	"context"
	"errors"
	"fmt"
	"math"
	"net"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	networkingv1alpha1 "k8s.io/api/networking/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	informers "k8s.io/client-go/informers/core/v1"
	networkinginformers "k8s.io/client-go/informers/networking/v1alpha1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	networkinglisters "k8s.io/client-go/listers/networking/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-base/metrics/prometheus/ratelimiter"
	nodeutil "k8s.io/component-helpers/node/util"
	"k8s.io/klog/v2"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/controller"
	cidrset "k8s.io/kubernetes/pkg/controller/nodeipam/ipam/multicidrset"
	controllerutil "k8s.io/kubernetes/pkg/controller/util/node"
	"k8s.io/kubernetes/pkg/util/slice"
	netutil "k8s.io/utils/net"
)

const (
	defaultClusterCIDRKey        = "kubernetes.io/clusterCIDR"
	defaultClusterCIDRValue      = "default"
	defaultClusterCIDRName       = "default-cluster-cidr"
	defaultClusterCIDRAPIVersion = "networking.k8s.io/v1alpha1"
	clusterCIDRFinalizer         = "networking.k8s.io/cluster-cidr-finalizer"
	ipv4MaxCIDRMask              = 32
	ipv6MaxCIDRMask              = 128
	minPerNodeHostBits           = 4
	maxRetries                   = 15
)

// used for testing
var testCIDRMap map[string][]*cidrset.ClusterCIDR

// CIDRs are reserved, then node resource is patched with them.
// multiCIDRNodeReservedCIDRs holds the reservation info for a node.
type multiCIDRNodeReservedCIDRs struct {
	nodeReservedCIDRs
	clusterCIDR *cidrset.ClusterCIDR
}

type multiCIDRRangeAllocator struct {
	client clientset.Interface
	// nodeLister is able to list/get nodes and is populated by the shared informer passed to controller.
	nodeLister corelisters.NodeLister
	// nodesSynced returns true if the node shared informer has been synced at least once.
	nodesSynced cache.InformerSynced
	// clusterCIDRLister is able to list/get clustercidrs and is populated by the shared informer passed to controller.
	clusterCIDRLister networkinglisters.ClusterCIDRLister
	// clusterCIDRSynced returns true if the clustercidr shared informer has been synced at least once.
	clusterCIDRSynced cache.InformerSynced
	recorder          record.EventRecorder
	// workerLoopPeriod is the time between worker runs. The workers
	// process the queue of nodes and clustercidrs changes
	workerLoopPeriod time.Duration
	// queues are where incoming work is placed to de-dup and to allow "easy"
	// rate limited requeues on errors
	cccQueue  workqueue.RateLimitingInterface
	nodeQueue workqueue.RateLimitingInterface

	// lock guards nodesInProcessing and cidrMap to avoid races in CIDR allocation.
	lock *sync.Mutex
	// cidrMap maps ClusterCIDR labels to internal ClusterCIDR objects.
	cidrMap map[string][]*cidrset.ClusterCIDR
}

// NewMultiCIDRRangeAllocator returns a CIDRAllocator to allocate CIDRs for node (one for each ip family).
// Caller must always pass in a list of existing nodes to the new allocator.
// NodeList is only nil in testing.
func NewMultiCIDRRangeAllocator(
	client clientset.Interface,
	nodeInformer informers.NodeInformer,
	clusterCIDRInformer networkinginformers.ClusterCIDRInformer,
	allocatorParams CIDRAllocatorParams,
	nodeList *v1.NodeList,
	cccList *networkingv1alpha1.ClusterCIDRList,
) (CIDRAllocator, error) {
	if client == nil {
		klog.Fatalf("client is nil")
	}

	eventBroadcaster := record.NewBroadcaster()
	eventSource := v1.EventSource{
		Component: "multiCIDRRangeAllocator",
	}
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, eventSource)
	eventBroadcaster.StartStructuredLogging(0)
	klog.V(0).Infof("Started sending events to API Server. (EventSource = %v)", eventSource)

	eventBroadcaster.StartRecordingToSink(
		&v1core.EventSinkImpl{
			Interface: client.CoreV1().Events(""),
		})

	if client.CoreV1().RESTClient().GetRateLimiter() != nil {
		ratelimiter.RegisterMetricAndTrackRateLimiterUsage("multi_cidr_range_allocator", client.CoreV1().RESTClient().GetRateLimiter())
	}

	ra := &multiCIDRRangeAllocator{
		client:            client,
		nodeLister:        nodeInformer.Lister(),
		nodesSynced:       nodeInformer.Informer().HasSynced,
		clusterCIDRLister: clusterCIDRInformer.Lister(),
		clusterCIDRSynced: clusterCIDRInformer.Informer().HasSynced,
		recorder:          recorder,
		workerLoopPeriod:  time.Second,
		cccQueue:          workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "multi_cidr_range_allocator_clustercidrs"),
		nodeQueue:         workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "multi_cidr_range_allocator_nodes"),
		lock:              &sync.Mutex{},
		cidrMap:           make(map[string][]*cidrset.ClusterCIDR, 0),
	}

	// testCIDRMap is only set for testing purposes.
	if len(testCIDRMap) > 0 {
		ra.cidrMap = testCIDRMap
		klog.Fatalf("testCIDRMap should only be set for testing purposes, if this is seen in production logs, it might be a misconfiguration or a bug.")
	}

	if cccList == nil {
		cccList = &networkingv1alpha1.ClusterCIDRList{}
	}
	createDefaultClusterCIDR(cccList, allocatorParams)

	// Regenerate the cidrMaps from the existing ClusterCIDRs.
	for _, clusterCIDR := range cccList.Items {
		klog.Infof("Regenerating existing ClusterCIDR: %v", clusterCIDR)
		// Create an event for invalid ClusterCIDRs, do not crash on failures.
		if err := ra.reconcileBootstrap(&clusterCIDR); err != nil {
			klog.Errorf("Error while regenerating existing ClusterCIDR: %v", err)
			ra.recorder.Event(&clusterCIDR, "Warning", "InvalidClusterCIDR encountered while regenerating ClusterCIDR during bootstrap.", err.Error())
		}
	}

	clusterCIDRInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    ra.addClusterCIDR,
		UpdateFunc: ra.updateClusterCIDR,
		DeleteFunc: ra.deleteClusterCIDR,
	})

	if allocatorParams.ServiceCIDR != nil {
		ra.filterOutServiceRange(allocatorParams.ServiceCIDR)
	} else {
		klog.V(0).Info("No Service CIDR provided. Skipping filtering out service addresses.")
	}

	if allocatorParams.SecondaryServiceCIDR != nil {
		ra.filterOutServiceRange(allocatorParams.SecondaryServiceCIDR)
	} else {
		klog.V(0).Info("No Secondary Service CIDR provided. Skipping filtering out secondary service addresses.")
	}

	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    ra.addNode,
		UpdateFunc: ra.updateNode,
		DeleteFunc: ra.deleteNode,
	})

	return ra, nil
}

func (r *multiCIDRRangeAllocator) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	klog.Infof("Starting Multi CIDR Range allocator")
	defer klog.Infof("Shutting down Multi CIDR Range allocator")

	if !cache.WaitForNamedCacheSync("multi_cidr_range_allocator", stopCh, r.nodesSynced, r.clusterCIDRSynced) {
		return
	}

	for i := 0; i < cidrUpdateWorkers; i++ {
		go wait.Until(r.nodeWorker, r.workerLoopPeriod, stopCh)
		go wait.Until(r.cccWorker, r.workerLoopPeriod, stopCh)
	}

	<-stopCh
}

func (r *multiCIDRRangeAllocator) nodeWorker() {
	for r.processNextNodeItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (r *multiCIDRRangeAllocator) processNextNodeItem() bool {
	cKey, quit := r.nodeQueue.Get()
	if quit {
		return false
	}
	defer r.nodeQueue.Done(cKey)

	err := r.syncNode(cKey.(string))
	r.handleNodeErr(err, cKey)

	return true
}

func (r *multiCIDRRangeAllocator) syncNode(key string) error {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing node %q. (%v)", key, time.Since(startTime))
	}()

	node, err := r.nodeLister.Get(key)
	if apierrors.IsNotFound(err) {
		// delete the assigned CIDR to that node
		// TODO, we may have to use a cache if we don't have this info already
	}
	if err != nil {
		return err
	}
	if !node.DeletionTimestamp.IsZero() {
		// node is being deleted
		return r.ReleaseCIDR(node)
	}

	return r.AllocateOrOccupyCIDR(node)
}

func (r *multiCIDRRangeAllocator) handleNodeErr(err error, key interface{}) {
	if err == nil {
		r.nodeQueue.Forget(key)
		return
	}

	if r.cccQueue.NumRequeues(key) < maxRetries {
		klog.Warningf("Error syncing clusterCIDR %q, retries(%d). Error: %v", key, r.nodeQueue.NumRequeues(key), err)
		r.nodeQueue.AddRateLimited(key)
		return
	}

	klog.Warningf("Error syncing ClusterCIDR %q, not retrying as retry budget (%d retries) has been exceeded: %v", key, maxRetries, err)
	r.nodeQueue.Forget(key)
	utilruntime.HandleError(err)
}

func (r *multiCIDRRangeAllocator) cccWorker() {
	for r.processNextCCCItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (r *multiCIDRRangeAllocator) processNextCCCItem() bool {
	cKey, quit := r.cccQueue.Get()
	if quit {
		return false
	}
	defer r.cccQueue.Done(cKey)

	err := r.syncClusterCIDR(cKey.(string))
	r.handleCCCErr(err, cKey)

	return true
}

func (r *multiCIDRRangeAllocator) handleCCCErr(err error, key interface{}) {
	if err == nil {
		r.cccQueue.Forget(key)
		return
	}

	if r.cccQueue.NumRequeues(key) < maxRetries {
		klog.Warningf("Error syncing clusterCIDR %q, retries(%d). Error: %v", key, r.cccQueue.NumRequeues(key), err)
		r.cccQueue.AddRateLimited(key)
		return
	}

	klog.Warningf("Error syncing ClusterCIDR %q, not retrying as retry budget (%d retries) has been exceeded: %v", key, maxRetries, err)
	r.cccQueue.Forget(key)
	utilruntime.HandleError(err)
}

func (r *multiCIDRRangeAllocator) syncClusterCIDR(key string) error {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing clusterCIDR request %q (%v)", key, time.Since(startTime))
	}()

	clusterCIDR, err := r.clusterCIDRLister.Get(key)
	if apierrors.IsNotFound(err) || !clusterCIDR.DeletionTimestamp.IsZero() {
		klog.V(3).Infof("clusterCIDR %v deleted", key)
		return r.reconcileDelete(clusterCIDR)
	}
	if err != nil {
		return err
	}
	return r.reconcileCreate(clusterCIDR)
}

// needToAddFinalizer checks if a finalizer should be added to the object.
func needToAddFinalizer(obj metav1.Object, finalizer string) bool {
	return obj.GetDeletionTimestamp() == nil && !slice.ContainsString(obj.GetFinalizers(),
		finalizer, nil)
}

// occupyCIDRs marks node.PodCIDRs[...] as used in allocator's tracked cidrSet.
func (r *multiCIDRRangeAllocator) occupyCIDRs(node *v1.Node) error {
	err := func(node *v1.Node) error {

		if len(node.Spec.PodCIDRs) == 0 {
			return nil
		}

		clusterCIDRList, err := r.orderedMatchingClusterCIDRs(node)
		if err != nil {
			return err
		}

		for _, clusterCIDR := range clusterCIDRList {
			occupiedCount := 0

			for _, cidr := range node.Spec.PodCIDRs {
				_, podCIDR, err := netutil.ParseCIDRSloppy(cidr)
				if err != nil {
					return fmt.Errorf("failed to parse CIDR %s on Node %v: %w", cidr, node.Name, err)
				}

				klog.Infof("occupy CIDR %s for node: %s", cidr, node.Name)

				if err := r.Occupy(clusterCIDR, podCIDR); err != nil {
					klog.V(3).Infof("Could not occupy cidr: %v, trying next range: %w", node.Spec.PodCIDRs, err)
					break
				}

				occupiedCount++
			}

			// Mark CIDRs as occupied only if the CCC is able to occupy all the node CIDRs.
			if occupiedCount == len(node.Spec.PodCIDRs) {
				clusterCIDR.AssociatedNodes[node.Name] = true
				return nil
			}
		}

		return fmt.Errorf("could not occupy cidrs: %v, No matching ClusterCIDRs found", node.Spec.PodCIDRs)
	}(node)

	return err
}

// associatedCIDRSet returns the CIDRSet, based on the ip family of the CIDR.
func (r *multiCIDRRangeAllocator) associatedCIDRSet(clusterCIDR *cidrset.ClusterCIDR, cidr *net.IPNet) (*cidrset.MultiCIDRSet, error) {
	switch {
	case netutil.IsIPv4CIDR(cidr):
		return clusterCIDR.IPv4CIDRSet, nil
	case netutil.IsIPv6CIDR(cidr):
		return clusterCIDR.IPv6CIDRSet, nil
	default:
		return nil, fmt.Errorf("invalid cidr: %v", cidr)
	}
}

// Occupy marks the CIDR as occupied in the allocatedCIDRMap of the cidrSet.
func (r *multiCIDRRangeAllocator) Occupy(clusterCIDR *cidrset.ClusterCIDR, cidr *net.IPNet) error {
	currCIDRSet, err := r.associatedCIDRSet(clusterCIDR, cidr)
	if err != nil {
		return err
	}

	if err := currCIDRSet.Occupy(cidr); err != nil {
		return fmt.Errorf("unable to occupy cidr %v in cidrSet", cidr)
	}

	return nil
}

// Release marks the CIDR as free in the cidrSet used bitmap,
// Also removes the CIDR from the allocatedCIDRSet.
func (r *multiCIDRRangeAllocator) Release(clusterCIDR *cidrset.ClusterCIDR, cidr *net.IPNet) error {
	currCIDRSet, err := r.associatedCIDRSet(clusterCIDR, cidr)
	if err != nil {
		return err
	}

	if err := currCIDRSet.Release(cidr); err != nil {
		klog.Infof("Unable to release cidr %v in cidrSet", cidr)
		return err
	}

	return nil
}

// AllocateOrOccupyCIDR allocates a CIDR to the node if the node doesn't have a
// CIDR already allocated, occupies the CIDR and marks as used if the node
// already has a PodCIDR assigned.
// WARNING: If you're adding any return calls or defer any more work from this
// function you have to make sure to update nodesInProcessing properly with the
// disposition of the node when the work is done.
func (r *multiCIDRRangeAllocator) AllocateOrOccupyCIDR(node *v1.Node) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	if node == nil {
		return nil
	}
	if len(node.Spec.PodCIDRs) > 0 {
		return r.occupyCIDRs(node)
	}

	cidrs, clusterCIDR, err := r.prioritizedCIDRs(node)
	if err != nil {
		controllerutil.RecordNodeStatusChange(r.recorder, node, "CIDRNotAvailable")
		return fmt.Errorf("failed to get cidrs for node %s", node.Name)
	}

	if len(cidrs) == 0 {
		controllerutil.RecordNodeStatusChange(r.recorder, node, "CIDRNotAvailable")
		return fmt.Errorf("no cidrSets with matching labels found for node %s", node.Name)
	}

	// allocate and queue the assignment.
	allocated := multiCIDRNodeReservedCIDRs{
		nodeReservedCIDRs: nodeReservedCIDRs{
			nodeName:       node.Name,
			allocatedCIDRs: cidrs,
		},
		clusterCIDR: clusterCIDR,
	}

	return r.updateCIDRsAllocation(allocated)
}

// ReleaseCIDR marks node.podCIDRs[...] as unused in our tracked cidrSets.
func (r *multiCIDRRangeAllocator) ReleaseCIDR(node *v1.Node) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	if node == nil || len(node.Spec.PodCIDRs) == 0 {
		return nil
	}

	clusterCIDR, err := r.allocatedClusterCIDR(node)
	if err != nil {
		return err
	}

	for _, cidr := range node.Spec.PodCIDRs {
		_, podCIDR, err := netutil.ParseCIDRSloppy(cidr)
		if err != nil {
			return fmt.Errorf("failed to parse CIDR %q on Node %q: %w", cidr, node.Name, err)
		}

		klog.Infof("release CIDR %s for node: %s", cidr, node.Name)
		if err := r.Release(clusterCIDR, podCIDR); err != nil {
			return fmt.Errorf("failed to release cidr %q from clusterCIDR %q for node %q: %w", cidr, clusterCIDR.Name, node.Name, err)
		}
	}

	// Remove the node from the ClusterCIDR AssociatedNodes.
	delete(clusterCIDR.AssociatedNodes, node.Name)

	return nil
}

// Marks all CIDRs with subNetMaskSize that belongs to serviceCIDR as used across all cidrs
// so that they won't be assignable.
func (r *multiCIDRRangeAllocator) filterOutServiceRange(serviceCIDR *net.IPNet) {
	// Checks if service CIDR has a nonempty intersection with cluster
	// CIDR. It is the case if either clusterCIDR contains serviceCIDR with
	// clusterCIDR's Mask applied (this means that clusterCIDR contains
	// serviceCIDR) or vice versa (which means that serviceCIDR contains
	// clusterCIDR).
	for _, clusterCIDRList := range r.cidrMap {
		for _, clusterCIDR := range clusterCIDRList {
			if err := r.occupyServiceCIDR(clusterCIDR, serviceCIDR); err != nil {
				klog.Errorf("unable to occupy service CIDR: %w", err)
			}
		}
	}
}

func (r *multiCIDRRangeAllocator) occupyServiceCIDR(clusterCIDR *cidrset.ClusterCIDR, serviceCIDR *net.IPNet) error {

	cidrSet, err := r.associatedCIDRSet(clusterCIDR, serviceCIDR)
	if err != nil {
		return err
	}

	cidr := cidrSet.ClusterCIDR

	// No need to occupy as Service CIDR doesn't intersect with the current ClusterCIDR.
	if !cidr.Contains(serviceCIDR.IP.Mask(cidr.Mask)) && !serviceCIDR.Contains(cidr.IP.Mask(serviceCIDR.Mask)) {
		return nil
	}

	if err := r.Occupy(clusterCIDR, serviceCIDR); err != nil {
		return fmt.Errorf("error filtering out service cidr %v from cluster cidr %v: %w", cidr, serviceCIDR, err)
	}

	return nil
}

// updateCIDRsAllocation assigns CIDR to Node and sends an update to the API server.
func (r *multiCIDRRangeAllocator) updateCIDRsAllocation(data multiCIDRNodeReservedCIDRs) error {
	err := func(data multiCIDRNodeReservedCIDRs) error {
		cidrsString := ipnetToStringList(data.allocatedCIDRs)
		node, err := r.nodeLister.Get(data.nodeName)
		if err != nil {
			klog.Errorf("Failed while getting node %v for updating Node.Spec.PodCIDRs: %v", data.nodeName, err)
			return err
		}

		// if cidr list matches the proposed,
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
				klog.V(4).Infof("Node %q already has allocated CIDR %q. It matches the proposed one.", node.Name, data.allocatedCIDRs)
				return nil
			}
		}

		// node has cidrs allocated, release the reserved.
		if len(node.Spec.PodCIDRs) != 0 {
			klog.Errorf("Node %q already has a CIDR allocated %q. Releasing the new one.", node.Name, node.Spec.PodCIDRs)
			for _, cidr := range data.allocatedCIDRs {
				if err := r.Release(data.clusterCIDR, cidr); err != nil {
					return fmt.Errorf("failed to release cidr %s from clusterCIDR %s for node: %s: %w", cidr, data.clusterCIDR.Name, node.Name, err)
				}
			}
			return nil
		}

		// If we reached here, it means that the node has no CIDR currently assigned. So we set it.
		for i := 0; i < cidrUpdateRetries; i++ {
			if err = nodeutil.PatchNodeCIDRs(r.client, types.NodeName(node.Name), cidrsString); err == nil {
				data.clusterCIDR.AssociatedNodes[node.Name] = true
				klog.Infof("Set node %q PodCIDR to %q", node.Name, cidrsString)
				return nil
			}
		}
		// failed release back to the pool.
		klog.Errorf("Failed to update node %q PodCIDR to %q after %d attempts: %v", node.Name, cidrsString, cidrUpdateRetries, err)
		controllerutil.RecordNodeStatusChange(r.recorder, node, "CIDRAssignmentFailed")
		// We accept the fact that we may leak CIDRs here. This is safer than releasing
		// them in case when we don't know if request went through.
		// NodeController restart will return all falsely allocated CIDRs to the pool.
		if !apierrors.IsServerTimeout(err) {
			klog.Errorf("CIDR assignment for node %q failed: %v. Releasing allocated CIDR", node.Name, err)
			for _, cidr := range data.allocatedCIDRs {
				if err := r.Release(data.clusterCIDR, cidr); err != nil {
					return fmt.Errorf("failed to release cidr %q from clusterCIDR %q for node: %q: %w", cidr, data.clusterCIDR.Name, node.Name, err)
				}
			}
		}
		return err
	}(data)

	return err
}

// defaultNodeSelector generates a label for the catch all use case.
func defaultNodeSelector() ([]byte, error) {
	nodeSelector := &v1.NodeSelector{
		NodeSelectorTerms: []v1.NodeSelectorTerm{
			{
				MatchExpressions: []v1.NodeSelectorRequirement{
					{
						Key:      defaultClusterCIDRKey,
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{defaultClusterCIDRValue},
					},
				},
			},
		},
	}

	marshalledSelector, err := nodeSelector.Marshal()
	if err != nil {
		return nil, err
	}

	return marshalledSelector, nil
}

// prioritizedCIDRs returns a list of CIDRs to be allocated to the node.
// Returns 1 CIDR  if single stack.
// Returns 2 CIDRs , 1 from each ip family if dual stack.
func (r *multiCIDRRangeAllocator) prioritizedCIDRs(node *v1.Node) ([]*net.IPNet, *cidrset.ClusterCIDR, error) {
	clusterCIDRList, err := r.orderedMatchingClusterCIDRs(node)
	if err != nil {
		return nil, nil, fmt.Errorf("unable to get a clusterCIDR for node %s: %w", node.Name, err)
	}

	for _, clusterCIDR := range clusterCIDRList {
		cidrs := make([]*net.IPNet, 0)
		if clusterCIDR.IPv4CIDRSet != nil {
			cidr, err := r.allocateCIDR(clusterCIDR, clusterCIDR.IPv4CIDRSet)
			if err != nil {
				klog.V(3).Infof("unable to allocate IPv4 CIDR, trying next range: %w", err)
				continue
			}
			cidrs = append(cidrs, cidr)
		}

		if clusterCIDR.IPv6CIDRSet != nil {
			cidr, err := r.allocateCIDR(clusterCIDR, clusterCIDR.IPv6CIDRSet)
			if err != nil {
				klog.V(3).Infof("unable to allocate IPv6 CIDR, trying next range: %w", err)
				continue
			}
			cidrs = append(cidrs, cidr)
		}

		return cidrs, clusterCIDR, nil
	}
	return nil, nil, fmt.Errorf("unable to get a clusterCIDR for node %s, no available CIDRs", node.Name)
}

func (r *multiCIDRRangeAllocator) allocateCIDR(clusterCIDR *cidrset.ClusterCIDR, cidrSet *cidrset.MultiCIDRSet) (*net.IPNet, error) {

	for evaluated := 0; evaluated < cidrSet.MaxCIDRs; evaluated++ {
		candidate, lastEvaluated, err := cidrSet.NextCandidate()
		if err != nil {
			return nil, err
		}

		evaluated += lastEvaluated

		if r.cidrInAllocatedList(candidate) {
			continue
		}

		// Deep Check.
		if r.cidrOverlapWithAllocatedList(candidate) {
			continue
		}

		// Mark the CIDR as occupied in the map.
		if err := r.Occupy(clusterCIDR, candidate); err != nil {
			return nil, err
		}
		// Increment the evaluated count metric.
		cidrSet.UpdateEvaluatedCount(evaluated)
		return candidate, nil
	}
	return nil, &cidrset.CIDRRangeNoCIDRsRemainingErr{
		CIDR: cidrSet.Label,
	}
}

func (r *multiCIDRRangeAllocator) cidrInAllocatedList(cidr *net.IPNet) bool {
	for _, clusterCIDRList := range r.cidrMap {
		for _, clusterCIDR := range clusterCIDRList {
			cidrSet, _ := r.associatedCIDRSet(clusterCIDR, cidr)
			if cidrSet != nil {
				if ok := cidrSet.AllocatedCIDRMap[cidr.String()]; ok {
					return true
				}
			}
		}
	}
	return false
}

func (r *multiCIDRRangeAllocator) cidrOverlapWithAllocatedList(cidr *net.IPNet) bool {
	for _, clusterCIDRList := range r.cidrMap {
		for _, clusterCIDR := range clusterCIDRList {
			cidrSet, _ := r.associatedCIDRSet(clusterCIDR, cidr)
			if cidrSet != nil {
				for allocated := range cidrSet.AllocatedCIDRMap {
					_, allocatedCIDR, _ := netutil.ParseCIDRSloppy(allocated)
					if cidr.Contains(allocatedCIDR.IP.Mask(cidr.Mask)) || allocatedCIDR.Contains(cidr.IP.Mask(allocatedCIDR.Mask)) {
						return true
					}
				}
			}
		}
	}
	return false
}

// allocatedClusterCIDR returns the ClusterCIDR from which the node CIDRs were allocated.
func (r *multiCIDRRangeAllocator) allocatedClusterCIDR(node *v1.Node) (*cidrset.ClusterCIDR, error) {
	clusterCIDRList, err := r.orderedMatchingClusterCIDRs(node)
	if err != nil {
		return nil, fmt.Errorf("unable to get a clusterCIDR for node %s: %w", node.Name, err)
	}

	for _, clusterCIDR := range clusterCIDRList {
		if ok := clusterCIDR.AssociatedNodes[node.Name]; ok {
			return clusterCIDR, nil
		}
	}
	return nil, fmt.Errorf("no clusterCIDR found associated with node: %s", node.Name)
}

// orderedMatchingClusterCIDRs returns a list of all the ClusterCIDRs matching the node labels.
// The list is ordered with the following priority, which act as tie-breakers.
// P0: ClusterCIDR with higher number of matching labels has the highest priority.
// P1: ClusterCIDR having cidrSet with fewer allocatable Pod CIDRs has higher priority.
// P2: ClusterCIDR with a PerNodeMaskSize having fewer IPs has higher priority.
// P3: ClusterCIDR having label with lower alphanumeric value has higher priority.
// P4: ClusterCIDR with a cidrSet having a smaller IP address value has a higher priority.
func (r *multiCIDRRangeAllocator) orderedMatchingClusterCIDRs(node *v1.Node) ([]*cidrset.ClusterCIDR, error) {
	matchingCIDRs := make([]*cidrset.ClusterCIDR, 0)
	pq := make(PriorityQueue, 0)

	for label, clusterCIDRList := range r.cidrMap {
		labelsMatch, matchCnt, err := r.matchCIDRLabels(node, []byte(label))
		if err != nil {
			return nil, err
		}

		if !labelsMatch {
			continue
		}

		for _, clusterCIDR := range clusterCIDRList {
			pqItem := &PriorityQueueItem{
				clusterCIDR:     clusterCIDR,
				labelMatchCount: matchCnt,
				selectorString:  label,
			}

			// Only push the CIDRsets which are not marked for termination.
			if !clusterCIDR.Terminating {
				heap.Push(&pq, pqItem)
			}
		}
	}

	// Remove the ClusterCIDRs from the PriorityQueue.
	// They arrive in descending order of matchCnt,
	// if matchCnt is equal it is ordered in ascending order of labels.
	for pq.Len() > 0 {
		pqItem := heap.Pop(&pq).(*PriorityQueueItem)
		matchingCIDRs = append(matchingCIDRs, pqItem.clusterCIDR)
	}

	// Append the catch all CIDR config.
	defaultSelector, err := defaultNodeSelector()
	if err != nil {
		return nil, err
	}
	if clusterCIDRList, ok := r.cidrMap[string(defaultSelector)]; ok {
		matchingCIDRs = append(matchingCIDRs, clusterCIDRList...)
	}
	return matchingCIDRs, nil
}

// matchCIDRLabels Matches the Node labels to CIDR Configs.
// Returns true only if all the labels match, also returns the count of matching labels.
func (r *multiCIDRRangeAllocator) matchCIDRLabels(node *v1.Node, label []byte) (bool, int, error) {
	var labelSet labels.Set
	var matchCnt int

	labelsMatch := false
	selector := &v1.NodeSelector{}
	err := selector.Unmarshal(label)
	if err != nil {
		klog.Errorf("Unable to unmarshal node selector for label %v: %v", label, err)
		return labelsMatch, 0, err
	}

	ls, err := v1helper.NodeSelectorAsSelector(selector)
	if err != nil {
		klog.Errorf("Unable to convert NodeSelector to labels.Selector: %v", err)
		return labelsMatch, 0, err
	}
	reqs, selectable := ls.Requirements()

	labelSet = node.ObjectMeta.Labels
	if selectable {
		matchCnt = 0
		for _, req := range reqs {
			if req.Matches(labelSet) {
				matchCnt += 1
			}
		}
		if matchCnt == len(reqs) {
			labelsMatch = true
		}
	}
	return labelsMatch, matchCnt, err
}

// Methods for handling ClusterCIDRs.

// createDefaultClusterCIDR creates a default ClusterCIDR if --cluster-cidr has
// been configured. It converts the --cluster-cidr and --per-node-mask-size* flags
// to appropriate ClusterCIDR fields.
func createDefaultClusterCIDR(existingConfigList *networkingv1alpha1.ClusterCIDRList,
	allocatorParams CIDRAllocatorParams) {
	// Create default ClusterCIDR only if --cluster-cidr has been configured
	if len(allocatorParams.ClusterCIDRs) == 0 {
		return
	}

	for _, clusterCIDR := range existingConfigList.Items {
		if clusterCIDR.Name == defaultClusterCIDRName {
			// Default ClusterCIDR already exists, no further action required.
			klog.V(3).Infof("Default ClusterCIDR %s already exists", defaultClusterCIDRName)
			return
		}
	}

	// Create a default ClusterCIDR as it is not already created.
	defaultCIDRConfig := &networkingv1alpha1.ClusterCIDR{
		TypeMeta: metav1.TypeMeta{
			APIVersion: defaultClusterCIDRAPIVersion,
			Kind:       "ClusterCIDR",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: defaultClusterCIDRName,
		},
		Spec: networkingv1alpha1.ClusterCIDRSpec{
			PerNodeHostBits: minPerNodeHostBits,
		},
	}

	ipv4PerNodeHostBits := int32(math.MinInt32)
	ipv6PerNodeHostBits := int32(math.MinInt32)
	isDualstack := false
	if len(allocatorParams.ClusterCIDRs) == 2 {
		isDualstack = true
	}

	for i, cidr := range allocatorParams.ClusterCIDRs {
		if netutil.IsIPv4CIDR(cidr) {
			defaultCIDRConfig.Spec.IPv4 = cidr.String()
			ipv4PerNodeHostBits = ipv4MaxCIDRMask - int32(allocatorParams.NodeCIDRMaskSizes[i])
			if !isDualstack && ipv4PerNodeHostBits > minPerNodeHostBits {
				defaultCIDRConfig.Spec.PerNodeHostBits = ipv4PerNodeHostBits
			}
		} else if netutil.IsIPv6CIDR(cidr) {
			defaultCIDRConfig.Spec.IPv6 = cidr.String()
			ipv6PerNodeHostBits = ipv6MaxCIDRMask - int32(allocatorParams.NodeCIDRMaskSizes[i])
			if !isDualstack && ipv6PerNodeHostBits > minPerNodeHostBits {
				defaultCIDRConfig.Spec.PerNodeHostBits = ipv6PerNodeHostBits
			}
		}
	}

	if isDualstack {
		// In case of dualstack CIDRs, currently the default values for PerNodeMaskSize are
		// 24 for IPv4 (PerNodeHostBits=8) and 64 for IPv6(PerNodeHostBits=64), there is no
		// requirement for the PerNodeHostBits to be equal for IPv4 and IPv6, However with
		// the introduction of ClusterCIDRs, we enforce the requirement for a single
		// PerNodeHostBits field, thus we choose the minimum PerNodeHostBits value, to avoid
		// overflow for IPv4 CIDRs.
		if ipv4PerNodeHostBits >= minPerNodeHostBits && ipv4PerNodeHostBits <= ipv6PerNodeHostBits {
			defaultCIDRConfig.Spec.PerNodeHostBits = ipv4PerNodeHostBits
		} else if ipv6PerNodeHostBits >= minPerNodeHostBits && ipv6PerNodeHostBits <= ipv4MaxCIDRMask {
			defaultCIDRConfig.Spec.PerNodeHostBits = ipv6PerNodeHostBits
		}
	}

	existingConfigList.Items = append(existingConfigList.Items, *defaultCIDRConfig)

	return
}

// reconcileCreate handles create ClusterCIDR events.
func (r *multiCIDRRangeAllocator) reconcileCreate(clusterCIDR *networkingv1alpha1.ClusterCIDR) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	if needToAddFinalizer(clusterCIDR, clusterCIDRFinalizer) {
		klog.V(3).Infof("Creating ClusterCIDR %s", clusterCIDR.Name)
		if err := r.createClusterCIDR(clusterCIDR, false); err != nil {
			klog.Errorf("Unable to create ClusterCIDR %s : %v", clusterCIDR.Name, err)
			return err
		}
	}
	return nil
}

// reconcileBootstrap handles creation of existing ClusterCIDRs.
// adds a finalizer if not already present.
func (r *multiCIDRRangeAllocator) reconcileBootstrap(clusterCIDR *networkingv1alpha1.ClusterCIDR) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	terminating := false
	// Create the ClusterCIDR only if the Spec has not been modified.
	if clusterCIDR.Generation > 1 {
		terminating = true
		err := fmt.Errorf("CIDRs from ClusterCIDR %s will not be used for allocation as it was modified", clusterCIDR.Name)
		klog.Errorf("ClusterCIDR Modified: %v", err)
	}

	klog.V(2).Infof("Creating ClusterCIDR %s during bootstrap", clusterCIDR.Name)
	if err := r.createClusterCIDR(clusterCIDR, terminating); err != nil {
		klog.Errorf("Unable to create ClusterCIDR %s: %v", clusterCIDR.Name, err)
		return err
	}

	return nil
}

// createClusterCIDR creates and maps the cidrSets in the cidrMap.
func (r *multiCIDRRangeAllocator) createClusterCIDR(clusterCIDR *networkingv1alpha1.ClusterCIDR, terminating bool) error {
	nodeSelector, err := r.nodeSelectorKey(clusterCIDR)
	if err != nil {
		return fmt.Errorf("unable to get labelSelector key: %w", err)
	}

	clusterCIDRSet, err := r.createClusterCIDRSet(clusterCIDR, terminating)
	if err != nil {
		return fmt.Errorf("invalid ClusterCIDR: %w", err)
	}

	if clusterCIDRSet.IPv4CIDRSet == nil && clusterCIDRSet.IPv6CIDRSet == nil {
		return errors.New("invalid ClusterCIDR: must provide IPv4 and/or IPv6 config")
	}

	if err := r.mapClusterCIDRSet(r.cidrMap, nodeSelector, clusterCIDRSet); err != nil {
		return fmt.Errorf("unable to map clusterCIDRSet: %w", err)
	}

	// Make a copy so we don't mutate the shared informer cache.
	updatedClusterCIDR := clusterCIDR.DeepCopy()
	if needToAddFinalizer(clusterCIDR, clusterCIDRFinalizer) {
		updatedClusterCIDR.ObjectMeta.Finalizers = append(clusterCIDR.ObjectMeta.Finalizers, clusterCIDRFinalizer)
	}

	if updatedClusterCIDR.ResourceVersion == "" {
		// Create is only used for creating default ClusterCIDR.
		if _, err := r.client.NetworkingV1alpha1().ClusterCIDRs().Create(context.TODO(), updatedClusterCIDR, metav1.CreateOptions{}); err != nil {
			klog.V(2).Infof("Error creating ClusterCIDR %s: %v", clusterCIDR.Name, err)
			return err
		}
	} else {
		// Update the ClusterCIDR object when called from reconcileCreate.
		if _, err := r.client.NetworkingV1alpha1().ClusterCIDRs().Update(context.TODO(), updatedClusterCIDR, metav1.UpdateOptions{}); err != nil {
			klog.V(2).Infof("Error creating ClusterCIDR %s: %v", clusterCIDR.Name, err)
			return err
		}
	}

	return nil
}

// createClusterCIDRSet creates and returns new cidrset.ClusterCIDR based on ClusterCIDR API object.
func (r *multiCIDRRangeAllocator) createClusterCIDRSet(clusterCIDR *networkingv1alpha1.ClusterCIDR, terminating bool) (*cidrset.ClusterCIDR, error) {

	clusterCIDRSet := &cidrset.ClusterCIDR{
		Name:            clusterCIDR.Name,
		AssociatedNodes: make(map[string]bool, 0),
		Terminating:     terminating,
	}

	if clusterCIDR.Spec.IPv4 != "" {
		_, ipv4CIDR, err := netutil.ParseCIDRSloppy(clusterCIDR.Spec.IPv4)
		if err != nil {
			return nil, fmt.Errorf("unable to parse provided IPv4 CIDR: %w", err)
		}
		clusterCIDRSet.IPv4CIDRSet, err = cidrset.NewMultiCIDRSet(ipv4CIDR, int(clusterCIDR.Spec.PerNodeHostBits))
		if err != nil {
			return nil, fmt.Errorf("unable to create IPv4 cidrSet: %w", err)
		}
	}

	if clusterCIDR.Spec.IPv6 != "" {
		_, ipv6CIDR, err := netutil.ParseCIDRSloppy(clusterCIDR.Spec.IPv6)
		if err != nil {
			return nil, fmt.Errorf("unable to parse provided IPv6 CIDR: %w", err)
		}
		clusterCIDRSet.IPv6CIDRSet, err = cidrset.NewMultiCIDRSet(ipv6CIDR, int(clusterCIDR.Spec.PerNodeHostBits))
		if err != nil {
			return nil, fmt.Errorf("unable to create IPv6 cidrSet: %w", err)
		}
	}

	return clusterCIDRSet, nil
}

// mapClusterCIDRSet maps the ClusterCIDRSet to the provided labelSelector in the cidrMap.
func (r *multiCIDRRangeAllocator) mapClusterCIDRSet(cidrMap map[string][]*cidrset.ClusterCIDR, nodeSelector string, clusterCIDRSet *cidrset.ClusterCIDR) error {
	if clusterCIDRSet == nil {
		return errors.New("invalid clusterCIDRSet, clusterCIDRSet cannot be nil")
	}

	if clusterCIDRSetList, ok := cidrMap[nodeSelector]; ok {
		cidrMap[nodeSelector] = append(clusterCIDRSetList, clusterCIDRSet)
	} else {
		cidrMap[nodeSelector] = []*cidrset.ClusterCIDR{clusterCIDRSet}
	}
	return nil
}

// reconcileDelete deletes the ClusterCIDR object and removes the finalizer.
func (r *multiCIDRRangeAllocator) reconcileDelete(clusterCIDR *networkingv1alpha1.ClusterCIDR) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	if slice.ContainsString(clusterCIDR.GetFinalizers(), clusterCIDRFinalizer, nil) {
		if err := r.removeClusterCIDR(clusterCIDR); err != nil {
			return err
		}
		// Remove the finalizer as delete is successful.
		cccCopy := clusterCIDR.DeepCopy()
		cccCopy.ObjectMeta.Finalizers = slice.RemoveString(cccCopy.ObjectMeta.Finalizers, clusterCIDRFinalizer, nil)
		if _, err := r.client.NetworkingV1alpha1().ClusterCIDRs().Update(context.TODO(), clusterCIDR, metav1.UpdateOptions{}); err != nil {
			klog.V(2).Infof("Error removing finalizer for ClusterCIDR %s: %v", clusterCIDR.Name, err)
			return err
		}
		klog.V(2).Infof("Removed finalizer for ClusterCIDR %s", clusterCIDR.Name)
	}
	return nil
}

// deleteClusterCIDR Deletes and unmaps the ClusterCIDRs from the cidrMap.
func (r *multiCIDRRangeAllocator) removeClusterCIDR(clusterCIDR *networkingv1alpha1.ClusterCIDR) error {

	labelSelector, err := r.nodeSelectorKey(clusterCIDR)
	if err != nil {
		return fmt.Errorf("unable to delete cidr: %w", err)
	}

	clusterCIDRSetList, ok := r.cidrMap[labelSelector]
	if !ok {
		klog.Infof("Label %s not found in CIDRMap, proceeding with delete", labelSelector)
		return nil
	}

	for i, clusterCIDRSet := range clusterCIDRSetList {
		if clusterCIDRSet.Name != clusterCIDR.Name {
			continue
		}

		// Mark clusterCIDRSet as terminating.
		clusterCIDRSet.Terminating = true

		// Allow deletion only if no nodes are associated with the ClusterCIDR.
		if len(clusterCIDRSet.AssociatedNodes) > 0 {
			return fmt.Errorf("ClusterCIDRSet %s marked as terminating, won't be deleted until all associated nodes are deleted", clusterCIDR.Name)
		}

		// Remove the label from the map if this was the only clusterCIDR associated
		// with it.
		if len(clusterCIDRSetList) == 1 {
			delete(r.cidrMap, labelSelector)
			return nil
		}

		clusterCIDRSetList = append(clusterCIDRSetList[:i], clusterCIDRSetList[i+1:]...)
		r.cidrMap[labelSelector] = clusterCIDRSetList
		return nil
	}
	klog.V(2).Info("clusterCIDR not found, proceeding with delete", "Name", clusterCIDR.Name, "label", labelSelector)
	return nil
}

func (r *multiCIDRRangeAllocator) nodeSelectorKey(clusterCIDR *networkingv1alpha1.ClusterCIDR) (string, error) {
	var nodeSelector []byte
	var err error

	if clusterCIDR.Spec.NodeSelector != nil {
		nodeSelector, err = clusterCIDR.Spec.NodeSelector.Marshal()
	} else {
		nodeSelector, err = defaultNodeSelector()
	}

	if err != nil {
		return "", err
	}

	return string(nodeSelector), nil
}

//

func (r *multiCIDRRangeAllocator) addNode(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}
	r.nodeQueue.Add(key)
}

func (r *multiCIDRRangeAllocator) updateNode(old, cur interface{}) {
	// TODO: Check if node has been already assigned a PodCIDR by this controller
	// if not, queueu to sync current status
	// node := cur.(*v1.Node)
	key, err := controller.KeyFunc(cur)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", cur, err))
		return
	}
	r.nodeQueue.Add(key)

}

func (r *multiCIDRRangeAllocator) deleteNode(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}
	r.nodeQueue.Add(key)
}

func (r *multiCIDRRangeAllocator) addClusterCIDR(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}
	r.cccQueue.Add(key)
}

func (r *multiCIDRRangeAllocator) updateClusterCIDR(old, cur interface{}) {
	// TODO: Check if node has been already assigned a PodCIDR by this controller
	// if not, queueu to sync current status
	// node := cur.(*v1.Node)
	key, err := controller.KeyFunc(cur)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", cur, err))
		return
	}
	r.cccQueue.Add(key)

}

func (r *multiCIDRRangeAllocator) deleteClusterCIDR(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}
	r.cccQueue.Add(key)
}
