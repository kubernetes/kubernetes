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
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
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
	nodeutil "k8s.io/component-helpers/node/util"
	"k8s.io/klog/v2"
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
)

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
	// Channel that is used to pass updating Nodes and their reserved CIDRs to the background.
	// This increases a throughput of CIDR assignment by not blocking on long operations.
	nodeCIDRUpdateChannel chan multiCIDRNodeReservedCIDRs
	broadcaster           record.EventBroadcaster
	recorder              record.EventRecorder
	// queues are where incoming work is placed to de-dup and to allow "easy"
	// rate limited requeues on errors
	cidrQueue workqueue.RateLimitingInterface
	nodeQueue workqueue.RateLimitingInterface

	// lock guards cidrMap to avoid races in CIDR allocation.
	lock *sync.Mutex
	// cidrMap maps ClusterCIDR labels to internal ClusterCIDR objects.
	cidrMap map[string][]*cidrset.ClusterCIDR
}

// NewMultiCIDRRangeAllocator returns a CIDRAllocator to allocate CIDRs for node (one for each ip family).
// Caller must always pass in a list of existing nodes to the new allocator.
// NodeList is only nil in testing.
func NewMultiCIDRRangeAllocator(
	ctx context.Context,
	client clientset.Interface,
	nodeInformer informers.NodeInformer,
	clusterCIDRInformer networkinginformers.ClusterCIDRInformer,
	allocatorParams CIDRAllocatorParams,
	nodeList *v1.NodeList,
	testCIDRMap map[string][]*cidrset.ClusterCIDR,
) (CIDRAllocator, error) {
	logger := klog.FromContext(ctx)
	if client == nil {
		logger.Error(nil, "kubeClient is nil when starting multi CIDRRangeAllocator")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}

	eventBroadcaster := record.NewBroadcaster()
	eventSource := v1.EventSource{
		Component: "multiCIDRRangeAllocator",
	}
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, eventSource)

	ra := &multiCIDRRangeAllocator{
		client:                client,
		nodeLister:            nodeInformer.Lister(),
		nodesSynced:           nodeInformer.Informer().HasSynced,
		clusterCIDRLister:     clusterCIDRInformer.Lister(),
		clusterCIDRSynced:     clusterCIDRInformer.Informer().HasSynced,
		nodeCIDRUpdateChannel: make(chan multiCIDRNodeReservedCIDRs, cidrUpdateQueueSize),
		broadcaster:           eventBroadcaster,
		recorder:              recorder,
		cidrQueue:             workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "multi_cidr_range_allocator_cidr"),
		nodeQueue:             workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "multi_cidr_range_allocator_node"),
		lock:                  &sync.Mutex{},
		cidrMap:               make(map[string][]*cidrset.ClusterCIDR, 0),
	}

	// testCIDRMap is only set for testing purposes.
	if len(testCIDRMap) > 0 {
		ra.cidrMap = testCIDRMap
		logger.Info("TestCIDRMap should only be set for testing purposes, if this is seen in production logs, it might be a misconfiguration or a bug")
	}

	ccList, err := listClusterCIDRs(ctx, client)
	if err != nil {
		return nil, err
	}

	if ccList == nil {
		ccList = &networkingv1alpha1.ClusterCIDRList{}
	}
	createDefaultClusterCIDR(logger, ccList, allocatorParams)

	// Regenerate the cidrMaps from the existing ClusterCIDRs.
	for _, clusterCIDR := range ccList.Items {
		logger.Info("Regenerating existing ClusterCIDR", "clusterCIDR", clusterCIDR)
		// Create an event for invalid ClusterCIDRs, do not crash on failures.
		if err := ra.reconcileBootstrap(ctx, &clusterCIDR); err != nil {
			logger.Error(err, "Error while regenerating existing ClusterCIDR")
			ra.recorder.Event(&clusterCIDR, "Warning", "InvalidClusterCIDR encountered while regenerating ClusterCIDR during bootstrap.", err.Error())
		}
	}

	clusterCIDRInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(obj)
			if err == nil {
				ra.cidrQueue.Add(key)
			}
		},
		UpdateFunc: func(old, new interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(new)
			if err == nil {
				ra.cidrQueue.Add(key)
			}
		},
		DeleteFunc: func(obj interface{}) {
			// IndexerInformer uses a delta nodeQueue, therefore for deletes we have to use this
			// key function.
			key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
			if err == nil {
				ra.cidrQueue.Add(key)
			}
		},
	})

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
			logger.Info("Node has CIDR, occupying it in CIDR map", "node", klog.KObj(&node), "podCIDRs", node.Spec.PodCIDRs)
			if err := ra.occupyCIDRs(logger, &node); err != nil {
				// This will happen if:
				// 1. We find garbage in the podCIDRs field. Retrying is useless.
				// 2. CIDR out of range: This means ClusterCIDR is not yet created
				// This error will keep crashing controller-manager until the
				// appropriate ClusterCIDR has been created
				return nil, err
			}
		}
	}

	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(obj)
			if err == nil {
				ra.nodeQueue.Add(key)
			}
		},
		UpdateFunc: func(old, new interface{}) {
			key, err := cache.MetaNamespaceKeyFunc(new)
			if err == nil {
				ra.nodeQueue.Add(key)
			}
		},
		DeleteFunc: func(obj interface{}) {
			// The informer cache no longer has the object, and since Node doesn't have a finalizer,
			// we don't see the Update with DeletionTimestamp != 0.
			// TODO: instead of executing the operation directly in the handler, build a small cache with key node.Name
			// and value PodCIDRs use ReleaseCIDR on the reconcile loop so we can retry on `ReleaseCIDR` failures.
			ra.ReleaseCIDR(logger, obj.(*v1.Node))
			// IndexerInformer uses a delta nodeQueue, therefore for deletes we have to use this
			// key function.
			key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
			if err == nil {
				ra.nodeQueue.Add(key)
			}
		},
	})

	return ra, nil
}

func (r *multiCIDRRangeAllocator) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()

	// Start event processing pipeline.
	logger := klog.FromContext(ctx)
	r.broadcaster.StartStructuredLogging(0)
	logger.Info("Started sending events to API Server")
	r.broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: r.client.CoreV1().Events("")})
	defer r.broadcaster.Shutdown()

	defer r.cidrQueue.ShutDown()
	defer r.nodeQueue.ShutDown()

	logger.Info("Starting Multi CIDR Range allocator")
	defer logger.Info("Shutting down Multi CIDR Range allocator")

	if !cache.WaitForNamedCacheSync("multi_cidr_range_allocator", ctx.Done(), r.nodesSynced, r.clusterCIDRSynced) {
		return
	}

	for i := 0; i < cidrUpdateWorkers; i++ {
		go wait.UntilWithContext(ctx, r.runCIDRWorker, time.Second)
		go wait.UntilWithContext(ctx, r.runNodeWorker, time.Second)
	}

	<-ctx.Done()
}

// runWorker is a long-running function that will continually call the
// processNextWorkItem function in order to read and process a message on the
// cidrQueue.
func (r *multiCIDRRangeAllocator) runCIDRWorker(ctx context.Context) {
	for r.processNextCIDRWorkItem(ctx) {
	}
}

// processNextWorkItem will read a single work item off the cidrQueue and
// attempt to process it, by calling the syncHandler.
func (r *multiCIDRRangeAllocator) processNextCIDRWorkItem(ctx context.Context) bool {
	obj, shutdown := r.cidrQueue.Get()
	if shutdown {
		return false
	}

	// We wrap this block in a func so we can defer c.cidrQueue.Done.
	err := func(ctx context.Context, obj interface{}) error {
		// We call Done here so the cidrQueue knows we have finished
		// processing this item. We also must remember to call Forget if we
		// do not want this work item being re-queued. For example, we do
		// not call Forget if a transient error occurs, instead the item is
		// put back on the cidrQueue and attempted again after a back-off
		// period.
		defer r.cidrQueue.Done(obj)
		var key string
		var ok bool
		// We expect strings to come off the cidrQueue. These are of the
		// form namespace/name. We do this as the delayed nature of the
		// cidrQueue means the items in the informer cache may actually be
		// more up to date that when the item was initially put onto the
		// cidrQueue.
		if key, ok = obj.(string); !ok {
			// As the item in the cidrQueue is actually invalid, we call
			// Forget here else we'd go into a loop of attempting to
			// process a work item that is invalid.
			r.cidrQueue.Forget(obj)
			utilruntime.HandleError(fmt.Errorf("expected string in cidrQueue but got %#v", obj))
			return nil
		}
		// Run the syncHandler, passing it the namespace/name string of the
		// Foo resource to be synced.
		if err := r.syncClusterCIDR(ctx, key); err != nil {
			// Put the item back on the cidrQueue to handle any transient errors.
			r.cidrQueue.AddRateLimited(key)
			return fmt.Errorf("error syncing '%s': %s, requeuing", key, err.Error())
		}
		// Finally, if no error occurs we Forget this item so it does not
		// get cidrQueued again until another change happens.
		r.cidrQueue.Forget(obj)
		klog.Infof("Successfully synced '%s'", key)
		return nil
	}(ctx, obj)

	if err != nil {
		utilruntime.HandleError(err)
		return true
	}

	return true
}

func (r *multiCIDRRangeAllocator) runNodeWorker(ctx context.Context) {
	for r.processNextNodeWorkItem(ctx) {
	}
}

// processNextWorkItem will read a single work item off the cidrQueue and
// attempt to process it, by calling the syncHandler.
func (r *multiCIDRRangeAllocator) processNextNodeWorkItem(ctx context.Context) bool {
	obj, shutdown := r.nodeQueue.Get()
	if shutdown {
		return false
	}

	// We wrap this block in a func so we can defer c.cidrQueue.Done.
	err := func(logger klog.Logger, obj interface{}) error {
		// We call Done here so the workNodeQueue knows we have finished
		// processing this item. We also must remember to call Forget if we
		// do not want this work item being re-queued. For example, we do
		// not call Forget if a transient error occurs, instead the item is
		// put back on the nodeQueue and attempted again after a back-off
		// period.
		defer r.nodeQueue.Done(obj)
		var key string
		var ok bool
		// We expect strings to come off the workNodeQueue. These are of the
		// form namespace/name. We do this as the delayed nature of the
		// workNodeQueue means the items in the informer cache may actually be
		// more up to date that when the item was initially put onto the
		// workNodeQueue.
		if key, ok = obj.(string); !ok {
			// As the item in the workNodeQueue is actually invalid, we call
			// Forget here else we'd go into a loop of attempting to
			// process a work item that is invalid.
			r.nodeQueue.Forget(obj)
			utilruntime.HandleError(fmt.Errorf("expected string in workNodeQueue but got %#v", obj))
			return nil
		}
		// Run the syncHandler, passing it the namespace/name string of the
		// Foo resource to be synced.
		if err := r.syncNode(logger, key); err != nil {
			// Put the item back on the cidrQueue to handle any transient errors.
			r.nodeQueue.AddRateLimited(key)
			return fmt.Errorf("error syncing '%s': %s, requeuing", key, err.Error())
		}
		// Finally, if no error occurs we Forget this item so it does not
		// get nodeQueue again until another change happens.
		r.nodeQueue.Forget(obj)
		klog.Infof("Successfully synced '%s'", key)
		return nil
	}(klog.FromContext(ctx), obj)

	if err != nil {
		utilruntime.HandleError(err)
		return true
	}

	return true
}

func (r *multiCIDRRangeAllocator) syncNode(logger klog.Logger, key string) error {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing Node request %q (%v)", key, time.Since(startTime))
	}()

	node, err := r.nodeLister.Get(key)
	if apierrors.IsNotFound(err) {
		klog.V(3).Infof("node has been deleted: %v", key)
		// TODO: obtain the node object information to call ReleaseCIDR from here
		// and retry if there is an error.
		return nil
	}
	if err != nil {
		return err
	}
	// Check the DeletionTimestamp to determine if object is under deletion.
	if !node.DeletionTimestamp.IsZero() {
		klog.V(3).Infof("node is being deleted: %v", key)
		return r.ReleaseCIDR(logger, node)
	}
	return r.AllocateOrOccupyCIDR(logger, node)
}

// needToAddFinalizer checks if a finalizer should be added to the object.
func needToAddFinalizer(obj metav1.Object, finalizer string) bool {
	return obj.GetDeletionTimestamp() == nil && !slice.ContainsString(obj.GetFinalizers(),
		finalizer, nil)
}

func (r *multiCIDRRangeAllocator) syncClusterCIDR(ctx context.Context, key string) error {
	startTime := time.Now()
	logger := klog.FromContext(ctx)
	defer func() {
		logger.V(4).Info("Finished syncing clusterCIDR request", "key", key, "latency", time.Since(startTime))
	}()

	clusterCIDR, err := r.clusterCIDRLister.Get(key)
	if apierrors.IsNotFound(err) {
		logger.V(3).Info("clusterCIDR has been deleted", "key", key)
		return nil
	}

	if err != nil {
		return err
	}

	// Check the DeletionTimestamp to determine if object is under deletion.
	if !clusterCIDR.DeletionTimestamp.IsZero() {
		return r.reconcileDelete(ctx, clusterCIDR)
	}
	return r.reconcileCreate(ctx, clusterCIDR)
}

// occupyCIDRs marks node.PodCIDRs[...] as used in allocator's tracked cidrSet.
func (r *multiCIDRRangeAllocator) occupyCIDRs(logger klog.Logger, node *v1.Node) error {

	err := func(node *v1.Node) error {

		if len(node.Spec.PodCIDRs) == 0 {
			return nil
		}
		clusterCIDRList, err := r.orderedMatchingClusterCIDRs(logger, node, true)
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

				logger.Info("occupy CIDR for node", "CIDR", cidr, "node", klog.KObj(node))

				if err := r.Occupy(clusterCIDR, podCIDR); err != nil {
					logger.V(3).Info("Could not occupy cidr, trying next range", "podCIDRs", node.Spec.PodCIDRs, "err", err)
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
func (r *multiCIDRRangeAllocator) Release(logger klog.Logger, clusterCIDR *cidrset.ClusterCIDR, cidr *net.IPNet) error {
	currCIDRSet, err := r.associatedCIDRSet(clusterCIDR, cidr)
	if err != nil {
		return err
	}

	if err := currCIDRSet.Release(cidr); err != nil {
		logger.Info("Unable to release cidr in cidrSet", "CIDR", cidr)
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
func (r *multiCIDRRangeAllocator) AllocateOrOccupyCIDR(logger klog.Logger, node *v1.Node) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	if node == nil {
		return nil
	}

	if len(node.Spec.PodCIDRs) > 0 {
		return r.occupyCIDRs(logger, node)
	}

	cidrs, clusterCIDR, err := r.prioritizedCIDRs(logger, node)
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

	return r.updateCIDRsAllocation(logger, allocated)
}

// ReleaseCIDR marks node.podCIDRs[...] as unused in our tracked cidrSets.
func (r *multiCIDRRangeAllocator) ReleaseCIDR(logger klog.Logger, node *v1.Node) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	if node == nil || len(node.Spec.PodCIDRs) == 0 {
		return nil
	}

	clusterCIDR, err := r.allocatedClusterCIDR(logger, node)
	if err != nil {
		return err
	}

	for _, cidr := range node.Spec.PodCIDRs {
		_, podCIDR, err := netutil.ParseCIDRSloppy(cidr)
		if err != nil {
			return fmt.Errorf("failed to parse CIDR %q on Node %q: %w", cidr, node.Name, err)
		}

		logger.Info("release CIDR for node", "CIDR", cidr, "node", klog.KObj(node))
		if err := r.Release(logger, clusterCIDR, podCIDR); err != nil {
			return fmt.Errorf("failed to release cidr %q from clusterCIDR %q for node %q: %w", cidr, clusterCIDR.Name, node.Name, err)
		}
	}

	// Remove the node from the ClusterCIDR AssociatedNodes.
	delete(clusterCIDR.AssociatedNodes, node.Name)

	return nil
}

// Marks all CIDRs with subNetMaskSize that belongs to serviceCIDR as used across all cidrs
// so that they won't be assignable.
func (r *multiCIDRRangeAllocator) filterOutServiceRange(logger klog.Logger, serviceCIDR *net.IPNet) {
	// Checks if service CIDR has a nonempty intersection with cluster
	// CIDR. It is the case if either clusterCIDR contains serviceCIDR with
	// clusterCIDR's Mask applied (this means that clusterCIDR contains
	// serviceCIDR) or vice versa (which means that serviceCIDR contains
	// clusterCIDR).
	for _, clusterCIDRList := range r.cidrMap {
		for _, clusterCIDR := range clusterCIDRList {
			if err := r.occupyServiceCIDR(clusterCIDR, serviceCIDR); err != nil {
				logger.Error(err, "Unable to occupy service CIDR")
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
func (r *multiCIDRRangeAllocator) updateCIDRsAllocation(logger klog.Logger, data multiCIDRNodeReservedCIDRs) error {
	err := func(data multiCIDRNodeReservedCIDRs) error {
		cidrsString := ipnetToStringList(data.allocatedCIDRs)
		node, err := r.nodeLister.Get(data.nodeName)
		if err != nil {
			logger.Error(err, "Failed while getting node for updating Node.Spec.PodCIDRs", "node", klog.KRef("", data.nodeName))
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
				logger.V(4).Info("Node already has allocated CIDR. It matches the proposed one.", "node", klog.KObj(node), "CIDRs", data.allocatedCIDRs)
				return nil
			}
		}

		// node has cidrs allocated, release the reserved.
		if len(node.Spec.PodCIDRs) != 0 {
			logger.Error(nil, "Node already has a CIDR allocated. Releasing the new one", "node", klog.KObj(node), "podCIDRs", node.Spec.PodCIDRs)
			for _, cidr := range data.allocatedCIDRs {
				if err := r.Release(logger, data.clusterCIDR, cidr); err != nil {
					return fmt.Errorf("failed to release cidr %s from clusterCIDR %s for node: %s: %w", cidr, data.clusterCIDR.Name, node.Name, err)
				}
			}
			return nil
		}

		// If we reached here, it means that the node has no CIDR currently assigned. So we set it.
		for i := 0; i < cidrUpdateRetries; i++ {
			if err = nodeutil.PatchNodeCIDRs(r.client, types.NodeName(node.Name), cidrsString); err == nil {
				data.clusterCIDR.AssociatedNodes[node.Name] = true
				logger.Info("Set node PodCIDR", "node", klog.KObj(node), "podCIDR", cidrsString)
				return nil
			}
		}
		// failed release back to the pool.
		logger.Error(err, "Failed to update node PodCIDR after attempts", "node", klog.KObj(node), "podCIDR", cidrsString, "retries", cidrUpdateRetries)
		controllerutil.RecordNodeStatusChange(r.recorder, node, "CIDRAssignmentFailed")
		// We accept the fact that we may leak CIDRs here. This is safer than releasing
		// them in case when we don't know if request went through.
		// NodeController restart will return all falsely allocated CIDRs to the pool.
		if !apierrors.IsServerTimeout(err) {
			logger.Error(err, "CIDR assignment for node failed. Releasing allocated CIDR", "node", klog.KObj(node))
			for _, cidr := range data.allocatedCIDRs {
				if err := r.Release(logger, data.clusterCIDR, cidr); err != nil {
					return fmt.Errorf("failed to release cidr %q from clusterCIDR %q for node: %q: %w", cidr, data.clusterCIDR.Name, node.Name, err)
				}
			}
		}
		return err
	}(data)

	return err
}

// defaultNodeSelector generates a label with defaultClusterCIDRKey as the key and
// defaultClusterCIDRValue as the value, it is an internal nodeSelector matching all
// nodes. Only used if no ClusterCIDR selects the node.
func defaultNodeSelector() *v1.NodeSelector {
	return &v1.NodeSelector{
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
}

// prioritizedCIDRs returns a list of CIDRs to be allocated to the node.
// Returns 1 CIDR  if single stack.
// Returns 2 CIDRs , 1 from each ip family if dual stack.
func (r *multiCIDRRangeAllocator) prioritizedCIDRs(logger klog.Logger, node *v1.Node) ([]*net.IPNet, *cidrset.ClusterCIDR, error) {
	clusterCIDRList, err := r.orderedMatchingClusterCIDRs(logger, node, true)
	if err != nil {
		return nil, nil, fmt.Errorf("unable to get a clusterCIDR for node %s: %w", node.Name, err)
	}

	for _, clusterCIDR := range clusterCIDRList {
		cidrs := make([]*net.IPNet, 0)
		if clusterCIDR.IPv4CIDRSet != nil {
			cidr, err := r.allocateCIDR(clusterCIDR, clusterCIDR.IPv4CIDRSet)
			if err != nil {
				logger.V(3).Info("Unable to allocate IPv4 CIDR, trying next range", "err", err)
				continue
			}
			cidrs = append(cidrs, cidr)
		}

		if clusterCIDR.IPv6CIDRSet != nil {
			cidr, err := r.allocateCIDR(clusterCIDR, clusterCIDR.IPv6CIDRSet)
			if err != nil {
				logger.V(3).Info("Unable to allocate IPv6 CIDR, trying next range", "err", err)
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
func (r *multiCIDRRangeAllocator) allocatedClusterCIDR(logger klog.Logger, node *v1.Node) (*cidrset.ClusterCIDR, error) {
	clusterCIDRList, err := r.orderedMatchingClusterCIDRs(logger, node, false)
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
//
// orderedMatchingClusterCIDRs takes `occupy` as an argument, it determines whether the function
// is called during an occupy or a release operation. For a release operation, a ClusterCIDR must
// be added to the matching ClusterCIDRs list, irrespective of whether the ClusterCIDR is terminating.
func (r *multiCIDRRangeAllocator) orderedMatchingClusterCIDRs(logger klog.Logger, node *v1.Node, occupy bool) ([]*cidrset.ClusterCIDR, error) {
	matchingCIDRs := make([]*cidrset.ClusterCIDR, 0)
	pq := make(PriorityQueue, 0)

	for label, clusterCIDRList := range r.cidrMap {
		labelsMatch, matchCnt, err := r.matchCIDRLabels(logger, node, label)
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
			// Always push the CIDRsets when marked for release.
			if !occupy || !clusterCIDR.Terminating {
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
	defaultSelector, err := nodeSelectorAsSelector(defaultNodeSelector())
	if err != nil {
		return nil, err
	}
	if clusterCIDRList, ok := r.cidrMap[defaultSelector.String()]; ok {
		matchingCIDRs = append(matchingCIDRs, clusterCIDRList...)
	}
	return matchingCIDRs, nil
}

// matchCIDRLabels Matches the Node labels to CIDR Configs.
// Returns true only if all the labels match, also returns the count of matching labels.
func (r *multiCIDRRangeAllocator) matchCIDRLabels(logger klog.Logger, node *v1.Node, label string) (bool, int, error) {
	var labelSet labels.Set
	var matchCnt int
	labelsMatch := false

	ls, err := labels.Parse(label)
	if err != nil {
		logger.Error(err, "Unable to parse label to labels.Selector", "label", label)
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
	return labelsMatch, matchCnt, nil
}

// Methods for handling ClusterCIDRs.

// createDefaultClusterCIDR creates a default ClusterCIDR if --cluster-cidr has
// been configured. It converts the --cluster-cidr and --per-node-mask-size* flags
// to appropriate ClusterCIDR fields.
func createDefaultClusterCIDR(logger klog.Logger, existingConfigList *networkingv1alpha1.ClusterCIDRList,
	allocatorParams CIDRAllocatorParams) {
	// Create default ClusterCIDR only if --cluster-cidr has been configured
	if len(allocatorParams.ClusterCIDRs) == 0 {
		return
	}

	for _, clusterCIDR := range existingConfigList.Items {
		if clusterCIDR.Name == defaultClusterCIDRName {
			// Default ClusterCIDR already exists, no further action required.
			logger.V(3).Info("Default ClusterCIDR already exists", "defaultClusterCIDRName", defaultClusterCIDRName)
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
func (r *multiCIDRRangeAllocator) reconcileCreate(ctx context.Context, clusterCIDR *networkingv1alpha1.ClusterCIDR) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	logger := klog.FromContext(ctx)
	if needToAddFinalizer(clusterCIDR, clusterCIDRFinalizer) {
		logger.V(3).Info("Creating ClusterCIDR", "clusterCIDR", clusterCIDR.Name)
		if err := r.createClusterCIDR(ctx, clusterCIDR, false); err != nil {
			logger.Error(err, "Unable to create ClusterCIDR", "clusterCIDR", clusterCIDR.Name)
			return err
		}
	}
	return nil
}

// reconcileBootstrap handles creation of existing ClusterCIDRs.
// adds a finalizer if not already present.
func (r *multiCIDRRangeAllocator) reconcileBootstrap(ctx context.Context, clusterCIDR *networkingv1alpha1.ClusterCIDR) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	logger := klog.FromContext(ctx)
	terminating := false
	// Create the ClusterCIDR only if the Spec has not been modified.
	if clusterCIDR.Generation > 1 {
		terminating = true
		err := fmt.Errorf("CIDRs from ClusterCIDR %s will not be used for allocation as it was modified", clusterCIDR.Name)
		logger.Error(err, "ClusterCIDR Modified")
	}

	logger.V(2).Info("Creating ClusterCIDR during bootstrap", "clusterCIDR", clusterCIDR.Name)
	if err := r.createClusterCIDR(ctx, clusterCIDR, terminating); err != nil {
		logger.Error(err, "Unable to create ClusterCIDR", "clusterCIDR", clusterCIDR.Name)
		return err
	}

	return nil
}

// createClusterCIDR creates and maps the cidrSets in the cidrMap.
func (r *multiCIDRRangeAllocator) createClusterCIDR(ctx context.Context, clusterCIDR *networkingv1alpha1.ClusterCIDR, terminating bool) error {
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

	logger := klog.FromContext(ctx)
	if updatedClusterCIDR.ResourceVersion == "" {
		// Create is only used for creating default ClusterCIDR.
		if _, err := r.client.NetworkingV1alpha1().ClusterCIDRs().Create(ctx, updatedClusterCIDR, metav1.CreateOptions{}); err != nil {
			logger.V(2).Info("Error creating ClusterCIDR", "clusterCIDR", klog.KObj(clusterCIDR), "err", err)
			return err
		}
	} else {
		// Update the ClusterCIDR object when called from reconcileCreate.
		if _, err := r.client.NetworkingV1alpha1().ClusterCIDRs().Update(ctx, updatedClusterCIDR, metav1.UpdateOptions{}); err != nil {
			logger.V(2).Info("Error creating ClusterCIDR", "clusterCIDR", clusterCIDR.Name, "err", err)
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

// reconcileDelete releases the assigned ClusterCIDR and removes the finalizer
// if the deletion timestamp is set.
func (r *multiCIDRRangeAllocator) reconcileDelete(ctx context.Context, clusterCIDR *networkingv1alpha1.ClusterCIDR) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	logger := klog.FromContext(ctx)
	if slice.ContainsString(clusterCIDR.GetFinalizers(), clusterCIDRFinalizer, nil) {
		logger.V(2).Info("Releasing ClusterCIDR", "clusterCIDR", clusterCIDR.Name)
		if err := r.deleteClusterCIDR(logger, clusterCIDR); err != nil {
			klog.V(2).Info("Error while deleting ClusterCIDR", "err", err)
			return err
		}
		// Remove the finalizer as delete is successful.
		cccCopy := clusterCIDR.DeepCopy()
		cccCopy.ObjectMeta.Finalizers = slice.RemoveString(cccCopy.ObjectMeta.Finalizers, clusterCIDRFinalizer, nil)
		if _, err := r.client.NetworkingV1alpha1().ClusterCIDRs().Update(ctx, cccCopy, metav1.UpdateOptions{}); err != nil {
			logger.V(2).Info("Error removing finalizer for ClusterCIDR", "clusterCIDR", clusterCIDR.Name, "err", err)
			return err
		}
		logger.V(2).Info("Removed finalizer for ClusterCIDR", "clusterCIDR", clusterCIDR.Name)
	}
	return nil
}

// deleteClusterCIDR Deletes and unmaps the ClusterCIDRs from the cidrMap.
func (r *multiCIDRRangeAllocator) deleteClusterCIDR(logger klog.Logger, clusterCIDR *networkingv1alpha1.ClusterCIDR) error {

	labelSelector, err := r.nodeSelectorKey(clusterCIDR)
	if err != nil {
		return fmt.Errorf("unable to delete cidr: %w", err)
	}

	clusterCIDRSetList, ok := r.cidrMap[labelSelector]
	if !ok {
		logger.Info("Label not found in CIDRMap, proceeding with delete", "labelSelector", labelSelector)
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
	logger.V(2).Info("clusterCIDR not found, proceeding with delete", "clusterCIDR", clusterCIDR.Name, "label", labelSelector)
	return nil
}

func (r *multiCIDRRangeAllocator) nodeSelectorKey(clusterCIDR *networkingv1alpha1.ClusterCIDR) (string, error) {
	var nodeSelector labels.Selector
	var err error

	if clusterCIDR.Spec.NodeSelector != nil {
		nodeSelector, err = nodeSelectorAsSelector(clusterCIDR.Spec.NodeSelector)
	} else {
		nodeSelector, err = nodeSelectorAsSelector(defaultNodeSelector())
	}

	if err != nil {
		return "", err
	}

	return nodeSelector.String(), nil
}

func listClusterCIDRs(ctx context.Context, kubeClient clientset.Interface) (*networkingv1alpha1.ClusterCIDRList, error) {
	var clusterCIDRList *networkingv1alpha1.ClusterCIDRList
	// We must poll because apiserver might not be up. This error causes
	// controller manager to restart.
	startTimestamp := time.Now()

	// start with 2s, multiply the duration by 1.6 each step, 11 steps = 9.7 minutes
	backoff := wait.Backoff{
		Duration: 2 * time.Second,
		Factor:   1.6,
		Steps:    11,
	}

	logger := klog.FromContext(ctx)
	if pollErr := wait.ExponentialBackoff(backoff, func() (bool, error) {
		var err error
		clusterCIDRList, err = kubeClient.NetworkingV1alpha1().ClusterCIDRs().List(ctx, metav1.ListOptions{
			FieldSelector: fields.Everything().String(),
			LabelSelector: labels.Everything().String(),
		})
		if err != nil {
			logger.Error(err, "Failed to list all clusterCIDRs")
			return false, nil
		}
		return true, nil
	}); pollErr != nil {
		logger.Error(nil, "Failed to list clusterCIDRs", "latency", time.Now().Sub(startTimestamp))
		return nil, fmt.Errorf("failed to list all clusterCIDRs in %v, cannot proceed without updating CIDR map",
			apiserverStartupGracePeriod)
	}
	return clusterCIDRList, nil
}

// nodeSelectorRequirementsAsLabelRequirements converts the NodeSelectorRequirement
// type to a labels.Requirement type.
func nodeSelectorRequirementsAsLabelRequirements(nsr v1.NodeSelectorRequirement) (*labels.Requirement, error) {
	var op selection.Operator
	switch nsr.Operator {
	case v1.NodeSelectorOpIn:
		op = selection.In
	case v1.NodeSelectorOpNotIn:
		op = selection.NotIn
	case v1.NodeSelectorOpExists:
		op = selection.Exists
	case v1.NodeSelectorOpDoesNotExist:
		op = selection.DoesNotExist
	case v1.NodeSelectorOpGt:
		op = selection.GreaterThan
	case v1.NodeSelectorOpLt:
		op = selection.LessThan
	default:
		return nil, fmt.Errorf("%q is not a valid node selector operator", nsr.Operator)
	}
	return labels.NewRequirement(nsr.Key, op, nsr.Values)
}

// TODO: nodeSelect and labelSelector semantics are different and the function
// doesn't translate them correctly, this has to be fixed before Beta
// xref: https://issues.k8s.io/116419
// nodeSelectorAsSelector converts the NodeSelector api type into a struct that
// implements labels.Selector
// Note: This function should be kept in sync with the selector methods in
// pkg/labels/selector.go
func nodeSelectorAsSelector(ns *v1.NodeSelector) (labels.Selector, error) {
	if ns == nil {
		return labels.Nothing(), nil
	}
	if len(ns.NodeSelectorTerms) == 0 {
		return labels.Everything(), nil
	}
	var requirements []labels.Requirement

	for _, nsTerm := range ns.NodeSelectorTerms {
		for _, expr := range nsTerm.MatchExpressions {
			req, err := nodeSelectorRequirementsAsLabelRequirements(expr)
			if err != nil {
				return nil, err
			}
			requirements = append(requirements, *req)
		}

		for _, field := range nsTerm.MatchFields {
			req, err := nodeSelectorRequirementsAsLabelRequirements(field)
			if err != nil {
				return nil, err
			}
			requirements = append(requirements, *req)
		}
	}

	selector := labels.NewSelector()
	selector = selector.Add(requirements...)
	return selector, nil
}
