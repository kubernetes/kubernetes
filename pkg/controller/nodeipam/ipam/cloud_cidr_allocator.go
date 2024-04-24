//go:build !providerless
// +build !providerless

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
	"math/rand"
	"net"
	"sync"
	"time"

	"github.com/google/go-cmp/cmp"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	informers "k8s.io/client-go/informers/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	cloudprovider "k8s.io/cloud-provider"
	nodeutil "k8s.io/component-helpers/node/util"
	controllerutil "k8s.io/kubernetes/pkg/controller/util/node"
	utiltaints "k8s.io/kubernetes/pkg/util/taints"
	"k8s.io/legacy-cloud-providers/gce"
	netutils "k8s.io/utils/net"
)

// nodeProcessingInfo tracks information related to current nodes in processing
type nodeProcessingInfo struct {
	retries int
}

// cloudCIDRAllocator allocates node CIDRs according to IP address aliases
// assigned by the cloud provider. In this case, the allocation and
// deallocation is delegated to the external provider, and the controller
// merely takes the assignment and updates the node spec.
type cloudCIDRAllocator struct {
	client clientset.Interface
	cloud  *gce.Cloud

	// nodeLister is able to list/get nodes and is populated by the shared informer passed to
	// NewCloudCIDRAllocator.
	nodeLister corelisters.NodeLister
	// nodesSynced returns true if the node shared informer has been synced at least once.
	nodesSynced cache.InformerSynced

	// Channel that is used to pass updating Nodes to the background.
	// This increases the throughput of CIDR assignment by parallelization
	// and not blocking on long operations (which shouldn't be done from
	// event handlers anyway).
	nodeUpdateChannel chan string
	broadcaster       record.EventBroadcaster
	recorder          record.EventRecorder

	// Keep a set of nodes that are currectly being processed to avoid races in CIDR allocation
	lock              sync.Mutex
	nodesInProcessing map[string]*nodeProcessingInfo
}

var _ CIDRAllocator = (*cloudCIDRAllocator)(nil)

// NewCloudCIDRAllocator creates a new cloud CIDR allocator.
func NewCloudCIDRAllocator(ctx context.Context, client clientset.Interface, cloud cloudprovider.Interface, nodeInformer informers.NodeInformer) (CIDRAllocator, error) {
	logger := klog.FromContext(ctx)
	if client == nil {
		logger.Error(nil, "kubeClient is nil when starting cloud CIDR allocator")
		klog.FlushAndExit(klog.ExitFlushTimeout, 1)
	}

	eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cidrAllocator"})

	gceCloud, ok := cloud.(*gce.Cloud)
	if !ok {
		err := fmt.Errorf("cloudCIDRAllocator does not support %v provider", cloud.ProviderName())
		return nil, err
	}

	ca := &cloudCIDRAllocator{
		client:            client,
		cloud:             gceCloud,
		nodeLister:        nodeInformer.Lister(),
		nodesSynced:       nodeInformer.Informer().HasSynced,
		nodeUpdateChannel: make(chan string, cidrUpdateQueueSize),
		broadcaster:       eventBroadcaster,
		recorder:          recorder,
		nodesInProcessing: map[string]*nodeProcessingInfo{},
	}

	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: controllerutil.CreateAddNodeHandler(
			func(node *v1.Node) error {
				return ca.AllocateOrOccupyCIDR(ctx, node)
			}),
		UpdateFunc: controllerutil.CreateUpdateNodeHandler(func(_, newNode *v1.Node) error {
			if newNode.Spec.PodCIDR == "" {
				return ca.AllocateOrOccupyCIDR(ctx, newNode)
			}
			// Even if PodCIDR is assigned, but NetworkUnavailable condition is
			// set to true, we need to process the node to set the condition.
			networkUnavailableTaint := &v1.Taint{Key: v1.TaintNodeNetworkUnavailable, Effect: v1.TaintEffectNoSchedule}
			_, cond := controllerutil.GetNodeCondition(&newNode.Status, v1.NodeNetworkUnavailable)
			if cond == nil || cond.Status != v1.ConditionFalse || utiltaints.TaintExists(newNode.Spec.Taints, networkUnavailableTaint) {
				return ca.AllocateOrOccupyCIDR(ctx, newNode)
			}
			return nil
		}),
		DeleteFunc: controllerutil.CreateDeleteNodeHandler(logger, func(node *v1.Node) error {
			return ca.ReleaseCIDR(logger, node)
		}),
	})
	logger.Info("Using cloud CIDR allocator", "provider", cloud.ProviderName())
	return ca, nil
}

func (ca *cloudCIDRAllocator) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()

	// Start event processing pipeline.
	ca.broadcaster.StartStructuredLogging(3)
	logger := klog.FromContext(ctx)
	logger.Info("Sending events to api server")
	ca.broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: ca.client.CoreV1().Events("")})
	defer ca.broadcaster.Shutdown()

	logger.Info("Starting cloud CIDR allocator")
	defer logger.Info("Shutting down cloud CIDR allocator")

	if !cache.WaitForNamedCacheSync("cidrallocator", ctx.Done(), ca.nodesSynced) {
		return
	}

	for i := 0; i < cidrUpdateWorkers; i++ {
		go ca.worker(ctx)
	}

	<-ctx.Done()
}

func (ca *cloudCIDRAllocator) worker(ctx context.Context) {
	logger := klog.FromContext(ctx)
	for {
		select {
		case workItem, ok := <-ca.nodeUpdateChannel:
			if !ok {
				logger.Info("Channel nodeCIDRUpdateChannel was unexpectedly closed")
				return
			}
			if err := ca.updateCIDRAllocation(ctx, workItem); err == nil {
				logger.V(3).Info("Updated CIDR", "workItem", workItem)
			} else {
				logger.Error(err, "Error updating CIDR", "workItem", workItem)
				if canRetry, timeout := ca.retryParams(logger, workItem); canRetry {
					logger.V(2).Info("Retrying update on next period", "workItem", workItem, "timeout", timeout)
					time.AfterFunc(timeout, func() {
						// Requeue the failed node for update again.
						ca.nodeUpdateChannel <- workItem
					})
					continue
				}
				logger.Error(nil, "Exceeded retry count, dropping from queue", "workItem", workItem)
			}
			ca.removeNodeFromProcessing(workItem)
		case <-ctx.Done():
			return
		}
	}
}

func (ca *cloudCIDRAllocator) insertNodeToProcessing(nodeName string) bool {
	ca.lock.Lock()
	defer ca.lock.Unlock()
	if _, found := ca.nodesInProcessing[nodeName]; found {
		return false
	}
	ca.nodesInProcessing[nodeName] = &nodeProcessingInfo{}
	return true
}

func (ca *cloudCIDRAllocator) retryParams(logger klog.Logger, nodeName string) (bool, time.Duration) {
	ca.lock.Lock()
	defer ca.lock.Unlock()

	entry, ok := ca.nodesInProcessing[nodeName]
	if !ok {
		logger.Error(nil, "Cannot get retryParams for node as entry does not exist", "node", klog.KRef("", nodeName))
		return false, 0
	}

	count := entry.retries + 1
	if count > updateMaxRetries {
		return false, 0
	}
	ca.nodesInProcessing[nodeName].retries = count

	return true, nodeUpdateRetryTimeout(count)
}

func nodeUpdateRetryTimeout(count int) time.Duration {
	timeout := updateRetryTimeout
	for i := 0; i < count && timeout < maxUpdateRetryTimeout; i++ {
		timeout *= 2
	}
	if timeout > maxUpdateRetryTimeout {
		timeout = maxUpdateRetryTimeout
	}
	return time.Duration(timeout.Nanoseconds()/2 + rand.Int63n(timeout.Nanoseconds()))
}

func (ca *cloudCIDRAllocator) removeNodeFromProcessing(nodeName string) {
	ca.lock.Lock()
	defer ca.lock.Unlock()
	delete(ca.nodesInProcessing, nodeName)
}

// WARNING: If you're adding any return calls or defer any more work from this
// function you have to make sure to update nodesInProcessing properly with the
// disposition of the node when the work is done.
func (ca *cloudCIDRAllocator) AllocateOrOccupyCIDR(ctx context.Context, node *v1.Node) error {
	if node == nil {
		return nil
	}

	logger := klog.FromContext(ctx)
	if !ca.insertNodeToProcessing(node.Name) {
		logger.V(2).Info("Node is already in a process of CIDR assignment", "node", klog.KObj(node))
		return nil
	}

	logger.V(4).Info("Putting node into the work queue", "node", klog.KObj(node))
	ca.nodeUpdateChannel <- node.Name
	return nil
}

// updateCIDRAllocation assigns CIDR to Node and sends an update to the API server.
func (ca *cloudCIDRAllocator) updateCIDRAllocation(ctx context.Context, nodeName string) error {
	logger := klog.FromContext(ctx)
	node, err := ca.nodeLister.Get(nodeName)
	if err != nil {
		if errors.IsNotFound(err) {
			return nil // node no longer available, skip processing
		}
		logger.Error(err, "Failed while getting the node for updating Node.Spec.PodCIDR", "node", klog.KRef("", nodeName))
		return err
	}
	if node.Spec.ProviderID == "" {
		return fmt.Errorf("node %s doesn't have providerID", nodeName)
	}

	cidrStrings, err := ca.cloud.AliasRangesByProviderID(node.Spec.ProviderID)
	if err != nil {
		controllerutil.RecordNodeStatusChange(logger, ca.recorder, node, "CIDRNotAvailable")
		return fmt.Errorf("failed to get cidr(s) from provider: %v", err)
	}
	if len(cidrStrings) == 0 {
		controllerutil.RecordNodeStatusChange(logger, ca.recorder, node, "CIDRNotAvailable")
		return fmt.Errorf("failed to allocate cidr: Node %v has no CIDRs", node.Name)
	}
	//Can have at most 2 ips (one for v4 and one for v6)
	if len(cidrStrings) > 2 {
		logger.Info("Got more than 2 ips, truncating to 2", "cidrStrings", cidrStrings)
		cidrStrings = cidrStrings[:2]
	}

	cidrs, err := netutils.ParseCIDRs(cidrStrings)
	if err != nil {
		return fmt.Errorf("failed to parse strings %v as CIDRs: %v", cidrStrings, err)
	}

	needUpdate, err := needPodCIDRsUpdate(logger, node, cidrs)
	if err != nil {
		return fmt.Errorf("err: %v, CIDRS: %v", err, cidrStrings)
	}
	if needUpdate {
		if node.Spec.PodCIDR != "" {
			logger.Error(nil, "PodCIDR being reassigned", "node", klog.KObj(node), "podCIDRs", node.Spec.PodCIDRs, "cidrStrings", cidrStrings)
			// We fall through and set the CIDR despite this error. This
			// implements the same logic as implemented in the
			// rangeAllocator.
			//
			// See https://github.com/kubernetes/kubernetes/pull/42147#discussion_r103357248
		}
		for i := 0; i < cidrUpdateRetries; i++ {
			if err = nodeutil.PatchNodeCIDRs(ctx, ca.client, types.NodeName(node.Name), cidrStrings); err == nil {
				logger.Info("Set the node PodCIDRs", "node", klog.KObj(node), "cidrStrings", cidrStrings)
				break
			}
		}
	}
	if err != nil {
		controllerutil.RecordNodeStatusChange(logger, ca.recorder, node, "CIDRAssignmentFailed")
		logger.Error(err, "Failed to update the node PodCIDR after multiple attempts", "node", klog.KObj(node), "cidrStrings", cidrStrings)
		return err
	}

	err = nodeutil.SetNodeCondition(ca.client, types.NodeName(node.Name), v1.NodeCondition{
		Type:               v1.NodeNetworkUnavailable,
		Status:             v1.ConditionFalse,
		Reason:             "RouteCreated",
		Message:            "NodeController create implicit route",
		LastTransitionTime: metav1.Now(),
	})
	if err != nil {
		logger.Error(err, "Error setting route status for the node", "node", klog.KObj(node))
	}
	return err
}

func needPodCIDRsUpdate(logger klog.Logger, node *v1.Node, podCIDRs []*net.IPNet) (bool, error) {
	if node.Spec.PodCIDR == "" {
		return true, nil
	}
	_, nodePodCIDR, err := netutils.ParseCIDRSloppy(node.Spec.PodCIDR)
	if err != nil {
		logger.Error(err, "Found invalid node.Spec.PodCIDR", "podCIDR", node.Spec.PodCIDR)
		// We will try to overwrite with new CIDR(s)
		return true, nil
	}
	nodePodCIDRs, err := netutils.ParseCIDRs(node.Spec.PodCIDRs)
	if err != nil {
		logger.Error(err, "Found invalid node.Spec.PodCIDRs", "podCIDRs", node.Spec.PodCIDRs)
		// We will try to overwrite with new CIDR(s)
		return true, nil
	}

	if len(podCIDRs) == 1 {
		if cmp.Equal(nodePodCIDR, podCIDRs[0]) {
			logger.V(4).Info("Node already has allocated CIDR. It matches the proposed one", "node", klog.KObj(node), "podCIDR", podCIDRs[0])
			return false, nil
		}
	} else if len(nodePodCIDRs) == len(podCIDRs) {
		if dualStack, _ := netutils.IsDualStackCIDRs(podCIDRs); !dualStack {
			return false, fmt.Errorf("IPs are not dual stack")
		}
		for idx, cidr := range podCIDRs {
			if !cmp.Equal(nodePodCIDRs[idx], cidr) {
				return true, nil
			}
		}
		logger.V(4).Info("Node already has allocated CIDRs. It matches the proposed one", "node", klog.KObj(node), "podCIDRs", podCIDRs)
		return false, nil
	}

	return true, nil
}

func (ca *cloudCIDRAllocator) ReleaseCIDR(logger klog.Logger, node *v1.Node) error {
	logger.V(2).Info("Node's PodCIDR will be released by external cloud provider (not managed by controller)",
		"node", klog.KObj(node), "podCIDR", node.Spec.PodCIDR)
	return nil
}
