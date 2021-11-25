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
	recorder          record.EventRecorder

	// Keep a set of nodes that are currectly being processed to avoid races in CIDR allocation
	lock              sync.Mutex
	nodesInProcessing map[string]*nodeProcessingInfo
}

var _ CIDRAllocator = (*cloudCIDRAllocator)(nil)

// NewCloudCIDRAllocator creates a new cloud CIDR allocator.
func NewCloudCIDRAllocator(client clientset.Interface, cloud cloudprovider.Interface, nodeInformer informers.NodeInformer) (CIDRAllocator, error) {
	if client == nil {
		klog.Fatalf("kubeClient is nil when starting NodeController")
	}

	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cidrAllocator"})
	eventBroadcaster.StartStructuredLogging(0)
	klog.V(0).Infof("Sending events to api server.")
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: client.CoreV1().Events("")})

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
		recorder:          recorder,
		nodesInProcessing: map[string]*nodeProcessingInfo{},
	}

	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: controllerutil.CreateAddNodeHandler(ca.AllocateOrOccupyCIDR),
		UpdateFunc: controllerutil.CreateUpdateNodeHandler(func(_, newNode *v1.Node) error {
			if newNode.Spec.PodCIDR == "" {
				return ca.AllocateOrOccupyCIDR(newNode)
			}
			// Even if PodCIDR is assigned, but NetworkUnavailable condition is
			// set to true, we need to process the node to set the condition.
			networkUnavailableTaint := &v1.Taint{Key: v1.TaintNodeNetworkUnavailable, Effect: v1.TaintEffectNoSchedule}
			_, cond := controllerutil.GetNodeCondition(&newNode.Status, v1.NodeNetworkUnavailable)
			if cond == nil || cond.Status != v1.ConditionFalse || utiltaints.TaintExists(newNode.Spec.Taints, networkUnavailableTaint) {
				return ca.AllocateOrOccupyCIDR(newNode)
			}
			return nil
		}),
		DeleteFunc: controllerutil.CreateDeleteNodeHandler(ca.ReleaseCIDR),
	})

	klog.V(0).Infof("Using cloud CIDR allocator (provider: %v)", cloud.ProviderName())
	return ca, nil
}

func (ca *cloudCIDRAllocator) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	klog.Infof("Starting cloud CIDR allocator")
	defer klog.Infof("Shutting down cloud CIDR allocator")

	if !cache.WaitForNamedCacheSync("cidrallocator", stopCh, ca.nodesSynced) {
		return
	}

	for i := 0; i < cidrUpdateWorkers; i++ {
		go ca.worker(stopCh)
	}

	<-stopCh
}

func (ca *cloudCIDRAllocator) worker(stopChan <-chan struct{}) {
	for {
		select {
		case workItem, ok := <-ca.nodeUpdateChannel:
			if !ok {
				klog.Warning("Channel nodeCIDRUpdateChannel was unexpectedly closed")
				return
			}
			if err := ca.updateCIDRAllocation(workItem); err == nil {
				klog.V(3).Infof("Updated CIDR for %q", workItem)
			} else {
				klog.Errorf("Error updating CIDR for %q: %v", workItem, err)
				if canRetry, timeout := ca.retryParams(workItem); canRetry {
					klog.V(2).Infof("Retrying update for %q after %v", workItem, timeout)
					time.AfterFunc(timeout, func() {
						// Requeue the failed node for update again.
						ca.nodeUpdateChannel <- workItem
					})
					continue
				}
				klog.Errorf("Exceeded retry count for %q, dropping from queue", workItem)
			}
			ca.removeNodeFromProcessing(workItem)
		case <-stopChan:
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

func (ca *cloudCIDRAllocator) retryParams(nodeName string) (bool, time.Duration) {
	ca.lock.Lock()
	defer ca.lock.Unlock()

	entry, ok := ca.nodesInProcessing[nodeName]
	if !ok {
		klog.Errorf("Cannot get retryParams for %q as entry does not exist", nodeName)
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
func (ca *cloudCIDRAllocator) AllocateOrOccupyCIDR(node *v1.Node) error {
	if node == nil {
		return nil
	}
	if !ca.insertNodeToProcessing(node.Name) {
		klog.V(2).InfoS("Node is already in a process of CIDR assignment", "node", klog.KObj(node))
		return nil
	}

	klog.V(4).Infof("Putting node %s into the work queue", node.Name)
	ca.nodeUpdateChannel <- node.Name
	return nil
}

// updateCIDRAllocation assigns CIDR to Node and sends an update to the API server.
func (ca *cloudCIDRAllocator) updateCIDRAllocation(nodeName string) error {
	node, err := ca.nodeLister.Get(nodeName)
	if err != nil {
		if errors.IsNotFound(err) {
			return nil // node no longer available, skip processing
		}
		klog.ErrorS(err, "Failed while getting the node for updating Node.Spec.PodCIDR", "nodeName", nodeName)
		return err
	}
	if node.Spec.ProviderID == "" {
		return fmt.Errorf("node %s doesn't have providerID", nodeName)
	}

	cidrStrings, err := ca.cloud.AliasRangesByProviderID(node.Spec.ProviderID)
	if err != nil {
		controllerutil.RecordNodeStatusChange(ca.recorder, node, "CIDRNotAvailable")
		return fmt.Errorf("failed to get cidr(s) from provider: %v", err)
	}
	if len(cidrStrings) == 0 {
		controllerutil.RecordNodeStatusChange(ca.recorder, node, "CIDRNotAvailable")
		return fmt.Errorf("failed to allocate cidr: Node %v has no CIDRs", node.Name)
	}
	//Can have at most 2 ips (one for v4 and one for v6)
	if len(cidrStrings) > 2 {
		klog.InfoS("Got more than 2 ips, truncating to 2", "cidrStrings", cidrStrings)
		cidrStrings = cidrStrings[:2]
	}

	cidrs, err := netutils.ParseCIDRs(cidrStrings)
	if err != nil {
		return fmt.Errorf("failed to parse strings %v as CIDRs: %v", cidrStrings, err)
	}

	needUpdate, err := needPodCIDRsUpdate(node, cidrs)
	if err != nil {
		return fmt.Errorf("err: %v, CIDRS: %v", err, cidrStrings)
	}
	if needUpdate {
		if node.Spec.PodCIDR != "" {
			klog.ErrorS(nil, "PodCIDR being reassigned!", "nodeName", node.Name, "node.Spec.PodCIDRs", node.Spec.PodCIDRs, "cidrStrings", cidrStrings)
			// We fall through and set the CIDR despite this error. This
			// implements the same logic as implemented in the
			// rangeAllocator.
			//
			// See https://github.com/kubernetes/kubernetes/pull/42147#discussion_r103357248
		}
		for i := 0; i < cidrUpdateRetries; i++ {
			if err = nodeutil.PatchNodeCIDRs(ca.client, types.NodeName(node.Name), cidrStrings); err == nil {
				klog.InfoS("Set the node PodCIDRs", "nodeName", node.Name, "cidrStrings", cidrStrings)
				break
			}
		}
	}
	if err != nil {
		controllerutil.RecordNodeStatusChange(ca.recorder, node, "CIDRAssignmentFailed")
		klog.ErrorS(err, "Failed to update the node PodCIDR after multiple attempts", "nodeName", node.Name, "cidrStrings", cidrStrings)
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
		klog.ErrorS(err, "Error setting route status for the node", "nodeName", node.Name)
	}
	return err
}

func needPodCIDRsUpdate(node *v1.Node, podCIDRs []*net.IPNet) (bool, error) {
	if node.Spec.PodCIDR == "" {
		return true, nil
	}
	_, nodePodCIDR, err := netutils.ParseCIDRSloppy(node.Spec.PodCIDR)
	if err != nil {
		klog.ErrorS(err, "Found invalid node.Spec.PodCIDR", "node.Spec.PodCIDR", node.Spec.PodCIDR)
		// We will try to overwrite with new CIDR(s)
		return true, nil
	}
	nodePodCIDRs, err := netutils.ParseCIDRs(node.Spec.PodCIDRs)
	if err != nil {
		klog.ErrorS(err, "Found invalid node.Spec.PodCIDRs", "node.Spec.PodCIDRs", node.Spec.PodCIDRs)
		// We will try to overwrite with new CIDR(s)
		return true, nil
	}

	if len(podCIDRs) == 1 {
		if cmp.Equal(nodePodCIDR, podCIDRs[0]) {
			klog.V(4).InfoS("Node already has allocated CIDR. It matches the proposed one.", "nodeName", node.Name, "podCIDRs[0]", podCIDRs[0])
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
		klog.V(4).InfoS("Node already has allocated CIDRs. It matches the proposed one.", "nodeName", node.Name, "podCIDRs", podCIDRs)
		return false, nil
	}

	return true, nil
}

func (ca *cloudCIDRAllocator) ReleaseCIDR(node *v1.Node) error {
	klog.V(2).Infof("Node %v PodCIDR (%v) will be released by external cloud provider (not managed by controller)",
		node.Name, node.Spec.PodCIDR)
	return nil
}
