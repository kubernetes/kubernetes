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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	informers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"

	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/controller/node/util"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
)

// cloudCIDRAllocator allocates node CIDRs according to IP address aliases
// assigned by the cloud provider. In this case, the allocation and
// deallocation is delegated to the external provider, and the controller
// merely takes the assignment and updates the node spec.
type cloudCIDRAllocator struct {
	client clientset.Interface
	cloud  *gce.GCECloud

	// Channel that is used to pass updating Nodes with assigned CIDRs to the background
	// This increases a throughput of CIDR assignment by not blocking on long operations.
	nodeCIDRUpdateChannel chan nodeAndCIDR
	recorder              record.EventRecorder

	// Keep a set of nodes that are currectly being processed to avoid races in CIDR allocation
	lock              sync.Mutex
	nodesInProcessing sets.String
}

var _ CIDRAllocator = (*cloudCIDRAllocator)(nil)

// NewCloudCIDRAllocator creates a new cloud CIDR allocator.
func NewCloudCIDRAllocator(client clientset.Interface, cloud cloudprovider.Interface) (CIDRAllocator, error) {
	if client == nil {
		glog.Fatalf("kubeClient is nil when starting NodeController")
	}

	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cidrAllocator"})
	eventBroadcaster.StartLogging(glog.Infof)
	glog.V(0).Infof("Sending events to api server.")
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(client.CoreV1().RESTClient()).Events("")})

	gceCloud, ok := cloud.(*gce.GCECloud)
	if !ok {
		err := fmt.Errorf("cloudCIDRAllocator does not support %v provider", cloud.ProviderName())
		return nil, err
	}

	ca := &cloudCIDRAllocator{
		client: client,
		cloud:  gceCloud,
		nodeCIDRUpdateChannel: make(chan nodeAndCIDR, cidrUpdateQueueSize),
		recorder:              recorder,
		nodesInProcessing:     sets.NewString(),
	}

	for i := 0; i < cidrUpdateWorkers; i++ {
		// TODO: Take stopChan as an argument to NewCloudCIDRAllocator and pass it to the worker.
		go ca.worker(wait.NeverStop)
	}

	glog.V(0).Infof("Using cloud CIDR allocator (provider: %v)", cloud.ProviderName())
	return ca, nil
}

func (ca *cloudCIDRAllocator) worker(stopChan <-chan struct{}) {
	for {
		select {
		case workItem, ok := <-ca.nodeCIDRUpdateChannel:
			if !ok {
				glog.Warning("Channel nodeCIDRUpdateChannel was unexpectedly closed")
				return
			}
			ca.updateCIDRAllocation(workItem)
		case <-stopChan:
			return
		}
	}
}

func (ca *cloudCIDRAllocator) insertNodeToProcessing(nodeName string) bool {
	ca.lock.Lock()
	defer ca.lock.Unlock()
	if ca.nodesInProcessing.Has(nodeName) {
		return false
	}
	ca.nodesInProcessing.Insert(nodeName)
	return true
}

func (ca *cloudCIDRAllocator) removeNodeFromProcessing(nodeName string) {
	ca.lock.Lock()
	defer ca.lock.Unlock()
	ca.nodesInProcessing.Delete(nodeName)
}

// WARNING: If you're adding any return calls or defer any more work from this
// function you have to make sure to update nodesInProcessing properly with the
// disposition of the node when the work is done.
func (ca *cloudCIDRAllocator) AllocateOrOccupyCIDR(node *v1.Node) error {
	if node == nil {
		return nil
	}
	if !ca.insertNodeToProcessing(node.Name) {
		glog.V(2).Infof("Node %v is already in a process of CIDR assignment.", node.Name)
		return nil
	}
	cidrs, err := ca.cloud.AliasRanges(types.NodeName(node.Name))
	if err != nil {
		ca.removeNodeFromProcessing(node.Name)
		util.RecordNodeStatusChange(ca.recorder, node, "CIDRNotAvailable")
		return fmt.Errorf("failed to allocate cidr: %v", err)
	}
	if len(cidrs) == 0 {
		ca.removeNodeFromProcessing(node.Name)
		util.RecordNodeStatusChange(ca.recorder, node, "CIDRNotAvailable")
		return fmt.Errorf("failed to allocate cidr: Node %v has no CIDRs", node.Name)
	}
	_, cidr, err := net.ParseCIDR(cidrs[0])
	if err != nil {
		return fmt.Errorf("failed to parse string '%s' as a CIDR: %v", cidrs[0], err)
	}

	glog.V(4).Infof("Putting node %s with CIDR %s into the work queue", node.Name, cidrs[0])
	ca.nodeCIDRUpdateChannel <- nodeAndCIDR{
		nodeName: node.Name,
		cidr:     cidr,
	}
	return nil
}

// updateCIDRAllocation assigns CIDR to Node and sends an update to the API server.
func (ca *cloudCIDRAllocator) updateCIDRAllocation(data nodeAndCIDR) error {
	var err error
	var node *v1.Node
	defer ca.removeNodeFromProcessing(data.nodeName)
	podCIDR := data.cidr.String()
	for rep := 0; rep < cidrUpdateRetries; rep++ {
		// TODO: change it to using PATCH instead of full Node updates.
		node, err = ca.client.CoreV1().Nodes().Get(data.nodeName, metav1.GetOptions{})
		if err != nil {
			glog.Errorf("Failed while getting node %v to retry updating Node.Spec.PodCIDR: %v", data.nodeName, err)
			continue
		}
		if node.Spec.PodCIDR != "" {
			if node.Spec.PodCIDR == podCIDR {
				glog.V(4).Infof("Node %v already has allocated CIDR %v. It matches the proposed one.", node.Name, podCIDR)
				return nil
			}
			glog.Errorf("PodCIDR being reassigned! Node %v spec has %v, but cloud provider has assigned %v",
				node.Name, node.Spec.PodCIDR, podCIDR)
			// We fall through and set the CIDR despite this error. This
			// implements the same logic as implemented in the
			// rangeAllocator.
			//
			// See https://github.com/kubernetes/kubernetes/pull/42147#discussion_r103357248
		}
		node.Spec.PodCIDR = podCIDR
		if _, err = ca.client.CoreV1().Nodes().Update(node); err == nil {
			glog.Infof("Set node %v PodCIDR to %v", node.Name, podCIDR)
			break
		}
		glog.Errorf("Failed to update node %v PodCIDR to %v (%d retries left): %v", node.Name, podCIDR, cidrUpdateRetries-rep-1, err)
	}
	if err != nil {
		util.RecordNodeStatusChange(ca.recorder, node, "CIDRAssignmentFailed")
		glog.Errorf("CIDR assignment for node %v failed: %v.", data.nodeName, err)
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
		glog.Errorf("Error setting route status for node %v: %v", node.Name, err)
	}
	return err
}

func (ca *cloudCIDRAllocator) ReleaseCIDR(node *v1.Node) error {
	glog.V(2).Infof("Node %v PodCIDR (%v) will be released by external cloud provider (not managed by controller)",
		node.Name, node.Spec.PodCIDR)
	return nil
}

func (ca *cloudCIDRAllocator) Register(nodeInformer informers.NodeInformer) {
	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: util.CreateAddNodeHandler(ca.AllocateOrOccupyCIDR),
		UpdateFunc: util.CreateUpdateNodeHandler(func(_, newNode *v1.Node) error {
			if newNode.Spec.PodCIDR == "" {
				return ca.AllocateOrOccupyCIDR(newNode)
			}
			return nil
		}),
		DeleteFunc: util.CreateDeleteNodeHandler(ca.ReleaseCIDR),
	})
}
