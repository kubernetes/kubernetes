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
	"sync"

	"github.com/golang/glog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"k8s.io/client-go/tools/record"

	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api"
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
	lock sync.Mutex

	client clientset.Interface
	cloud  *gce.GCECloud

	recorder record.EventRecorder
}

var _ CIDRAllocator = (*cloudCIDRAllocator)(nil)

// NewCloudCIDRAllocator creates a new cloud CIDR allocator.
func NewCloudCIDRAllocator(
	client clientset.Interface,
	cloud cloudprovider.Interface) (ca CIDRAllocator, err error) {

	gceCloud, ok := cloud.(*gce.GCECloud)
	if !ok {
		err = fmt.Errorf("cloudCIDRAllocator does not support %v provider", cloud.ProviderName())
		return
	}

	ca = &cloudCIDRAllocator{
		client: client,
		cloud:  gceCloud,
		recorder: record.NewBroadcaster().NewRecorder(
			api.Scheme,
			v1.EventSource{Component: "cidrAllocator"}),
	}

	glog.V(0).Infof("Using cloud CIDR allocator (provider: %v)", cloud.ProviderName())

	return
}

func (ca *cloudCIDRAllocator) AllocateOrOccupyCIDR(node *v1.Node) error {
	glog.V(2).Infof("Updating PodCIDR for node %v", node.Name)

	cidrs, err := ca.cloud.AliasRanges(types.NodeName(node.Name))

	if err != nil {
		util.RecordNodeStatusChange(ca.recorder, node, "CIDRNotAvailable")
		return fmt.Errorf("failed to allocate cidr: %v", err)
	}

	if len(cidrs) == 0 {
		util.RecordNodeStatusChange(ca.recorder, node, "CIDRNotAvailable")
		glog.V(2).Infof("Node %v has no CIDRs", node.Name)
		return fmt.Errorf("failed to allocate cidr (none exist)")
	}

	node, err = ca.client.Core().Nodes().Get(node.Name, metav1.GetOptions{})
	if err != nil {
		glog.Errorf("Could not get Node object from Kubernetes: %v", err)
		return err
	}

	podCIDR := cidrs[0]

	if node.Spec.PodCIDR != "" {
		if node.Spec.PodCIDR == podCIDR {
			glog.V(3).Infof("Node %v has PodCIDR %v", node.Name, podCIDR)
			return nil
		}
		glog.Errorf("PodCIDR cannot be reassigned, node %v spec has %v, but cloud provider has assigned %v",
			node.Name, node.Spec.PodCIDR, podCIDR)
		// We fall through and set the CIDR despite this error. This
		// implements the same logic as implemented in the
		// rangeAllocator.
		//
		// See https://github.com/kubernetes/kubernetes/pull/42147#discussion_r103357248
	}

	node.Spec.PodCIDR = cidrs[0]
	if _, err := ca.client.Core().Nodes().Update(node); err == nil {
		glog.V(2).Infof("Node %v PodCIDR set to %v", node.Name, podCIDR)
	} else {
		glog.Errorf("Could not update node %v PodCIDR to %v: %v",
			node.Name, podCIDR, err)
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
		glog.Errorf("Error setting route status for node %v: %v",
			node.Name, err)
	}

	return err
}

func (ca *cloudCIDRAllocator) ReleaseCIDR(node *v1.Node) error {
	glog.V(2).Infof("Node %v PodCIDR (%v) will be released by external cloud provider (not managed by controller)",
		node.Name, node.Spec.PodCIDR)
	return nil
}
