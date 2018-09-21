/*
Copyright 2014 The Kubernetes Authors.

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

package nodeipam

import (
	"net"
	"time"

	"github.com/golang/glog"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"

	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"

	"k8s.io/api/core/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam"
	nodesync "k8s.io/kubernetes/pkg/controller/nodeipam/ipam/sync"
	"k8s.io/kubernetes/pkg/util/metrics"
)

func init() {
	// Register prometheus metrics
	Register()
}

const (
	// ipamResyncInterval is the amount of time between when the cloud and node
	// CIDR range assignments are synchronized.
	ipamResyncInterval = 30 * time.Second
	// ipamMaxBackoff is the maximum backoff for retrying synchronization of a
	// given in the error state.
	ipamMaxBackoff = 10 * time.Second
	// ipamInitialRetry is the initial retry interval for retrying synchronization of a
	// given in the error state.
	ipamInitialBackoff = 250 * time.Millisecond
)

// Controller is the controller that manages node ipam state.
type Controller struct {
	allocatorType ipam.CIDRAllocatorType

	cloud       cloudprovider.Interface
	clusterCIDR *net.IPNet
	serviceCIDR *net.IPNet
	kubeClient  clientset.Interface
	// Method for easy mocking in unittest.
	lookupIP func(host string) ([]net.IP, error)

	nodeLister         corelisters.NodeLister
	nodeInformerSynced cache.InformerSynced

	cidrAllocator ipam.CIDRAllocator

	forcefullyDeletePod func(*v1.Pod) error
}

// NewNodeIpamController returns a new node IP Address Management controller to
// sync instances from cloudprovider.
// This method returns an error if it is unable to initialize the CIDR bitmap with
// podCIDRs it has already allocated to nodes. Since we don't allow podCIDR changes
// currently, this should be handled as a fatal error.
func NewNodeIpamController(
	nodeInformer coreinformers.NodeInformer,
	cloud cloudprovider.Interface,
	kubeClient clientset.Interface,
	clusterCIDR *net.IPNet,
	serviceCIDR *net.IPNet,
	nodeCIDRMaskSize int,
	allocatorType ipam.CIDRAllocatorType) (*Controller, error) {

	if kubeClient == nil {
		glog.Fatalf("kubeClient is nil when starting Controller")
	}

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)

	glog.V(0).Infof("Sending events to api server.")
	eventBroadcaster.StartRecordingToSink(
		&v1core.EventSinkImpl{
			Interface: kubeClient.CoreV1().Events(""),
		})

	if kubeClient != nil && kubeClient.CoreV1().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("node_ipam_controller", kubeClient.CoreV1().RESTClient().GetRateLimiter())
	}

	if clusterCIDR == nil {
		glog.Fatal("Controller: Must specify --cluster-cidr if --allocate-node-cidrs is set")
	}
	mask := clusterCIDR.Mask
	if allocatorType != ipam.CloudAllocatorType {
		// Cloud CIDR allocator does not rely on clusterCIDR or nodeCIDRMaskSize for allocation.
		if maskSize, _ := mask.Size(); maskSize > nodeCIDRMaskSize {
			glog.Fatal("Controller: Invalid --cluster-cidr, mask size of cluster CIDR must be less than --node-cidr-mask-size")
		}
	}

	ic := &Controller{
		cloud:         cloud,
		kubeClient:    kubeClient,
		lookupIP:      net.LookupIP,
		clusterCIDR:   clusterCIDR,
		serviceCIDR:   serviceCIDR,
		allocatorType: allocatorType,
	}

	// TODO: Abstract this check into a generic controller manager should run method.
	if ic.allocatorType == ipam.IPAMFromClusterAllocatorType || ic.allocatorType == ipam.IPAMFromCloudAllocatorType {
		cfg := &ipam.Config{
			Resync:       ipamResyncInterval,
			MaxBackoff:   ipamMaxBackoff,
			InitialRetry: ipamInitialBackoff,
		}
		switch ic.allocatorType {
		case ipam.IPAMFromClusterAllocatorType:
			cfg.Mode = nodesync.SyncFromCluster
		case ipam.IPAMFromCloudAllocatorType:
			cfg.Mode = nodesync.SyncFromCloud
		}
		ipamc, err := ipam.NewController(cfg, kubeClient, cloud, clusterCIDR, serviceCIDR, nodeCIDRMaskSize)
		if err != nil {
			glog.Fatalf("Error creating ipam controller: %v", err)
		}
		if err := ipamc.Start(nodeInformer); err != nil {
			glog.Fatalf("Error trying to Init(): %v", err)
		}
	} else {
		var err error
		ic.cidrAllocator, err = ipam.New(
			kubeClient, cloud, nodeInformer, ic.allocatorType, ic.clusterCIDR, ic.serviceCIDR, nodeCIDRMaskSize)
		if err != nil {
			return nil, err
		}
	}

	ic.nodeLister = nodeInformer.Lister()
	ic.nodeInformerSynced = nodeInformer.Informer().HasSynced

	return ic, nil
}

// Run starts an asynchronous loop that monitors the status of cluster nodes.
func (nc *Controller) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	glog.Infof("Starting ipam controller")
	defer glog.Infof("Shutting down ipam controller")

	if !controller.WaitForCacheSync("node", stopCh, nc.nodeInformerSynced) {
		return
	}

	if nc.allocatorType != ipam.IPAMFromClusterAllocatorType && nc.allocatorType != ipam.IPAMFromCloudAllocatorType {
		go nc.cidrAllocator.Run(stopCh)
	}

	<-stopCh
}
