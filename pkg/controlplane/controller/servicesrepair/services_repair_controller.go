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

package servicesrepair

import (
	"fmt"
	"net"
	"sync"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	coreinformers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes"
	listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/registry/core/rangeallocation"
	servicecontroller "k8s.io/kubernetes/pkg/registry/core/service/ipallocator/controller"
	portallocatorcontroller "k8s.io/kubernetes/pkg/registry/core/service/portallocator/controller"
)

// Controller deletes expired apiserver leases.
type Controller struct {
	client        kubernetes.Interface
	serviceLister listers.ServiceLister
	serviceSynced cache.InformerSynced

	serviceClusterIPRegistry          rangeallocation.RangeRegistry
	serviceClusterIPRange             net.IPNet
	secondaryServiceClusterIPRegistry rangeallocation.RangeRegistry
	secondaryServiceClusterIPRange    net.IPNet

	serviceNodePortRegistry rangeallocation.RangeRegistry
	serviceNodePortRange    utilnet.PortRange

	interval time.Duration
}

// NewController creates a new Controller to ensure system namespaces exist.
func NewController(
	clientset kubernetes.Interface,
	serviceInformer coreinformers.ServiceInformer,
	serviceClusterIPRegistry rangeallocation.RangeRegistry,
	serviceClusterIPRange net.IPNet,
	secondaryServiceClusterIPRegistry rangeallocation.RangeRegistry,
	secondaryServiceClusterIPRange net.IPNet,
	serviceNodePortRegistry rangeallocation.RangeRegistry,
	serviceNodePortRange utilnet.PortRange,
	interval time.Duration) *Controller {
	return &Controller{
		client:                            clientset,
		serviceLister:                     serviceInformer.Lister(),
		serviceSynced:                     serviceInformer.Informer().HasSynced,
		serviceClusterIPRegistry:          serviceClusterIPRegistry,
		serviceClusterIPRange:             serviceClusterIPRange,
		secondaryServiceClusterIPRegistry: secondaryServiceClusterIPRegistry,
		secondaryServiceClusterIPRange:    secondaryServiceClusterIPRange,
		serviceNodePortRegistry:           serviceNodePortRegistry,
		serviceNodePortRange:              serviceNodePortRange,
		interval:                          interval,
	}
}

// Run starts one worker.
func (c *Controller) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer klog.Infof("Shutting services repair controller")

	klog.Infof("Starting services repair controller")

	if !cache.WaitForCacheSync(stopCh, c.serviceSynced) {
		utilruntime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
		return
	}

	repairClusterIPs := servicecontroller.NewRepair(
		c.interval,
		c.serviceLister,
		c.client.EventsV1(),
		&c.serviceClusterIPRange,
		c.serviceClusterIPRegistry,
		&c.secondaryServiceClusterIPRange,
		c.secondaryServiceClusterIPRegistry)

	repairNodePorts := portallocatorcontroller.NewRepair(
		c.interval,
		c.serviceLister,
		c.client.EventsV1(),
		c.serviceNodePortRange,
		c.serviceNodePortRegistry)

	// We start both repairClusterIPs and repairNodePorts to ensure repair
	// loops of ClusterIPs and NodePorts.
	// We run both repair loops using RunUntil public interface.
	// However, we want to fail liveness/readiness until the first
	// successful repair loop, so we basically pass appropriate
	// callbacks to RunUtil methods.
	// Additionally, we ensure that we don't wait for it for longer
	// than 1 minute for backward compatibility of failing the whole
	// apiserver if we can't repair them.
	wg := sync.WaitGroup{}
	wg.Add(2)

	runRepairClusterIPs := func(stopCh <-chan struct{}) {
		repairClusterIPs.RunUntil(wg.Done, stopCh)
	}
	runRepairNodePorts := func(stopCh <-chan struct{}) {
		repairNodePorts.RunUntil(wg.Done, stopCh)
	}

	go runRepairClusterIPs(stopCh)
	go runRepairNodePorts(stopCh)

	// For backward compatibility, we ensure that if we never are able
	// to repair clusterIPs and/or nodeports, we not only fail the liveness
	// and/or readiness, but also explicitly fail.
	done := make(chan struct{})
	go func() {
		defer close(done)
		wg.Wait()
	}()
	select {
	case <-done:
	case <-time.After(time.Minute):
		klog.Fatalf("Unable to perform initial IP and Port allocation check")
	}

	<-stopCh
}
