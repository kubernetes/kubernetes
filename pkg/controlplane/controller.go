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

package controlplane

import (
	"net"
	"sync"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	genericapiserver "k8s.io/apiserver/pkg/server"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/registry/core/rangeallocation"
	corerest "k8s.io/kubernetes/pkg/registry/core/rest"
	servicecontroller "k8s.io/kubernetes/pkg/registry/core/service/ipallocator/controller"
	portallocatorcontroller "k8s.io/kubernetes/pkg/registry/core/service/portallocator/controller"
	"k8s.io/kubernetes/pkg/util/async"
)

const (
	kubernetesServiceName = "kubernetes"
)

// Controller is the controller manager for the core bootstrap Kubernetes
// controller loops, which provide the IP repair check on service IPs
type Controller struct {
	client    kubernetes.Interface
	informers informers.SharedInformerFactory

	ServiceClusterIPRegistry          rangeallocation.RangeRegistry
	ServiceClusterIPRange             net.IPNet
	SecondaryServiceClusterIPRegistry rangeallocation.RangeRegistry
	SecondaryServiceClusterIPRange    net.IPNet

	ServiceClusterIPInterval time.Duration

	ServiceNodePortRegistry rangeallocation.RangeRegistry
	ServiceNodePortInterval time.Duration
	ServiceNodePortRange    utilnet.PortRange

	runner *async.Runner
}

// NewBootstrapController returns a controller for watching the core capabilities of the master
func (c *completedConfig) NewBootstrapController(legacyRESTStorage corerest.LegacyRESTStorage, client kubernetes.Interface) (*Controller, error) {
	return &Controller{
		client:    client,
		informers: c.ExtraConfig.VersionedInformers,

		ServiceClusterIPRegistry:          legacyRESTStorage.ServiceClusterIPAllocator,
		ServiceClusterIPRange:             c.ExtraConfig.ServiceIPRange,
		SecondaryServiceClusterIPRegistry: legacyRESTStorage.SecondaryServiceClusterIPAllocator,
		SecondaryServiceClusterIPRange:    c.ExtraConfig.SecondaryServiceIPRange,

		ServiceClusterIPInterval: c.ExtraConfig.RepairServicesInterval,

		ServiceNodePortRegistry: legacyRESTStorage.ServiceNodePortAllocator,
		ServiceNodePortRange:    c.ExtraConfig.ServiceNodePortRange,
		ServiceNodePortInterval: c.ExtraConfig.RepairServicesInterval,
	}, nil
}

// PostStartHook initiates the core controller loops that must exist for bootstrapping.
func (c *Controller) PostStartHook(hookContext genericapiserver.PostStartHookContext) error {
	c.Start()
	return nil
}

// PreShutdownHook triggers the actions needed to shut down the API Server cleanly.
func (c *Controller) PreShutdownHook() error {
	c.Stop()
	return nil
}

// Start begins the core controller loops that must exist for bootstrapping
// a cluster.
func (c *Controller) Start() {
	if c.runner != nil {
		return
	}

	repairNodePorts := portallocatorcontroller.NewRepair(c.ServiceNodePortInterval, c.client.CoreV1(), c.client.EventsV1(), c.ServiceNodePortRange, c.ServiceNodePortRegistry)

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
	wg.Add(1)

	runRepairNodePorts := func(stopCh chan struct{}) {
		repairNodePorts.RunUntil(wg.Done, stopCh)
	}

	wg.Add(1)
	var runRepairClusterIPs func(stopCh chan struct{})
	if !utilfeature.DefaultFeatureGate.Enabled(features.MultiCIDRServiceAllocator) {
		repairClusterIPs := servicecontroller.NewRepair(c.ServiceClusterIPInterval,
			c.client.CoreV1(),
			c.client.EventsV1(),
			&c.ServiceClusterIPRange,
			c.ServiceClusterIPRegistry,
			&c.SecondaryServiceClusterIPRange,
			c.SecondaryServiceClusterIPRegistry)
		runRepairClusterIPs = func(stopCh chan struct{}) {
			repairClusterIPs.RunUntil(wg.Done, stopCh)
		}
	} else {
		repairClusterIPs := servicecontroller.NewRepairIPAddress(c.ServiceClusterIPInterval,
			c.client,
			&c.ServiceClusterIPRange,
			&c.SecondaryServiceClusterIPRange,
			c.informers.Core().V1().Services(),
			c.informers.Networking().V1alpha1().IPAddresses(),
		)
		runRepairClusterIPs = func(stopCh chan struct{}) {
			repairClusterIPs.RunUntil(wg.Done, stopCh)
		}
	}
	c.runner = async.NewRunner(runRepairClusterIPs, runRepairNodePorts)
	c.runner.Start()

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
}

// Stop cleans up this API Servers endpoint reconciliation leases so another master can take over more quickly.
func (c *Controller) Stop() {
	if c.runner != nil {
		c.runner.Stop()
	}
}
