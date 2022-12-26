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
	"fmt"
	"net"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/registry/core/rangeallocation"
	corerest "k8s.io/kubernetes/pkg/registry/core/rest"
	servicecontroller "k8s.io/kubernetes/pkg/registry/core/service/ipallocator/controller"
	portallocatorcontroller "k8s.io/kubernetes/pkg/registry/core/service/portallocator/controller"
	"k8s.io/kubernetes/pkg/util/async"
)

// Controller is the controller manager for the core bootstrap Kubernetes
// controller loops, the "default", "kube-system" and "kube-public" namespaces,
// and provide the IP repair check on service IPs
type Controller struct {
	client kubernetes.Interface

	ServiceClusterIPRegistry          rangeallocation.RangeRegistry
	ServiceClusterIPRange             net.IPNet
	SecondaryServiceClusterIPRegistry rangeallocation.RangeRegistry
	SecondaryServiceClusterIPRange    net.IPNet

	ServiceClusterIPInterval time.Duration

	ServiceNodePortRegistry rangeallocation.RangeRegistry
	ServiceNodePortInterval time.Duration
	ServiceNodePortRange    utilnet.PortRange

	SystemNamespaces         []string
	SystemNamespacesInterval time.Duration

	runner *async.Runner
}

// NewBootstrapController returns a controller for watching the core capabilities of the master
func (c *completedConfig) NewBootstrapController(legacyRESTStorage corerest.LegacyRESTStorage, client kubernetes.Interface) (*Controller, error) {
	systemNamespaces := []string{metav1.NamespaceSystem, metav1.NamespacePublic, corev1.NamespaceNodeLease}

	return &Controller{
		client:                   client,
		SystemNamespaces:         systemNamespaces,
		SystemNamespacesInterval: 1 * time.Minute,

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

	repairClusterIPs := servicecontroller.NewRepair(c.ServiceClusterIPInterval, c.client.CoreV1(), c.client.EventsV1(), &c.ServiceClusterIPRange, c.ServiceClusterIPRegistry, &c.SecondaryServiceClusterIPRange, c.SecondaryServiceClusterIPRegistry)
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
	wg.Add(2)

	runRepairClusterIPs := func(stopCh chan struct{}) {
		repairClusterIPs.RunUntil(wg.Done, stopCh)
	}
	runRepairNodePorts := func(stopCh chan struct{}) {
		repairNodePorts.RunUntil(wg.Done, stopCh)
	}

	c.runner = async.NewRunner(c.RunKubernetesNamespaces, runRepairClusterIPs, runRepairNodePorts)
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

// RunKubernetesNamespaces periodically makes sure that all internal namespaces exist
func (c *Controller) RunKubernetesNamespaces(ch chan struct{}) {
	wait.Until(func() {
		// Loop the system namespace list, and create them if they do not exist
		for _, ns := range c.SystemNamespaces {
			if err := createNamespaceIfNeeded(c.client.CoreV1(), ns); err != nil {
				runtime.HandleError(fmt.Errorf("unable to create required kubernetes system namespace %s: %v", ns, err))
			}
		}
	}, c.SystemNamespacesInterval, ch)
}
