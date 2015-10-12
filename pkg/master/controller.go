/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package master

import (
	"fmt"
	"net"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/endpoints"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/registry/endpoint"
	"k8s.io/kubernetes/pkg/registry/namespace"
	"k8s.io/kubernetes/pkg/registry/service"
	servicecontroller "k8s.io/kubernetes/pkg/registry/service/ipallocator/controller"
	portallocatorcontroller "k8s.io/kubernetes/pkg/registry/service/portallocator/controller"
	"k8s.io/kubernetes/pkg/util"

	"github.com/golang/glog"
)

// Controller is the controller manager for the core bootstrap Kubernetes controller
// loops, which manage creating the "kubernetes" service, the "default"
// namespace, and provide the IP repair check on service IPs
type Controller struct {
	NamespaceRegistry namespace.Registry
	ServiceRegistry   service.Registry
	// TODO: MasterCount is yucky
	MasterCount int

	ServiceClusterIPRegistry service.RangeRegistry
	ServiceClusterIPInterval time.Duration
	ServiceClusterIPRange    *net.IPNet

	ServiceNodePortRegistry service.RangeRegistry
	ServiceNodePortInterval time.Duration
	ServiceNodePortRange    util.PortRange

	EndpointRegistry endpoint.Registry
	EndpointInterval time.Duration

	PublicIP net.IP

	ServiceIP         net.IP
	ServicePort       int
	PublicServicePort int

	runner *util.Runner
}

// Start begins the core controller loops that must exist for bootstrapping
// a cluster.
func (c *Controller) Start() {
	if c.runner != nil {
		return
	}

	repairClusterIPs := servicecontroller.NewRepair(c.ServiceClusterIPInterval, c.ServiceRegistry, c.ServiceClusterIPRange, c.ServiceClusterIPRegistry)
	repairNodePorts := portallocatorcontroller.NewRepair(c.ServiceNodePortInterval, c.ServiceRegistry, c.ServiceNodePortRange, c.ServiceNodePortRegistry)

	// run all of the controllers once prior to returning from Start.
	if err := repairClusterIPs.RunOnce(); err != nil {
		// If we fail to repair cluster IPs apiserver is useless. We should restart and retry.
		glog.Fatalf("Unable to perform initial IP allocation check: %v", err)
	}
	if err := repairNodePorts.RunOnce(); err != nil {
		// If we fail to repair node ports apiserver is useless. We should restart and retry.
		glog.Fatalf("Unable to perform initial service nodePort check: %v", err)
	}
	if err := c.UpdateKubernetesService(); err != nil {
		glog.Errorf("Unable to perform initial Kubernetes service initialization: %v", err)
	}

	c.runner = util.NewRunner(c.RunKubernetesService, repairClusterIPs.RunUntil, repairNodePorts.RunUntil)
	c.runner.Start()
}

// RunKubernetesService periodically updates the kubernetes service
func (c *Controller) RunKubernetesService(ch chan struct{}) {
	util.Until(func() {
		if err := c.UpdateKubernetesService(); err != nil {
			util.HandleError(fmt.Errorf("unable to sync kubernetes service: %v", err))
		}
	}, c.EndpointInterval, ch)
}

// UpdateKubernetesService attempts to update the default Kube service.
func (c *Controller) UpdateKubernetesService() error {
	// Update service & endpoint records.
	// TODO: when it becomes possible to change this stuff,
	// stop polling and start watching.
	// TODO: add endpoints of all replicas, not just the elected master.
	if err := c.CreateNamespaceIfNeeded(api.NamespaceDefault); err != nil {
		return err
	}
	if c.ServiceIP != nil {
		if err := c.CreateMasterServiceIfNeeded("kubernetes", c.ServiceIP, c.ServicePort); err != nil {
			return err
		}
		if err := c.SetEndpoints("kubernetes", c.PublicIP, c.PublicServicePort); err != nil {
			return err
		}
	}
	return nil
}

// CreateNamespaceIfNeeded will create the namespace that contains the master services if it doesn't already exist
func (c *Controller) CreateNamespaceIfNeeded(ns string) error {
	ctx := api.NewContext()
	if _, err := c.NamespaceRegistry.GetNamespace(ctx, api.NamespaceDefault); err == nil {
		// the namespace already exists
		return nil
	}
	newNs := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:      ns,
			Namespace: "",
		},
	}
	err := c.NamespaceRegistry.CreateNamespace(ctx, newNs)
	if err != nil && errors.IsAlreadyExists(err) {
		err = nil
	}
	return err
}

// CreateMasterServiceIfNeeded will create the specified service if it
// doesn't already exist.
func (c *Controller) CreateMasterServiceIfNeeded(serviceName string, serviceIP net.IP, servicePort int) error {
	ctx := api.NewDefaultContext()
	if _, err := c.ServiceRegistry.GetService(ctx, serviceName); err == nil {
		// The service already exists.
		return nil
	}
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:      serviceName,
			Namespace: api.NamespaceDefault,
			Labels:    map[string]string{"provider": "kubernetes", "component": "apiserver"},
		},
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{{Port: servicePort, Protocol: api.ProtocolTCP, TargetPort: util.NewIntOrStringFromInt(servicePort)}},
			// maintained by this code, not by the pod selector
			Selector:        nil,
			ClusterIP:       serviceIP.String(),
			SessionAffinity: api.ServiceAffinityNone,
			Type:            api.ServiceTypeClusterIP,
		},
	}

	if err := rest.BeforeCreate(rest.Services, ctx, svc); err != nil {
		return err
	}

	_, err := c.ServiceRegistry.CreateService(ctx, svc)
	if err != nil && errors.IsAlreadyExists(err) {
		err = nil
	}
	return err
}

// SetEndpoints sets the endpoints for the given apiserver service (ro or rw).
// SetEndpoints expects that the endpoints objects it manages will all be
// managed only by SetEndpoints; therefore, to understand this, you need only
// understand the requirements and the body of this function.
//
// Requirements:
//  * All apiservers MUST use the same ports for their {rw, ro} services.
//  * All apiservers MUST use SetEndpoints and only SetEndpoints to manage the
//      endpoints for their {rw, ro} services.
//  * All apiservers MUST know and agree on the number of apiservers expected
//      to be running (c.masterCount).
//  * SetEndpoints is called periodically from all apiservers.
//
func (c *Controller) SetEndpoints(serviceName string, ip net.IP, port int) error {
	ctx := api.NewDefaultContext()
	e, err := c.EndpointRegistry.GetEndpoints(ctx, serviceName)
	if err != nil {
		e = &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:      serviceName,
				Namespace: api.NamespaceDefault,
			},
		}
	}

	// First, determine if the endpoint is in the format we expect (one
	// subset, one port, N IP addresses).
	formatCorrect, ipCorrect := checkEndpointSubsetFormat(e, ip.String(), port, c.MasterCount)
	if !formatCorrect {
		// Something is egregiously wrong, just re-make the endpoints record.
		e.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: ip.String()}},
			Ports:     []api.EndpointPort{{Port: port, Protocol: api.ProtocolTCP}},
		}}
		glog.Warningf("Resetting endpoints for master service %q to %v", serviceName, e)
		return c.EndpointRegistry.UpdateEndpoints(ctx, e)
	} else if !ipCorrect {
		// We *always* add our own IP address; if there are too many IP
		// addresses, we remove the ones lexicographically after our
		// own IP address.  Given the requirements stated at the top of
		// this function, this should cause the list of IP addresses to
		// become eventually correct.
		e.Subsets[0].Addresses = append(e.Subsets[0].Addresses, api.EndpointAddress{IP: ip.String()})
		e.Subsets = endpoints.RepackSubsets(e.Subsets)
		if addrs := &e.Subsets[0].Addresses; len(*addrs) > c.MasterCount {
			// addrs is a pointer because we're going to mutate it.
			for i, addr := range *addrs {
				if addr.IP == ip.String() {
					for len(*addrs) > c.MasterCount {
						remove := (i + 1) % len(*addrs)
						*addrs = append((*addrs)[:remove], (*addrs)[remove+1:]...)
					}
					break
				}
			}
		}
		return c.EndpointRegistry.UpdateEndpoints(ctx, e)
	}
	// We didn't make any changes, no need to actually call update.
	return nil
}

// Determine if the endpoint is in the format SetEndpoints expect (one subset,
// one port, N IP addresses); and if the specified IP address is present and
// the correct number of ip addresses are found.
func checkEndpointSubsetFormat(e *api.Endpoints, ip string, port int, count int) (formatCorrect, ipCorrect bool) {
	if len(e.Subsets) != 1 {
		return false, false
	}
	sub := &e.Subsets[0]
	if len(sub.Ports) != 1 {
		return false, false
	}
	p := &sub.Ports[0]
	if p.Port != port || p.Protocol != api.ProtocolTCP {
		return false, false
	}
	for _, addr := range sub.Addresses {
		if addr.IP == ip {
			return true, len(sub.Addresses) == count
		}
	}
	return true, false
}
