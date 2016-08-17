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

package master

import (
	"fmt"
	"net"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/endpoints"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/registry/endpoint"
	"k8s.io/kubernetes/pkg/registry/namespace"
	"k8s.io/kubernetes/pkg/registry/rangeallocation"
	"k8s.io/kubernetes/pkg/registry/service"
	servicecontroller "k8s.io/kubernetes/pkg/registry/service/ipallocator/controller"
	portallocatorcontroller "k8s.io/kubernetes/pkg/registry/service/portallocator/controller"
	"k8s.io/kubernetes/pkg/util/async"
	"k8s.io/kubernetes/pkg/util/intstr"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
)

// Controller is the controller manager for the core bootstrap Kubernetes controller
// loops, which manage creating the "kubernetes" service, the "default" and "kube-system"
// namespace, and provide the IP repair check on service IPs
type Controller struct {
	NamespaceRegistry namespace.Registry
	ServiceRegistry   service.Registry

	ServiceClusterIPRegistry rangeallocation.RangeRegistry
	ServiceClusterIPInterval time.Duration
	ServiceClusterIPRange    *net.IPNet

	ServiceNodePortRegistry rangeallocation.RangeRegistry
	ServiceNodePortInterval time.Duration
	ServiceNodePortRange    utilnet.PortRange

	EndpointReconciler EndpointReconciler
	EndpointInterval   time.Duration

	SystemNamespaces         []string
	SystemNamespacesInterval time.Duration

	PublicIP net.IP

	ServiceIP                 net.IP
	ServicePort               int
	ExtraServicePorts         []api.ServicePort
	ExtraEndpointPorts        []api.EndpointPort
	PublicServicePort         int
	KubernetesServiceNodePort int

	runner *async.Runner
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
	// Service definition is reconciled during first run to correct port and type per expectations.
	if err := c.UpdateKubernetesService(true); err != nil {
		glog.Errorf("Unable to perform initial Kubernetes service initialization: %v", err)
	}

	c.runner = async.NewRunner(c.RunKubernetesNamespaces, c.RunKubernetesService, repairClusterIPs.RunUntil, repairNodePorts.RunUntil)
	c.runner.Start()
}

// RunKubernetesNamespaces periodically makes sure that all internal namespaces exist
func (c *Controller) RunKubernetesNamespaces(ch chan struct{}) {
	wait.Until(func() {
		// Loop the system namespace list, and create them if they do not exist
		for _, ns := range c.SystemNamespaces {
			if err := c.CreateNamespaceIfNeeded(ns); err != nil {
				runtime.HandleError(fmt.Errorf("unable to create required kubernetes system namespace %s: %v", ns, err))
			}
		}
	}, c.SystemNamespacesInterval, ch)
}

// RunKubernetesService periodically updates the kubernetes service
func (c *Controller) RunKubernetesService(ch chan struct{}) {
	wait.Until(func() {
		// Service definition is not reconciled after first
		// run, ports and type will be corrected only during
		// start.
		if err := c.UpdateKubernetesService(false); err != nil {
			runtime.HandleError(fmt.Errorf("unable to sync kubernetes service: %v", err))
		}
	}, c.EndpointInterval, ch)
}

// UpdateKubernetesService attempts to update the default Kube service.
func (c *Controller) UpdateKubernetesService(reconcile bool) error {
	// Update service & endpoint records.
	// TODO: when it becomes possible to change this stuff,
	// stop polling and start watching.
	// TODO: add endpoints of all replicas, not just the elected master.
	if err := c.CreateNamespaceIfNeeded(api.NamespaceDefault); err != nil {
		return err
	}
	if c.ServiceIP != nil {
		servicePorts, serviceType := createPortAndServiceSpec(c.ServicePort, c.KubernetesServiceNodePort, "https", c.ExtraServicePorts)
		if err := c.CreateOrUpdateMasterServiceIfNeeded("kubernetes", c.ServiceIP, servicePorts, serviceType, reconcile); err != nil {
			return err
		}
		endpointPorts := createEndpointPortSpec(c.PublicServicePort, "https", c.ExtraEndpointPorts)
		if err := c.EndpointReconciler.ReconcileEndpoints("kubernetes", c.PublicIP, endpointPorts, reconcile); err != nil {
			return err
		}
	}
	return nil
}

// CreateNamespaceIfNeeded will create a namespace if it doesn't already exist
func (c *Controller) CreateNamespaceIfNeeded(ns string) error {
	ctx := api.NewContext()
	if _, err := c.NamespaceRegistry.GetNamespace(ctx, ns); err == nil {
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

// createPortAndServiceSpec creates an array of service ports.
// If the NodePort value is 0, just the servicePort is used, otherwise, a node port is exposed.
func createPortAndServiceSpec(servicePort int, nodePort int, servicePortName string, extraServicePorts []api.ServicePort) ([]api.ServicePort, api.ServiceType) {
	//Use the Cluster IP type for the service port if NodePort isn't provided.
	//Otherwise, we will be binding the master service to a NodePort.
	servicePorts := []api.ServicePort{{Protocol: api.ProtocolTCP,
		Port:       int32(servicePort),
		Name:       servicePortName,
		TargetPort: intstr.FromInt(servicePort)}}
	serviceType := api.ServiceTypeClusterIP
	if nodePort > 0 {
		servicePorts[0].NodePort = int32(nodePort)
		serviceType = api.ServiceTypeNodePort
	}
	if extraServicePorts != nil {
		servicePorts = append(servicePorts, extraServicePorts...)
	}
	return servicePorts, serviceType
}

// createEndpointPortSpec creates an array of endpoint ports
func createEndpointPortSpec(endpointPort int, endpointPortName string, extraEndpointPorts []api.EndpointPort) []api.EndpointPort {
	endpointPorts := []api.EndpointPort{{Protocol: api.ProtocolTCP,
		Port: int32(endpointPort),
		Name: endpointPortName,
	}}
	if extraEndpointPorts != nil {
		endpointPorts = append(endpointPorts, extraEndpointPorts...)
	}
	return endpointPorts
}

// CreateMasterServiceIfNeeded will create the specified service if it
// doesn't already exist.
func (c *Controller) CreateOrUpdateMasterServiceIfNeeded(serviceName string, serviceIP net.IP, servicePorts []api.ServicePort, serviceType api.ServiceType, reconcile bool) error {
	ctx := api.NewDefaultContext()
	if s, err := c.ServiceRegistry.GetService(ctx, serviceName); err == nil {
		// The service already exists.
		if reconcile {
			if svc, updated := getMasterServiceUpdateIfNeeded(s, servicePorts, serviceType); updated {
				glog.Warningf("Resetting master service %q to %#v", serviceName, svc)
				_, err := c.ServiceRegistry.UpdateService(ctx, svc)
				return err
			}
		}
		return nil
	}
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:      serviceName,
			Namespace: api.NamespaceDefault,
			Labels:    map[string]string{"provider": "kubernetes", "component": "apiserver"},
		},
		Spec: api.ServiceSpec{
			Ports: servicePorts,
			// maintained by this code, not by the pod selector
			Selector:        nil,
			ClusterIP:       serviceIP.String(),
			SessionAffinity: api.ServiceAffinityClientIP,
			Type:            serviceType,
		},
	}
	if err := rest.BeforeCreate(service.Strategy, ctx, svc); err != nil {
		return err
	}

	_, err := c.ServiceRegistry.CreateService(ctx, svc)
	if err != nil && errors.IsAlreadyExists(err) {
		err = nil
	}
	return err
}

// EndpointReconciler knows how to reconcile the endpoints for the apiserver service.
type EndpointReconciler interface {
	// ReconcileEndpoints sets the endpoints for the given apiserver service (ro or rw).
	// ReconcileEndpoints expects that the endpoints objects it manages will all be
	// managed only by ReconcileEndpoints; therefore, to understand this, you need only
	// understand the requirements.
	//
	// Requirements:
	//  * All apiservers MUST use the same ports for their {rw, ro} services.
	//  * All apiservers MUST use ReconcileEndpoints and only ReconcileEndpoints to manage the
	//      endpoints for their {rw, ro} services.
	//  * ReconcileEndpoints is called periodically from all apiservers.
	ReconcileEndpoints(serviceName string, ip net.IP, endpointPorts []api.EndpointPort, reconcilePorts bool) error
}

// masterCountEndpointReconciler reconciles endpoints based on a specified expected number of
// masters. masterCountEndpointReconciler implements EndpointReconciler.
type masterCountEndpointReconciler struct {
	masterCount      int
	endpointRegistry endpoint.Registry
}

var _ EndpointReconciler = &masterCountEndpointReconciler{}

// NewMasterCountEndpointReconciler creates a new EndpointReconciler that reconciles based on a
// specified expected number of masters.
func NewMasterCountEndpointReconciler(masterCount int, endpointRegistry endpoint.Registry) *masterCountEndpointReconciler {
	return &masterCountEndpointReconciler{
		masterCount:      masterCount,
		endpointRegistry: endpointRegistry,
	}
}

// ReconcileEndpoints sets the endpoints for the given apiserver service (ro or rw).
// ReconcileEndpoints expects that the endpoints objects it manages will all be
// managed only by ReconcileEndpoints; therefore, to understand this, you need only
// understand the requirements and the body of this function.
//
// Requirements:
//  * All apiservers MUST use the same ports for their {rw, ro} services.
//  * All apiservers MUST use ReconcileEndpoints and only ReconcileEndpoints to manage the
//      endpoints for their {rw, ro} services.
//  * All apiservers MUST know and agree on the number of apiservers expected
//      to be running (c.masterCount).
//  * ReconcileEndpoints is called periodically from all apiservers.
func (r *masterCountEndpointReconciler) ReconcileEndpoints(serviceName string, ip net.IP, endpointPorts []api.EndpointPort, reconcilePorts bool) error {
	ctx := api.NewDefaultContext()
	e, err := r.endpointRegistry.GetEndpoints(ctx, serviceName)
	if err != nil {
		e = &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:      serviceName,
				Namespace: api.NamespaceDefault,
			},
		}
	}
	if errors.IsNotFound(err) {
		// Simply create non-existing endpoints for the service.
		e.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: ip.String()}},
			Ports:     endpointPorts,
		}}
		return r.endpointRegistry.UpdateEndpoints(ctx, e)
	}

	// First, determine if the endpoint is in the format we expect (one
	// subset, ports matching endpointPorts, N IP addresses).
	formatCorrect, ipCorrect, portsCorrect := checkEndpointSubsetFormat(e, ip.String(), endpointPorts, r.masterCount, reconcilePorts)
	if !formatCorrect {
		// Something is egregiously wrong, just re-make the endpoints record.
		e.Subsets = []api.EndpointSubset{{
			Addresses: []api.EndpointAddress{{IP: ip.String()}},
			Ports:     endpointPorts,
		}}
		glog.Warningf("Resetting endpoints for master service %q to %#v", serviceName, e)
		return r.endpointRegistry.UpdateEndpoints(ctx, e)
	}
	if ipCorrect && portsCorrect {
		return nil
	}
	if !ipCorrect {
		// We *always* add our own IP address.
		e.Subsets[0].Addresses = append(e.Subsets[0].Addresses, api.EndpointAddress{IP: ip.String()})

		// Lexicographic order is retained by this step.
		e.Subsets = endpoints.RepackSubsets(e.Subsets)

		// If too many IP addresses, remove the ones lexicographically after our
		// own IP address.  Given the requirements stated at the top of
		// this function, this should cause the list of IP addresses to
		// become eventually correct.
		if addrs := &e.Subsets[0].Addresses; len(*addrs) > r.masterCount {
			// addrs is a pointer because we're going to mutate it.
			for i, addr := range *addrs {
				if addr.IP == ip.String() {
					for len(*addrs) > r.masterCount {
						// wrap around if necessary.
						remove := (i + 1) % len(*addrs)
						*addrs = append((*addrs)[:remove], (*addrs)[remove+1:]...)
					}
					break
				}
			}
		}
	}
	if !portsCorrect {
		// Reset ports.
		e.Subsets[0].Ports = endpointPorts
	}
	glog.Warningf("Resetting endpoints for master service %q to %v", serviceName, e)
	return r.endpointRegistry.UpdateEndpoints(ctx, e)
}

// Determine if the endpoint is in the format ReconcileEndpoints expects.
//
// Return values:
// * formatCorrect is true if exactly one subset is found.
// * ipCorrect is true when current master's IP is found and the number
//     of addresses is less than or equal to the master count.
// * portsCorrect is true when endpoint ports exactly match provided ports.
//     portsCorrect is only evaluated when reconcilePorts is set to true.
func checkEndpointSubsetFormat(e *api.Endpoints, ip string, ports []api.EndpointPort, count int, reconcilePorts bool) (formatCorrect bool, ipCorrect bool, portsCorrect bool) {
	if len(e.Subsets) != 1 {
		return false, false, false
	}
	sub := &e.Subsets[0]
	portsCorrect = true
	if reconcilePorts {
		if len(sub.Ports) != len(ports) {
			portsCorrect = false
		}
		for i, port := range ports {
			if len(sub.Ports) <= i || port != sub.Ports[i] {
				portsCorrect = false
				break
			}
		}
	}
	for _, addr := range sub.Addresses {
		if addr.IP == ip {
			ipCorrect = len(sub.Addresses) <= count
			break
		}
	}
	return true, ipCorrect, portsCorrect
}

// * getMasterServiceUpdateIfNeeded sets service attributes for the
//     given apiserver service.
// * getMasterServiceUpdateIfNeeded expects that the service object it
//     manages will be managed only by getMasterServiceUpdateIfNeeded;
//     therefore, to understand this, you need only understand the
//     requirements and the body of this function.
// * getMasterServiceUpdateIfNeeded ensures that the correct ports are
//     are set.
//
// Requirements:
// * All apiservers MUST use getMasterServiceUpdateIfNeeded and only
//     getMasterServiceUpdateIfNeeded to manage service attributes
// * updateMasterService is called periodically from all apiservers.
func getMasterServiceUpdateIfNeeded(svc *api.Service, servicePorts []api.ServicePort, serviceType api.ServiceType) (s *api.Service, updated bool) {
	// Determine if the service is in the format we expect
	// (servicePorts are present and service type matches)
	formatCorrect := checkServiceFormat(svc, servicePorts, serviceType)
	if formatCorrect {
		return svc, false
	}
	svc.Spec.Ports = servicePorts
	svc.Spec.Type = serviceType
	return svc, true
}

// Determine if the service is in the correct format
// getMasterServiceUpdateIfNeeded expects (servicePorts are correct
// and service type matches).
func checkServiceFormat(s *api.Service, ports []api.ServicePort, serviceType api.ServiceType) (formatCorrect bool) {
	if s.Spec.Type != serviceType {
		return false
	}
	if len(ports) != len(s.Spec.Ports) {
		return false
	}
	for i, port := range ports {
		if port != s.Spec.Ports[i] {
			return false
		}
	}
	return true
}
