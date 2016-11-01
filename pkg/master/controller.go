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
	"reflect"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/endpoints"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/registry/core/namespace"
	"k8s.io/kubernetes/pkg/registry/core/rangeallocation"
	corerest "k8s.io/kubernetes/pkg/registry/core/rest"
	"k8s.io/kubernetes/pkg/registry/core/service"
	servicecontroller "k8s.io/kubernetes/pkg/registry/core/service/ipallocator/controller"
	portallocatorcontroller "k8s.io/kubernetes/pkg/registry/core/service/portallocator/controller"
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
	ServiceClusterIPRange    net.IPNet

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

// NewBootstrapController returns a controller for watching the core capabilities of the master
func (c *Config) NewBootstrapController(legacyRESTStorage corerest.LegacyRESTStorage) *Controller {
	return &Controller{
		NamespaceRegistry: legacyRESTStorage.NamespaceRegistry,
		ServiceRegistry:   legacyRESTStorage.ServiceRegistry,

		EndpointReconciler: c.EndpointReconcilerConfig.Reconciler,
		EndpointInterval:   c.EndpointReconcilerConfig.Interval,

		SystemNamespaces:         []string{api.NamespaceSystem},
		SystemNamespacesInterval: 1 * time.Minute,

		ServiceClusterIPRegistry: legacyRESTStorage.ServiceClusterIPAllocator,
		ServiceClusterIPRange:    c.ServiceIPRange,
		ServiceClusterIPInterval: 3 * time.Minute,

		ServiceNodePortRegistry: legacyRESTStorage.ServiceNodePortAllocator,
		ServiceNodePortRange:    c.ServiceNodePortRange,
		ServiceNodePortInterval: 3 * time.Minute,

		PublicIP: c.GenericConfig.PublicAddress,

		ServiceIP:                 c.APIServerServiceIP,
		ServicePort:               c.APIServerServicePort,
		ExtraServicePorts:         c.ExtraServicePorts,
		ExtraEndpointPorts:        c.ExtraEndpointPorts,
		PublicServicePort:         c.GenericConfig.ReadWritePort,
		KubernetesServiceNodePort: c.KubernetesServiceNodePort,
	}
}

func (c *Controller) PostStartHook(hookContext genericapiserver.PostStartHookContext) error {
	c.Start()
	return nil
}

// Start begins the core controller loops that must exist for bootstrapping
// a cluster.
func (c *Controller) Start() {
	if c.runner != nil {
		return
	}

	repairClusterIPs := servicecontroller.NewRepair(c.ServiceClusterIPInterval, c.ServiceRegistry, &c.ServiceClusterIPRange, c.ServiceClusterIPRegistry)
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
		if err := c.EndpointReconciler.ReconcileEndpoints("apiservers", "kubernetes", c.PublicIP, endpointPorts, reconcile, time.Now()); err != nil {
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
	// It uses auxilary config map to keep expiration times for apiservers.
	// Each apiserver separately update its expiration time and removes all expired apiservers.
	//
	// Requirements:
	//  * All apiservers MUST use the same ports for their {rw, ro} services.
	//  * All apiservers MUST use ReconcileEndpoints and only ReconcileEndpoints to manage the
	//      endpoints for their {rw, ro} services.
	//  * ReconcileEndpoints is called periodically from all apiservers.
	ReconcileEndpoints(configmapName, serviceName string, ip net.IP, endpointPorts []api.EndpointPort, reconcilePorts bool, now time.Time) error
}

// dynamicEndpointReconciler reconciles endpoints based on a config map with expiration times.
// dynamicEndpointReconciler implements EndpointReconciler.
type dynamicEndpointReconciler struct {
	endpointClient  coreclient.EndpointsGetter
	configmapClient coreclient.ConfigMapsGetter
}

var _ EndpointReconciler = &dynamicEndpointReconciler{}

// NewDynamicEndpointReconciler creates a new EndpointReconciler that reconciles based on an auxiliary config map.
func NewDynamicEndpointReconciler(endpointClient coreclient.EndpointsGetter,
	configmapClient coreclient.ConfigMapsGetter) *dynamicEndpointReconciler {
	return &dynamicEndpointReconciler{
		endpointClient:  endpointClient,
		configmapClient: configmapClient,
	}
}

// updateCM updates TTL configmap for API servers: renews TTL for key server and removes all expired entries.
func updateCM(key string, m *api.ConfigMap, now time.Time) {
	const format = "20060102 150405 MST"
	window := 3 * DefaultEndpointReconcilerInterval
	newData := map[string]string{}
	for k, v := range m.Data {
		ttl, err := time.Parse(format, v)
		if err != nil {
			continue
		}
		if now.Before(ttl) {
			newData[k] = v
		}
	}
	newData[key] = now.Add(window).Format(format)
	m.Data = newData
}

// ReconcileEndpoints sets the endpoints for the given apiserver service (ro or rw).
// ReconcileEndpoints expects that the endpoints objects it manages will all be
// managed only by ReconcileEndpoints; therefore, to understand this, you need only
// understand the requirements and the body of this function.
//
// It uses auxilary config map to keep expiration times for apiservers.
// Each apiserver separately update its expiration time and removes all expired apiservers.
//
// Requirements:
//  * All apiservers MUST use the same ports for their {rw, ro} services.
//  * All apiservers MUST use ReconcileEndpoints and only ReconcileEndpoints to manage the
//      endpoints for their {rw, ro} services.
//  * ReconcileEndpoints is called periodically from all apiservers.
func (r *dynamicEndpointReconciler) ReconcileEndpoints(configmapName, serviceName string, ip net.IP, endpointPorts []api.EndpointPort, reconcilePorts bool, now time.Time) error {
	// Handle config map
	m, err := r.configmapClient.ConfigMaps(api.NamespaceSystem).Get(configmapName)

	if err != nil && !errors.IsNotFound(err) {
		return fmt.Errorf("couldn't read apiservers configmap: %v", err)
	}

	createCM := false
	if err != nil && errors.IsNotFound(err) {
		m = &api.ConfigMap{
			ObjectMeta: api.ObjectMeta{
				Name:      configmapName,
				Namespace: api.NamespaceSystem,
			},
		}
		createCM = true
	}

	updateCM(ip.String(), m, now)

	if createCM {
		_, err = r.configmapClient.ConfigMaps(api.NamespaceSystem).Create(m)
		if err != nil {
			return fmt.Errorf("couldn't create cm: %v", err)
		}
	} else {
		_, err = r.configmapClient.ConfigMaps(api.NamespaceSystem).Update(m)
		if err != nil {
			return fmt.Errorf("couldn't update cm: %v", err)
		}
	}

	// Handle endpoints
	e, err := r.endpointClient.Endpoints(api.NamespaceDefault).Get(serviceName)
	if err != nil && !errors.IsNotFound(err) {
		return fmt.Errorf("couldn't read kubernetes service endpoints: %v", err)
	}

	createE := false
	if errors.IsNotFound(err) {
		e = &api.Endpoints{
			ObjectMeta: api.ObjectMeta{
				Name:      serviceName,
				Namespace: api.NamespaceDefault,
			},
		}
		createE = true
	}
	e.Subsets = endpoints.RepackSubsets(e.Subsets)
	ePorts := endpointPorts
	if len(e.Subsets) == 1 {
		ePorts = e.Subsets[0].Ports
	}

	newSubsets := []api.EndpointSubset{}
	for key := range m.Data {
		newSubsets = append(newSubsets, api.EndpointSubset{
			Addresses: []api.EndpointAddress{{IP: key}},
			Ports:     ePorts,
		})
	}
	newSubsets = endpoints.RepackSubsets(newSubsets)

	updateE := false
	if !reflect.DeepEqual(e.Subsets, newSubsets) {
		glog.Infof("Endpoints for kubernetes service changed: %v", newSubsets)
		e.Subsets = newSubsets
		updateE = true
	}

	if !reflect.DeepEqual(e.Subsets[0].Ports, endpointPorts) && reconcilePorts {
		glog.Warningf("Mismatched ports for kubernetes service: %v / %v", e.Subsets[0].Ports, endpointPorts)
		e.Subsets[0].Ports = endpointPorts
		updateE = true
	}

	if createE {
		_, err = r.endpointClient.Endpoints(api.NamespaceDefault).Create(e)
		return err
	}

	if updateE {
		_, err = r.endpointClient.Endpoints(api.NamespaceDefault).Update(e)
		return err
	}

	return nil
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
