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

package service

import (
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"strconv"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/helper"
	apiservice "k8s.io/kubernetes/pkg/api/service"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/registry/core/endpoint"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
)

// ServiceRest includes storage for services and all sub resources
type ServiceRest struct {
	Service *REST
	Proxy   *ProxyREST
}

// REST adapts a service registry into apiserver's RESTStorage model.
type REST struct {
	registry         Registry
	endpoints        endpoint.Registry
	serviceIPs       ipallocator.Interface
	serviceNodePorts portallocator.Interface
	proxyTransport   http.RoundTripper
}

// NewStorage returns a new REST.
func NewStorage(registry Registry, endpoints endpoint.Registry, serviceIPs ipallocator.Interface,
	serviceNodePorts portallocator.Interface, proxyTransport http.RoundTripper) *ServiceRest {
	rest := &REST{
		registry:         registry,
		endpoints:        endpoints,
		serviceIPs:       serviceIPs,
		serviceNodePorts: serviceNodePorts,
		proxyTransport:   proxyTransport,
	}
	return &ServiceRest{
		Service: rest,
		Proxy:   &ProxyREST{ServiceRest: rest, ProxyTransport: proxyTransport},
	}
}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (rs *REST) ShortNames() []string {
	return []string{"svc"}
}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (rs *REST) Categories() []string {
	return []string{"all"}
}

// TODO: implement includeUninitialized by refactoring this to move to store
func (rs *REST) Create(ctx genericapirequest.Context, obj runtime.Object, includeUninitialized bool) (runtime.Object, error) {
	service := obj.(*api.Service)

	if err := rest.BeforeCreate(Strategy, ctx, obj); err != nil {
		return nil, err
	}

	// TODO: this should probably move to strategy.PrepareForCreate()
	releaseServiceIP := false
	defer func() {
		if releaseServiceIP {
			if helper.IsServiceIPSet(service) {
				rs.serviceIPs.Release(net.ParseIP(service.Spec.ClusterIP))
			}
		}
	}()

	var err error
	if service.Spec.Type != api.ServiceTypeExternalName {
		if releaseServiceIP, err = rs.initClusterIP(service); err != nil {
			return nil, err
		}
	}

	nodePortOp := portallocator.StartOperation(rs.serviceNodePorts)
	defer nodePortOp.Finish()

	if service.Spec.Type == api.ServiceTypeNodePort || service.Spec.Type == api.ServiceTypeLoadBalancer {
		if err := rs.initNodePorts(service, nodePortOp); err != nil {
			return nil, err
		}
	}

	// Handle ExternalTraiffc related fields during service creation.
	if utilfeature.DefaultFeatureGate.Enabled(features.ExternalTrafficLocalOnly) {
		if apiservice.NeedsHealthCheck(service) {
			if err := rs.allocateHealthCheckNodePort(service, nodePortOp); err != nil {
				return nil, errors.NewInternalError(err)
			}
		}
		if errs := validation.ValidateServiceExternalTrafficFieldsCombination(service); len(errs) > 0 {
			return nil, errors.NewInvalid(api.Kind("Service"), service.Name, errs)
		}
	}

	out, err := rs.registry.CreateService(ctx, service)
	if err != nil {
		err = rest.CheckGeneratedNameError(Strategy, err, service)
	}

	if err == nil {
		el := nodePortOp.Commit()
		if el != nil {
			// these should be caught by an eventual reconciliation / restart
			glog.Errorf("error(s) committing service node-ports changes: %v", el)
		}

		releaseServiceIP = false
	}

	return out, err
}

func (rs *REST) Delete(ctx genericapirequest.Context, id string) (runtime.Object, error) {
	service, err := rs.registry.GetService(ctx, id, &metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	err = rs.registry.DeleteService(ctx, id)
	if err != nil {
		return nil, err
	}

	// TODO: can leave dangling endpoints, and potentially return incorrect
	// endpoints if a new service is created with the same name
	err = rs.endpoints.DeleteEndpoints(ctx, id)
	if err != nil && !errors.IsNotFound(err) {
		return nil, err
	}

	if helper.IsServiceIPSet(service) {
		rs.serviceIPs.Release(net.ParseIP(service.Spec.ClusterIP))
	}

	for _, nodePort := range CollectServiceNodePorts(service) {
		err := rs.serviceNodePorts.Release(nodePort)
		if err != nil {
			// these should be caught by an eventual reconciliation / restart
			glog.Errorf("Error releasing service %s node port %d: %v", service.Name, nodePort, err)
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.ExternalTrafficLocalOnly) &&
		apiservice.NeedsHealthCheck(service) {
		nodePort := service.Spec.HealthCheckNodePort
		if nodePort > 0 {
			err := rs.serviceNodePorts.Release(int(nodePort))
			if err != nil {
				// these should be caught by an eventual reconciliation / restart
				utilruntime.HandleError(fmt.Errorf("Error releasing service %s health check node port %d: %v", service.Name, nodePort, err))
			}
		}
	}
	return &metav1.Status{Status: metav1.StatusSuccess}, nil
}

func (rs *REST) Get(ctx genericapirequest.Context, id string, options *metav1.GetOptions) (runtime.Object, error) {
	return rs.registry.GetService(ctx, id, options)
}

func (rs *REST) List(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (runtime.Object, error) {
	return rs.registry.ListServices(ctx, options)
}

// Watch returns Services events via a watch.Interface.
// It implements rest.Watcher.
func (rs *REST) Watch(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return rs.registry.WatchServices(ctx, options)
}

// Export returns Service stripped of cluster-specific information.
// It implements rest.Exporter.
func (rs *REST) Export(ctx genericapirequest.Context, name string, opts metav1.ExportOptions) (runtime.Object, error) {
	return rs.registry.ExportService(ctx, name, opts)
}

func (*REST) New() runtime.Object {
	return &api.Service{}
}

func (*REST) NewList() runtime.Object {
	return &api.ServiceList{}
}

// externalTrafficPolicyUpdate adjusts ExternalTrafficPolicy during service update if needed.
// It is necessary because we default ExternalTrafficPolicy field to different values.
// (NodePort / LoadBalancer: default is Global; Other types: default is empty.)
func externalTrafficPolicyUpdate(oldService, service *api.Service) {
	var neededExternalTraffic, needsExternalTraffic bool
	if oldService.Spec.Type == api.ServiceTypeNodePort ||
		oldService.Spec.Type == api.ServiceTypeLoadBalancer {
		neededExternalTraffic = true
	}
	if service.Spec.Type == api.ServiceTypeNodePort ||
		service.Spec.Type == api.ServiceTypeLoadBalancer {
		needsExternalTraffic = true
	}
	if neededExternalTraffic && !needsExternalTraffic {
		// Clear ExternalTrafficPolicy to prevent confusion from ineffective field.
		service.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyType("")
	}
}

// healthCheckNodePortUpdate handles HealthCheckNodePort allocation/release
// and adjusts HealthCheckNodePort during service update if needed.
func (rs *REST) healthCheckNodePortUpdate(oldService, service *api.Service, nodePortOp *portallocator.PortAllocationOperation) (bool, error) {
	neededHealthCheckNodePort := apiservice.NeedsHealthCheck(oldService)
	oldHealthCheckNodePort := oldService.Spec.HealthCheckNodePort

	needsHealthCheckNodePort := apiservice.NeedsHealthCheck(service)
	newHealthCheckNodePort := service.Spec.HealthCheckNodePort

	switch {
	// Case 1: Transition from don't need HealthCheckNodePort to needs HealthCheckNodePort.
	// Allocate a health check node port or attempt to reserve the user-specified one if provided.
	// Insert health check node port into the service's HealthCheckNodePort field if needed.
	case !neededHealthCheckNodePort && needsHealthCheckNodePort:
		glog.Infof("Transition to LoadBalancer type service with ExternalTrafficPolicy=Local")
		if err := rs.allocateHealthCheckNodePort(service, nodePortOp); err != nil {
			return false, errors.NewInternalError(err)
		}

	// Case 2: Transition from needs HealthCheckNodePort to don't need HealthCheckNodePort.
	// Free the existing healthCheckNodePort and clear the HealthCheckNodePort field.
	case neededHealthCheckNodePort && !needsHealthCheckNodePort:
		glog.Infof("Transition to non LoadBalancer type service or LoadBalancer type service with ExternalTrafficPolicy=Global")
		glog.V(4).Infof("Releasing healthCheckNodePort: %d", oldHealthCheckNodePort)
		nodePortOp.ReleaseDeferred(int(oldHealthCheckNodePort))
		// Clear the HealthCheckNodePort field.
		service.Spec.HealthCheckNodePort = 0

	// Case 3: Remain in needs HealthCheckNodePort.
	// Reject changing the value of the HealthCheckNodePort field.
	case neededHealthCheckNodePort && needsHealthCheckNodePort:
		if oldHealthCheckNodePort != newHealthCheckNodePort {
			glog.Warningf("Attempt to change value of health check node port DENIED")
			fldPath := field.NewPath("spec", "healthCheckNodePort")
			el := field.ErrorList{field.Invalid(fldPath, newHealthCheckNodePort,
				"cannot change healthCheckNodePort on loadBalancer service with externalTraffic=Local during update")}
			return false, errors.NewInvalid(api.Kind("Service"), service.Name, el)
		}
	}
	return true, nil
}

func (rs *REST) Update(ctx genericapirequest.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	oldService, err := rs.registry.GetService(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, false, err
	}

	obj, err := objInfo.UpdatedObject(ctx, oldService)
	if err != nil {
		return nil, false, err
	}

	service := obj.(*api.Service)
	if !rest.ValidNamespace(ctx, &service.ObjectMeta) {
		return nil, false, errors.NewConflict(api.Resource("services"), service.Namespace, fmt.Errorf("Service.Namespace does not match the provided context"))
	}

	// Copy over non-user fields
	// TODO: make this a merge function
	if errs := validation.ValidateServiceUpdate(service, oldService); len(errs) > 0 {
		return nil, false, errors.NewInvalid(api.Kind("Service"), service.Name, errs)
	}

	// TODO: this should probably move to strategy.PrepareForCreate()
	releaseServiceIP := false
	defer func() {
		if releaseServiceIP {
			if helper.IsServiceIPSet(service) {
				rs.serviceIPs.Release(net.ParseIP(service.Spec.ClusterIP))
			}
		}
	}()

	nodePortOp := portallocator.StartOperation(rs.serviceNodePorts)
	defer nodePortOp.Finish()

	// Update service from ExternalName to non-ExternalName, should initialize ClusterIP.
	if oldService.Spec.Type == api.ServiceTypeExternalName && service.Spec.Type != api.ServiceTypeExternalName {
		if releaseServiceIP, err = rs.initClusterIP(service); err != nil {
			return nil, false, err
		}
	}
	// Update service from non-ExternalName to ExternalName, should release ClusterIP if exists.
	if oldService.Spec.Type != api.ServiceTypeExternalName && service.Spec.Type == api.ServiceTypeExternalName {
		if helper.IsServiceIPSet(oldService) {
			rs.serviceIPs.Release(net.ParseIP(oldService.Spec.ClusterIP))
		}
	}
	// Update service from NodePort or LoadBalancer to ExternalName or ClusterIP, should release NodePort if exists.
	if (oldService.Spec.Type == api.ServiceTypeNodePort || oldService.Spec.Type == api.ServiceTypeLoadBalancer) &&
		(service.Spec.Type == api.ServiceTypeExternalName || service.Spec.Type == api.ServiceTypeClusterIP) {
		rs.releaseNodePorts(oldService, nodePortOp)
	}
	// Update service from any type to NodePort or LoadBalancer, should update NodePort.
	if service.Spec.Type == api.ServiceTypeNodePort || service.Spec.Type == api.ServiceTypeLoadBalancer {
		if err := rs.updateNodePorts(oldService, service, nodePortOp); err != nil {
			return nil, false, err
		}
	}
	// Update service from LoadBalancer to non-LoadBalancer, should remove any LoadBalancerStatus.
	if service.Spec.Type != api.ServiceTypeLoadBalancer {
		// Although loadbalancer delete is actually asynchronous, we don't need to expose the user to that complexity.
		service.Status.LoadBalancer = api.LoadBalancerStatus{}
	}

	// Handle ExternalTraiffc related updates.
	if utilfeature.DefaultFeatureGate.Enabled(features.ExternalTrafficLocalOnly) {
		success, err := rs.healthCheckNodePortUpdate(oldService, service, nodePortOp)
		if !success || err != nil {
			return nil, false, err
		}
		externalTrafficPolicyUpdate(oldService, service)
		if errs := validation.ValidateServiceExternalTrafficFieldsCombination(service); len(errs) > 0 {
			return nil, false, errors.NewInvalid(api.Kind("Service"), service.Name, errs)
		}
	}

	out, err := rs.registry.UpdateService(ctx, service)
	if err == nil {
		el := nodePortOp.Commit()
		if el != nil {
			// problems should be fixed by an eventual reconciliation / restart
			glog.Errorf("error(s) committing NodePorts changes: %v", el)
		}

		releaseServiceIP = false
	}

	return out, false, err
}

// Implement Redirector.
var _ = rest.Redirector(&REST{})

// ResourceLocation returns a URL to which one can send traffic for the specified service.
func (rs *REST) ResourceLocation(ctx genericapirequest.Context, id string) (*url.URL, http.RoundTripper, error) {
	// Allow ID as "svcname", "svcname:port", or "scheme:svcname:port".
	svcScheme, svcName, portStr, valid := utilnet.SplitSchemeNamePort(id)
	if !valid {
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("invalid service request %q", id))
	}

	// If a port *number* was specified, find the corresponding service port name
	if portNum, err := strconv.ParseInt(portStr, 10, 64); err == nil {
		svc, err := rs.registry.GetService(ctx, svcName, &metav1.GetOptions{})
		if err != nil {
			return nil, nil, err
		}
		found := false
		for _, svcPort := range svc.Spec.Ports {
			if int64(svcPort.Port) == portNum {
				// use the declared port's name
				portStr = svcPort.Name
				found = true
				break
			}
		}
		if !found {
			return nil, nil, errors.NewServiceUnavailable(fmt.Sprintf("no service port %d found for service %q", portNum, svcName))
		}
	}

	eps, err := rs.endpoints.GetEndpoints(ctx, svcName, &metav1.GetOptions{})
	if err != nil {
		return nil, nil, err
	}
	if len(eps.Subsets) == 0 {
		return nil, nil, errors.NewServiceUnavailable(fmt.Sprintf("no endpoints available for service %q", svcName))
	}
	// Pick a random Subset to start searching from.
	ssSeed := rand.Intn(len(eps.Subsets))
	// Find a Subset that has the port.
	for ssi := 0; ssi < len(eps.Subsets); ssi++ {
		ss := &eps.Subsets[(ssSeed+ssi)%len(eps.Subsets)]
		if len(ss.Addresses) == 0 {
			continue
		}
		for i := range ss.Ports {
			if ss.Ports[i].Name == portStr {
				// Pick a random address.
				ip := ss.Addresses[rand.Intn(len(ss.Addresses))].IP
				port := int(ss.Ports[i].Port)
				return &url.URL{
					Scheme: svcScheme,
					Host:   net.JoinHostPort(ip, strconv.Itoa(port)),
				}, rs.proxyTransport, nil
			}
		}
	}
	return nil, nil, errors.NewServiceUnavailable(fmt.Sprintf("no endpoints available for service %q", id))
}

// This is O(N), but we expect haystack to be small;
// so small that we expect a linear search to be faster
func contains(haystack []int, needle int) bool {
	for _, v := range haystack {
		if v == needle {
			return true
		}
	}
	return false
}

func CollectServiceNodePorts(service *api.Service) []int {
	servicePorts := []int{}
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		if servicePort.NodePort != 0 {
			servicePorts = append(servicePorts, int(servicePort.NodePort))
		}
	}
	return servicePorts
}

// Loop through the service ports list, find one with the same port number and
// NodePort specified, return this NodePort otherwise return 0.
func findRequestedNodePort(port int, servicePorts []api.ServicePort) int {
	for i := range servicePorts {
		servicePort := servicePorts[i]
		if port == int(servicePort.Port) && servicePort.NodePort != 0 {
			return int(servicePort.NodePort)
		}
	}
	return 0
}

// allocateHealthCheckNodePort allocates health check node port to service.
func (rs *REST) allocateHealthCheckNodePort(service *api.Service, nodePortOp *portallocator.PortAllocationOperation) error {
	healthCheckNodePort := service.Spec.HealthCheckNodePort
	if healthCheckNodePort != 0 {
		// If the request has a health check nodePort in mind, attempt to reserve it.
		err := nodePortOp.Allocate(int(healthCheckNodePort))
		if err != nil {
			return fmt.Errorf("failed to allocate requested HealthCheck NodePort %v: %v",
				healthCheckNodePort, err)
		}
		glog.V(4).Infof("Reserved user requested healthCheckNodePort: %d", healthCheckNodePort)
	} else {
		// If the request has no health check nodePort specified, allocate any.
		healthCheckNodePort, err := nodePortOp.AllocateNext()
		if err != nil {
			return fmt.Errorf("failed to allocate a HealthCheck NodePort %v: %v", healthCheckNodePort, err)
		}
		service.Spec.HealthCheckNodePort = int32(healthCheckNodePort)
		glog.V(4).Infof("Reserved allocated healthCheckNodePort: %d", healthCheckNodePort)
	}
	return nil
}

// The return bool value indicates if a cluster IP is allocated successfully.
func (rs *REST) initClusterIP(service *api.Service) (bool, error) {
	switch {
	case service.Spec.ClusterIP == "":
		// Allocate next available.
		ip, err := rs.serviceIPs.AllocateNext()
		if err != nil {
			// TODO: what error should be returned here?  It's not a
			// field-level validation failure (the field is valid), and it's
			// not really an internal error.
			return false, errors.NewInternalError(fmt.Errorf("failed to allocate a serviceIP: %v", err))
		}
		service.Spec.ClusterIP = ip.String()
		return true, nil
	case service.Spec.ClusterIP != api.ClusterIPNone && service.Spec.ClusterIP != "":
		// Try to respect the requested IP.
		if err := rs.serviceIPs.Allocate(net.ParseIP(service.Spec.ClusterIP)); err != nil {
			// TODO: when validation becomes versioned, this gets more complicated.
			el := field.ErrorList{field.Invalid(field.NewPath("spec", "clusterIP"), service.Spec.ClusterIP, err.Error())}
			return false, errors.NewInvalid(api.Kind("Service"), service.Name, el)
		}
		return true, nil
	}

	return false, nil
}

func (rs *REST) initNodePorts(service *api.Service, nodePortOp *portallocator.PortAllocationOperation) error {
	svcPortToNodePort := map[int]int{}
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		allocatedNodePort := svcPortToNodePort[int(servicePort.Port)]
		if allocatedNodePort == 0 {
			// This will only scan forward in the service.Spec.Ports list because any matches
			// before the current port would have been found in svcPortToNodePort. This is really
			// looking for any user provided values.
			np := findRequestedNodePort(int(servicePort.Port), service.Spec.Ports)
			if np != 0 {
				err := nodePortOp.Allocate(np)
				if err != nil {
					// TODO: when validation becomes versioned, this gets more complicated.
					el := field.ErrorList{field.Invalid(field.NewPath("spec", "ports").Index(i).Child("nodePort"), np, err.Error())}
					return errors.NewInvalid(api.Kind("Service"), service.Name, el)
				}
				servicePort.NodePort = int32(np)
				svcPortToNodePort[int(servicePort.Port)] = np
			} else {
				nodePort, err := nodePortOp.AllocateNext()
				if err != nil {
					// TODO: what error should be returned here?  It's not a
					// field-level validation failure (the field is valid), and it's
					// not really an internal error.
					return errors.NewInternalError(fmt.Errorf("failed to allocate a nodePort: %v", err))
				}
				servicePort.NodePort = int32(nodePort)
				svcPortToNodePort[int(servicePort.Port)] = nodePort
			}
		} else if int(servicePort.NodePort) != allocatedNodePort {
			// TODO(xiangpengzhao): do we need to allocate a new NodePort in this case?
			// Note: the current implementation is better, because it saves a NodePort.
			if servicePort.NodePort == 0 {
				servicePort.NodePort = int32(allocatedNodePort)
			} else {
				err := nodePortOp.Allocate(int(servicePort.NodePort))
				if err != nil {
					// TODO: when validation becomes versioned, this gets more complicated.
					el := field.ErrorList{field.Invalid(field.NewPath("spec", "ports").Index(i).Child("nodePort"), servicePort.NodePort, err.Error())}
					return errors.NewInvalid(api.Kind("Service"), service.Name, el)
				}
			}
		}
	}

	return nil
}

func (rs *REST) updateNodePorts(oldService, newService *api.Service, nodePortOp *portallocator.PortAllocationOperation) error {
	oldNodePorts := CollectServiceNodePorts(oldService)

	newNodePorts := []int{}
	for i := range newService.Spec.Ports {
		servicePort := &newService.Spec.Ports[i]
		nodePort := int(servicePort.NodePort)
		if nodePort != 0 {
			if !contains(oldNodePorts, nodePort) {
				err := nodePortOp.Allocate(nodePort)
				if err != nil {
					el := field.ErrorList{field.Invalid(field.NewPath("spec", "ports").Index(i).Child("nodePort"), nodePort, err.Error())}
					return errors.NewInvalid(api.Kind("Service"), newService.Name, el)
				}
			}
		} else {
			nodePort, err := nodePortOp.AllocateNext()
			if err != nil {
				// TODO: what error should be returned here?  It's not a
				// field-level validation failure (the field is valid), and it's
				// not really an internal error.
				return errors.NewInternalError(fmt.Errorf("failed to allocate a nodePort: %v", err))
			}
			servicePort.NodePort = int32(nodePort)
		}
		// Detect duplicate node ports; this should have been caught by validation, so we panic
		if contains(newNodePorts, nodePort) {
			panic("duplicate node port")
		}
		newNodePorts = append(newNodePorts, nodePort)
	}

	// The comparison loops are O(N^2), but we don't expect N to be huge
	// (there's a hard-limit at 2^16, because they're ports; and even 4 ports would be a lot)
	for _, oldNodePort := range oldNodePorts {
		if contains(newNodePorts, oldNodePort) {
			continue
		}
		nodePortOp.ReleaseDeferred(oldNodePort)
	}

	return nil
}

func (rs *REST) releaseNodePorts(service *api.Service, nodePortOp *portallocator.PortAllocationOperation) {
	nodePorts := CollectServiceNodePorts(service)

	for _, nodePort := range nodePorts {
		nodePortOp.ReleaseDeferred(nodePort)
	}
}
