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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/registry/endpoint"
	"k8s.io/kubernetes/pkg/registry/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/service/portallocator"
	"k8s.io/kubernetes/pkg/runtime"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/watch"
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

func (rs *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	service := obj.(*api.Service)

	if err := rest.BeforeCreate(Strategy, ctx, obj); err != nil {
		return nil, err
	}

	// TODO: this should probably move to strategy.PrepareForCreate()
	releaseServiceIP := false
	defer func() {
		if releaseServiceIP {
			if api.IsServiceIPSet(service) {
				rs.serviceIPs.Release(net.ParseIP(service.Spec.ClusterIP))
			}
		}
	}()

	nodePortOp := portallocator.StartOperation(rs.serviceNodePorts)
	defer nodePortOp.Finish()

	if api.IsServiceIPRequested(service) {
		// Allocate next available.
		ip, err := rs.serviceIPs.AllocateNext()
		if err != nil {
			// TODO: what error should be returned here?  It's not a
			// field-level validation failure (the field is valid), and it's
			// not really an internal error.
			return nil, errors.NewInternalError(fmt.Errorf("failed to allocate a serviceIP: %v", err))
		}
		service.Spec.ClusterIP = ip.String()
		releaseServiceIP = true
	} else if api.IsServiceIPSet(service) {
		// Try to respect the requested IP.
		if err := rs.serviceIPs.Allocate(net.ParseIP(service.Spec.ClusterIP)); err != nil {
			// TODO: when validation becomes versioned, this gets more complicated.
			el := field.ErrorList{field.Invalid(field.NewPath("spec", "clusterIP"), service.Spec.ClusterIP, err.Error())}
			return nil, errors.NewInvalid(api.Kind("Service"), service.Name, el)
		}
		releaseServiceIP = true
	}

	assignNodePorts := shouldAssignNodePorts(service)
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		if servicePort.NodePort != 0 {
			err := nodePortOp.Allocate(int(servicePort.NodePort))
			if err != nil {
				// TODO: when validation becomes versioned, this gets more complicated.
				el := field.ErrorList{field.Invalid(field.NewPath("spec", "ports").Index(i).Child("nodePort"), servicePort.NodePort, err.Error())}
				return nil, errors.NewInvalid(api.Kind("Service"), service.Name, el)
			}
		} else if assignNodePorts {
			nodePort, err := nodePortOp.AllocateNext()
			if err != nil {
				// TODO: what error should be returned here?  It's not a
				// field-level validation failure (the field is valid), and it's
				// not really an internal error.
				return nil, errors.NewInternalError(fmt.Errorf("failed to allocate a nodePort: %v", err))
			}
			servicePort.NodePort = int32(nodePort)
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

func (rs *REST) Delete(ctx api.Context, id string) (runtime.Object, error) {
	service, err := rs.registry.GetService(ctx, id)
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

	if api.IsServiceIPSet(service) {
		rs.serviceIPs.Release(net.ParseIP(service.Spec.ClusterIP))
	}

	for _, nodePort := range CollectServiceNodePorts(service) {
		err := rs.serviceNodePorts.Release(nodePort)
		if err != nil {
			// these should be caught by an eventual reconciliation / restart
			glog.Errorf("Error releasing service %s node port %d: %v", service.Name, nodePort, err)
		}
	}

	return &unversioned.Status{Status: unversioned.StatusSuccess}, nil
}

func (rs *REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	return rs.registry.GetService(ctx, id)
}

func (rs *REST) List(ctx api.Context, options *api.ListOptions) (runtime.Object, error) {
	return rs.registry.ListServices(ctx, options)
}

// Watch returns Services events via a watch.Interface.
// It implements rest.Watcher.
func (rs *REST) Watch(ctx api.Context, options *api.ListOptions) (watch.Interface, error) {
	return rs.registry.WatchServices(ctx, options)
}

// Export returns Service stripped of cluster-specific information.
// It implements rest.Exporter.
func (rs *REST) Export(ctx api.Context, name string, opts unversioned.ExportOptions) (runtime.Object, error) {
	return rs.registry.ExportService(ctx, name, opts)
}

func (*REST) New() runtime.Object {
	return &api.Service{}
}

func (*REST) NewList() runtime.Object {
	return &api.ServiceList{}
}

func (rs *REST) Update(ctx api.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	oldService, err := rs.registry.GetService(ctx, name)
	if err != nil {
		return nil, false, err
	}

	obj, err := objInfo.UpdatedObject(ctx, oldService)
	if err != nil {
		return nil, false, err
	}

	service := obj.(*api.Service)
	if !api.ValidNamespace(ctx, &service.ObjectMeta) {
		return nil, false, errors.NewConflict(api.Resource("services"), service.Namespace, fmt.Errorf("Service.Namespace does not match the provided context"))
	}

	// Copy over non-user fields
	// TODO: make this a merge function
	if errs := validation.ValidateServiceUpdate(service, oldService); len(errs) > 0 {
		return nil, false, errors.NewInvalid(api.Kind("Service"), service.Name, errs)
	}

	nodePortOp := portallocator.StartOperation(rs.serviceNodePorts)
	defer nodePortOp.Finish()

	assignNodePorts := shouldAssignNodePorts(service)

	oldNodePorts := CollectServiceNodePorts(oldService)

	newNodePorts := []int{}
	if assignNodePorts {
		for i := range service.Spec.Ports {
			servicePort := &service.Spec.Ports[i]
			nodePort := int(servicePort.NodePort)
			if nodePort != 0 {
				if !contains(oldNodePorts, nodePort) {
					err := nodePortOp.Allocate(nodePort)
					if err != nil {
						el := field.ErrorList{field.Invalid(field.NewPath("spec", "ports").Index(i).Child("nodePort"), nodePort, err.Error())}
						return nil, false, errors.NewInvalid(api.Kind("Service"), service.Name, el)
					}
				}
			} else {
				nodePort, err = nodePortOp.AllocateNext()
				if err != nil {
					// TODO: what error should be returned here?  It's not a
					// field-level validation failure (the field is valid), and it's
					// not really an internal error.
					return nil, false, errors.NewInternalError(fmt.Errorf("failed to allocate a nodePort: %v", err))
				}
				servicePort.NodePort = int32(nodePort)
			}
			// Detect duplicate node ports; this should have been caught by validation, so we panic
			if contains(newNodePorts, nodePort) {
				panic("duplicate node port")
			}
			newNodePorts = append(newNodePorts, nodePort)
		}
	} else {
		// Validate should have validated that nodePort == 0
	}

	// The comparison loops are O(N^2), but we don't expect N to be huge
	// (there's a hard-limit at 2^16, because they're ports; and even 4 ports would be a lot)
	for _, oldNodePort := range oldNodePorts {
		if !contains(newNodePorts, oldNodePort) {
			continue
		}
		nodePortOp.ReleaseDeferred(oldNodePort)
	}

	// Remove any LoadBalancerStatus now if Type != LoadBalancer;
	// although loadbalancer delete is actually asynchronous, we don't need to expose the user to that complexity.
	if service.Spec.Type != api.ServiceTypeLoadBalancer {
		service.Status.LoadBalancer = api.LoadBalancerStatus{}
	}

	out, err := rs.registry.UpdateService(ctx, service)

	if err == nil {
		el := nodePortOp.Commit()
		if el != nil {
			// problems should be fixed by an eventual reconciliation / restart
			glog.Errorf("error(s) committing NodePorts changes: %v", el)
		}
	}

	return out, false, err
}

// Implement Redirector.
var _ = rest.Redirector(&REST{})

// ResourceLocation returns a URL to which one can send traffic for the specified service.
func (rs *REST) ResourceLocation(ctx api.Context, id string) (*url.URL, http.RoundTripper, error) {
	// Allow ID as "svcname", "svcname:port", or "scheme:svcname:port".
	svcScheme, svcName, portStr, valid := utilnet.SplitSchemeNamePort(id)
	if !valid {
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("invalid service request %q", id))
	}

	// If a port *number* was specified, find the corresponding service port name
	if portNum, err := strconv.ParseInt(portStr, 10, 64); err == nil {
		svc, err := rs.registry.GetService(ctx, svcName)
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

	eps, err := rs.endpoints.GetEndpoints(ctx, svcName)
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

func shouldAssignNodePorts(service *api.Service) bool {
	switch service.Spec.Type {
	case api.ServiceTypeLoadBalancer:
		return true
	case api.ServiceTypeNodePort:
		return true
	case api.ServiceTypeClusterIP:
		return false
	default:
		glog.Errorf("Unknown service type: %v", service.Spec.Type)
		return false
	}
}
