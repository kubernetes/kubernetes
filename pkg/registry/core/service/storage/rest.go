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

package storage

import (
	"context"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"strconv"

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
	"k8s.io/apiserver/pkg/util/dryrun"
	"k8s.io/klog"

	apiservice "k8s.io/kubernetes/pkg/api/service"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	registry "k8s.io/kubernetes/pkg/registry/core/service"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	netutil "k8s.io/utils/net"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

// REST adapts a service registry into apiserver's RESTStorage model.
type REST struct {
	services               ServiceStorage
	endpoints              EndpointsStorage
	serviceIPs             ipallocator.Interface
	secondaryServiceIPs    ipallocator.Interface
	defaultServiceIPFamily api.IPFamily
	serviceNodePorts       portallocator.Interface
	proxyTransport         http.RoundTripper
	pods                   rest.Getter
}

// ServiceNodePort includes protocol and port number of a service NodePort.
type ServiceNodePort struct {
	// The IP protocol for this port. Supports "TCP" and "UDP".
	Protocol api.Protocol

	// The port on each node on which this service is exposed.
	// Default is to auto-allocate a port if the ServiceType of this Service requires one.
	NodePort int32
}

type ServiceStorage interface {
	rest.Scoper
	rest.Getter
	rest.Lister
	rest.CreaterUpdater
	rest.GracefulDeleter
	rest.Watcher
	rest.Exporter
	rest.StorageVersionProvider
}

type EndpointsStorage interface {
	rest.Getter
	rest.GracefulDeleter
}

// NewREST returns a wrapper around the underlying generic storage and performs
// allocations and deallocations of various service related resources like ports.
// TODO: all transactional behavior should be supported from within generic storage
//   or the strategy.
func NewREST(
	services ServiceStorage,
	endpoints EndpointsStorage,
	pods rest.Getter,
	serviceIPs ipallocator.Interface,
	secondaryServiceIPs ipallocator.Interface,
	serviceNodePorts portallocator.Interface,
	proxyTransport http.RoundTripper,
) (*REST, *registry.ProxyREST) {
	// detect this cluster default Service IPFamily (ipfamily of --service-cluster-ip-range)
	// we do it once here, to avoid having to do it over and over during ipfamily assignment
	serviceIPFamily := api.IPv4Protocol
	cidr := serviceIPs.CIDR()
	if netutil.IsIPv6CIDR(&cidr) {
		serviceIPFamily = api.IPv6Protocol
	}

	klog.V(0).Infof("the default service ipfamily for this cluster is: %s", string(serviceIPFamily))

	rest := &REST{
		services:               services,
		endpoints:              endpoints,
		serviceIPs:             serviceIPs,
		secondaryServiceIPs:    secondaryServiceIPs,
		serviceNodePorts:       serviceNodePorts,
		defaultServiceIPFamily: serviceIPFamily,
		proxyTransport:         proxyTransport,
		pods:                   pods,
	}
	return rest, &registry.ProxyREST{Redirector: rest, ProxyTransport: proxyTransport}
}

var (
	_ ServiceStorage              = &REST{}
	_ rest.CategoriesProvider     = &REST{}
	_ rest.ShortNamesProvider     = &REST{}
	_ rest.StorageVersionProvider = &REST{}
)

func (rs *REST) StorageVersion() runtime.GroupVersioner {
	return rs.services.StorageVersion()
}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (rs *REST) ShortNames() []string {
	return []string{"svc"}
}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (rs *REST) Categories() []string {
	return []string{"all"}
}

func (rs *REST) NamespaceScoped() bool {
	return rs.services.NamespaceScoped()
}

func (rs *REST) New() runtime.Object {
	return rs.services.New()
}

func (rs *REST) NewList() runtime.Object {
	return rs.services.NewList()
}

func (rs *REST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return rs.services.Get(ctx, name, options)
}

func (rs *REST) List(ctx context.Context, options *metainternalversion.ListOptions) (runtime.Object, error) {
	return rs.services.List(ctx, options)
}

func (rs *REST) Watch(ctx context.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return rs.services.Watch(ctx, options)
}

func (rs *REST) Export(ctx context.Context, name string, opts metav1.ExportOptions) (runtime.Object, error) {
	return rs.services.Export(ctx, name, opts)
}

func (rs *REST) Create(ctx context.Context, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	service := obj.(*api.Service)

	// set the service ip family, if it was not already set
	if utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) && service.Spec.IPFamily == nil {
		service.Spec.IPFamily = &rs.defaultServiceIPFamily
	}

	if err := rest.BeforeCreate(registry.Strategy, ctx, obj); err != nil {
		return nil, err
	}

	// TODO: this should probably move to strategy.PrepareForCreate()
	releaseServiceIP := false
	defer func() {
		if releaseServiceIP {
			if helper.IsServiceIPSet(service) {
				allocator := rs.getAllocatorByClusterIP(service)
				allocator.Release(net.ParseIP(service.Spec.ClusterIP))
			}
		}
	}()

	var err error
	if !dryrun.IsDryRun(options.DryRun) {
		if service.Spec.Type != api.ServiceTypeExternalName {
			allocator := rs.getAllocatorBySpec(service)
			if releaseServiceIP, err = initClusterIP(service, allocator); err != nil {
				return nil, err
			}
		}
	}

	nodePortOp := portallocator.StartOperation(rs.serviceNodePorts, dryrun.IsDryRun(options.DryRun))
	defer nodePortOp.Finish()

	if service.Spec.Type == api.ServiceTypeNodePort || service.Spec.Type == api.ServiceTypeLoadBalancer {
		if err := initNodePorts(service, nodePortOp); err != nil {
			return nil, err
		}
	}

	// Handle ExternalTraffic related fields during service creation.
	if apiservice.NeedsHealthCheck(service) {
		if err := allocateHealthCheckNodePort(service, nodePortOp); err != nil {
			return nil, errors.NewInternalError(err)
		}
	}
	if errs := validation.ValidateServiceExternalTrafficFieldsCombination(service); len(errs) > 0 {
		return nil, errors.NewInvalid(api.Kind("Service"), service.Name, errs)
	}

	out, err := rs.services.Create(ctx, service, createValidation, options)
	if err != nil {
		err = rest.CheckGeneratedNameError(registry.Strategy, err, service)
	}

	if err == nil {
		el := nodePortOp.Commit()
		if el != nil {
			// these should be caught by an eventual reconciliation / restart
			utilruntime.HandleError(fmt.Errorf("error(s) committing service node-ports changes: %v", el))
		}

		releaseServiceIP = false
	}

	return out, err
}

func (rs *REST) Delete(ctx context.Context, id string, deleteValidation rest.ValidateObjectFunc, options *metav1.DeleteOptions) (runtime.Object, bool, error) {
	// TODO: handle graceful
	obj, _, err := rs.services.Delete(ctx, id, deleteValidation, options)
	if err != nil {
		return nil, false, err
	}

	svc := obj.(*api.Service)

	// Only perform the cleanup if this is a non-dryrun deletion
	if !dryrun.IsDryRun(options.DryRun) {
		// TODO: can leave dangling endpoints, and potentially return incorrect
		// endpoints if a new service is created with the same name
		_, _, err = rs.endpoints.Delete(ctx, id, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
		if err != nil && !errors.IsNotFound(err) {
			return nil, false, err
		}

		rs.releaseAllocatedResources(svc)
	}

	// TODO: this is duplicated from the generic storage, when this wrapper is fully removed we can drop this
	details := &metav1.StatusDetails{
		Name: svc.Name,
		UID:  svc.UID,
	}
	if info, ok := genericapirequest.RequestInfoFrom(ctx); ok {
		details.Group = info.APIGroup
		details.Kind = info.Resource // legacy behavior
	}
	status := &metav1.Status{Status: metav1.StatusSuccess, Details: details}
	return status, true, nil
}

func (rs *REST) releaseAllocatedResources(svc *api.Service) {
	if helper.IsServiceIPSet(svc) {
		allocator := rs.getAllocatorByClusterIP(svc)
		allocator.Release(net.ParseIP(svc.Spec.ClusterIP))
	}

	for _, nodePort := range collectServiceNodePorts(svc) {
		err := rs.serviceNodePorts.Release(nodePort)
		if err != nil {
			// these should be caught by an eventual reconciliation / restart
			utilruntime.HandleError(fmt.Errorf("Error releasing service %s node port %d: %v", svc.Name, nodePort, err))
		}
	}

	if apiservice.NeedsHealthCheck(svc) {
		nodePort := svc.Spec.HealthCheckNodePort
		if nodePort > 0 {
			err := rs.serviceNodePorts.Release(int(nodePort))
			if err != nil {
				// these should be caught by an eventual reconciliation / restart
				utilruntime.HandleError(fmt.Errorf("Error releasing service %s health check node port %d: %v", svc.Name, nodePort, err))
			}
		}
	}
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
		klog.Infof("Transition to LoadBalancer type service with ExternalTrafficPolicy=Local")
		if err := allocateHealthCheckNodePort(service, nodePortOp); err != nil {
			return false, errors.NewInternalError(err)
		}

	// Case 2: Transition from needs HealthCheckNodePort to don't need HealthCheckNodePort.
	// Free the existing healthCheckNodePort and clear the HealthCheckNodePort field.
	case neededHealthCheckNodePort && !needsHealthCheckNodePort:
		klog.Infof("Transition to non LoadBalancer type service or LoadBalancer type service with ExternalTrafficPolicy=Global")
		klog.V(4).Infof("Releasing healthCheckNodePort: %d", oldHealthCheckNodePort)
		nodePortOp.ReleaseDeferred(int(oldHealthCheckNodePort))
		// Clear the HealthCheckNodePort field.
		service.Spec.HealthCheckNodePort = 0

	// Case 3: Remain in needs HealthCheckNodePort.
	// Reject changing the value of the HealthCheckNodePort field.
	case neededHealthCheckNodePort && needsHealthCheckNodePort:
		if oldHealthCheckNodePort != newHealthCheckNodePort {
			klog.Warningf("Attempt to change value of health check node port DENIED")
			fldPath := field.NewPath("spec", "healthCheckNodePort")
			el := field.ErrorList{field.Invalid(fldPath, newHealthCheckNodePort,
				"cannot change healthCheckNodePort on loadBalancer service with externalTraffic=Local during update")}
			return false, errors.NewInvalid(api.Kind("Service"), service.Name, el)
		}
	}
	return true, nil
}

func (rs *REST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	oldObj, err := rs.services.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		// Support create on update, if forced to.
		if forceAllowCreate {
			obj, err := objInfo.UpdatedObject(ctx, nil)
			if err != nil {
				return nil, false, err
			}
			createdObj, err := rs.Create(ctx, obj, createValidation, &metav1.CreateOptions{DryRun: options.DryRun})
			if err != nil {
				return nil, false, err
			}
			return createdObj, true, nil
		}
		return nil, false, err
	}
	oldService := oldObj.(*api.Service)

	obj, err := objInfo.UpdatedObject(ctx, oldService)
	if err != nil {
		return nil, false, err
	}

	service := obj.(*api.Service)

	if !rest.ValidNamespace(ctx, &service.ObjectMeta) {
		return nil, false, errors.NewConflict(api.Resource("services"), service.Namespace, fmt.Errorf("Service.Namespace does not match the provided context"))
	}

	// Copy over non-user fields
	if err := rest.BeforeUpdate(registry.Strategy, ctx, service, oldService); err != nil {
		return nil, false, err
	}

	// TODO: this should probably move to strategy.PrepareForCreate()
	releaseServiceIP := false
	defer func() {
		if releaseServiceIP {
			if helper.IsServiceIPSet(service) {
				allocator := rs.getAllocatorByClusterIP(service)
				allocator.Release(net.ParseIP(service.Spec.ClusterIP))
			}
		}
	}()

	nodePortOp := portallocator.StartOperation(rs.serviceNodePorts, dryrun.IsDryRun(options.DryRun))
	defer nodePortOp.Finish()

	if !dryrun.IsDryRun(options.DryRun) {
		// Update service from ExternalName to non-ExternalName, should initialize ClusterIP.
		// Since we don't support changing the ip family of a service we don't need to handle
		// oldService.Spec.ServiceIPFamily != service.Spec.ServiceIPFamily
		if oldService.Spec.Type == api.ServiceTypeExternalName && service.Spec.Type != api.ServiceTypeExternalName {
			allocator := rs.getAllocatorBySpec(service)
			if releaseServiceIP, err = initClusterIP(service, allocator); err != nil {
				return nil, false, err
			}
		}
		// Update service from non-ExternalName to ExternalName, should release ClusterIP if exists.
		if oldService.Spec.Type != api.ServiceTypeExternalName && service.Spec.Type == api.ServiceTypeExternalName {
			if helper.IsServiceIPSet(oldService) {
				allocator := rs.getAllocatorByClusterIP(service)
				allocator.Release(net.ParseIP(oldService.Spec.ClusterIP))
			}
		}
	}
	// Update service from NodePort or LoadBalancer to ExternalName or ClusterIP, should release NodePort if exists.
	if (oldService.Spec.Type == api.ServiceTypeNodePort || oldService.Spec.Type == api.ServiceTypeLoadBalancer) &&
		(service.Spec.Type == api.ServiceTypeExternalName || service.Spec.Type == api.ServiceTypeClusterIP) {
		releaseNodePorts(oldService, nodePortOp)
	}
	// Update service from any type to NodePort or LoadBalancer, should update NodePort.
	if service.Spec.Type == api.ServiceTypeNodePort || service.Spec.Type == api.ServiceTypeLoadBalancer {
		if err := updateNodePorts(oldService, service, nodePortOp); err != nil {
			return nil, false, err
		}
	}
	// Update service from LoadBalancer to non-LoadBalancer, should remove any LoadBalancerStatus.
	if service.Spec.Type != api.ServiceTypeLoadBalancer {
		// Although loadbalancer delete is actually asynchronous, we don't need to expose the user to that complexity.
		service.Status.LoadBalancer = api.LoadBalancerStatus{}
	}

	// Handle ExternalTraffic related updates.
	success, err := rs.healthCheckNodePortUpdate(oldService, service, nodePortOp)
	if !success || err != nil {
		return nil, false, err
	}
	externalTrafficPolicyUpdate(oldService, service)
	if errs := validation.ValidateServiceExternalTrafficFieldsCombination(service); len(errs) > 0 {
		return nil, false, errors.NewInvalid(api.Kind("Service"), service.Name, errs)
	}

	out, created, err := rs.services.Update(ctx, service.Name, rest.DefaultUpdatedObjectInfo(service), createValidation, updateValidation, forceAllowCreate, options)
	if err == nil {
		el := nodePortOp.Commit()
		if el != nil {
			// problems should be fixed by an eventual reconciliation / restart
			utilruntime.HandleError(fmt.Errorf("error(s) committing NodePorts changes: %v", el))
		}

		releaseServiceIP = false
	}

	return out, created, err
}

// Implement Redirector.
var _ = rest.Redirector(&REST{})

// ResourceLocation returns a URL to which one can send traffic for the specified service.
func (rs *REST) ResourceLocation(ctx context.Context, id string) (*url.URL, http.RoundTripper, error) {
	// Allow ID as "svcname", "svcname:port", or "scheme:svcname:port".
	svcScheme, svcName, portStr, valid := utilnet.SplitSchemeNamePort(id)
	if !valid {
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("invalid service request %q", id))
	}

	// If a port *number* was specified, find the corresponding service port name
	if portNum, err := strconv.ParseInt(portStr, 10, 64); err == nil {
		obj, err := rs.services.Get(ctx, svcName, &metav1.GetOptions{})
		if err != nil {
			return nil, nil, err
		}
		svc := obj.(*api.Service)
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

	obj, err := rs.endpoints.Get(ctx, svcName, &metav1.GetOptions{})
	if err != nil {
		return nil, nil, err
	}
	eps := obj.(*api.Endpoints)
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
				addrSeed := rand.Intn(len(ss.Addresses))
				// This is a little wonky, but it's expensive to test for the presence of a Pod
				// So we repeatedly try at random and validate it, this means that for an invalid
				// service with a lot of endpoints we're going to potentially make a lot of calls,
				// but in the expected case we'll only make one.
				for try := 0; try < len(ss.Addresses); try++ {
					addr := ss.Addresses[(addrSeed+try)%len(ss.Addresses)]
					if err := isValidAddress(ctx, &addr, rs.pods); err != nil {
						utilruntime.HandleError(fmt.Errorf("Address %v isn't valid (%v)", addr, err))
						continue
					}
					ip := addr.IP
					port := int(ss.Ports[i].Port)
					return &url.URL{
						Scheme: svcScheme,
						Host:   net.JoinHostPort(ip, strconv.Itoa(port)),
					}, rs.proxyTransport, nil
				}
				utilruntime.HandleError(fmt.Errorf("Failed to find a valid address, skipping subset: %v", ss))
			}
		}
	}
	return nil, nil, errors.NewServiceUnavailable(fmt.Sprintf("no endpoints available for service %q", id))
}

func (r *REST) ConvertToTable(ctx context.Context, object runtime.Object, tableOptions runtime.Object) (*metav1.Table, error) {
	return r.services.ConvertToTable(ctx, object, tableOptions)
}

// When allocating we always use BySpec, when releasing we always use ByClusterIP
func (r *REST) getAllocatorByClusterIP(service *api.Service) ipallocator.Interface {
	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) || r.secondaryServiceIPs == nil {
		return r.serviceIPs
	}

	secondaryAllocatorCIDR := r.secondaryServiceIPs.CIDR()
	if netutil.IsIPv6String(service.Spec.ClusterIP) && netutil.IsIPv6CIDR(&secondaryAllocatorCIDR) {
		return r.secondaryServiceIPs
	}

	return r.serviceIPs
}

func (r *REST) getAllocatorBySpec(service *api.Service) ipallocator.Interface {
	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) ||
		service.Spec.IPFamily == nil ||
		r.secondaryServiceIPs == nil {
		return r.serviceIPs
	}

	secondaryAllocatorCIDR := r.secondaryServiceIPs.CIDR()
	if *(service.Spec.IPFamily) == api.IPv6Protocol && netutil.IsIPv6CIDR(&secondaryAllocatorCIDR) {
		return r.secondaryServiceIPs
	}

	return r.serviceIPs
}

func isValidAddress(ctx context.Context, addr *api.EndpointAddress, pods rest.Getter) error {
	if addr.TargetRef == nil {
		return fmt.Errorf("Address has no target ref, skipping: %v", addr)
	}
	if genericapirequest.NamespaceValue(ctx) != addr.TargetRef.Namespace {
		return fmt.Errorf("Address namespace doesn't match context namespace")
	}
	obj, err := pods.Get(ctx, addr.TargetRef.Name, &metav1.GetOptions{})
	if err != nil {
		return err
	}
	pod, ok := obj.(*api.Pod)
	if !ok {
		return fmt.Errorf("failed to cast to pod: %v", obj)
	}
	if pod == nil {
		return fmt.Errorf("pod is missing, skipping (%s/%s)", addr.TargetRef.Namespace, addr.TargetRef.Name)
	}
	if pod.Status.PodIPs[0].IP != addr.IP {
		return fmt.Errorf("pod ip doesn't match endpoint ip, skipping: %s vs %s (%s/%s)", pod.Status.PodIPs[0].IP, addr.IP, addr.TargetRef.Namespace, addr.TargetRef.Name)
	}
	return nil
}

// This is O(N), but we expect haystack to be small;
// so small that we expect a linear search to be faster
func containsNumber(haystack []int, needle int) bool {
	for _, v := range haystack {
		if v == needle {
			return true
		}
	}
	return false
}

// This is O(N), but we expect serviceNodePorts to be small;
// so small that we expect a linear search to be faster
func containsNodePort(serviceNodePorts []ServiceNodePort, serviceNodePort ServiceNodePort) bool {
	for _, snp := range serviceNodePorts {
		if snp == serviceNodePort {
			return true
		}
	}
	return false
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
func allocateHealthCheckNodePort(service *api.Service, nodePortOp *portallocator.PortAllocationOperation) error {
	healthCheckNodePort := service.Spec.HealthCheckNodePort
	if healthCheckNodePort != 0 {
		// If the request has a health check nodePort in mind, attempt to reserve it.
		err := nodePortOp.Allocate(int(healthCheckNodePort))
		if err != nil {
			return fmt.Errorf("failed to allocate requested HealthCheck NodePort %v: %v",
				healthCheckNodePort, err)
		}
		klog.V(4).Infof("Reserved user requested healthCheckNodePort: %d", healthCheckNodePort)
	} else {
		// If the request has no health check nodePort specified, allocate any.
		healthCheckNodePort, err := nodePortOp.AllocateNext()
		if err != nil {
			return fmt.Errorf("failed to allocate a HealthCheck NodePort %v: %v", healthCheckNodePort, err)
		}
		service.Spec.HealthCheckNodePort = int32(healthCheckNodePort)
		klog.V(4).Infof("Reserved allocated healthCheckNodePort: %d", healthCheckNodePort)
	}
	return nil
}

// The return bool value indicates if a cluster IP is allocated successfully.
func initClusterIP(service *api.Service, allocator ipallocator.Interface) (bool, error) {
	switch {
	case service.Spec.ClusterIP == "":
		// Allocate next available.
		ip, err := allocator.AllocateNext()
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
		if err := allocator.Allocate(net.ParseIP(service.Spec.ClusterIP)); err != nil {
			// TODO: when validation becomes versioned, this gets more complicated.
			el := field.ErrorList{field.Invalid(field.NewPath("spec", "clusterIP"), service.Spec.ClusterIP, err.Error())}
			return false, errors.NewInvalid(api.Kind("Service"), service.Name, el)
		}
		return true, nil
	}

	return false, nil
}

func initNodePorts(service *api.Service, nodePortOp *portallocator.PortAllocationOperation) error {
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

func updateNodePorts(oldService, newService *api.Service, nodePortOp *portallocator.PortAllocationOperation) error {
	oldNodePortsNumbers := collectServiceNodePorts(oldService)
	newNodePorts := []ServiceNodePort{}
	portAllocated := map[int]bool{}

	for i := range newService.Spec.Ports {
		servicePort := &newService.Spec.Ports[i]
		nodePort := ServiceNodePort{Protocol: servicePort.Protocol, NodePort: servicePort.NodePort}
		if nodePort.NodePort != 0 {
			if !containsNumber(oldNodePortsNumbers, int(nodePort.NodePort)) && !portAllocated[int(nodePort.NodePort)] {
				err := nodePortOp.Allocate(int(nodePort.NodePort))
				if err != nil {
					el := field.ErrorList{field.Invalid(field.NewPath("spec", "ports").Index(i).Child("nodePort"), nodePort.NodePort, err.Error())}
					return errors.NewInvalid(api.Kind("Service"), newService.Name, el)
				}
				portAllocated[int(nodePort.NodePort)] = true
			}
		} else {
			nodePortNumber, err := nodePortOp.AllocateNext()
			if err != nil {
				// TODO: what error should be returned here?  It's not a
				// field-level validation failure (the field is valid), and it's
				// not really an internal error.
				return errors.NewInternalError(fmt.Errorf("failed to allocate a nodePort: %v", err))
			}
			servicePort.NodePort = int32(nodePortNumber)
			nodePort.NodePort = servicePort.NodePort
		}
		if containsNodePort(newNodePorts, nodePort) {
			return fmt.Errorf("duplicate nodePort: %v", nodePort)
		}
		newNodePorts = append(newNodePorts, nodePort)
	}

	newNodePortsNumbers := collectServiceNodePorts(newService)

	// The comparison loops are O(N^2), but we don't expect N to be huge
	// (there's a hard-limit at 2^16, because they're ports; and even 4 ports would be a lot)
	for _, oldNodePortNumber := range oldNodePortsNumbers {
		if containsNumber(newNodePortsNumbers, oldNodePortNumber) {
			continue
		}
		nodePortOp.ReleaseDeferred(int(oldNodePortNumber))
	}

	return nil
}

func releaseNodePorts(service *api.Service, nodePortOp *portallocator.PortAllocationOperation) {
	nodePorts := collectServiceNodePorts(service)

	for _, nodePort := range nodePorts {
		nodePortOp.ReleaseDeferred(nodePort)
	}
}

func collectServiceNodePorts(service *api.Service) []int {
	servicePorts := []int{}
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		if servicePort.NodePort != 0 {
			servicePorts = append(servicePorts, int(servicePort.NodePort))
		}
	}
	return servicePorts
}
