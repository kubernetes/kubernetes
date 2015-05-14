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

package service

import (
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/endpoint"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/minion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service/ipallocator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/fielderrors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

// REST adapts a service registry into apiserver's RESTStorage model.
type REST struct {
	registry    Registry
	machines    minion.Registry
	endpoints   endpoint.Registry
	portals     ipallocator.Interface
	portMgr     *portAllocator
	clusterName string
}

// NewStorage returns a new REST.
func NewStorage(registry Registry, machines minion.Registry, endpoints endpoint.Registry, portals ipallocator.Interface,
	publicServicePorts util.PortRange, clusterName string) *REST {

	// TODO: Before we can replicate masters, this has to be synced (e.g. lives in etcd)
	pa := newPortAllocator(&publicServicePorts)
	if pa == nil {
		glog.Fatalf("Failed to create a port allocator. Is port-range '%v' valid?", publicServicePorts)
	}

	reloadServiceStateFromStorage(pa, registry)

	return &REST{
		registry:    registry,
		machines:    machines,
		endpoints:   endpoints,
		portals:     portals,
		portMgr:     pa,
		clusterName: clusterName,
	}
}

// Helper: mark all previously allocated IPs & ports in the allocator.
func reloadServiceStateFromStorage(pa *portAllocator, registry Registry) {
	services, err := registry.ListServices(api.NewContext())
	if err != nil {
		// This is really bad.
		glog.Errorf("can't list services to init service REST: %v", err)
		return
	}
	for i := range services.Items {
		service := &services.Items[i]
		for j := range service.Spec.Ports {
			servicePort := &service.Spec.Ports[j]
			if servicePort.PublicPort != 0 {
				if err := pa.Allocate(servicePort.PublicPort); err != nil {
					// This is really bad.
					glog.Errorf("service %q ServicePort %s could not be allocated: %v", service.Name, servicePort.PublicPort, err)
				}
			}
		}
	}
}

// Encapsulates the semantics of a port allocation 'transaction'
// it is better to leak ports than to double-allocate them
// so we allocate immediately, but defer release
// on commit we best-effort release the deferred releases
// on rollback we best-effort release any allocations we did
type portAllocationOperation struct {
	pa              *portAllocator
	allocated       []int
	releaseDeferred []int
	ShouldRollback  bool
}

func (op *portAllocationOperation) Init(pa *portAllocator) {
	op.pa = pa
	op.allocated = []int{}
	op.releaseDeferred = []int{}
	op.ShouldRollback = true
}

func (op *portAllocationOperation) Rollback() []error {
	errors := []error{}

	for _, allocated := range op.allocated {
		err := op.pa.Release(allocated)
		if err != nil {
			errors = append(errors, err)
		}
	}

	if len(errors) == 0 {
		return nil
	}
	return errors
}

func (op *portAllocationOperation) Commit() []error {
	errors := []error{}

	for _, release := range op.releaseDeferred {
		err := op.pa.Release(release)
		if err != nil {
			errors = append(errors, err)
		}
	}

	if len(errors) == 0 {
		return nil
	}
	return errors
}

func (op *portAllocationOperation) Allocate(port int) error {
	err := op.pa.Allocate(port)
	if err == nil {
		op.allocated = append(op.allocated, port)
	}
	return err
}

func (op *portAllocationOperation) AllocateNext() (int, error) {
	port, err := op.pa.AllocateNext()
	if err == nil {
		op.allocated = append(op.allocated, port)
	}
	return port, err
}

func (op *portAllocationOperation) ReleaseDeferred(port int) {
	op.releaseDeferred = append(op.releaseDeferred, port)
}

func contains(haystack []int, needle int) bool {
	for _, v := range haystack {
		if v == needle {
			return true
		}
	}
	return false
}

func (rs *REST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	service := obj.(*api.Service)

	if err := rest.BeforeCreate(rest.Services, ctx, obj); err != nil {
		return nil, err
	}

	releaseServiceIP := false
	defer func() {
		if releaseServiceIP {
			if api.IsServiceIPSet(service) {
				rs.portals.Release(net.ParseIP(service.Spec.PortalIP))
			}
		}
	}()

	var publicPortOp portAllocationOperation
	publicPortOp.Init(rs.portMgr)
	defer func() {
		if publicPortOp.ShouldRollback {
			publicPortOp.Rollback()
		}
	}()

	if api.IsServiceIPRequested(service) {
		// Allocate next available.
		ip, err := rs.portals.AllocateNext()
		if err != nil {
			el := fielderrors.ValidationErrorList{fielderrors.NewFieldInvalid("spec.portalIP", service.Spec.PortalIP, err.Error())}
			return nil, errors.NewInvalid("Service", service.Name, el)
		}
		service.Spec.PortalIP = ip.String()
		releaseServiceIP = true
	} else if api.IsServiceIPSet(service) {
		// Try to respect the requested IP.
		if err := rs.portals.Allocate(net.ParseIP(service.Spec.PortalIP)); err != nil {
			el := fielderrors.ValidationErrorList{fielderrors.NewFieldInvalid("spec.portalIP", service.Spec.PortalIP, err.Error())}
			return nil, errors.NewInvalid("Service", service.Name, el)
		}
		releaseServiceIP = true
	}

	assignPublicPorts := shouldAssignPublicPorts(service)
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		if servicePort.PublicPort != 0 {
			err := publicPortOp.Allocate(servicePort.PublicPort)
			if err != nil {
				return nil, err
			}
		} else if assignPublicPorts {
			publicPort, err := publicPortOp.AllocateNext()
			if err != nil {
				return nil, err
			}
			servicePort.PublicPort = publicPort
		}
	}

	out, err := rs.registry.CreateService(ctx, service)
	if err != nil {
		err = rest.CheckGeneratedNameError(rest.Services, err, service)
	}

	if err == nil {
		el := publicPortOp.Commit()
		if el != nil {
			// these should be caught by an eventual reconciliation / restart
			glog.Errorf("error(s) committing public-ports changes: %v", el)
		}
		publicPortOp.ShouldRollback = false

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

	if api.IsServiceIPSet(service) {
		rs.portals.Release(net.ParseIP(service.Spec.PortalIP))
	}

	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		if servicePort.PublicPort != 0 {
			err := rs.portMgr.Release(servicePort.PublicPort)
			if err != nil {
				// these should be caught by an eventual reconciliation / restart
				glog.Errorf("Error releasing service %s public port %d: %v", service.Name, servicePort.PublicPort, err)
			}
		}
	}

	return &api.Status{Status: api.StatusSuccess}, nil
}

func (rs *REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	service, err := rs.registry.GetService(ctx, id)
	if err != nil {
		return nil, err
	}
	return service, err
}

func (rs *REST) List(ctx api.Context, label labels.Selector, field fields.Selector) (runtime.Object, error) {
	list, err := rs.registry.ListServices(ctx)
	if err != nil {
		return nil, err
	}
	var filtered []api.Service
	for _, service := range list.Items {
		if label.Matches(labels.Set(service.Labels)) {
			filtered = append(filtered, service)
		}
	}
	list.Items = filtered
	return list, err
}

// Watch returns Services events via a watch.Interface.
// It implements rest.Watcher.
func (rs *REST) Watch(ctx api.Context, label labels.Selector, field fields.Selector, resourceVersion string) (watch.Interface, error) {
	return rs.registry.WatchServices(ctx, label, field, resourceVersion)
}

func (*REST) New() runtime.Object {
	return &api.Service{}
}

func (*REST) NewList() runtime.Object {
	return &api.ServiceList{}
}

func (rs *REST) Update(ctx api.Context, obj runtime.Object) (runtime.Object, bool, error) {
	service := obj.(*api.Service)
	if !api.ValidNamespace(ctx, &service.ObjectMeta) {
		return nil, false, errors.NewConflict("service", service.Namespace, fmt.Errorf("Service.Namespace does not match the provided context"))
	}

	oldService, err := rs.registry.GetService(ctx, service.Name)
	if err != nil {
		return nil, false, err
	}

	// Copy over non-user fields
	// TODO: make this a merge function
	if errs := validation.ValidateServiceUpdate(oldService, service); len(errs) > 0 {
		return nil, false, errors.NewInvalid("service", service.Name, errs)
	}

	var publicPortOp portAllocationOperation
	publicPortOp.Init(rs.portMgr)
	defer func() {
		if publicPortOp.ShouldRollback {
			publicPortOp.Rollback()
		}
	}()

	assignPublicPorts := shouldAssignPublicPorts(service)

	oldPublicPorts := collectServicePublicPorts(oldService)

	newPublicPorts := []int{}
	if assignPublicPorts {
		for i := range service.Spec.Ports {
			servicePort := &service.Spec.Ports[i]
			publicPort := servicePort.PublicPort
			if publicPort != 0 {
				if !contains(oldPublicPorts, publicPort) {
					err := publicPortOp.Allocate(publicPort)
					if err != nil {
						return nil, false, err
					}
				}
			} else {
				publicPort, err = publicPortOp.AllocateNext()
				if err != nil {
					return nil, false, err
				}
				servicePort.PublicPort = publicPort
			}
			// Detect duplicate public ports; this should have been caught by validation, so we panic
			if contains(newPublicPorts, publicPort) {
				panic("duplicate public port")
			}
			newPublicPorts = append(newPublicPorts, publicPort)
		}
	} else {
		// Validate should have validated that publicPort == 0
	}

	// The comparison loops are O(N^2), but we don't expect N to be huge
	// (there's a hard-limit at 2^16, because they're ports; and even 4 ports would be a lot)
	for _, oldPublicPort := range oldPublicPorts {
		if !contains(newPublicPorts, oldPublicPort) {
			continue
		}
		publicPortOp.ReleaseDeferred(oldPublicPort)
	}

	out, err := rs.registry.UpdateService(ctx, service)

	if err == nil {
		el := publicPortOp.Commit()
		if el != nil {
			// problems should be fixed by an eventual reconciliation / restart
			glog.Errorf("error(s) committing public-ports changes: %v", el)
		}
		publicPortOp.ShouldRollback = false
	}

	return out, false, err
}

func collectServicePublicPorts(service *api.Service) []int {
	servicePorts := []int{}
	for i := range service.Spec.Ports {
		servicePort := &service.Spec.Ports[i]
		if servicePort.PublicPort != 0 {
			servicePorts = append(servicePorts, servicePort.PublicPort)
		}
	}
	return servicePorts
}

// Implement Redirector.
var _ = rest.Redirector(&REST{})

// ResourceLocation returns a URL to which one can send traffic for the specified service.
func (rs *REST) ResourceLocation(ctx api.Context, id string) (*url.URL, http.RoundTripper, error) {
	// Allow ID as "svcname" or "svcname:port".
	svcName, portStr, valid := util.SplitPort(id)
	if !valid {
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("invalid service request %q", id))
	}

	eps, err := rs.endpoints.GetEndpoints(ctx, svcName)
	if err != nil {
		return nil, nil, err
	}
	if len(eps.Subsets) == 0 {
		return nil, nil, fmt.Errorf("no endpoints available for %q", svcName)
	}
	// Pick a random Subset to start searching from.
	ssSeed := rand.Intn(len(eps.Subsets))
	// Find a Subset that has the port.
	for ssi := 0; ssi < len(eps.Subsets); ssi++ {
		ss := &eps.Subsets[(ssSeed+ssi)%len(eps.Subsets)]
		for i := range ss.Ports {
			if ss.Ports[i].Name == portStr {
				// Pick a random address.
				ip := ss.Addresses[rand.Intn(len(ss.Addresses))].IP
				port := ss.Ports[i].Port
				// We leave off the scheme ('http://') because we have no idea what sort of server
				// is listening at this endpoint.
				return &url.URL{
					Host: net.JoinHostPort(ip, strconv.Itoa(port)),
				}, nil, nil
			}
		}
	}
	return nil, nil, fmt.Errorf("no endpoints available for %q", id)
}

func shouldAssignPublicPorts(service *api.Service) bool {
	switch service.Spec.Visibility {
	case api.VisibilityTypeLoadBalancer:
		return true
	case api.VisibilityTypePublic:
		return true
	case api.VisibilityTypeCluster:
		return false
	default:
		glog.Errorf("Unknown visibility value: %v", service.Spec.Visibility)
		return false
	}
}
