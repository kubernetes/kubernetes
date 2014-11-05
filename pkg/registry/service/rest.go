/*
Copyright 2014 Google Inc. All rights reserved.

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
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/minion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

// REST adapts a service registry into apiserver's RESTStorage model.
type REST struct {
	registry  Registry
	cloud     cloudprovider.Interface
	machines  minion.Registry
	portalMgr *ipAllocator
}

// NewREST returns a new REST.
func NewREST(registry Registry, cloud cloudprovider.Interface, machines minion.Registry, portalNet *net.IPNet) *REST {
	// TODO: Before we can replicate masters, this has to be synced (e.g. lives in etcd)
	ipa := newIPAllocator(portalNet)
	if ipa == nil {
		glog.Fatalf("Failed to create an IP allocator. Is subnet '%v' valid?", portalNet)
	}
	reloadIPsFromStorage(ipa, registry)

	return &REST{
		registry:  registry,
		cloud:     cloud,
		machines:  machines,
		portalMgr: ipa,
	}
}

// Helper: mark all previously allocated IPs in the allocator.
func reloadIPsFromStorage(ipa *ipAllocator, registry Registry) {
	services, err := registry.ListServices(api.NewContext())
	if err != nil {
		// This is really bad.
		glog.Errorf("can't list services to init service REST: %s", err)
		return
	}
	for i := range services.Items {
		service := &services.Items[i]
		if service.Spec.PortalIP == "" {
			glog.Warningf("service %q has no PortalIP", service.Name)
			continue
		}
		if err := ipa.Allocate(net.ParseIP(service.Spec.PortalIP)); err != nil {
			// This is really bad.
			glog.Errorf("service %q PortalIP %s could not be allocated: %s", service.Name, service.Spec.PortalIP, err)
		}
	}
}

func (rs *REST) Create(ctx api.Context, obj runtime.Object) (<-chan apiserver.RESTResult, error) {

	service := obj.(*api.Service)
	if !api.ValidNamespace(ctx, &service.ObjectMeta) {
		return nil, errors.NewConflict("service", service.Namespace, fmt.Errorf("Service.Namespace does not match the provided context"))
	}
	if errs := validation.ValidateService(service, rs.registry, ctx); len(errs) > 0 {
		return nil, errors.NewInvalid("service", service.Name, errs)
	}

	service.CreationTimestamp = util.Now()

	if service.Spec.PortalIP == "" {
		// Allocate next available.
		if ip, err := rs.portalMgr.AllocateNext(); err != nil {
			return nil, err
		} else {
			service.Spec.PortalIP = ip.String()
		}
	} else {
		// Try to respect the requested IP.
		if err := rs.portalMgr.Allocate(net.ParseIP(service.Spec.PortalIP)); err != nil {
			// TODO: Differentiate "IP already allocated" from real errors.
			el := errors.ValidationErrorList{errors.NewFieldInvalid("spec.portalIP", service.Spec.PortalIP)}
			return nil, errors.NewInvalid("service", service.Name, el)
		}
	}

	return apiserver.MakeAsync(func() (runtime.Object, error) {
		// TODO: Consider moving this to a rectification loop, so that we make/remove external load balancers
		// correctly no matter what http operations happen.
		service.Spec.ProxyPort = 0
		if service.Spec.CreateExternalLoadBalancer {
			if rs.cloud == nil {
				return nil, fmt.Errorf("requested an external service, but no cloud provider supplied.")
			}
			balancer, ok := rs.cloud.TCPLoadBalancer()
			if !ok {
				return nil, fmt.Errorf("The cloud provider does not support external TCP load balancers.")
			}
			zones, ok := rs.cloud.Zones()
			if !ok {
				return nil, fmt.Errorf("The cloud provider does not support zone enumeration.")
			}
			hosts, err := rs.machines.ListMinions(ctx)
			if err != nil {
				return nil, err
			}
			zone, err := zones.GetZone()
			if err != nil {
				return nil, err
			}
			err = balancer.CreateTCPLoadBalancer(service.Name, zone.Region, service.Spec.Port, hostsFromMinionList(hosts))
			if err != nil {
				return nil, err
			}
			// External load-balancers require a known port for the service proxy.
			// TODO: If we end up brokering HostPorts between Pods and Services, this can be any port.
			service.Spec.ProxyPort = service.Spec.Port
		}
		err := rs.registry.CreateService(ctx, service)
		if err != nil {
			return nil, err
		}
		return rs.registry.GetService(ctx, service.Name)
	}), nil
}

func hostsFromMinionList(list *api.MinionList) []string {
	result := make([]string, len(list.Items))
	for ix := range list.Items {
		result[ix] = list.Items[ix].Name
	}
	return result
}

func (rs *REST) Delete(ctx api.Context, id string) (<-chan apiserver.RESTResult, error) {
	service, err := rs.registry.GetService(ctx, id)
	if err != nil {
		return nil, err
	}
	rs.portalMgr.Release(net.ParseIP(service.Spec.PortalIP))
	return apiserver.MakeAsync(func() (runtime.Object, error) {
		rs.deleteExternalLoadBalancer(service)
		return &api.Status{Status: api.StatusSuccess}, rs.registry.DeleteService(ctx, id)
	}), nil
}

func (rs *REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	service, err := rs.registry.GetService(ctx, id)
	if err != nil {
		return nil, err
	}
	return service, err
}

// TODO: implement field selector?
func (rs *REST) List(ctx api.Context, label, field labels.Selector) (runtime.Object, error) {
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
// It implements apiserver.ResourceWatcher.
func (rs *REST) Watch(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error) {
	return rs.registry.WatchServices(ctx, label, field, resourceVersion)
}

func (*REST) New() runtime.Object {
	return &api.Service{}
}

// GetServiceEnvironmentVariables populates a list of environment variables that are use
// in the container environment to get access to services.
func GetServiceEnvironmentVariables(ctx api.Context, registry Registry, machine string) ([]api.EnvVar, error) {
	var result []api.EnvVar
	services, err := registry.ListServices(ctx)
	if err != nil {
		return result, err
	}
	for _, service := range services.Items {
		// Host
		name := makeEnvVariableName(service.Name) + "_SERVICE_HOST"
		result = append(result, api.EnvVar{Name: name, Value: service.Spec.PortalIP})
		// Port
		name = makeEnvVariableName(service.Name) + "_SERVICE_PORT"
		result = append(result, api.EnvVar{Name: name, Value: strconv.Itoa(service.Spec.Port)})
		// Docker-compatible vars.
		result = append(result, makeLinkVariables(service)...)
	}
	return result, nil
}

func (rs *REST) Update(ctx api.Context, obj runtime.Object) (<-chan apiserver.RESTResult, error) {
	service := obj.(*api.Service)
	if !api.ValidNamespace(ctx, &service.ObjectMeta) {
		return nil, errors.NewConflict("service", service.Namespace, fmt.Errorf("Service.Namespace does not match the provided context"))
	}
	if errs := validation.ValidateService(service, rs.registry, ctx); len(errs) > 0 {
		return nil, errors.NewInvalid("service", service.Name, errs)
	}
	return apiserver.MakeAsync(func() (runtime.Object, error) {
		cur, err := rs.registry.GetService(ctx, service.Name)
		if err != nil {
			return nil, err
		}
		if service.Spec.PortalIP != cur.Spec.PortalIP {
			// TODO: Would be nice to pass "field is immutable" to users.
			el := errors.ValidationErrorList{errors.NewFieldInvalid("spec.portalIP", service.Spec.PortalIP)}
			return nil, errors.NewInvalid("service", service.Name, el)
		}
		// Copy over non-user fields.
		service.Spec.ProxyPort = cur.Spec.ProxyPort
		// TODO: check to see if external load balancer status changed
		err = rs.registry.UpdateService(ctx, service)
		if err != nil {
			return nil, err
		}
		return rs.registry.GetService(ctx, service.Name)
	}), nil
}

// ResourceLocation returns a URL to which one can send traffic for the specified service.
func (rs *REST) ResourceLocation(ctx api.Context, id string) (string, error) {
	e, err := rs.registry.GetEndpoints(ctx, id)
	if err != nil {
		return "", err
	}
	if len(e.Endpoints) == 0 {
		return "", fmt.Errorf("no endpoints available for %v", id)
	}
	// We leave off the scheme ('http://') because we have no idea what sort of server
	// is listening at this endpoint.
	return e.Endpoints[rand.Intn(len(e.Endpoints))], nil
}

func (rs *REST) deleteExternalLoadBalancer(service *api.Service) error {
	if !service.Spec.CreateExternalLoadBalancer || rs.cloud == nil {
		return nil
	}
	zones, ok := rs.cloud.Zones()
	if !ok {
		// We failed to get zone enumerator.
		// As this should have failed when we tried in "create" too,
		// assume external load balancer was never created.
		return nil
	}
	balancer, ok := rs.cloud.TCPLoadBalancer()
	if !ok {
		// See comment above.
		return nil
	}
	zone, err := zones.GetZone()
	if err != nil {
		return err
	}
	if err := balancer.DeleteTCPLoadBalancer(service.Name, zone.Region); err != nil {
		return err
	}
	return nil
}

func makeEnvVariableName(str string) string {
	return strings.ToUpper(strings.Replace(str, "-", "_", -1))
}

func makeLinkVariables(service api.Service) []api.EnvVar {
	prefix := makeEnvVariableName(service.Name)
	protocol := string(api.ProtocolTCP)
	if service.Spec.Protocol != "" {
		protocol = string(service.Spec.Protocol)
	}
	portPrefix := fmt.Sprintf("%s_PORT_%d_%s", prefix, service.Spec.Port, strings.ToUpper(protocol))
	return []api.EnvVar{
		{
			Name:  prefix + "_PORT",
			Value: fmt.Sprintf("%s://%s:%d", strings.ToLower(protocol), service.Spec.PortalIP, service.Spec.Port),
		},
		{
			Name:  portPrefix,
			Value: fmt.Sprintf("%s://%s:%d", strings.ToLower(protocol), service.Spec.PortalIP, service.Spec.Port),
		},
		{
			Name:  portPrefix + "_PROTO",
			Value: strings.ToLower(protocol),
		},
		{
			Name:  portPrefix + "_PORT",
			Value: strconv.Itoa(service.Spec.Port),
		},
		{
			Name:  portPrefix + "_ADDR",
			Value: service.Spec.PortalIP,
		},
	}
}
