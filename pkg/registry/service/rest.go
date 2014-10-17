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
		s := &services.Items[i]
		if s.PortalIP == "" {
			glog.Warningf("service %q has no PortalIP", s.ID)
			continue
		}
		if err := ipa.Allocate(net.ParseIP(s.PortalIP)); err != nil {
			// This is really bad.
			glog.Errorf("service %q PortalIP %s could not be allocated: %s", s.ID, s.PortalIP, err)
		}
	}
}

func (rs *REST) Create(ctx api.Context, obj runtime.Object) (<-chan runtime.Object, error) {
	srv := obj.(*api.Service)
	if !api.ValidNamespace(ctx, &srv.TypeMeta) {
		return nil, errors.NewConflict("service", srv.Namespace, fmt.Errorf("Service.Namespace does not match the provided context"))
	}
	if errs := validation.ValidateService(srv); len(errs) > 0 {
		return nil, errors.NewInvalid("service", srv.ID, errs)
	}

	srv.CreationTimestamp = util.Now()

	if ip, err := rs.portalMgr.AllocateNext(); err != nil {
		return nil, err
	} else {
		srv.PortalIP = ip.String()
	}

	return apiserver.MakeAsync(func() (runtime.Object, error) {
		// TODO: Consider moving this to a rectification loop, so that we make/remove external load balancers
		// correctly no matter what http operations happen.
		srv.ProxyPort = 0
		if srv.CreateExternalLoadBalancer {
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
			err = balancer.CreateTCPLoadBalancer(srv.ID, zone.Region, srv.Port, hostsFromMinionList(hosts))
			if err != nil {
				return nil, err
			}
			// External load-balancers require a known port for the service proxy.
			// TODO: If we end up brokering HostPorts between Pods and Services, this can be any port.
			srv.ProxyPort = srv.Port
		}
		err := rs.registry.CreateService(ctx, srv)
		if err != nil {
			return nil, err
		}
		return rs.registry.GetService(ctx, srv.ID)
	}), nil
}

func hostsFromMinionList(list *api.MinionList) []string {
	result := make([]string, len(list.Items))
	for ix := range list.Items {
		result[ix] = list.Items[ix].ID
	}
	return result
}

func (rs *REST) Delete(ctx api.Context, id string) (<-chan runtime.Object, error) {
	service, err := rs.registry.GetService(ctx, id)
	if err != nil {
		return nil, err
	}
	rs.portalMgr.Release(net.ParseIP(service.PortalIP))
	return apiserver.MakeAsync(func() (runtime.Object, error) {
		rs.deleteExternalLoadBalancer(service)
		return &api.Status{Status: api.StatusSuccess}, rs.registry.DeleteService(ctx, id)
	}), nil
}

func (rs *REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	s, err := rs.registry.GetService(ctx, id)
	if err != nil {
		return nil, err
	}
	return s, err
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
		name := makeEnvVariableName(service.ID) + "_SERVICE_HOST"
		result = append(result, api.EnvVar{Name: name, Value: service.PortalIP})
		// Port
		name = makeEnvVariableName(service.ID) + "_SERVICE_PORT"
		result = append(result, api.EnvVar{Name: name, Value: strconv.Itoa(service.Port)})
		// Docker-compatible vars.
		result = append(result, makeLinkVariables(service)...)
	}
	return result, nil
}

func (rs *REST) Update(ctx api.Context, obj runtime.Object) (<-chan runtime.Object, error) {
	srv := obj.(*api.Service)
	if !api.ValidNamespace(ctx, &srv.TypeMeta) {
		return nil, errors.NewConflict("service", srv.Namespace, fmt.Errorf("Service.Namespace does not match the provided context"))
	}
	if errs := validation.ValidateService(srv); len(errs) > 0 {
		return nil, errors.NewInvalid("service", srv.ID, errs)
	}
	return apiserver.MakeAsync(func() (runtime.Object, error) {
		cur, err := rs.registry.GetService(ctx, srv.ID)
		if err != nil {
			return nil, err
		}
		// Copy over non-user fields.
		srv.PortalIP = cur.PortalIP
		srv.ProxyPort = cur.ProxyPort
		// TODO: check to see if external load balancer status changed
		err = rs.registry.UpdateService(ctx, srv)
		if err != nil {
			return nil, err
		}
		return rs.registry.GetService(ctx, srv.ID)
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
	if !service.CreateExternalLoadBalancer || rs.cloud == nil {
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
	if err := balancer.DeleteTCPLoadBalancer(service.TypeMeta.ID, zone.Region); err != nil {
		return err
	}
	return nil
}

func makeEnvVariableName(str string) string {
	return strings.ToUpper(strings.Replace(str, "-", "_", -1))
}

func makeLinkVariables(service api.Service) []api.EnvVar {
	prefix := makeEnvVariableName(service.ID)
	protocol := string(api.ProtocolTCP)
	if service.Protocol != "" {
		protocol = string(service.Protocol)
	}
	portPrefix := fmt.Sprintf("%s_PORT_%d_%s", prefix, service.Port, strings.ToUpper(protocol))
	return []api.EnvVar{
		{
			Name:  prefix + "_PORT",
			Value: fmt.Sprintf("%s://%s:%d", strings.ToLower(protocol), service.PortalIP, service.Port),
		},
		{
			Name:  portPrefix,
			Value: fmt.Sprintf("%s://%s:%d", strings.ToLower(protocol), service.PortalIP, service.Port),
		},
		{
			Name:  portPrefix + "_PROTO",
			Value: strings.ToLower(protocol),
		},
		{
			Name:  portPrefix + "_PORT",
			Value: strconv.Itoa(service.Port),
		},
		{
			Name:  portPrefix + "_ADDR",
			Value: service.PortalIP,
		},
	}
}
