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
	"net"
	"strconv"
	"strings"
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/minion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// RegistryStorage adapts a service registry into apiserver's RESTStorage model.
type RegistryStorage struct {
	registry  Registry
	cloud     cloudprovider.Interface
	machines  minion.Registry
	portalMgr *IPAllocator
}

//FIXME: needs a new pkg, maybe util
type IPAllocator struct {
	subnet net.IPNet
	//FIXME: needs to be smarter, for now a bitmap will suffice
	lock sync.Mutex
	//FIXME: assumes /24
	used [32]byte
}

// NewIPAllocator creates and intializes a new IPAllocator object.
// FIXME: resync from storage at startup.
func NewIPAllocator(subnet net.IPNet) *IPAllocator {
	return &IPAllocator{
		subnet: subnet,
	}
}

// Allocate allocates and returns a new IP.
//FIXME: error if we are "full"?
func (ipa *IPAllocator) Allocate() net.IP {
	ipa.lock.Lock()
	defer ipa.lock.Unlock()
	for i := range ipa.used {
		if ipa.used[i] != 0xff {
			free := ^ipa.used[i]
			//FIXME: not right - need to get bit index, not value
			//FIXME: don't use the network address
			next := free & ^(free - 1)
			ip := ipa.subnet.IP
			ip[3] = byte(i*8) + next
			//FIXME: check not to use the broadcast addr
			return ip
		}
	}
	//FIXME: error
	return nil
}

// Release de-allocates an IP.
//FIXME: error if that IP is nil, not allocated, or IP subnet does not match
func (ipa *IPAllocator) Release(ip net.IP) {
	ipa.lock.Lock()
	defer ipa.lock.Unlock()
	host := ip[3]
	i := host / 8
	m := byte(1 << byte(host%8))
	ipa.used[i] = ipa.used[i] &^ m
}

// NewRegistryStorage returns a new RegistryStorage.
func NewRegistryStorage(registry Registry, cloud cloudprovider.Interface, machines minion.Registry, portalNet net.IPNet) apiserver.RESTStorage {
	return &RegistryStorage{
		registry:  registry,
		cloud:     cloud,
		machines:  machines,
		portalMgr: NewIPAllocator(portalNet),
	}
}

func (rs *RegistryStorage) Create(obj interface{}) (<-chan interface{}, error) {
	srv := obj.(*api.Service)
	if errs := api.ValidateService(srv); len(errs) > 0 {
		return nil, apiserver.NewInvalidErr("service", srv.ID, errs)
	}

	srv.CreationTimestamp = util.Now()
	srv.PortalIP = rs.portalMgr.Allocate()
	return apiserver.MakeAsync(func() (interface{}, error) {
		// TODO: Consider moving this to a rectification loop, so that we make/remove external load balancers
		// correctly no matter what http operations happen.
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
			hosts, err := rs.machines.List()
			if err != nil {
				return nil, err
			}
			zone, err := zones.GetZone()
			if err != nil {
				return nil, err
			}
			err = balancer.CreateTCPLoadBalancer(srv.ID, zone.Region, srv.Port, hosts)
			if err != nil {
				return nil, err
			}
		}
		err := rs.registry.CreateService(*srv)
		if err != nil {
			return nil, err
		}
		return rs.registry.GetService(srv.ID)
	}), nil
}

func (rs *RegistryStorage) Delete(id string) (<-chan interface{}, error) {
	service, err := rs.registry.GetService(id)
	if err != nil {
		return nil, err
	}
	rs.portalMgr.Release(service.PortalIP)
	return apiserver.MakeAsync(func() (interface{}, error) {
		rs.deleteExternalLoadBalancer(service)
		return &api.Status{Status: api.StatusSuccess}, rs.registry.DeleteService(id)
	}), nil
}

func (rs *RegistryStorage) Get(id string) (interface{}, error) {
	s, err := rs.registry.GetService(id)
	if err != nil {
		return nil, err
	}
	return s, err
}

func (rs *RegistryStorage) List(selector labels.Selector) (interface{}, error) {
	list, err := rs.registry.ListServices()
	if err != nil {
		return nil, err
	}
	var filtered []api.Service
	for _, service := range list.Items {
		if selector.Matches(labels.Set(service.Labels)) {
			filtered = append(filtered, service)
		}
	}
	list.Items = filtered
	return list, err
}

func (rs RegistryStorage) New() interface{} {
	return &api.Service{}
}

// GetServiceEnvironmentVariables populates a list of environment variables that are use
// in the container environment to get access to services.
func GetServiceEnvironmentVariables(registry Registry, machine string) ([]api.EnvVar, error) {
	var result []api.EnvVar
	services, err := registry.ListServices()
	if err != nil {
		return result, err
	}
	for _, service := range services.Items {
		name := makeEnvVariableName(service.ID) + "_SERVICE_PORT"
		value := strconv.Itoa(service.Port)
		result = append(result, api.EnvVar{Name: name, Value: value})
		result = append(result, makeLinkVariables(service, machine)...)
	}
	result = append(result, api.EnvVar{Name: "SERVICE_HOST", Value: machine})
	return result, nil
}

func (rs *RegistryStorage) Update(obj interface{}) (<-chan interface{}, error) {
	srv := obj.(*api.Service)
	if errs := api.ValidateService(srv); len(errs) > 0 {
		return nil, apiserver.NewInvalidErr("service", srv.ID, errs)
	}
	return apiserver.MakeAsync(func() (interface{}, error) {
		// TODO: check to see if external load balancer status changed
		err := rs.registry.UpdateService(*srv)
		if err != nil {
			return nil, err
		}
		return rs.registry.GetService(srv.ID)
	}), nil
}

func (rs *RegistryStorage) deleteExternalLoadBalancer(service *api.Service) error {
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
	if err := balancer.DeleteTCPLoadBalancer(service.JSONBase.ID, zone.Region); err != nil {
		return err
	}
	return nil
}

func makeEnvVariableName(str string) string {
	return strings.ToUpper(strings.Replace(str, "-", "_", -1))
}

func makeLinkVariables(service api.Service, machine string) []api.EnvVar {
	prefix := makeEnvVariableName(service.ID)
	var port string
	if service.ContainerPort.Kind == util.IntstrString {
		port = service.ContainerPort.StrVal
	} else {
		port = strconv.Itoa(service.ContainerPort.IntVal)
	}
	portPrefix := prefix + "_PORT_" + makeEnvVariableName(port) + "_TCP"
	return []api.EnvVar{
		{
			Name:  prefix + "_PORT",
			Value: fmt.Sprintf("tcp://%s:%d", machine, service.Port),
		},
		{
			Name:  portPrefix,
			Value: fmt.Sprintf("tcp://%s:%d", machine, service.Port),
		},
		{
			Name:  portPrefix + "_PROTO",
			Value: "tcp",
		},
		{
			Name:  portPrefix + "_PORT",
			Value: strconv.Itoa(service.Port),
		},
		{
			Name:  portPrefix + "_ADDR",
			Value: machine,
		},
	}
}
