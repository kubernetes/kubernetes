/*
Copyright 2017 The Kubernetes Authors.

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

package testing

import (
	"fmt"

	utilipvs "k8s.io/kubernetes/pkg/util/ipvs"

)

// no-op implementation of ipvs Interface
type FakeIPVS struct {
	Scheduler    string
	Services     map[serviceKey]*utilipvs.Service
	Destinations map[serviceKey][]*utilipvs.Destination
}

type serviceKey struct {
	IP       string
	Port     uint16
	Protocol string
}

func (s *serviceKey) String() string {
	return fmt.Sprintf("%s:%d/%s", s.IP, s.Port, s.Protocol)
}

func NewFake() *FakeIPVS {
	return &FakeIPVS{
		Services: make(map[serviceKey]*utilipvs.Service),
		Destinations: make(map[serviceKey][]*utilipvs.Destination),
	}
}

func toServiceKey(serv *utilipvs.Service) serviceKey {
	return serviceKey{
		IP: serv.Address.To4().String(),
		Port: serv.Port,
		Protocol: serv.Protocol,
	}
}

func (*FakeIPVS) InitIpvsInterface() error {
	return nil
}

func (*FakeIPVS) CreateAliasDevice(aliasDev string) error {
	return nil
}

func (*FakeIPVS) DeleteAliasDevice(aliasDev string) error {
	return nil
}

func (*FakeIPVS) SetAlias(serv *utilipvs.Service) error {
	return nil
}

func (*FakeIPVS) UnSetAlias(serv *utilipvs.Service) error {
	return nil
}

func (f *FakeIPVS) AddService(serv *utilipvs.Service) error {
	if serv == nil {
		return fmt.Errorf("Failed to add service: service can't be nil")
	}
	key := toServiceKey(serv)
	f.Services[key] = serv
	// make sure no destination present when creating new service
	f.Destinations = make(map[serviceKey][]*utilipvs.Destination)
	return nil
}

func (f *FakeIPVS) UpdateService(serv *utilipvs.Service) error {
	if serv == nil {
		return fmt.Errorf("Failed to update service, service can't be nil")
	}
	return nil
}

func (f *FakeIPVS) DeleteService(serv *utilipvs.Service) error {
	if serv == nil {
		return fmt.Errorf("Failed to delete service: service can't be nil")
	}
	key := toServiceKey(serv)
	delete(f.Services, key)
	// clear specific destinations as well
	f.Destinations[key] = nil
	return nil
}

func (f *FakeIPVS) GetService(serv *utilipvs.Service) (*utilipvs.Service, error) {
	if serv == nil {
		return nil, fmt.Errorf("Failed to get service: service can't be nil")
	}
	key := toServiceKey(serv)
	svc, found := f.Services[key]
	if found {
		return svc, nil
	} else {
		return nil, fmt.Errorf("Not found serv: %v", key.String())
	}
}

func (f *FakeIPVS) GetServices() ([]*utilipvs.Service, error) {
	res := make([]*utilipvs.Service, 0)
	for _, svc := range f.Services {
		res = append(res, svc)
	}
	return res, nil
}

func (f *FakeIPVS) Flush() error {
	// directly drop old data
	f.Services = nil
	f.Destinations = nil
	return nil
}

func (*FakeIPVS) AddReloadFunc(reloadFunc func()) {}

func (*FakeIPVS) Destroy() {}

func (f *FakeIPVS) AddDestination(serv *utilipvs.Service, dest *utilipvs.Destination) error {
	if serv == nil || dest == nil {
		return fmt.Errorf("Failed to add destination for service, neither service nor destination shouldn't be nil")
	}
	key := toServiceKey(serv)
	if _, ok := f.Services[key]; !ok {
		return fmt.Errorf("Failed to add destination for service %v, service not found", key.String())
	}
	dests := f.Destinations[key]
	if dests == nil {
		dests = make([]*utilipvs.Destination, 0)
		f.Destinations[key] = dests
	}
	f.Destinations[key] = append(f.Destinations[key], dest)
	return nil
}

func (f *FakeIPVS) GetDestinations(serv *utilipvs.Service) ([]*utilipvs.Destination, error) {
	if serv == nil {
		return nil, fmt.Errorf("Failed to get destination for nil service")
	}
	key := toServiceKey(serv)
	if _, ok := f.Services[key]; !ok {
		return nil, fmt.Errorf("Failed to get destinations for service %v, service not found", key.String())
	}
	return f.Destinations[key], nil
}

func (*FakeIPVS) UpdateDestination(serv *utilipvs.Service, dest *utilipvs.Destination) error {
	if serv == nil || dest == nil {
		return fmt.Errorf("Failed to update destination, neither service nor destination can't be nil")
	}
	return nil
}

func (*FakeIPVS) DeleteDestination(serv *utilipvs.Service, dest *utilipvs.Destination) error {
	if serv == nil || dest == nil {
		return fmt.Errorf("Failed to delete destination, neither service nor destination can't be nil")
	}
	return nil
}

func (*FakeIPVS) CheckAliasDevice(string) error {
	return nil
}

var _ = utilipvs.Interface(&FakeIPVS{})