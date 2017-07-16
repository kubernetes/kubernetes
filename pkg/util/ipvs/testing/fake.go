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

//FakeIPVS no-op implementation of ipvs Interface
type FakeIPVS struct {
	Scheduler    string
	Services     map[serviceKey]*utilipvs.InternalService
	Destinations map[serviceKey][]*utilipvs.InternalDestination
}

type serviceKey struct {
	IP       string
	Port     uint16
	Protocol string
}

func (s *serviceKey) String() string {
	return fmt.Sprintf("%s:%d/%s", s.IP, s.Port, s.Protocol)
}

//NewFake creates a fake ipvs strucuter
func NewFake() *FakeIPVS {
	return &FakeIPVS{
		Services:     make(map[serviceKey]*utilipvs.InternalService),
		Destinations: make(map[serviceKey][]*utilipvs.InternalDestination),
	}
}

func toServiceKey(serv *utilipvs.InternalService) serviceKey {
	return serviceKey{
		IP:       serv.Address.To4().String(),
		Port:     serv.Port,
		Protocol: serv.Protocol,
	}
}

//EnsureDummyDevice creates dummy device
func (*FakeIPVS) EnsureDummyDevice(dev string) (exist bool, err error) {
	return true, nil
}

//DeleteDummyDevice deletes a dummy device
func (*FakeIPVS) DeleteDummyDevice(dev string) error {
	return nil
}

//EnsureServiceAddressBind is a fake implementation
func (*FakeIPVS) EnsureServiceAddressBind(serv *utilipvs.InternalService, dev string) (exist bool, err error) {
	return true, nil
}

//UnBindServiceAddress is a fake implementation
func (*FakeIPVS) UnBindServiceAddress(serv *utilipvs.InternalService, dev string) error {
	return nil
}

//AddService is a fake implementation
func (f *FakeIPVS) AddService(serv *utilipvs.InternalService) error {
	if serv == nil {
		return fmt.Errorf("Failed to add service: service can't be nil")
	}
	key := toServiceKey(serv)
	f.Services[key] = serv
	// make sure no destination present when creating new service
	f.Destinations = make(map[serviceKey][]*utilipvs.InternalDestination)
	return nil
}

//UpdateService is a fake implementation
func (f *FakeIPVS) UpdateService(serv *utilipvs.InternalService) error {
	if serv == nil {
		return fmt.Errorf("Failed to update service, service can't be nil")
	}
	return nil
}

//DeleteService is a fake implementation
func (f *FakeIPVS) DeleteService(serv *utilipvs.InternalService) error {
	if serv == nil {
		return fmt.Errorf("Failed to delete service: service can't be nil")
	}
	key := toServiceKey(serv)
	delete(f.Services, key)
	// clear specific destinations as well
	f.Destinations[key] = nil
	return nil
}

//GetService is a fake implementation
func (f *FakeIPVS) GetService(serv *utilipvs.InternalService) (*utilipvs.InternalService, error) {
	if serv == nil {
		return nil, fmt.Errorf("Failed to get service: service can't be nil")
	}
	key := toServiceKey(serv)
	svc, found := f.Services[key]
	if found {
		return svc, nil
	}
	return nil, fmt.Errorf("Not found serv: %v", key.String())
}

//GetServices is a fake implementation
func (f *FakeIPVS) GetServices() ([]*utilipvs.InternalService, error) {
	res := make([]*utilipvs.InternalService, 0)
	for _, svc := range f.Services {
		res = append(res, svc)
	}
	return res, nil
}

//Flush is a fake implementation
func (f *FakeIPVS) Flush() error {
	// directly drop old data
	f.Services = nil
	f.Destinations = nil
	return nil
}

//AddDestination is a fake implementation
func (f *FakeIPVS) AddDestination(serv *utilipvs.InternalService, dest *utilipvs.InternalDestination) error {
	if serv == nil || dest == nil {
		return fmt.Errorf("Failed to add destination for service, neither service nor destination shouldn't be nil")
	}
	key := toServiceKey(serv)
	if _, ok := f.Services[key]; !ok {
		return fmt.Errorf("Failed to add destination for service %v, service not found", key.String())
	}
	dests := f.Destinations[key]
	if dests == nil {
		dests = make([]*utilipvs.InternalDestination, 0)
		f.Destinations[key] = dests
	}
	f.Destinations[key] = append(f.Destinations[key], dest)
	return nil
}

//GetDestinations is a fake implementation
func (f *FakeIPVS) GetDestinations(serv *utilipvs.InternalService) ([]*utilipvs.InternalDestination, error) {
	if serv == nil {
		return nil, fmt.Errorf("Failed to get destination for nil service")
	}
	key := toServiceKey(serv)
	if _, ok := f.Services[key]; !ok {
		return nil, fmt.Errorf("Failed to get destinations for service %v, service not found", key.String())
	}
	return f.Destinations[key], nil
}

//UpdateDestination is a fake implementation
func (*FakeIPVS) UpdateDestination(serv *utilipvs.InternalService, dest *utilipvs.InternalDestination) error {
	if serv == nil || dest == nil {
		return fmt.Errorf("Failed to update destination, neither service nor destination can't be nil")
	}
	return nil
}

//DeleteDestination is a fake implementation
func (*FakeIPVS) DeleteDestination(serv *utilipvs.InternalService, dest *utilipvs.InternalDestination) error {
	if serv == nil || dest == nil {
		return fmt.Errorf("Failed to delete destination, neither service nor destination can't be nil")
	}
	return nil
}

var _ = utilipvs.Interface(&FakeIPVS{})
