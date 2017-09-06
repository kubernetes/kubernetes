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
	Services     map[serviceKey]*utilipvs.VirtualServer
	Destinations map[serviceKey][]*utilipvs.RealServer
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
		Services:     make(map[serviceKey]*utilipvs.VirtualServer),
		Destinations: make(map[serviceKey][]*utilipvs.RealServer),
	}
}

func toServiceKey(serv *utilipvs.VirtualServer) serviceKey {
	return serviceKey{
		IP:       serv.Address.To4().String(),
		Port:     serv.Port,
		Protocol: serv.Protocol,
	}
}

//EnsureVirtualServerAddressBind is a fake implementation
func (*FakeIPVS) EnsureVirtualServerAddressBind(serv *utilipvs.VirtualServer, dev string) (exist bool, err error) {
	return true, nil
}

//UnbindVirtualServerAddress is a fake implementation
func (*FakeIPVS) UnbindVirtualServerAddress(serv *utilipvs.VirtualServer, dev string) error {
	return nil
}

//AddVirtualServer is a fake implementation
func (f *FakeIPVS) AddVirtualServer(serv *utilipvs.VirtualServer) error {
	if serv == nil {
		return fmt.Errorf("Failed to add service: service can't be nil")
	}
	key := toServiceKey(serv)
	f.Services[key] = serv
	// make sure no destination present when creating new service
	f.Destinations = make(map[serviceKey][]*utilipvs.RealServer)
	return nil
}

//UpdateVirtualServer is a fake implementation
func (f *FakeIPVS) UpdateVirtualServer(serv *utilipvs.VirtualServer) error {
	if serv == nil {
		return fmt.Errorf("Failed to update service, service can't be nil")
	}
	return nil
}

//DeleteVirtualServer is a fake implementation
func (f *FakeIPVS) DeleteVirtualServer(serv *utilipvs.VirtualServer) error {
	if serv == nil {
		return fmt.Errorf("Failed to delete service: service can't be nil")
	}
	key := toServiceKey(serv)
	delete(f.Services, key)
	// clear specific destinations as well
	f.Destinations[key] = nil
	return nil
}

//GetVirtualServer is a fake implementation
func (f *FakeIPVS) GetVirtualServer(serv *utilipvs.VirtualServer) (*utilipvs.VirtualServer, error) {
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

//GetVirtualServers is a fake implementation
func (f *FakeIPVS) GetVirtualServers() ([]*utilipvs.VirtualServer, error) {
	res := make([]*utilipvs.VirtualServer, 0)
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

//AddRealServer is a fake implementation
func (f *FakeIPVS) AddRealServer(serv *utilipvs.VirtualServer, dest *utilipvs.RealServer) error {
	if serv == nil || dest == nil {
		return fmt.Errorf("Failed to add destination for service, neither service nor destination shouldn't be nil")
	}
	key := toServiceKey(serv)
	if _, ok := f.Services[key]; !ok {
		return fmt.Errorf("Failed to add destination for service %v, service not found", key.String())
	}
	dests := f.Destinations[key]
	if dests == nil {
		dests = make([]*utilipvs.RealServer, 0)
		f.Destinations[key] = dests
	}
	f.Destinations[key] = append(f.Destinations[key], dest)
	return nil
}

//GetRealServers is a fake implementation
func (f *FakeIPVS) GetRealServers(serv *utilipvs.VirtualServer) ([]*utilipvs.RealServer, error) {
	if serv == nil {
		return nil, fmt.Errorf("Failed to get destination for nil service")
	}
	key := toServiceKey(serv)
	if _, ok := f.Services[key]; !ok {
		return nil, fmt.Errorf("Failed to get destinations for service %v, service not found", key.String())
	}
	return f.Destinations[key], nil
}

//DeleteRealServer is a fake implementation
func (*FakeIPVS) DeleteRealServer(serv *utilipvs.VirtualServer, dest *utilipvs.RealServer) error {
	if serv == nil || dest == nil {
		return fmt.Errorf("Failed to delete destination, neither service nor destination can't be nil")
	}
	return nil
}

var _ = utilipvs.Interface(&FakeIPVS{})
